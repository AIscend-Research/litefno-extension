r"""Are the confidence scores honest? Reliability, ECE and MCE (ext29)

Board task: "Core novel metric 2: Uncertainty Calibration -- reliability diagrams
(confidence vs. actual accuracy); measure expected calibration error (ECE) and
max calibration error (MCE) to show the confidence scores are honest."

ext28 established that ensemble disagreement *ranks* the surrogate's errors
almost exactly as well as an oracle reading the true error does. Ranking is
discrimination, and discrimination is indifferent to scale: a signal that
reports every uncertainty at a tenth of its true size ranks identically to one
that reports it correctly. This asks the other question. When the ensemble says
"plus or minus sigma", is sigma the truth?

The two are independent, and the pairing is the point. A perfectly ranked but
badly scaled signal is exactly what you get from a small deep ensemble, and it
is dangerous in a way a badly ranked one is not -- it is trusted, and it is
wrong by a factor nobody measured.

Reliability for a regression
----------------------------
A classifier's reliability diagram bins predictions by confidence and asks
whether 70%-confident predictions are right 70% of the time. This repo predicts
fields, so the same question is asked two ways, both reported.

**Quantile calibration** (the standard regression analog). Treat the ensemble as
a Gaussian predictive distribution with mean mu and standard deviation sigma. For
a nominal central coverage p, the interval is `mu +/- z * sigma` with
`z = Phi^-1((1+p)/2)`. A calibrated model has the truth land inside that interval
p of the time. Plotting observed against nominal coverage is the reliability
diagram; the gap between the curve and the diagonal is the calibration error.

    ECE = mean over nominal levels of |observed - nominal|
    MCE = max  over nominal levels of |observed - nominal|

**Error-vs-sigma calibration** (the literal reading, "confidence vs accuracy").
Bin predictions by the sigma they claim, and compare the sigma each bin claims to
the RMSE it actually achieves. A calibrated bin sits on the diagonal. Reported
normalised by the overall RMSE so it is scale-free and comparable across splits.

Granularity, and why it is per pixel
------------------------------------
Calibration is a property of individual predictions, so each pixel of each
channel of each step is one prediction with its own (mu, sigma, y). A held-out
fold of 118 steps is 118 * 2 * 32 * 32 = 241664 predictions, which is enough to
estimate a coverage curve; 118 would not be.

The Gaussian assumption is doing real work here and is stated rather than hidden.
With four members the ensemble spread is a noisy variance estimate, and the
sample standard deviation of four numbers is biased low even when the variance
estimate is unbiased. Both push the same way -- toward apparent overconfidence --
so a finding of overconfidence must be read with that in mind. A finding of
*under*confidence would be the surprising one.

The comparison that matters
---------------------------
Calibration is measured on both splits from the same models: the seen regimes
(in-distribution) and the held-out regime (out-of-distribution). Confidence
scores that are honest in-distribution and dishonest off it are the common case
and the dangerous one, and reporting only one split would hide it.

Then the obvious repair: fit a single scale factor `s` on the in-distribution
split, minimising its calibration error, and apply it unchanged to the held-out
split. If one constant fixes both, the miscalibration is a scale bug and cheap
to fix. If it fixes in-distribution and not held-out, the dishonesty is a
property of the shift, and no amount of in-distribution recalibration reaches it.
The scale is fit where a practitioner could actually fit it -- on data from the
regimes they have -- not on the held-out regime, which would be assuming the
answer.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_BR_PATH = Path(__file__).resolve().parent / "baseline_reference.py"


def _load_br():
    spec = importlib.util.spec_from_file_location("baseline_reference", _BR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["baseline_reference"] = module
    spec.loader.exec_module(module)
    return module


br = _load_br()
import torch                                        # noqa: E402
from torch import nn                                # noqa: E402

# nominal coverage levels for the reliability diagram
LEVELS = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.80, 0.90, 0.95, 0.99])


# --------------------------------------------------------------------------
# regime bookkeeping (same manifest contract as cross_regime.py)
# --------------------------------------------------------------------------


def regime_labels(manifest: dict, split: str) -> np.ndarray:
    labels = []
    for entry in manifest["splits"][split]["files"]:
        labels.extend([entry["regime"]] * len(entry["trajectories"]))
    return np.array(labels)


def load_dataset(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    out = {}
    for split in ("train", "test"):
        arr = br.load_split(data_dir / f"{split}.h5")
        lab = regime_labels(manifest, split)
        assert len(lab) == arr.shape[0], (split, len(lab), arr.shape)
        out[split] = (arr, lab)
    return out, manifest


# --------------------------------------------------------------------------
# the ensemble's predictive distribution
# --------------------------------------------------------------------------


def train_member(xtr, ytr, device: str, epochs: int, seed: int,
                 log_every: int = 20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, _ = br.build_model("litefno", xtr.shape[1], xtr.shape[1])
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=br.LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=br.LR_STEP,
                                            gamma=br.LR_GAMMA)
    loss_fn = nn.MSELoss()
    n, t0 = len(xtr), time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, br.BATCH):
            idx = perm[i:i + br.BATCH]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            print(f"        epoch {epoch + 1}/{epochs} ({time.time() - t0:.0f}s)",
                  flush=True)
    return model


def predictive(models: list, x, y, device: str, batch: int = 64):
    """Per-pixel (residual, sigma) over the whole split.

    ``sigma`` is the ensemble's spread, computed from predictions only. The
    unbiased (Bessel-corrected) variance is used; with four members its square
    root is still biased low, which the module docstring flags as pushing toward
    apparent overconfidence.
    """
    for m in models:
        m.eval()
    residuals, sigmas = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(device)
            yb = y[i:i + batch].to(device)
            preds = torch.stack([m(xb) for m in models])       # (M, B, C, H, W)
            mu = preds.mean(dim=0)
            sd = preds.std(dim=0, unbiased=True)
            residuals.append((yb - mu).flatten().cpu())
            sigmas.append(sd.flatten().cpu())
    return (torch.cat(residuals).numpy().astype(np.float64),
            torch.cat(sigmas).numpy().astype(np.float64))


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def coverage_curve(residual: np.ndarray, sigma: np.ndarray,
                   scale: float = 1.0, levels: np.ndarray = LEVELS) -> list[dict]:
    """Observed vs nominal central coverage under a Gaussian predictive law.

    A sigma of exactly zero would make every interval degenerate; those pixels
    are counted as covered only when the residual is also zero, which is the
    honest reading of "the model claims no uncertainty at all".
    """
    from scipy.stats import norm
    s = sigma * scale
    out = []
    for p in levels:
        z = norm.ppf((1.0 + p) / 2.0)
        half = z * s
        covered = np.abs(residual) <= half
        # degenerate intervals: claimed certainty is only right if it was right
        degenerate = half <= 0
        covered = np.where(degenerate, residual == 0, covered)
        obs = float(covered.mean())
        out.append({"nominal": float(p), "observed": obs,
                    "signed_gap": obs - float(p),
                    "abs_gap": abs(obs - float(p))})
    return out


def ece_mce(curve: list[dict]) -> tuple[float, float]:
    """Expected and maximum calibration error over the nominal levels."""
    gaps = np.array([c["abs_gap"] for c in curve], dtype=float)
    if not len(gaps):
        return float("nan"), float("nan")
    return float(gaps.mean()), float(gaps.max())


def fit_scale(residual: np.ndarray, sigma: np.ndarray,
              levels: np.ndarray = LEVELS) -> float:
    """One constant multiplying sigma, chosen to minimise ECE.

    A coarse-to-fine scan rather than an optimiser: ECE is piecewise constant in
    the scale (coverage only changes when a pixel crosses an interval edge), so
    gradient methods have nothing to descend.
    """
    best, best_ece = 1.0, float("inf")
    grid = np.geomspace(0.05, 20.0, 60)
    for _ in range(2):
        for s in grid:
            e, _ = ece_mce(coverage_curve(residual, sigma, float(s), levels))
            if e < best_ece:
                best_ece, best = e, float(s)
        lo, hi = best / 1.5, best * 1.5
        grid = np.geomspace(lo, hi, 60)
    return best


def sigma_bins(residual: np.ndarray, sigma: np.ndarray, n_bins: int = 12,
               scale: float = 1.0) -> list[dict]:
    """Claimed sigma against achieved RMSE, in equal-count bins.

    The literal "confidence vs accuracy" diagram: each bin claims a typical
    sigma and achieves a typical error, and calibration means those agree.
    """
    s = sigma * scale
    order = np.argsort(s, kind="stable")
    chunks = np.array_split(order, n_bins)
    overall = float(np.sqrt((residual ** 2).mean()))
    out = []
    for k, idx in enumerate(chunks):
        if not len(idx):
            continue
        claimed = float(s[idx].mean())
        achieved = float(np.sqrt((residual[idx] ** 2).mean()))
        out.append({
            "bin": k, "n": int(len(idx)),
            "claimed_sigma": claimed, "achieved_rmse": achieved,
            "ratio": achieved / claimed if claimed > 0 else float("nan"),
            "gap_normalised": (achieved - claimed) / overall if overall else float("nan"),
        })
    return out


def sigma_ece_mce(bins: list[dict]) -> tuple[float, float]:
    """Count-weighted mean and max |claimed - achieved|, normalised by RMSE."""
    if not bins:
        return float("nan"), float("nan")
    n = np.array([b["n"] for b in bins], dtype=float)
    g = np.abs(np.array([b["gap_normalised"] for b in bins], dtype=float))
    return float((n * g).sum() / n.sum()), float(g.max())


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_curve(name: str, curve: list[dict]) -> None:
    e, m = ece_mce(curve)
    print(f"\n    {name}   ECE {e:.3f}   MCE {m:.3f}")
    hdr = f"      {'nominal':>8s} {'observed':>9s} {'gap':>8s}"
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))
    for c in curve:
        print(f"      {c['nominal']:>7.0%} {c['observed']:>9.1%} "
              f"{c['signed_gap']:>+8.1%}")


def print_report(res: dict, held: str) -> None:
    print(f"\n=== Uncertainty calibration | held out: {held} ===")
    print("    a calibrated model's observed coverage matches its nominal "
          "coverage;\n    observed below nominal means the intervals are too "
          "narrow -- overconfident")

    for split in ("seen", "held"):
        label = ("in-distribution (seen regimes)" if split == "seen"
                 else f"out-of-distribution (held-out {held})")
        print_curve(label, res[split]["curve"])

    print("\n=== Confidence vs actual accuracy (equal-count sigma bins) ===")
    for split in ("seen", "held"):
        bins = res[split]["bins"]
        e, m = sigma_ece_mce(bins)
        label = "in-distribution" if split == "seen" else "out-of-distribution"
        print(f"\n    {label}   sigma-ECE {e:.3f}   sigma-MCE {m:.3f}   "
              f"(normalised by RMSE)")
        hdr = (f"      {'bin':>4s} {'claimed':>11s} {'achieved':>11s} "
               f"{'achieved/claimed':>17s}")
        print(hdr)
        print("      " + "-" * (len(hdr) - 6))
        for b in bins:
            print(f"      {b['bin']:>4d} {b['claimed_sigma']:>11.3e} "
                  f"{b['achieved_rmse']:>11.3e} {b['ratio']:>17.1f}")

    print("\n=== Does one scale factor fix it? ===")
    s = res["scale"]
    print(f"    scale fit on in-distribution data only: sigma *= {s:.2f}")
    hdr = (f"      {'split':>20s} {'ECE before':>11s} {'ECE after':>10s} "
           f"{'MCE before':>11s} {'MCE after':>10s}")
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))
    for split in ("seen", "held"):
        label = "in-distribution" if split == "seen" else "out-of-distribution"
        e0, m0 = ece_mce(res[split]["curve"])
        e1, m1 = ece_mce(res[split]["curve_scaled"])
        print(f"      {label:>20s} {e0:>11.3f} {e1:>10.3f} {m0:>11.3f} "
              f"{m1:>10.3f}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def plot(res: dict, held: str, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping plot: {exc}")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for split, style, lbl in (("seen", "o-", "in-distribution"),
                              ("held", "s-", f"held out: {held}")):
        c = res[split]["curve"]
        ax1.plot([x["nominal"] for x in c], [x["observed"] for x in c],
                 style, label=lbl)
        cs = res[split]["curve_scaled"]
        ax1.plot([x["nominal"] for x in cs], [x["observed"] for x in cs],
                 style, alpha=0.4, label=f"{lbl}, rescaled")
    ax1.set_xlabel("nominal coverage")
    ax1.set_ylabel("observed coverage")
    ax1.set_title("Reliability diagram")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for split, style, lbl in (("seen", "o-", "in-distribution"),
                              ("held", "s-", f"held out: {held}")):
        b = res[split]["bins"]
        ax2.plot([x["claimed_sigma"] for x in b],
                 [x["achieved_rmse"] for x in b], style, label=lbl)
    lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
            max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims, lims, "k--", lw=1, label="perfect")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("claimed sigma")
    ax2.set_ylabel("achieved RMSE")
    ax2.set_title("Confidence vs actual accuracy")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/processed/gray_scott_streamed"))
    ap.add_argument("--held-out", default="bubbles")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/extensions"))
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    data, _ = load_dataset(args.data_dir)
    train_arr, train_lab = data["train"]
    test_arr, test_lab = data["test"]

    xtr, ytr = br.to_pairs(train_arr[train_lab != args.held_out])
    x_held, y_held = br.to_pairs(test_arr[test_lab == args.held_out])
    x_seen, y_seen = br.to_pairs(test_arr[test_lab != args.held_out])
    print(f"device={device}  held out {args.held_out}  "
          f"{args.members} members x {args.epochs} epochs")

    models = []
    for k in range(args.members):
        print(f"      member {k + 1}/{args.members}", flush=True)
        models.append(train_member(xtr, ytr, device, args.epochs,
                                   args.seed + 1000 * k))

    r_seen, s_seen = predictive(models, x_seen, y_seen, device)
    r_held, s_held = predictive(models, x_held, y_held, device)
    print(f"  predictions: {len(r_seen)} in-distribution, {len(r_held)} held-out")

    scale = fit_scale(r_seen, s_seen)
    res = {
        "seen": {"curve": coverage_curve(r_seen, s_seen),
                 "curve_scaled": coverage_curve(r_seen, s_seen, scale),
                 "bins": sigma_bins(r_seen, s_seen, args.bins)},
        "held": {"curve": coverage_curve(r_held, s_held),
                 "curve_scaled": coverage_curve(r_held, s_held, scale),
                 "bins": sigma_bins(r_held, s_held, args.bins)},
        "scale": scale,
    }
    print_report(res, args.held_out)

    rows = []
    for split in ("seen", "held"):
        for key in ("curve", "curve_scaled"):
            for c in res[split][key]:
                rows.append({"held_out": args.held_out, "split": split,
                             "rescaled": key.endswith("scaled"),
                             "scale": scale if key.endswith("scaled") else 1.0,
                             **c})
    write_csv(args.out_dir / f"ext29_reliability_{args.held_out}.csv", rows)

    brows = []
    for split in ("seen", "held"):
        for b in res[split]["bins"]:
            brows.append({"held_out": args.held_out, "split": split, **b})
    write_csv(args.out_dir / f"ext29_sigma_bins_{args.held_out}.csv", brows)

    summary = []
    for split in ("seen", "held"):
        e0, m0 = ece_mce(res[split]["curve"])
        e1, m1 = ece_mce(res[split]["curve_scaled"])
        se, sm = sigma_ece_mce(res[split]["bins"])
        summary.append({"held_out": args.held_out, "split": split,
                        "members": args.members, "epochs": args.epochs,
                        "seed": args.seed, "scale": scale,
                        "ece": e0, "mce": m0,
                        "ece_rescaled": e1, "mce_rescaled": m1,
                        "sigma_ece": se, "sigma_mce": sm})
    write_csv(args.out_dir / f"ext29_summary_{args.held_out}.csv", summary)
    plot(res, args.held_out, args.fig_dir / f"ext29_calibration_{args.held_out}.png")


if __name__ == "__main__":
    sys.exit(main())
