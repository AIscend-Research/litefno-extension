r"""Safe Deferral Rate: buying back out-of-distribution accuracy by abstaining (ext28)

Board task: "Core novel metric 1: Safe Deferral Rate -- plot
accuracy/sensitivity/specificity as a function of confidence threshold T; show
that with T tuned, you can recover near in-distribution accuracy by deferring
just the ambiguous cases."

ext26 measured the price of leaving the training distribution: a LiteFNO trained
on five Gray-Scott regimes is worse on the sixth. This asks the operational
follow-up. If the surrogate could tell *which* of its own predictions are the
bad ones, it could hand those back to the real solver and keep the rest -- and
the question becomes how few it has to hand back before what it keeps is as good
as in-distribution work.

That framing is what makes deferral cost something here. Deferring is not free
abstention: a deferred step means running the ground-truth simulation, which is
the expense the surrogate existed to avoid. So the metric is a rate, not a
score. The headline, the **Safe Deferral Rate**, is the smallest fraction of
held-out steps that must be deferred before the retained ones match
in-distribution error.

Regression, not classification
------------------------------
Sensitivity and specificity need a binary event, and this repo predicts fields.
The event used here is "this prediction would have been unacceptably wrong":
a held-out step is *unsafe* when its per-sample error exceeds a tolerance tau,
and tau is set from the in-distribution error distribution rather than picked --
it is the ``--tau-quantile`` quantile of the seen regimes' per-sample errors, so
"unacceptable" means "worse than in-distribution work almost ever is".

    sensitivity = P(deferred | unsafe)     the bad steps we caught
    specificity = P(kept     | safe)       the good steps we didn't waste

Both are needed because either alone is trivial to win: deferring everything
scores perfect sensitivity, deferring nothing scores perfect specificity.

The confidence signal
---------------------
Ensemble disagreement -- the spread of the members' predictions at that step,
which is what ext26's `robust+unc` arm already builds. No extra model is
trained to estimate it and no label is used to compute it, so it is available at
deployment time, which a signal derived from the true error would not be.

Two controls, because the curve is meaningless alone
----------------------------------------------------
A deferral curve that only shows "error falls as we defer more" proves nothing:
it has to be beaten against what deferring *without* a signal would do, and
measured against what a perfect signal could do.

    random    defer a uniformly random subset. This is the null. Under the
              fixed normalisation below it is flat in expectation, so any real
              descent in the confidence curve is the signal doing work.
    oracle    defer the genuinely worst steps, ranked by true error. Not
              achievable -- it reads the answer -- but it bounds how much of the
              available gain the confidence signal actually captures.

A normalisation that would otherwise fake the result
----------------------------------------------------
VRMSE divides by the variance of the target. Recomputing that variance on each
retained subset would make the denominator move with the deferral rate, and the
curve would then partly measure which targets got dropped rather than how good
the kept predictions are -- a subset can look better purely by retaining
lower-variance fields.

So every retained score is normalised by one fixed variance, the full held-out
target variance, computed once before any deferral:

    score(q) = sqrt( mean_{kept} per_sample_mse / V_full )

At q = 0 this is exactly the standard VRMSE, so the curve starts at ext26's
number and stays comparable to it all the way along. The test suite pins both
properties.
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
# ensemble
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


def ensemble_stats(models: list, x, y, device: str, batch: int = 128):
    """Per-sample error of the ensemble mean, and per-sample disagreement.

    Returns ``(mse, disagreement, target_variance)``:

      mse            per-sample mean squared error of the ensemble mean
      disagreement   per-sample std across members, averaged over the field.
                     Computed from predictions only -- no targets -- so it is
                     a signal a deployed model would actually have.
      target_variance  one number for the whole split, the fixed denominator
    """
    for m in models:
        m.eval()
    mses, disagreements = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(device)
            yb = y[i:i + batch].to(device)
            preds = torch.stack([m(xb) for m in models])      # (M, B, C, H, W)
            mean = preds.mean(dim=0)
            flat = tuple(range(1, mean.dim()))
            mses.append(((mean - yb) ** 2).mean(dim=flat).cpu())
            if len(models) > 1:
                spread = preds.std(dim=0, unbiased=True)
            else:
                spread = torch.zeros_like(mean)
            disagreements.append(spread.mean(dim=flat).cpu())
    return (torch.cat(mses).numpy(),
            torch.cat(disagreements).numpy(),
            float(y.var()))


# --------------------------------------------------------------------------
# the deferral curve
# --------------------------------------------------------------------------


def retained_score(mse: np.ndarray, keep: np.ndarray, variance: float) -> float:
    """VRMSE over the kept samples, normalised by a FIXED variance.

    Returns NaN when nothing is kept -- an empty retained set has no error, and
    reporting 0.0 there would draw a curve that plunges to perfection at 100%
    deferral, which is the opposite of what deferring everything means.
    """
    if keep.sum() == 0:
        return float("nan")
    return float(np.sqrt(mse[keep].mean() / variance))


def deferral_curve(mse: np.ndarray, confidence: np.ndarray, variance: float,
                   reference: float, tau: float,
                   fractions: np.ndarray) -> list[dict]:
    """Sweep the deferral fraction; report accuracy and detection at each.

    ``confidence`` is a score where *higher means more doubtful* (ensemble
    disagreement). At deferral fraction q the threshold T is the (1-q) quantile
    of that score, and everything above T is deferred -- so sweeping q and
    sweeping T are the same sweep, reported by the axis a reader can act on.
    """
    unsafe = mse > tau
    rows = []
    for q in fractions:
        if q <= 0:
            thresh, defer = float("inf"), np.zeros(len(mse), dtype=bool)
        else:
            thresh = float(np.quantile(confidence, 1.0 - q))
            defer = confidence > thresh
            # ties at the threshold would otherwise silently defer more (or
            # fewer) than q; take the q most doubtful by rank instead
            if defer.sum() != int(round(q * len(mse))):
                k = int(round(q * len(mse)))
                order = np.argsort(-confidence, kind="stable")
                defer = np.zeros(len(mse), dtype=bool)
                defer[order[:k]] = True
                thresh = (float(confidence[order[k - 1]]) if k > 0
                          else float("inf"))
        keep = ~defer
        tp = int((defer & unsafe).sum())
        fn = int((keep & unsafe).sum())
        tn = int((keep & ~unsafe).sum())
        fp = int((defer & ~unsafe).sum())
        rows.append({
            "deferral_rate": float(defer.mean()),
            "threshold_T": thresh,
            "retained_vrmse": retained_score(mse, keep, variance),
            "reference_vrmse": reference,
            "n_kept": int(keep.sum()), "n_deferred": int(defer.sum()),
            "n_unsafe": int(unsafe.sum()),
            "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
            "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        })
    return rows


def safe_deferral_rate(rows: list[dict]) -> float:
    """Smallest deferral rate whose retained error reaches in-distribution.

    NaN if the sweep never gets there -- reporting the best achieved rate
    instead would read as success at whatever the sweep happened to stop at.
    """
    for r in sorted(rows, key=lambda r: r["deferral_rate"]):
        if np.isfinite(r["retained_vrmse"]) and \
                r["retained_vrmse"] <= r["reference_vrmse"]:
            return r["deferral_rate"]
    return float("nan")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_report(held: str, curves: dict, reference: float, base: float,
                 tau: float, n_unsafe: int, n_total: int) -> None:
    print(f"\n=== Safe Deferral Rate | held out: {held} ===")
    print(f"    in-distribution (seen regimes) VRMSE : {reference:.5f}")
    print(f"    held-out VRMSE, no deferral          : {base:.5f}  "
          f"({base / reference:.2f}x)")
    print(f"    unsafe tolerance tau                 : {tau:.3e}  "
          f"({n_unsafe}/{n_total} held-out steps = {n_unsafe / n_total:.0%})")

    rows = curves["confidence"]
    hdr = (f"    {'defer':>7s} {'T':>10s} {'retained':>10s} {'vs in-dist':>11s} "
           f"{'sens':>7s} {'spec':>7s}")
    print("\n    signal: ensemble disagreement")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in rows:
        ratio = (r["retained_vrmse"] / reference
                 if np.isfinite(r["retained_vrmse"]) else float("nan"))
        rv = ("       n/a" if not np.isfinite(r["retained_vrmse"])
              else f"{r['retained_vrmse']:>10.5f}")
        rr = "        n/a" if not np.isfinite(ratio) else f"{ratio:>10.2f}x"
        print(f"    {r['deferral_rate']:>6.0%} {r['threshold_T']:>10.3e} {rv} "
              f"{rr} {r['sensitivity']:>7.2f} {r['specificity']:>7.2f}")

    print("\n    Safe Deferral Rate (first rate reaching in-distribution error)")
    for name in ("oracle", "confidence", "random"):
        sdr = safe_deferral_rate(curves[name])
        label = {"oracle": "oracle (upper bound, reads the answer)",
                 "confidence": "ensemble disagreement (deployable)",
                 "random": "random (null: no signal)"}[name]
        got = "never reached" if not np.isfinite(sdr) else f"{sdr:.0%}"
        print(f"      {got:>14s}   {label}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def plot(curves: dict, reference: float, held: str, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping plot: {exc}")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    style = {"oracle": ("--", "oracle (upper bound)"),
             "confidence": ("o-", "ensemble disagreement"),
             "random": (":", "random (null)")}
    for name, (fmt, label) in style.items():
        rows = curves[name]
        x = [r["deferral_rate"] * 100 for r in rows]
        ax1.plot(x, [r["retained_vrmse"] for r in rows], fmt, label=label)
    ax1.axhline(reference, color="k", lw=1, label="in-distribution")
    ax1.set_xlabel("deferral rate (%)")
    ax1.set_ylabel("retained VRMSE (fixed normalisation)")
    ax1.set_title(f"Accuracy vs deferral | held out: {held}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    rows = curves["confidence"]
    x = [r["deferral_rate"] * 100 for r in rows]
    ax2.plot(x, [r["sensitivity"] for r in rows], "o-", label="sensitivity")
    ax2.plot(x, [r["specificity"] for r in rows], "s-", label="specificity")
    ax2.set_xlabel("deferral rate (%)")
    ax2.set_ylabel("rate")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("Detection of unsafe steps")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/processed/gray_scott_streamed"))
    ap.add_argument("--held-out", default="maze",
                    help="regime to treat as out-of-distribution")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-quantile", type=float, default=0.95,
                    help="tolerance for 'unsafe', as a quantile of the "
                         "in-distribution per-sample error")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/extensions"))
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    data, _ = load_dataset(args.data_dir)
    train_arr, train_lab = data["train"]
    test_arr, test_lab = data["test"]

    seen_mask = train_lab != args.held_out
    xtr, ytr = br.to_pairs(train_arr[seen_mask])
    x_held, y_held = br.to_pairs(test_arr[test_lab == args.held_out])
    x_seen, y_seen = br.to_pairs(test_arr[test_lab != args.held_out])
    print(f"device={device}  held out {args.held_out}  "
          f"{args.members} members x {args.epochs} epochs")
    print(f"  train {len(xtr)} pairs, held-out {len(x_held)}, seen {len(x_seen)}")

    models = []
    for k in range(args.members):
        print(f"      member {k + 1}/{args.members}", flush=True)
        models.append(train_member(xtr, ytr, device, args.epochs,
                                   args.seed + 1000 * k))

    mse_h, conf_h, var_h = ensemble_stats(models, x_held, y_held, device)
    mse_s, _, var_s = ensemble_stats(models, x_seen, y_seen, device)
    reference = float(np.sqrt(mse_s.mean() / var_s))
    base = float(np.sqrt(mse_h.mean() / var_h))
    tau = float(np.quantile(mse_s, args.tau_quantile))

    fractions = np.arange(0.0, 0.85, 0.05)
    rng = np.random.default_rng(args.seed)
    curves = {
        # higher = more doubtful, so the oracle ranks by true error and random
        # ranks by noise. All three use the identical sweep machinery.
        "confidence": deferral_curve(mse_h, conf_h, var_h, reference, tau,
                                     fractions),
        "oracle": deferral_curve(mse_h, mse_h, var_h, reference, tau, fractions),
        "random": deferral_curve(mse_h, rng.random(len(mse_h)), var_h,
                                 reference, tau, fractions),
    }

    print_report(args.held_out, curves, reference, base, tau,
                 int((mse_h > tau).sum()), len(mse_h))

    rows = []
    for name, cur in curves.items():
        for r in cur:
            rows.append({"held_out": args.held_out, "signal": name,
                         "members": args.members, "epochs": args.epochs,
                         "seed": args.seed, **r})
    # the held-out regime is part of the filename: which fold a deferral curve
    # came from changes what it means, and a fold with no gap (maze, 0.94x)
    # must not silently overwrite one that has a gap
    write_csv(args.out_dir / f"ext28_deferral_{args.held_out}.csv", rows)
    write_csv(args.out_dir / f"ext28_summary_{args.held_out}.csv", [{
        "held_out": args.held_out, "signal": name,
        "safe_deferral_rate": safe_deferral_rate(cur),
        "reference_vrmse": reference, "base_held_vrmse": base,
        "gap_ratio": base / reference, "tau": tau,
        "tau_quantile": args.tau_quantile,
    } for name, cur in curves.items()])
    plot(curves, reference, args.held_out,
         args.fig_dir / f"ext28_deferral_{args.held_out}.png")


if __name__ == "__main__":
    sys.exit(main())
