r"""Data-efficiency curve: accuracy vs training-set size, harmonic vs plain (ext30)

Roadmap task: "Add the data-efficiency curve: accuracy vs training-set size,
harmonic vs plain."

ext15 built the harmonic-conditioned arm and its control and measured them at one
training-set size -- or rather, it built them and was never run at all. The
question this asks is different from "is the conditioned arm better": it is
whether the conditioning buys *data*, which is the quantity a low-resource
scientist actually spends.

An architectural prior that supplies structure the data would otherwise have to
teach should show up as a leftward shift of the error-vs-size curve, not merely
as a lower point on it. So the headline is a **data multiplier**: how much
training data the plain arm needs to reach the accuracy the harmonic arm reaches
at a given size. A multiplier of 2 means the prior is worth doubling the dataset.

The two arms
------------
    plain        CP-factorized spectral LiteFNO
    harmonic     the same, plus a learnable complex bias on the Turing shells

They are bit-identical at initialisation -- the bias starts at zero -- so at a
given seed the two runs share initialisation, subset and data order, and the
comparison is paired. That matters more than usual here, because the repo's own
prior (ext9, ext12, and ext15's own docstring) says the effect should be small,
and an unpaired comparison at this scale would be swamped by seed variance.

Sizes are counted in trajectories, balanced across regimes
----------------------------------------------------------
Training-set size is varied by *trajectory*, not by frame. Frames within one
trajectory are consecutive states of the same system and are nowhere near
independent, so subsampling frames would shrink the nominal size far more than
the information.

The subset is balanced across the six regimes -- k trajectories from each -- for
the reason ext10 established: the regimes differ by two orders of magnitude in
where they keep their variance, so an unbalanced draw would confound size with
composition, and a small unbalanced subset could omit a regime entirely.

Which trajectories are drawn varies with the seed, so the curve is not hostage to
one particular subset, and both arms at a given seed get the identical draw.

Reading the multiplier honestly
-------------------------------
The multiplier is obtained by interpolating the plain arm's curve to find the
size at which it would match the harmonic arm's error. Interpolation is done in
log(size), which is the axis the curve is closer to linear in.

It is reported **only where the target error falls inside the measured range of
the plain curve**. Extrapolating past the largest size measured would invent a
multiplier out of the curve's slope, which is precisely the number a reader would
most want to trust and least be able to check. Out-of-range cases are reported as
such rather than as a number.
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
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_br():
    spec = importlib.util.spec_from_file_location("baseline_reference", _BR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["baseline_reference"] = module
    spec.loader.exec_module(module)
    return module


br = _load_br()
import torch                                          # noqa: E402
from torch import nn                                  # noqa: E402

from litefno.models.harmonic import HarmonicLiteFNO   # noqa: E402

ARMS = ("plain", "harmonic")


# --------------------------------------------------------------------------
# data
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


def balanced_subset(labels: np.ndarray, per_regime: int, seed: int) -> np.ndarray:
    """Indices of ``per_regime`` trajectories from each regime.

    Balanced by construction: a size is the same composition at every point on
    the curve, so the curve measures size rather than which regimes got lucky.
    The draw depends on the seed, and both arms at that seed share it.
    """
    rng = np.random.default_rng(seed)
    picked = []
    for regime in dict.fromkeys(labels):
        idx = np.flatnonzero(labels == regime)
        take = min(per_regime, len(idx))
        picked.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(picked))


# --------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------


def build(arm: str, in_ch: int, cfg: dict):
    torch.manual_seed(cfg["seed"])
    return HarmonicLiteFNO(
        in_ch, in_ch, width=cfg["width"], modes=cfg["modes"],
        layers=cfg["layers"], rank=cfg["rank"],
        harmonic_bias=(arm == "harmonic"),
        fundamental=cfg["fundamental"], n_harmonics=cfg["n_harmonics"])


def per_regime_vrmse(model, arr, labels, device) -> dict:
    out = {}
    for regime in dict.fromkeys(labels):
        x, y = br.to_pairs(arr[labels == regime])
        out[regime] = br.evaluate_one_step(model, x, y, device)
    return out


def train_one(arm: str, cfg: dict, xtr, ytr, xte, yte, device: str,
              log_every: int = 30):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = build(arm, xtr.shape[1], cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=br.LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=br.LR_STEP,
                                            gamma=br.LR_GAMMA)
    loss_fn = nn.MSELoss()

    n, t0 = len(xtr), time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, br.BATCH):
            idx = perm[i:i + br.BATCH]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        if (epoch + 1) % log_every == 0 or epoch == cfg["epochs"] - 1:
            print(f"        epoch {epoch + 1}/{cfg['epochs']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return model, params, round(time.time() - t0, 1)


# --------------------------------------------------------------------------
# the multiplier
# --------------------------------------------------------------------------


def data_multiplier(curve: list[dict], target_error: float,
                    at_size: int) -> dict:
    """How much data the plain arm needs to reach ``target_error``.

    ``curve`` is the plain arm's (size, error) points, which must be sorted and
    are assumed monotonically decreasing in error; interpolation is linear in
    log(size).

    Returns a dict whose ``status`` is one of:
      ok            the target falls between two measured sizes
      out_of_range  the plain arm never reaches the target within the sizes
                    measured -- reported, not extrapolated
      degenerate    fewer than two usable points
    """
    pts = sorted([(c["n_traj"], c["vrmse"]) for c in curve
                  if np.isfinite(c["vrmse"])])
    if len(pts) < 2:
        return {"status": "degenerate", "multiplier": float("nan"),
                "equivalent_n": float("nan")}
    sizes = np.array([p[0] for p in pts], dtype=float)
    errs = np.array([p[1] for p in pts], dtype=float)

    if target_error >= errs[0]:
        # the smallest measured plain run is already at or below the target
        return {"status": "out_of_range_low", "multiplier": float("nan"),
                "equivalent_n": float("nan")}
    if target_error < errs[-1]:
        # the plain arm never gets this good within the sizes measured
        return {"status": "out_of_range_high", "multiplier": float("nan"),
                "equivalent_n": float("nan")}

    for (s0, e0), (s1, e1) in zip(pts, pts[1:]):
        lo, hi = min(e0, e1), max(e0, e1)
        if lo <= target_error <= hi:
            if e0 == e1:
                equiv = float(s0)
            else:
                frac = (e0 - target_error) / (e0 - e1)
                equiv = float(np.exp(np.log(s0)
                                     + frac * (np.log(s1) - np.log(s0))))
            return {"status": "ok", "equivalent_n": equiv,
                    "multiplier": equiv / at_size if at_size else float("nan")}
    return {"status": "out_of_range_high", "multiplier": float("nan"),
            "equivalent_n": float("nan")}


def paired_summary(rows: list[dict]) -> list[dict]:
    """Per size: mean error of each arm, and the paired per-seed difference.

    The paired difference is the quantity that matters. Both arms at a seed
    share initialisation, subset and data order, so their difference removes the
    seed variance that would otherwise swamp a small effect. ``n_seeds_helped``
    reports sign consistency, which at three seeds is more informative than a
    standard deviation.
    """
    sizes = sorted({r["n_traj"] for r in rows})
    out = []
    for n in sizes:
        byarm = {a: {r["seed"]: r["vrmse"] for r in rows
                     if r["n_traj"] == n and r["arm"] == a} for a in ARMS}
        seeds = sorted(set(byarm["plain"]) & set(byarm["harmonic"]))
        if not seeds:
            continue
        p = np.array([byarm["plain"][s] for s in seeds], dtype=float)
        h = np.array([byarm["harmonic"][s] for s in seeds], dtype=float)
        diff = h - p                     # negative means harmonic is better
        out.append({
            "n_traj": n, "n_seeds": len(seeds),
            "plain_mean": float(p.mean()), "plain_std": float(p.std(ddof=0)),
            "harmonic_mean": float(h.mean()),
            "harmonic_std": float(h.std(ddof=0)),
            "paired_diff_mean": float(diff.mean()),
            "paired_diff_std": float(diff.std(ddof=0)),
            "rel_change": float(diff.mean() / p.mean()) if p.mean() else float("nan"),
            "n_seeds_helped": int((diff < 0).sum()),
            "exceeds_seed_spread": bool(abs(diff.mean()) > p.std(ddof=0)),
        })
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_report(summary: list[dict], rows: list[dict], mults: list[dict],
                 per_regime: list[dict], params: dict) -> None:
    print("\n=== Data-efficiency curve (one-step test VRMSE) ===")
    print(f"    plain {params['plain']:,} params | harmonic "
          f"{params['harmonic']:,} params "
          f"(+{100 * (params['harmonic'] - params['plain']) / params['plain']:.2f}%)")
    hdr = (f"    {'traj':>5s} {'pairs':>7s} {'plain':>18s} {'harmonic':>18s} "
           f"{'paired diff':>13s} {'helped':>7s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for s in summary:
        pairs = next((r["n_pairs"] for r in rows if r["n_traj"] == s["n_traj"]), 0)
        print(f"    {s['n_traj']:>5d} {pairs:>7d} "
              f"{s['plain_mean']:>10.5f}+-{s['plain_std']:<6.5f} "
              f"{s['harmonic_mean']:>10.5f}+-{s['harmonic_std']:<6.5f} "
              f"{s['rel_change']:>+12.1%} "
              f"{s['n_seeds_helped']}/{s['n_seeds']:>5d}")
    print("    'helped' counts seeds where harmonic beat plain on the identical "
          "subset and init")

    print("\n=== Data multiplier: how much data plain needs to match harmonic ===")
    print("    reported only where the target falls inside the measured range; "
          "extrapolation\n    past the largest size would invent the number "
          "from the curve's slope")
    hdr = (f"    {'at traj':>8s} {'harmonic err':>13s} "
           f"{'plain needs':>12s} {'multiplier':>11s} {'status':>18s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for m in mults:
        eq = ("      n/a" if not np.isfinite(m["equivalent_n"])
              else f"{m['equivalent_n']:>9.1f}")
        mul = ("       n/a" if not np.isfinite(m["multiplier"])
               else f"{m['multiplier']:>10.2f}x")
        print(f"    {m['n_traj']:>8d} {m['harmonic_vrmse']:>13.5f} {eq} {mul} "
              f"{m['status']:>18s}")

    if per_regime:
        print("\n=== Per regime at the largest size "
              "(ext15 predicts maze and spots benefit) ===")
        hdr = (f"    {'regime':>9s} {'var<mode8':>10s} {'plain':>10s} "
               f"{'harmonic':>10s} {'change':>9s}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for r in per_regime:
            v = ("     n/a" if not np.isfinite(r["var_below_mode8"])
                 else f"{r['var_below_mode8']:>9.1%}")
            print(f"    {r['regime']:>9s} {v} {r['plain']:>10.5f} "
                  f"{r['harmonic']:>10.5f} {r['change']:>+9.1%}")


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


def plot(summary: list[dict], rows: list[dict], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping plot: {exc}")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    n = [s["n_traj"] for s in summary]
    for arm, style, col in (("plain", "o-", "tab:blue"),
                            ("harmonic", "s--", "tab:orange")):
        m = np.array([s[f"{arm}_mean"] for s in summary])
        sd = np.array([s[f"{arm}_std"] for s in summary])
        ax1.plot(n, m, style, color=col, label=arm)
        ax1.fill_between(n, m - sd, m + sd, color=col, alpha=0.15)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("training trajectories")
    ax1.set_ylabel("one-step test VRMSE")
    ax1.set_title("Data-efficiency curve")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    d = [s["paired_diff_mean"] for s in summary]
    sd = [s["paired_diff_std"] for s in summary]
    ax2.axhline(0, color="k", lw=1)
    ax2.errorbar(n, d, yerr=sd, fmt="o-", color="tab:purple", capsize=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("training trajectories")
    ax2.set_ylabel("harmonic - plain (paired)")
    ax2.set_title("Paired difference (negative = harmonic better)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def load_ext10(path: Path) -> dict:
    """Per-regime share of spatial variance below mode 8, from ext10.

    Same column and same filter as ``harmonic_ab.py`` -- the settled segment of
    field A -- so the two scripts order the regimes by the identical quantity.
    """
    if not path.exists():
        return {}
    with path.open() as f:
        return {r["scenario"]: float(r["spatial_var_at_modes_8"])
                for r in csv.DictReader(f)
                if r.get("segment") == "settled" and r.get("field") == "A"
                and r.get("spatial_var_at_modes_8")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/processed/gray_scott_streamed"))
    ap.add_argument("--per-regime", type=int, nargs="*", default=[1, 2, 3, 4],
                    help="trajectories per regime at each point on the curve")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--fundamental", type=float, default=4.0)
    ap.add_argument("--n-harmonics", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/extensions"))
    ap.add_argument("--ext10-csv", type=Path,
                    default=Path("results/extensions/"
                                 "ext10_harmonic_summary_gray_scott.csv"))
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    data, _ = load_dataset(args.data_dir)
    train_arr, train_lab = data["train"]
    test_arr, test_lab = data["test"]
    xte, yte = br.to_pairs(test_arr)
    print(f"device={device}  sizes(per regime)={args.per_regime}  "
          f"seeds={args.seeds}  epochs={args.epochs}")
    print(f"  train pool {len(train_arr)} traj, test {len(xte)} pairs")

    rows, params, models_at_max = [], {}, {}
    for k in args.per_regime:
        for seed in args.seeds:
            idx = balanced_subset(train_lab, k, seed)
            sub = train_arr[idx]
            xtr, ytr = br.to_pairs(sub)
            for arm in ARMS:
                cfg = {"seed": seed, "epochs": args.epochs, "width": args.width,
                       "modes": args.modes, "layers": args.layers,
                       "rank": args.rank, "fundamental": args.fundamental,
                       "n_harmonics": args.n_harmonics}
                print(f"\n  [{k}/regime = {len(idx)} traj | seed {seed} | {arm}]",
                      flush=True)
                model, p, secs = train_one(arm, cfg, xtr, ytr, xte, yte, device)
                v = br.evaluate_one_step(model, xte, yte, device)
                params[arm] = p
                rows.append({"arm": arm, "n_traj": int(len(idx)),
                             "per_regime": k, "seed": seed,
                             "n_pairs": int(len(xtr)), "vrmse": v,
                             "params": p, "epochs": args.epochs,
                             "train_s": secs})
                print(f"    -> test VRMSE {v:.5f} ({secs:.0f}s)", flush=True)
                if k == max(args.per_regime) and seed == args.seeds[0]:
                    models_at_max[arm] = model
                write_csv(args.out_dir / "ext30_curve.csv", rows)

    summary = paired_summary(rows)
    plain_curve = [{"n_traj": s["n_traj"], "vrmse": s["plain_mean"]}
                   for s in summary]
    mults = []
    for s in summary:
        m = data_multiplier(plain_curve, s["harmonic_mean"], s["n_traj"])
        mults.append({"n_traj": s["n_traj"],
                      "harmonic_vrmse": s["harmonic_mean"], **m})

    per_regime = []
    if len(models_at_max) == 2:
        ext10 = load_ext10(args.ext10_csv)
        pr = {a: per_regime_vrmse(models_at_max[a], test_arr, test_lab, device)
              for a in ARMS}
        for regime in pr["plain"]:
            p, h = pr["plain"][regime], pr["harmonic"][regime]
            per_regime.append({
                "regime": regime,
                "var_below_mode8": ext10.get(regime, float("nan")),
                "plain": p, "harmonic": h,
                "change": (h - p) / p if p else float("nan")})
        per_regime.sort(key=lambda r: (-r["var_below_mode8"]
                                       if np.isfinite(r["var_below_mode8"])
                                       else 0))

    print_report(summary, rows, mults, per_regime, params)
    write_csv(args.out_dir / "ext30_curve.csv", rows)
    write_csv(args.out_dir / "ext30_summary.csv", summary)
    write_csv(args.out_dir / "ext30_multiplier.csv", mults)
    if per_regime:
        write_csv(args.out_dir / "ext30_per_regime.csv", per_regime)
    plot(summary, rows, args.fig_dir / "ext30_data_efficiency.png")


if __name__ == "__main__":
    sys.exit(main())
