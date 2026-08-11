r"""Controlled A/B of harmonic conditioning, broken down by regime.

Board task: "Add harmonic conditioning: modify LITEFNO's spectral factorization
to add a harmonic-mode bias." The model change lives in
``src/litefno/models/harmonic.py``; this script is the measurement.

Two arms, identical in every respect except one flag:

    control       CP-factorized spectral LiteFNO
    conditioned   the same, plus a learnable complex bias on the harmonic
                  shells (a fundamental wavenumber and its multiples)

They share seeds, initialisation, data order, optimiser and schedule, and at
initialisation they are bit-identical -- the bias starts at zero, and a test in
``tests/test_harmonic.py`` pins that the two produce equal output before any
training. The bias adds 1-2% to the parameter count, so a difference cannot be
attributed to model size.

The measurement that matters is per regime, not the aggregate
-------------------------------------------------------------
An aggregate VRMSE would be the wrong readout here, and would probably show
nothing. The conditioning targets a specific band: ext10 found maze and spots
keep ~99% of their spatial variance *above* mode 8, in a narrow ring at the
Turing wavelength, while spirals and gliders are low-wavenumber. So the
prediction is differential, not uniform -- harmonic conditioning on the Turing
shells should help maze and spots and do close to nothing for spirals and
gliders.

A uniform improvement would actually be evidence against the mechanism: it would
suggest the extra parameters are helping generically rather than by supplying
structure at the wavenumbers that need it.

So the reported quantity is per-regime relative change, with the aggregate shown
alongside for context and explicitly not used as the verdict.

What the repo's earlier measurements predict
--------------------------------------------
Small. ext9/PR #15 killed the spatial harmonic prior in its low-wavenumber form.
ext12 found that even with documented, exactly periodic forcing, harmonics carry
5.4% of temporal variance globally. This intervention differs in targeting a
mid-spectrum band that two regimes genuinely concentrate in, which is a narrower
claim -- but the prior on effect size should be low, and saying so before the
run is the point of writing it here.
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


# Regime bookkeeping is duplicated from scripts/cross_regime.py, which is on a
# separate branch; the two should be pulled into the package once both land.
def regime_labels(manifest: dict, split: str) -> "np.ndarray":
    """Per-trajectory regime name, in the order the split was concatenated."""
    labels = []
    for entry in manifest["splits"][split]["files"]:
        labels.extend([entry["regime"]] * len(entry["trajectories"]))
    return np.array(labels)


def load_dataset(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    out = {}
    for split in ("train", "valid", "test"):
        arr = br.load_split(data_dir / f"{split}.h5")
        lab = regime_labels(manifest, split)
        assert len(lab) == arr.shape[0], (split, len(lab), arr.shape)
        out[split] = (arr, lab)
    return out, manifest


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return float(np.corrcoef(rx, ry)[0, 1])

import torch                                          # noqa: E402
from torch import nn                                  # noqa: E402

from litefno.models.harmonic import HarmonicLiteFNO   # noqa: E402


def build(arm: str, in_ch: int, cfg: dict):
    torch.manual_seed(cfg["seed"])
    return HarmonicLiteFNO(
        in_ch, in_ch, width=cfg["width"], modes=cfg["modes"],
        layers=cfg["layers"], rank=cfg["rank"],
        harmonic_bias=(arm == "conditioned"),
        fundamental=cfg["fundamental"], n_harmonics=cfg["n_harmonics"])


def per_regime_vrmse(model, arr, labels, device) -> dict:
    out = {}
    for regime in dict.fromkeys(labels):
        x, y = br.to_pairs(arr[labels == regime])
        out[regime] = br.evaluate_one_step(model, x, y, device)
    return out


def train_arm(arm: str, cfg: dict, data: dict, device: str) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train_arr, _ = data["train"]
    test_arr, test_lab = data["test"]
    xtr, ytr = br.to_pairs(train_arr)
    xte, yte = br.to_pairs(test_arr)

    model = build(arm, xtr.shape[1], cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=br.LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=br.LR_STEP,
                                            gamma=br.LR_GAMMA)
    loss_fn = nn.MSELoss()

    n, t0, curve = len(xtr), time.time(), []
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, br.BATCH):
            idx = perm[i:i + br.BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx].to(device)), ytr[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)
        sched.step()
        if (epoch + 1) % cfg["log_every"] == 0 or epoch == cfg["epochs"] - 1:
            v = br.evaluate_one_step(model, xte, yte, device)
            curve.append({"epoch": epoch + 1, "train_mse": total / n,
                          "test_vrmse": v})
            print(f"      epoch {epoch + 1:>4d}/{cfg['epochs']}  "
                  f"train_mse={total / n:.3e}  test_vrmse={v:.5f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    rec = {"arm": arm, "seed": cfg["seed"], "params": n_params,
           "harmonic_modes": model.n_harmonic_modes(),
           "shells": json.dumps(model.shells),
           "epochs": cfg["epochs"], "train_s": round(time.time() - t0, 1),
           "test_vrmse": br.evaluate_one_step(model, xte, yte, device)}
    rec.update({f"vrmse_{k}": v
                for k, v in per_regime_vrmse(model, test_arr, test_lab,
                                             device).items()})
    rec["curve"] = curve
    return rec


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def summarise(records: list[dict], regimes: list[str]) -> list[dict]:
    out = []
    for arm in ("control", "conditioned"):
        rs = [r for r in records if r["arm"] == arm]
        if not rs:
            continue
        row = {"arm": arm, "n_seeds": len(rs), "params": rs[0]["params"],
               "harmonic_modes": rs[0]["harmonic_modes"]}
        for key in ["test_vrmse"] + [f"vrmse_{r}" for r in regimes]:
            v = np.array([r[key] for r in rs], dtype=float)
            row[f"{key}_mean"] = float(v.mean())
            row[f"{key}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    return out


def print_report(summary: list[dict], records: list[dict], regimes: list[str],
                 ext10_csv: Path) -> list[dict]:
    if len(summary) < 2:
        print("\n(need both arms for the A/B)")
        return []
    ctrl = next(s for s in summary if s["arm"] == "control")
    cond = next(s for s in summary if s["arm"] == "conditioned")

    print("\n=== Harmonic conditioning A/B (one-step test VRMSE) ===")
    print(f"    control {ctrl['params']:,} params | conditioned "
          f"{cond['params']:,} params "
          f"(+{100 * (cond['params'] - ctrl['params']) / ctrl['params']:.1f}%, "
          f"{cond['harmonic_modes']} biased modes)")
    print(f"    aggregate: {ctrl['test_vrmse_mean']:.5f} -> "
          f"{cond['test_vrmse_mean']:.5f}  "
          f"({100 * (cond['test_vrmse_mean'] / ctrl['test_vrmse_mean'] - 1):+.1f}%)"
          "   [context only, not the verdict]")

    # the differential test
    v = {}
    if ext10_csv.exists():
        with ext10_csv.open() as f:
            v = {r["scenario"]: float(r["spatial_var_at_modes_8"])
                 for r in csv.DictReader(f)
                 if r["segment"] == "settled" and r["field"] == "A"
                 and r.get("spatial_var_at_modes_8")}

    print("\n    per regime (the measurement that matters):")
    hdr = (f"    {'regime':>9s} {'var<mode8':>10s} {'control':>10s} "
           f"{'conditioned':>12s} {'change':>9s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    rows = []
    for regime in sorted(regimes, key=lambda r: v.get(r, 1.0)):
        c = ctrl[f"vrmse_{regime}_mean"]
        d = cond[f"vrmse_{regime}_mean"]
        rel = 100 * (d / c - 1) if c else float("nan")
        print(f"    {regime:>9s} {v.get(regime, float('nan')):>9.1%} "
              f"{c:>10.5f} {d:>12.5f} {rel:>+8.1f}%")
        rows.append({"regime": regime, "var_below_mode8": v.get(regime),
                     "control_vrmse": c, "conditioned_vrmse": d,
                     "relative_change_pct": rel})

    if v and len(rows) >= 3:
        have = [r for r in rows if r["var_below_mode8"] is not None]
        rho = spearman([r["var_below_mode8"] for r in have],
                       [r["relative_change_pct"] for r in have])
        print(f"\n    prediction: conditioning on the Turing shells helps the "
              f"regimes that live there")
        print(f"    (maze and spots, lowest var<mode8) and does little for "
              f"spirals and gliders")
        print(f"    rho(var below mode 8, relative change) = {rho:+.3f}  "
              f"-- positive means the prediction holds")
        for r in rows:
            r["spearman_rho"] = rho
    return rows


def write_csv(path: Path, rows: list[dict], drop=("curve",)) -> None:
    flat = [{k: val for k, val in r.items() if k not in drop} for r in rows]
    if not flat:
        return
    keys, seen = [], set()
    for r in flat:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    print(f"wrote {path}")


def plot(summary, per_regime, records, regimes, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped figure")
        return
    if len(summary) < 2:
        return
    ctrl = next(s for s in summary if s["arm"] == "control")
    cond = next(s for s in summary if s["arm"] == "conditioned")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    order = sorted(regimes, key=lambda r: ctrl[f"vrmse_{r}_mean"])
    x = np.arange(len(order))
    ax.bar(x - 0.2, [ctrl[f"vrmse_{r}_mean"] for r in order], 0.4,
           yerr=[ctrl[f"vrmse_{r}_std"] for r in order], capsize=3,
           label="control", color="tab:blue", edgecolor="k", linewidth=0.4)
    ax.bar(x + 0.2, [cond[f"vrmse_{r}_mean"] for r in order], 0.4,
           yerr=[cond[f"vrmse_{r}_std"] for r in order], capsize=3,
           label="conditioned", color="tab:orange", edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("one-step test VRMSE")
    ax.set_title("Per-regime error, control vs conditioned", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    if per_regime:
        pr = [p for p in per_regime if p["var_below_mode8"] is not None]
        pr.sort(key=lambda p: p["var_below_mode8"])
        cols = ["tab:green" if p["relative_change_pct"] < 0 else "tab:red"
                for p in pr]
        ax.barh(range(len(pr)), [p["relative_change_pct"] for p in pr],
                color=cols, edgecolor="k", linewidth=0.4)
        ax.set_yticks(range(len(pr)))
        ax.set_yticklabels([f"{p['regime']}\n({p['var_below_mode8']:.1%})"
                            for p in pr], fontsize=7)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("relative change in VRMSE (%)  — negative is better")
        ax.set_title("Effect by regime, ordered by how much\nenergy sits above "
                     "mode 8", fontsize=10)

    ax = axes[2]
    for r in records:
        style = "-" if r["arm"] == "control" else "--"
        col = "tab:blue" if r["arm"] == "control" else "tab:orange"
        ax.semilogy([c["epoch"] for c in r["curve"]],
                    [c["test_vrmse"] for c in r["curve"]], style, color=col,
                    lw=1.2, alpha=0.8)
    ax.plot([], [], "-", color="tab:blue", label="control")
    ax.plot([], [], "--", color="tab:orange", label="conditioned")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test VRMSE")
    ax.set_title("Convergence, all seeds", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Harmonic-mode bias on the Turing shells — controlled A/B",
                 y=1.03, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/processed/gray_scott_streamed"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--fundamental", type=float, default=4.0,
                    help="Turing wavelength after the pipeline's 4x downsample")
    ap.add_argument("--n-harmonics", type=int, default=3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/baseline"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/baseline"))
    ap.add_argument("--ext10-csv", type=Path,
                    default=Path("results/extensions/"
                                 "ext10_harmonic_summary_gray_scott.csv"))
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    data, _ = load_dataset(args.data_dir)
    regimes = list(dict.fromkeys(data["test"][1]))
    print(f"device={device}  regimes={regimes}")
    print(f"  width={args.width} modes={args.modes} layers={args.layers} "
          f"rank={args.rank}  fundamental={args.fundamental} "
          f"n_harmonics={args.n_harmonics}")

    records = []
    for seed in args.seeds:
        cfg = {"seed": seed, "epochs": args.epochs, "log_every": args.log_every,
               "width": args.width, "modes": args.modes, "layers": args.layers,
               "rank": args.rank, "fundamental": args.fundamental,
               "n_harmonics": args.n_harmonics}
        for arm in ("control", "conditioned"):
            print(f"\n  [{arm} seed {seed}]", flush=True)
            rec = train_arm(arm, cfg, data, device)
            print(f"    -> test VRMSE {rec['test_vrmse']:.5f} "
                  f"({rec['params']:,} params, {rec['train_s']:.0f}s)",
                  flush=True)
            records.append(rec)
            write_csv(args.out_dir / "ext15_harmonic_ab_seeds.csv", records)

    summary = summarise(records, regimes)
    per_regime = print_report(summary, records, regimes, args.ext10_csv)

    write_csv(args.out_dir / "ext15_harmonic_ab_summary.csv", summary)
    if per_regime:
        write_csv(args.out_dir / "ext15_harmonic_ab_per_regime.csv", per_regime)
    plot(summary, per_regime, records, regimes,
         args.fig_dir / "ext15_harmonic_ab.png")


if __name__ == "__main__":
    sys.exit(main())
