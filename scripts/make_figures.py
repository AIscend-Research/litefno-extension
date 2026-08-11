#!/usr/bin/env python3
"""Generate all publication figures and LaTeX tables for the LiteFNO study.

Sources:
  * REAL Well-derived logs in outputs/logs/ and logs/ (epoch study + FNO-S baseline)
  * PROXY sweep results in experiments/ (width / rank / seed / modes)

Outputs:
  * paper/figures/*.pdf   (vector, publication quality)
  * paper/tables/*.tex    (booktabs table bodies, \\input-able)
  * paper/data/*.csv       (curve data for reference)
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
PDATA = ROOT / "paper" / "data"
for d in (FIG, TAB, PDATA):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
})
BLUE, ORANGE, GREEN, RED, PURPLE = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def epoch_rows(path: Path):
    rows = [r for r in read_jsonl(path) if r.get("step", -2) >= 0 and "valid_vrmse" in r]
    return sorted(rows, key=lambda r: r["step"])


def read_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- REAL logs
REAL_LITE = ROOT / "outputs/logs/gray_scott_reaction_diffusion_litefno.jsonl"
REAL_FNOS = ROOT / "outputs/logs/gray_scott_reaction_diffusion_fno_s.jsonl"


def fig_real_learning_curves():
    """Fig 1+2: real 200-epoch LiteFNO learning curves (train vs valid RMSE & VRMSE)."""
    rows = epoch_rows(REAL_LITE)
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(steps, [r["train_rmse"] for r in rows], color=BLUE, label="Train")
    axes[0].plot(steps, [r["valid_rmse"] for r in rows], color=ORANGE, label="Valid")
    axes[0].set(xlabel="Epoch", ylabel="RMSE", title="RMSE")
    axes[1].plot(steps, [r["train_vrmse"] for r in rows], color=BLUE, label="Train")
    axes[1].plot(steps, [r["valid_vrmse"] for r in rows], color=ORANGE, label="Valid")
    axes[1].set(xlabel="Epoch", ylabel="VRMSE", title="VRMSE")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "real_learning_curves.pdf")
    plt.close(fig)
    # also export csv for reference
    with (PDATA / "real_litefno_200ep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "train_rmse", "valid_rmse", "train_vrmse", "valid_vrmse"])
        for r in rows:
            w.writerow([r["step"], r["train_rmse"], r["valid_rmse"], r["train_vrmse"], r["valid_vrmse"]])


def fig_epoch_study():
    """Fig 6: epoch study from the real 200-epoch log; mark 33/100/200 checkpoints."""
    rows = epoch_rows(REAL_LITE)
    steps = [r["step"] for r in rows]
    vrmse = [r["valid_vrmse"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(steps, vrmse, color=BLUE, lw=1.5, label="Valid VRMSE (200-epoch run)")
    marks = {}
    for e in (33, 100, 200):
        idx = min(range(len(steps)), key=lambda i: abs(steps[i] - (e - 1)))
        marks[e] = vrmse[idx]
        ax.scatter([steps[idx]], [vrmse[idx]], zorder=5, s=45)
        ax.annotate(f"e{e}: {vrmse[idx]:.4f}", (steps[idx], vrmse[idx]),
                    textcoords="offset points", xytext=(6, 10), fontsize=9)
    ax.set(xlabel="Epoch", ylabel="Validation VRMSE",
           title="Training-duration study (real Well data)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "epoch_study.pdf")
    plt.close(fig)
    return marks


def fig_baseline_compare():
    """Real FNO-S vs LiteFNO valid VRMSE over 200 epochs."""
    lite = epoch_rows(REAL_LITE)
    fnos = epoch_rows(REAL_FNOS)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot([r["step"] for r in lite], [r["valid_vrmse"] for r in lite],
            color=BLUE, label="LiteFNO (107.8k params)")
    ax.plot([r["step"] for r in fnos], [r["valid_vrmse"] for r in fnos],
            color=RED, label="FNO-S (9.47M params)")
    ax.set(xlabel="Epoch", ylabel="Validation VRMSE",
           title="LiteFNO vs FNO-S (real Well data)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "baseline_compare.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- PROXY sweeps
ALL = ROOT / "experiments" / "all_results.csv"


def load_results():
    if not ALL.exists():
        return {}
    return {r["name"]: r for r in read_csv(ALL)}


def fig_proxy_learning_curves(res):
    p = ROOT / "experiments/baseline/litefno_w64_r32_s1337/metrics.csv"
    if not p.exists():
        return
    rows = read_csv(p)
    steps = [int(r["step"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(steps, [f(r["train_rmse"]) for r in rows], color=BLUE, label="Train")
    axes[0].plot(steps, [f(r["valid_rmse"]) for r in rows], color=ORANGE, label="Valid")
    axes[0].set(xlabel="Epoch", ylabel="RMSE", title="Proxy RMSE")
    axes[1].plot(steps, [f(r["train_vrmse"]) for r in rows], color=BLUE, label="Train")
    axes[1].plot(steps, [f(r["valid_vrmse"]) for r in rows], color=ORANGE, label="Valid")
    axes[1].set(xlabel="Epoch", ylabel="VRMSE", title="Proxy VRMSE")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "proxy_learning_curves.pdf")
    plt.close(fig)


def _twin_metric_param(names, res, xkey, xlabel, title, outfile, xvals):
    pts = [(xvals[n], f(res[n]["test_vrmse"]), int(res[n]["params"])) for n in names if n in res]
    if not pts:
        return
    pts.sort()
    xs, vr, pr = zip(*pts)
    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    ax1.plot(xs, vr, "o-", color=BLUE, label="Test VRMSE")
    ax1.set(xlabel=xlabel, ylabel="Test VRMSE")
    ax1.set_xticks(xs)
    ax2 = ax1.twinx()
    ax2.plot(xs, [p / 1000 for p in pr], "s--", color=GREEN, label="Params (k)")
    ax2.set_ylabel("Parameters (thousands)")
    ax2.grid(False)
    ax1.set_title(title)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / outfile)
    plt.close(fig)


def fig_width(res):
    names = ["litefno_w32", "litefno_w64_r32_s1337", "litefno_w128"]
    xvals = {"litefno_w32": 32, "litefno_w64_r32_s1337": 64, "litefno_w128": 128}
    _twin_metric_param(names, res, "width", "Width", "Width sweep (proxy)",
                       "width_sweep.pdf", xvals)


def fig_rank(res):
    names = ["litefno_r16", "litefno_w64_r32_s1337", "litefno_r64"]
    xvals = {"litefno_r16": 16, "litefno_w64_r32_s1337": 32, "litefno_r64": 64}
    _twin_metric_param(names, res, "rank", "Rank", "Rank sweep (proxy)",
                       "rank_sweep.pdf", xvals)


def fig_modes(res):
    names = ["fno_s_m6", "fno_s_m12", "fno_s_m18"]
    xvals = {"fno_s_m6": 6, "fno_s_m12": 12, "fno_s_m18": 18}
    _twin_metric_param(names, res, "modes", "Fourier modes (per axis)",
                       "Fourier-mode ablation (FNO-S, proxy)", "modes_ablation.pdf", xvals)


def fig_seeds(res):
    names = ["litefno_w64_r32_s1337", "litefno_s2025", "litefno_s42"]
    vals = [(res[n]["seed"], f(res[n]["test_vrmse"])) for n in names if n in res]
    if not vals:
        return None
    seeds = [v[0] for v in vals]
    vrmse = [v[1] for v in vals]
    mean = statistics.mean(vrmse)
    sd = statistics.pstdev(vrmse) if len(vrmse) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.bar(range(len(seeds)), vrmse, color=PURPLE, alpha=0.8, width=0.55)
    ax.axhline(mean, color="k", ls="--", lw=1, label=f"mean={mean:.4f}")
    ax.axhspan(mean - sd, mean + sd, color="gray", alpha=0.2, label=f"±1 std ({sd:.1e})")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set(ylabel="Test VRMSE", title="Seed variability (proxy)")
    ax.set_ylim(0, max(vrmse) * 1.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "seed_variability.pdf")
    plt.close(fig)
    return mean, sd


# ---------------------------------------------------------------- LaTeX tables
def tex_num(x, nd=5):
    v = f(x)
    return "--" if v is None else f"{v:.{nd}f}"


def table_sweep(res, names, label_fn, head, outfile):
    lines = []
    for n in names:
        if n not in res:
            continue
        r = res[n]
        lines.append(
            f"    {label_fn(r)} & {int(r['params']):,} & "
            f"{tex_num(r['best_valid_rmse'])} & {tex_num(r['best_valid_vrmse'])} & "
            f"{tex_num(r['test_rmse'])} & {tex_num(r['test_vrmse'])} & {r['train_time_s']} \\\\"
        )
    body = (head + "\n" + "\n".join(lines) + "\n")
    (TAB / outfile).write_text(body)


def main():
    # real-data figures (always available)
    fig_real_learning_curves()
    marks = fig_epoch_study()
    fig_baseline_compare()

    res = load_results()
    if res:
        fig_proxy_learning_curves(res)
        fig_width(res)
        fig_rank(res)
        fig_modes(res)
        seed_stats = fig_seeds(res)

        table_sweep(res, ["litefno_w32", "litefno_w64_r32_s1337", "litefno_w128"],
                    lambda r: f"{r['width']}", None, "width_sweep.tex")
        table_sweep(res, ["litefno_r16", "litefno_w64_r32_s1337", "litefno_r64"],
                    lambda r: f"{r['rank']}", None, "rank_sweep.tex")
        table_sweep(res, ["fno_s_m6", "fno_s_m12", "fno_s_m18"],
                    lambda r: f"{r['modes']}", None, "modes.tex")
        table_sweep(res, ["litefno_w64_r32_s1337", "litefno_s2025", "litefno_s42"],
                    lambda r: f"{r['seed']}", None, "seeds.tex")
        print("seed stats (mean, std):", seed_stats)

    print("epoch-study marks (valid VRMSE @ 33/100/200):", marks)
    print("Wrote figures to", FIG)
    print("Wrote tables to", TAB)


if __name__ == "__main__":
    main()
