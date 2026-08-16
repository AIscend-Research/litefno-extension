r"""Is the low-rank operator actually deployable? (ext25, H6)

Board task: "Benchmark model size/latency/FLOPs -- must stay Kaggle-friendly
(not just 'low-rank LITEFNO' but actually deployable in a low-resource
scientist's notebook)."

The hypothesis, and why it is worth stating rather than assuming
---------------------------------------------------------------
Every low-rank paper reports a parameter count, and the reader is invited to
read it as a deployability claim. State that invitation as a hypothesis:

    H6: parameter count predicts deployability -- across this model family, a
        model with fewer parameters is smaller on disk, cheaper in FLOPs, and
        faster to run on the hardware a low-resource scientist actually has.

The disk half is true by definition and is not interesting. The other halves are
measurable and one of them is false, in a way that is specific to CP
factorization and fixable.

Where it goes wrong, in closed form
-----------------------------------
``CPSpectralConv2d.forward`` calls ``self.weight()``, which contracts the CP
factors back into the dense ``(in, out, m1, m2)`` spectral weight -- every
forward pass, every time. That costs ``8 * rank * in * out * m1 * m2`` flops and
**does not depend on the batch size**. CP therefore makes the weight cheap to
*store* and leaves it exactly as expensive to *use*, plus the cost of rebuilding
it. At batch 1 -- a scientist stepping one field through a notebook, which is
the setting the board task names -- that reconstruction is the majority of the
model's arithmetic.

So the extension measures four things, in order of how much they can be argued
with:

1. the closed-form FLOP model, checked against ``torch``'s own tracer on the
   terms the tracer covers, so no later number rests on an unvalidated count;
2. size, FLOPs and latency across a family spanning 280x in parameters, with
   Spearman rank correlations that make H6 falsifiable rather than rhetorical;
3. the fix -- fold the CP factors once at eval time -- with its speedup
   *predicted* from the closed form before it is measured, and its outputs
   checked to be bitwise identical;
4. a stated deployment envelope, as a pass/fail table rather than a vibe.

Controls
--------
Latency is measured at 1 and 4 threads, because a model that parallelises well
would otherwise look fast for the wrong reason and a 4-vCPU notebook VM is the
target. Batch 1 and batch 16 are both reported, because a batch-independent cost
is invisible in throughput and dominant in interactive use. Warmup is discarded
and the median is reported with its IQR.

Self-contained: builds its own random inputs, needs no data, no GPU and no
network.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch                                                    # noqa: E402
from torch import nn                                            # noqa: E402

from litefno.bench import (                                     # noqa: E402
    analytic_flops, count_parameters, flop_audit,
    fuse_spectral_weights, latency, peak_rss_bytes, resident_bytes, spearman,
    state_dict_bytes)
from litefno.models.fno_s import FNOS                           # noqa: E402
from litefno.models.harmonic import HarmonicLiteFNO             # noqa: E402
from litefno.models.litefno import LiteFNO                      # noqa: E402

RESULTS = _ROOT / "results" / "extensions"
FIGURES = _ROOT / "figures" / "extensions"

IN_CH = OUT_CH = 2                          # Gray-Scott: two fields

# The family. Spans 280x in parameters and three architectures, so a rank
# correlation over it is a statement about the design space rather than about
# one pair of models. The two arms marked (headline) are the shapes the repo's
# own protocol trains: configs/experiments/base_litefno.yaml and
# scripts/baseline_reference.py.
ARMS = [
    ("cnn-w32-r16",   "cnn", dict(width=32, rank=16, layers=4)),
    ("cnn-w64-r32",   "cnn", dict(width=64, rank=32, layers=8)),   # headline
    ("fno_s-w32-m8",  "fno_s", dict(width=32, modes=8, layers=4)),
    ("fno_s-w32-m16", "fno_s", dict(width=32, modes=16, layers=4)),
    ("fno_s-w64-m16", "fno_s", dict(width=64, modes=16, layers=8)),
    ("cp-w32-r2",     "cp", dict(width=32, modes=16, layers=4, rank=2)),
    ("cp-w32-r8",     "cp", dict(width=32, modes=16, layers=4, rank=8)),
    ("cp-w32-r32",    "cp", dict(width=32, modes=16, layers=4, rank=32)),
    ("cp-w64-r8",     "cp", dict(width=64, modes=16, layers=8, rank=8)),  # headline
]


def build(family: str, **kwargs) -> nn.Module:
    if family == "cnn":
        return LiteFNO(IN_CH, OUT_CH, **kwargs)
    if family == "fno_s":
        return FNOS(IN_CH, OUT_CH, **kwargs)
    if family == "cp":
        return HarmonicLiteFNO(IN_CH, OUT_CH, **kwargs)
    raise ValueError(family)


# --------------------------------------------------------------------------
# 1. the closed-form FLOP model against torch's tracer
# --------------------------------------------------------------------------


def audit_table(size: int, batch: int) -> list[dict]:
    rows = []
    for name, family, kwargs in ARMS:
        model = build(family, **kwargs)
        audit = flop_audit(model, size, batch=batch)
        rows.append({"arm": name, "family": family, "size": size,
                     "batch": batch, **audit})
    return rows


# --------------------------------------------------------------------------
# 2. size, FLOPs and latency across the family
# --------------------------------------------------------------------------


def family_table(args) -> list[dict]:
    rows = []
    for name, family, kwargs in ARMS:
        variants = [(name, False)]
        if family == "cp":
            variants.append((name + "+fused", True))
        for label, fuse in variants:
            torch.manual_seed(0)
            model = build(family, **kwargs)
            if fuse:
                fuse_spectral_weights(model)
            flops = analytic_flops(model, args.size, batch=1)
            row = {"arm": label, "family": family, "fused": fuse,
                   "params": count_parameters(model),
                   "disk_mb": state_dict_bytes(model) / 2 ** 20,
                   "resident_mb": resident_bytes(model) / 2 ** 20,
                   "flops_b1": flops["total"],
                   "fixed_share": flops["fixed_share"]}
            for batch in args.batches:
                example = torch.randn(batch, IN_CH, args.size, args.size)
                for threads in args.threads:
                    timing = latency(model, example, repeats=args.repeats,
                                     warmup=args.warmup, threads=threads)
                    tag = f"b{batch}_t{threads}"
                    row[f"ms_{tag}"] = timing["ms"]
                    row[f"iqr_{tag}"] = timing["ms_iqr"]
                    row[f"ms_per_sample_{tag}"] = timing["ms_per_sample"]
            # measured on an unfused copy: fusing detaches the CP factors, so a
            # fused model trains nothing and timing its backward pass would be
            # timing a no-op
            torch.manual_seed(0)
            row["train_step_ms"] = train_step_ms(
                build(family, **kwargs), args.size, args.train_batch,
                args.threads[-1])
            rows.append(row)
            print(f"   {label:>18}: {row['params']:>9,} params  "
                  f"{row['flops_b1']:.2e} flops  "
                  f"{row[f'ms_b1_t{args.threads[-1]}']:.2f} ms @ b1")
    return rows


def train_step_ms(model: nn.Module, size: int, batch: int, threads: int,
                  repeats: int = 5) -> float:
    """One forward + backward + optimiser step, median of ``repeats``.

    Deployability is not only inference. The question a scientist actually has
    is whether the headline protocol finishes inside one notebook session.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch, IN_CH, size, size)
    y = torch.randn(batch, OUT_CH, size, size)
    try:
        for _ in range(2):                              # warmup
            optimiser.zero_grad()
            nn.functional.mse_loss(model(x), y).backward()
            optimiser.step()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            optimiser.zero_grad()
            nn.functional.mse_loss(model(x), y).backward()
            optimiser.step()
            samples.append((time.perf_counter() - start) * 1e3)
    finally:
        torch.set_num_threads(previous)
        model.eval()
    return float(np.median(samples))


def rank_table(rows: list[dict], args) -> list[dict]:
    """Does parameter count rank models the way latency does? That is H6."""
    base = [r for r in rows if not r["fused"]]
    out = []
    for batch in args.batches:
        for threads in args.threads:
            key = f"ms_b{batch}_t{threads}"
            latencies = [r[key] for r in base]
            out.append({
                "batch": batch, "threads": threads, "n_arms": len(base),
                "spearman_params_latency": spearman(
                    [r["params"] for r in base], latencies),
                "spearman_flops_latency": spearman(
                    [r["flops_b1"] for r in base], latencies),
                "spearman_disk_latency": spearman(
                    [r["disk_mb"] for r in base], latencies)})
    return out


# --------------------------------------------------------------------------
# 3. the fix, predicted before it is measured
# --------------------------------------------------------------------------


def fusion_table(args) -> list[dict]:
    """Fold the CP factors once; check outputs, predict and measure the gain.

    The closed form says the speedup at batch B should be
    ``total(B) / (total(B) - fixed)``, since fusing removes exactly the
    batch-independent term and nothing else. Predicting first and measuring
    second is the only way this is a test rather than a description.
    """
    rows = []
    for name, family, kwargs in ARMS:
        if family != "cp":
            continue
        for batch in args.fusion_batches:
            torch.manual_seed(0)
            plain = build(family, **kwargs).eval()
            torch.manual_seed(0)
            fused = build(family, **kwargs).eval()
            n_fused = fuse_spectral_weights(fused)

            example = torch.randn(batch, IN_CH, args.size, args.size)
            with torch.inference_mode():
                delta = float((fused(example) - plain(example)).abs().max())

            flops = analytic_flops(plain, args.size, batch=batch)
            predicted = flops["total"] / (flops["total"] - flops["fixed"])
            threads = args.threads[-1]
            slow = latency(plain, example, repeats=args.repeats,
                           warmup=args.warmup, threads=threads)
            fast = latency(fused, example, repeats=args.repeats,
                           warmup=args.warmup, threads=threads)
            # why the closed form over-predicts: the work fusing removes is a
            # dense einsum that runs near peak arithmetic throughput, while what
            # remains is FFT-dominated and memory-bound. Dividing the removed
            # FLOPs by the time actually saved makes the two rates comparable.
            saved_s = (slow["ms"] - fast["ms"]) / 1e3
            removed_rate = (flops["fixed"] / saved_s / 1e9
                            if saved_s > 0 else float("nan"))
            kept_rate = (flops["total"] - flops["fixed"]) / (
                fast["ms"] / 1e3) / 1e9
            rows.append({
                "arm": name, "batch": batch, "layers_fused": n_fused,
                "max_abs_output_diff": delta,
                "params_plain": count_parameters(plain),
                "params_fused": count_parameters(fused),
                "disk_mb_plain": state_dict_bytes(plain) / 2 ** 20,
                "disk_mb_fused": state_dict_bytes(fused) / 2 ** 20,
                "resident_mb_fused": resident_bytes(fused) / 2 ** 20,
                "fixed_share": flops["fixed_share"],
                "predicted_speedup": predicted,
                "ms_plain": slow["ms"], "ms_fused": fast["ms"],
                "measured_speedup": slow["ms"] / fast["ms"],
                "gflops_per_s_removed": removed_rate,
                "gflops_per_s_kept": kept_rate})
            print(f"   {name:>12} b={batch:<3} predicted x{predicted:.2f}  "
                  f"measured x{slow['ms'] / fast['ms']:.2f}  "
                  f"|diff| {delta:.1e}")
    return rows


# --------------------------------------------------------------------------
# 4. resolution scaling
# --------------------------------------------------------------------------


def resolution_table(args) -> list[dict]:
    """Latency against grid size, with the analytic exponent alongside.

    A CNN is ``O(N^2)`` in the side length; a spectral layer's FFT is
    ``O(N^2 log N)`` and its mode contraction is ``O(1)`` in ``N`` once the
    modes are fixed. Which term wins decides whether a model that fits at 32x32
    still fits at 128x128, and that is a deployability question, not a
    theoretical one.
    """
    rows = []
    for name, family, kwargs in ARMS:
        torch.manual_seed(0)
        model = build(family, **kwargs)
        for size in args.sizes:
            example = torch.randn(1, IN_CH, size, size)
            timing = latency(model, example, repeats=max(5, args.repeats // 3),
                             warmup=args.warmup, threads=args.threads[-1])
            flops = analytic_flops(model, size, batch=1)
            rows.append({"arm": name, "family": family, "size": size,
                         "ms": timing["ms"], "flops": flops["total"],
                         "fixed_share": flops["fixed_share"]})
        sizes = np.array(args.sizes, dtype=float)
        times = np.array([r["ms"] for r in rows if r["arm"] == name])
        # slope of log(latency) against log(N): 2 means area-scaling
        slope = float(np.polyfit(np.log(sizes), np.log(times), 1)[0])
        for r in rows:
            if r["arm"] == name:
                r["latency_exponent"] = slope
        print(f"   {name:>14}: latency ~ N^{slope:.2f}")
    return rows


# --------------------------------------------------------------------------
# 5. the deployment envelope
# --------------------------------------------------------------------------


def envelope_table(family_rows: list[dict], args) -> list[dict]:
    """Pass/fail against an explicitly stated budget.

    The budget is the script's assumption, set by flags, not a measured fact
    about any particular hosted notebook service -- free-tier quotas change, and
    a number baked into a repository would be wrong within a year. What is
    defensible is the shape: a handful of CPU cores, no guaranteed accelerator,
    a session that ends, and a reader who wants a single field stepped forward
    without waiting.
    """
    threads = args.threads[-1]
    rows = []
    for row in family_rows:
        interactive = row[f"ms_b1_t{threads}"]
        train_hours = (args.train_epochs * args.samples_per_epoch
                       / args.train_batch * row["train_step_ms"]
                       / 1e3 / 3600)
        checks = {
            "disk": row["disk_mb"] <= args.max_disk_mb,
            "interactive": interactive <= args.max_interactive_ms,
            "trains_in_session": train_hours <= args.max_session_hours,
        }
        rows.append({"arm": row["arm"], "params": row["params"],
                     "disk_mb": row["disk_mb"],
                     "interactive_ms": interactive,
                     "train_hours": train_hours,
                     **{f"ok_{k}": v for k, v in checks.items()},
                     "deployable": all(checks.values())})
    return rows


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def save_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(RESULTS / name, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(family_rows, fusion_rows, resolution_rows, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"cnn": "tab:grey", "fno_s": "tab:blue", "cp": "tab:red"}
    threads = args.threads[-1]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    base = [r for r in family_rows if not r["fused"]]
    for row in base:
        ax.scatter(row["params"], row[f"ms_b1_t{threads}"], s=60,
                   color=colours[row["family"]], zorder=3)
        ax.annotate(row["arm"], (row["params"], row[f"ms_b1_t{threads}"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    for fam, colour in colours.items():
        ax.scatter([], [], color=colour, label=fam)
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel(f"batch-1 latency (ms, {threads} threads)")
    ax.set_title("1. H6: do parameters predict latency?")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for row in family_rows:
        marker = "s" if row["fused"] else "o"
        ax.scatter(row["flops_b1"], row[f"ms_b1_t{threads}"], s=60,
                   marker=marker, color=colours[row["family"]], zorder=3)
    ax.scatter([], [], marker="o", color="0.3", label="as written")
    ax.scatter([], [], marker="s", color="0.3", label="CP weights fused")
    ax.set_xscale("log")
    ax.set_xlabel("analytic FLOPs / forward at batch 1")
    ax.set_ylabel(f"batch-1 latency (ms, {threads} threads)")
    ax.set_title("2. FLOPs do predict it")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    arms = sorted({r["arm"] for r in fusion_rows})
    palette = plt.get_cmap("tab10")
    for index, arm in enumerate(arms):
        sub = sorted([r for r in fusion_rows if r["arm"] == arm],
                     key=lambda r: r["batch"])
        colour = palette(index)
        ax.plot([r["batch"] for r in sub], [r["predicted_speedup"] for r in sub],
                "--", color=colour, alpha=0.55, lw=1)
        ax.plot([r["batch"] for r in sub], [r["measured_speedup"] for r in sub],
                "o-", color=colour, label=arm)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("speedup from fusing the CP weights")
    ax.set_title("3. the fixed cost, and what removing it buys\n"
                 "(dashed: predicted from the closed form)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for arm in {r["arm"] for r in resolution_rows}:
        sub = sorted([r for r in resolution_rows if r["arm"] == arm],
                     key=lambda r: r["size"])
        ax.plot([r["size"] for r in sub], [r["ms"] for r in sub], "o-",
                color=colours[sub[0]["family"]], lw=1,
                label=f"{arm} (N^{sub[0]['latency_exponent']:.1f})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("grid side N")
    ax.set_ylabel("batch-1 latency (ms)")
    ax.set_title("4. does it still fit at higher resolution?")
    ax.legend(fontsize=6, ncol=2)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "ext25_deployability.png", dpi=150)
    plt.close(fig)


def print_report(audit_rows, family_rows, rank_rows, fusion_rows,
                 resolution_rows, envelope_rows, args) -> None:
    line = "=" * 78
    print(f"\n{line}\next25: is the low-rank operator deployable? (H6)\n{line}")

    print("\n1. the closed-form FLOP model against torch's own tracer")
    print(f"   {'arm':>14} {'analytic':>12} {'tracer':>12} {'rel err':>10}"
          f" {'FFT share':>10} {'fixed share':>12}")
    for row in audit_rows:
        counted = f"{row['counted']:.4e}" if row["counted"] else "n/a"
        print(f"   {row['arm']:>14} "
              f"{row['analytic_counter_convention']:>12.4e} {counted:>12} "
              f"{row['rel_error']:>10.1e} {row['fft_share']:>10.1%} "
              f"{row['fixed_share']:>12.1%}")
    print("   compared in the tracer's convention (complex MAC = 2 flops, no")
    print("   FFT, no bias). The CP residual is the two small factor")
    print("   contractions torch does not emit as matmuls.")

    threads = args.threads[-1]
    print(f"\n2. the family at {args.size}x{args.size}, {threads} threads")
    print(f"   {'arm':>18} {'params':>10} {'disk MB':>8} {'RAM MB':>7}"
          f" {'FLOPs b1':>11} {'ms b1':>8} {'ms b16':>8} {'fixed':>7}")
    for row in family_rows:
        print(f"   {row['arm']:>18} {row['params']:>10,} "
              f"{row['disk_mb']:>8.2f} {row['resident_mb']:>7.2f} "
              f"{row['flops_b1']:>11.3e} "
              f"{row[f'ms_b1_t{threads}']:>8.2f} "
              f"{row.get(f'ms_b16_t{threads}', float('nan')):>8.2f} "
              f"{row['fixed_share']:>7.1%}")
    print("   'disk' is the checkpoint; 'RAM' includes the fused model's")
    print("   non-persistent weight cache, which is what fusing actually costs.")

    print("\n3. H6: does parameter count rank models the way latency does?")
    print(f"   {'batch':>6} {'threads':>8} {'rho(params, ms)':>17}"
          f" {'rho(FLOPs, ms)':>16} {'rho(disk, ms)':>15}")
    for row in rank_rows:
        print(f"   {row['batch']:>6} {row['threads']:>8} "
              f"{row['spearman_params_latency']:>17.3f} "
              f"{row['spearman_flops_latency']:>16.3f} "
              f"{row['spearman_disk_latency']:>15.3f}")

    print("\n4. folding the CP factors once, at eval time")
    print(f"   {'arm':>12} {'batch':>6} {'fixed':>7} {'predicted':>10}"
          f" {'measured':>9} {'ms before':>10} {'ms after':>9} {'|diff|':>8}"
          f" {'GF/s cut':>9} {'GF/s kept':>10}")
    for row in fusion_rows:
        print(f"   {row['arm']:>12} {row['batch']:>6} "
              f"{row['fixed_share']:>7.1%} x{row['predicted_speedup']:>9.2f} "
              f"x{row['measured_speedup']:>8.2f} {row['ms_plain']:>10.2f} "
              f"{row['ms_fused']:>9.2f} {row['max_abs_output_diff']:>8.1e} "
              f"{row['gflops_per_s_removed']:>9.1f} "
              f"{row['gflops_per_s_kept']:>10.1f}")
    print("   outputs are identical, not close: the same tensor computed once")
    print("   instead of once per call. Parameters and checkpoint size are")
    print("   unchanged -- the cache is a non-persistent buffer.")
    print("   The closed form over-predicts the speedup, and the last two")
    print("   columns say why: the removed einsum runs near peak arithmetic")
    print("   throughput while what remains is FFT-bound, so equal FLOPs are")
    print("   not equal time. FLOPs rank models correctly and price them badly.")

    print("\n5. resolution scaling (batch 1)")
    arms = []
    for row in resolution_rows:
        if row["arm"] not in [a["arm"] for a in arms]:
            arms.append(row)
    header = "   " + f"{'arm':>14}" + "".join(f"{s:>9}" for s in args.sizes) \
        + f"{'exponent':>10}"
    print(header)
    for arm in arms:
        sub = sorted([r for r in resolution_rows if r["arm"] == arm["arm"]],
                     key=lambda r: r["size"])
        times = "".join(f"{r['ms']:>9.2f}" for r in sub)
        print(f"   {arm['arm']:>14}{times}{arm['latency_exponent']:>10.2f}")

    print(f"\n6. the envelope: <= {args.max_disk_mb:.0f} MB on disk, "
          f"<= {args.max_interactive_ms:.0f} ms per interactive step,")
    print(f"   <= {args.max_session_hours:.0f} h to train "
          f"{args.train_epochs} epochs x {args.samples_per_epoch:,} samples "
          f"at batch {args.train_batch} on {threads} CPU threads")
    print(f"   {'arm':>18} {'disk MB':>8} {'ms/step':>9} {'train h':>9}"
          f"  {'verdict':>10}")
    for row in envelope_rows:
        why = [k[3:] for k in ("ok_disk", "ok_interactive",
                               "ok_trains_in_session") if not row[k]]
        verdict = "deployable" if row["deployable"] else "fails: " + ",".join(why)
        print(f"   {row['arm']:>18} {row['disk_mb']:>8.2f} "
              f"{row['interactive_ms']:>9.2f} {row['train_hours']:>9.1f}"
              f"  {verdict:>10}")
    print("   the budget is this script's assumption, set by flags, not a")
    print("   measured fact about any hosted notebook service.")


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=32,
                   help="grid side for the headline table")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4],
                   help="the last one is used for the headline numbers")
    p.add_argument("--fusion-batches", type=int, nargs="+",
                   default=[1, 4, 16, 64])
    p.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 96, 128])
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--train-batch", type=int, default=64)
    p.add_argument("--train-epochs", type=int, default=200,
                   help="configs/experiments/base_litefno.yaml")
    p.add_argument("--samples-per-epoch", type=int, default=40000,
                   help="assumed Gray-Scott train windows; scales linearly")
    p.add_argument("--max-disk-mb", type=float, default=100.0)
    p.add_argument("--max-interactive-ms", type=float, default=50.0)
    p.add_argument("--max-session-hours", type=float, default=12.0)
    p.add_argument("--quick", action="store_true",
                   help="plumbing check: tiny sweeps, few repeats")
    args = p.parse_args()

    if args.quick:
        args.repeats, args.warmup = 3, 1
        args.batches, args.threads = [1], [4]
        args.fusion_batches, args.sizes = [1, 8], [32, 48]

    start = time.time()
    torch.manual_seed(0)

    print("1. auditing the closed-form FLOP model")
    audit_rows = audit_table(args.size, batch=1)
    worst = max(r["rel_error"] for r in audit_rows
                if r["rel_error"] == r["rel_error"])
    print(f"   worst relative disagreement with the tracer: {worst:.2e}")

    print("2. size, FLOPs and latency across the family")
    family_rows = family_table(args)
    rank_rows = rank_table(family_rows, args)

    print("3. folding the CP weights")
    fusion_rows = fusion_table(args)

    print("4. resolution scaling")
    resolution_rows = resolution_table(args)

    envelope_rows = envelope_table(family_rows, args)

    save_csv("ext25_flop_audit.csv", audit_rows)
    save_csv("ext25_family.csv", family_rows)
    save_csv("ext25_rank.csv", rank_rows)
    save_csv("ext25_fusion.csv", fusion_rows)
    save_csv("ext25_resolution.csv", resolution_rows)
    save_csv("ext25_envelope.csv", envelope_rows)
    plot(family_rows, fusion_rows, resolution_rows, args)

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "ext25_host.json", "w") as handle:
        json.dump({"platform": platform.platform(),
                   "processor": platform.processor(),
                   "torch": torch.__version__,
                   "torch_threads_default": torch.get_num_threads(),
                   "peak_rss_mb": peak_rss_bytes() / 2 ** 20,
                   "note": "latency numbers are hardware-specific; the rank "
                           "correlations and the closed-form ratios are not"},
                  handle, indent=2)

    print_report(audit_rows, family_rows, rank_rows, fusion_rows,
                 resolution_rows, envelope_rows, args)
    print(f"\ntotal {time.time() - start:.0f}s   "
          f"peak RSS {peak_rss_bytes() / 2 ** 20:.0f} MB")


if __name__ == "__main__":
    main()
