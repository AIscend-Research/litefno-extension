"""Plot params vs test_vrmse for the LiteFNO rank sweep on a single dataset.

Reads each ``r<rank>.jsonl`` produced by ``scripts/run_rank_sweep.sh``, takes
the parameter count from any training-step record and the final-epoch
``test_vrmse`` from the trailing summary line, and plots one point per rank
on a log-x axis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_run(path: Path) -> tuple[int, float, int] | None:
    """Return (params, test_vrmse, rank) for a single sweep run, or None.

    The trailing record (written by ``test_at_end``) carries ``test_vrmse``
    but not ``params``; pull params from any earlier training-step record.
    """
    params = None
    test_vrmse = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if params is None and "params" in rec:
                params = int(rec["params"])
            if "test_vrmse" in rec:
                test_vrmse = float(rec["test_vrmse"])
    if params is None or test_vrmse is None:
        return None
    rank = int(path.stem.lstrip("r"))
    return params, test_vrmse, rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("outputs/logs/gray_scott_reaction_diffusion_rank_sweep"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/figures/gray_scott_rank_pareto.png"),
    )
    parser.add_argument("--title", default="LiteFNO on gray_scott: params vs test vRMSE")
    args = parser.parse_args()

    paths = sorted(args.log_dir.glob("r*.jsonl"), key=lambda p: int(p.stem.lstrip("r")))
    if not paths:
        raise SystemExit(f"No r*.jsonl files in {args.log_dir}")

    points = [load_run(p) for p in paths]
    points = [pt for pt in points if pt is not None]
    if not points:
        raise SystemExit(f"No completed runs (no test_vrmse) found in {args.log_dir}")
    points.sort()  # by params

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ranks = [p[2] for p in points]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    for x, y, r in zip(xs, ys, ranks):
        ax.annotate(f"r={r}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("test vRMSE")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for x, y, r in zip(xs, ys, ranks):
        print(f"  rank={r:>3}  params={x:>9}  test_vrmse={y:.6e}")


if __name__ == "__main__":
    main()
