"""Spectral variance decomposition of the ground-truth field.

Board task: "use Fourier decomposition to quantify how much variance is captured
by low-frequency harmonic modes vs. high-frequency noise".

Reads the ground-truth radial PSD already committed in
``results/extensions/ext6_energy_spectrum.csv`` and converts it into a
cumulative variance fraction over wavenumber.

Important: ``radial_psd`` in phase3_extensions.ipynb returns the MEAN power per
radial shell (bincount(r, P) / bincount(r)). Total energy in shell k is therefore
psd(k) * N(k), where N(k) is the number of Fourier modes falling in that shell.
Summing psd(k) directly would silently under-weight high-k shells, which contain
many more modes. We reconstruct N(k) with the same integer-radius binning the
notebook uses so the weighting matches exactly.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def shell_counts(grid: int) -> np.ndarray:
    """Replicate the notebook's integer-radius binning on a grid x grid field."""
    yy, xx = np.indices((grid, grid))
    r = np.sqrt((yy - grid / 2) ** 2 + (xx - grid / 2) ** 2).astype(int)
    return np.bincount(r.ravel())


def infer_grid(max_k: int) -> int:
    """Smallest even grid whose max integer radius reaches max_k."""
    for g in range(4, 2048, 2):
        if len(shell_counts(g)) - 1 >= max_k:
            return g
    raise ValueError(f"no grid found for max_k={max_k}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd-csv", type=Path,
                    default=Path("results/extensions/ext6_energy_spectrum.csv"))
    ap.add_argument("--column", default="psd_true")
    ap.add_argument("--grid", type=int, default=None,
                    help="spatial grid size; inferred from max k if omitted")
    ap.add_argument("--out-csv", type=Path,
                    default=Path("results/extensions/ext9_variance_decomposition.csv"))
    ap.add_argument("--out-png", type=Path,
                    default=Path("figures/extensions/ext9_variance_decomposition.png"))
    args = ap.parse_args()

    with args.psd_csv.open() as f:
        rows = list(csv.DictReader(f))
    k = np.array([int(r["k"]) for r in rows])
    psd = np.array([float(r[args.column]) for r in rows])

    grid = args.grid or infer_grid(int(k.max()))
    counts = shell_counts(grid)
    n_k = counts[k]

    energy = psd * n_k
    frac = energy / energy.sum()
    cum = np.cumsum(frac)

    print(f"grid inferred: {grid}x{grid}   shells k={k.min()}..{k.max()}")
    print(f"{'k':>4} {'modes':>7} {'shell share':>12} {'cumulative':>11}")
    for kk, nn, ff, cc in zip(k, n_k, frac, cum):
        print(f"{kk:>4} {nn:>7} {ff:>11.4%} {cc:>10.4%}")

    print("\nvariance captured below cutoff:")
    for cutoff in (2, 4, 8, 12, 16):
        if cutoff <= k.max():
            print(f"  k <= {cutoff:>2}: {cum[k <= cutoff][-1]:>7.2%}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "n_modes", "shell_energy_share", "cumulative_share"])
        for kk, nn, ff, cc in zip(k, n_k, frac, cum):
            w.writerow([kk, nn, ff, cc])
    print(f"\nwrote {args.out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.loglog(k, psd, "k-", lw=2)
        a1.set_xlabel("wavenumber k"); a1.set_ylabel("mean power per mode")
        a1.set_title("Radial PSD (ground truth)")
        a2.plot(k, cum * 100, "k-", lw=2)
        a2.axhline(95, ls=":", c="grey"); a2.axhline(99, ls=":", c="grey")
        a2.set_xlabel("wavenumber cutoff k"); a2.set_ylabel("cumulative variance (%)")
        a2.set_title("Variance captured below cutoff"); a2.set_ylim(0, 101)
        fig.tight_layout()
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=150)
        print(f"wrote {args.out_png}")
    except ImportError:
        print("matplotlib unavailable; skipped figure")


if __name__ == "__main__":
    main()
