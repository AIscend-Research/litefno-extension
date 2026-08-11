#!/usr/bin/env python3
"""Generate a *self-simulated* Gray-Scott reaction--diffusion dataset.

This produces a local proxy for The Well's ``gray_scott_reaction_diffusion``
dataset by numerically integrating the Gray-Scott PDE on a periodic grid. It is
NOT the original Well data: absolute error magnitudes are therefore not directly
comparable to the published numbers. It exists so that the architecture/training
*sweeps* (width, rank, seed, Fourier modes) can be run as genuine experiments and
their *relative* trends measured on identical, reproducible inputs.

Gray-Scott system (two chemical concentrations u, v):

    du/dt = Du * lap(u) - u v^2 + F (1 - u)
    dv/dt = Dv * lap(v) + u v^2 - (F + k) v

Different feed/kill rates (F, k) produce qualitatively different patterns
(spots, worms, mazes, solitons), mirroring the multi-regime structure of the
original benchmark. Trajectories are simulated at 2x resolution and block-mean
downsampled to the target grid (anti-aliased, matching the preprocessing fix
documented in notes_deviations.md).

Output: HDF5 files {train,valid,test}.h5 each holding a 5D array under key
``data`` with shape (n_traj, n_steps, H, W, fields=2), matching the loader in
``litefno.data``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

# Pearson-style (F, k) regimes that yield visually distinct dynamics.
REGIMES = [
    (0.0367, 0.0649),  # mitosis / dividing spots
    (0.0545, 0.0620),  # maze / worms
    (0.0180, 0.0510),  # solitons / gliders
    (0.0140, 0.0540),  # moving spots
    (0.0220, 0.0510),  # worms
    (0.0260, 0.0550),  # stable spots
]

DU = 0.16
DV = 0.08
DT = 1.0


def laplacian(a: np.ndarray) -> np.ndarray:
    """5-point periodic Laplacian over the last two axes."""
    return (
        np.roll(a, 1, axis=-2)
        + np.roll(a, -1, axis=-2)
        + np.roll(a, 1, axis=-1)
        + np.roll(a, -1, axis=-1)
        - 4.0 * a
    )


def seed_initial(n: int, h: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """u=1, v=0 background with a few randomly placed seeded blobs of reactant."""
    u = np.ones((n, h, h), dtype=np.float64)
    v = np.zeros((n, h, h), dtype=np.float64)
    for i in range(n):
        n_blobs = rng.integers(1, 4)
        for _ in range(n_blobs):
            r = rng.integers(h // 12, h // 6)
            cy, cx = rng.integers(r, h - r, size=2)
            u[i, cy - r : cy + r, cx - r : cx + r] = 0.50
            v[i, cy - r : cy + r, cx - r : cx + r] = 0.25
    # small symmetry-breaking noise
    u += 0.01 * rng.standard_normal(u.shape)
    v += 0.01 * rng.standard_normal(v.shape)
    return np.clip(u, 0, 1), np.clip(v, 0, 1)


def simulate_split(
    n_traj: int,
    n_steps: int,
    target_h: int,
    sim_factor: int,
    sub_steps: int,
    warmup: int,
    seed: int,
) -> np.ndarray:
    """Simulate ``n_traj`` Gray-Scott trajectories, evenly split across regimes."""
    rng = np.random.default_rng(seed)
    sim_h = target_h * sim_factor
    out = np.empty((n_traj, n_steps, target_h, target_h, 2), dtype=np.float32)

    # Assign each trajectory a regime round-robin so every split is stratified.
    regime_ids = np.array([t % len(REGIMES) for t in range(n_traj)])

    for rid in range(len(REGIMES)):
        idx = np.where(regime_ids == rid)[0]
        if idx.size == 0:
            continue
        F, k = REGIMES[rid]
        u, v = seed_initial(idx.size, sim_h, rng)

        # warmup so the initial transient starts forming structure
        for _ in range(warmup):
            uvv = u * v * v
            u += DT * (DU * laplacian(u) - uvv + F * (1.0 - u))
            v += DT * (DV * laplacian(v) + uvv - (F + k) * v)

        for s in range(n_steps):
            # block-mean downsample sim_h -> target_h (anti-aliased)
            ub = u.reshape(idx.size, target_h, sim_factor, target_h, sim_factor).mean(axis=(2, 4))
            vb = v.reshape(idx.size, target_h, sim_factor, target_h, sim_factor).mean(axis=(2, 4))
            out[idx, s, :, :, 0] = ub.astype(np.float32)
            out[idx, s, :, :, 1] = vb.astype(np.float32)
            for _ in range(sub_steps):
                uvv = u * v * v
                u += DT * (DU * laplacian(u) - uvv + F * (1.0 - u))
                v += DT * (DV * laplacian(v) + uvv - (F + k) * v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate proxy Gray-Scott dataset")
    ap.add_argument("--out-dir", default="data/processed/gray_scott_proxy", type=Path)
    ap.add_argument("--target-h", type=int, default=32)
    ap.add_argument("--sim-factor", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--sub-steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--train", type=int, default=180)
    ap.add_argument("--valid", type=int, default=60)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20240601)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": args.train, "valid": args.valid, "test": args.test}
    for offset, (split, n) in enumerate(splits.items()):
        arr = simulate_split(
            n_traj=n,
            n_steps=args.n_steps,
            target_h=args.target_h,
            sim_factor=args.sim_factor,
            sub_steps=args.sub_steps,
            warmup=args.warmup,
            seed=args.seed + 101 * offset,
        )
        path = args.out_dir / f"{split}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=arr, compression="gzip", compression_opts=4)
        print(
            f"[{split}] {path}  shape={arr.shape}  "
            f"u[min/mean/max]={arr[...,0].min():.3f}/{arr[...,0].mean():.3f}/{arr[...,0].max():.3f}  "
            f"v[min/mean/max]={arr[...,1].min():.3f}/{arr[...,1].mean():.3f}/{arr[...,1].max():.3f}"
        )


if __name__ == "__main__":
    main()
