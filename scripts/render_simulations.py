"""Render the six Gray-Scott regimes as publication-quality imagery.

The trained models in this repo consume 32x32 downsampled fields from The Well.
That resolution is right for training and useless for looking at. This script
re-simulates the same six regimes from the governing equations at full
resolution so the patterns can actually be seen.

    du/dt = Du * lap(u) - u*v^2 + F*(1 - u)
    dv/dt = Dv * lap(v) + u*v^2 - (F + k)*v

IMPORTANT: these renders are *illustrations of the regimes*, not the training
data. The (F, k) values below were chosen empirically so that each named pattern
class appears under this file's Du/Dv and forward-Euler integrator -- they are
not read off The Well's metadata, and several published (F, k) pairs for these
names die out at this diffusion scaling. Trajectories will therefore not match
the Zenodo files frame-for-frame; the qualitative pattern class is what carries
over. Anything making a quantitative claim must use the real data.

Outputs (under figures/simulations/):
    gs_<regime>.png          single high-resolution final frame
    gs_<regime>_strip.png    filmstrip, pattern formation over time
    gs_<regime>.gif          looping animation
    gs_atlas.png             all six regimes in one panel

Usage:
    python scripts/render_simulations.py                # all six, default size
    python scripts/render_simulations.py --regime maze --size 512
    python scripts/render_simulations.py --quick        # fast smoke test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Rendering is import-guarded so the simulation core stays usable headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402


# --------------------------------------------------------------------------
# Regimes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    """One Gray-Scott parameter point, named after The Well's regime label."""

    name: str
    F: float
    k: float
    steps: int
    blurb: str
    init: str = "patches"  # "patches" or "front"; see seed_fields


# (F, k) per regime, verified alive at Du/Dv = 0.16/0.08 with dt = 1. The
# high-feed values usually quoted for worms and bubbles (0.078/0.061 and
# 0.098/0.057) collapse to the trivial u = 1 state at this scaling, so the
# nearest living point in the same pattern class is used instead.
REGIMES: tuple[Regime, ...] = (
    Regime("gliders", 0.014, 0.054, 24000, "Self-propelling blobs that travel and collide."),
    Regime("bubbles", 0.012, 0.050, 24000, "Rounded cells that inflate and crowd."),
    Regime("maze", 0.029, 0.057, 24000, "Labyrinthine corridors, frozen once formed."),
    Regime("worms", 0.054, 0.063, 20000, "Elongating filaments that branch and merge."),
    Regime("spirals", 0.016, 0.049, 26000, "Curling wave tips -- the one true oscillator.",
           init="front"),
    Regime("spots", 0.030, 0.062, 20000, "Discrete spots that divide and fill space."),
)

REGIMES_BY_NAME = {r.name: r for r in REGIMES}

DU = 0.16
DV = 0.08
DT = 1.0


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def laplacian(a: np.ndarray) -> np.ndarray:
    """Five-point periodic Laplacian on unit grid spacing."""
    return (
        np.roll(a, 1, axis=0)
        + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1)
        + np.roll(a, -1, axis=1)
        - 4.0 * a
    )


def seed_fields(
    size: int, rng: np.random.Generator, mode: str = "patches"
) -> tuple[np.ndarray, np.ndarray]:
    """Initial condition. u saturated, v seeded, symmetry broken by noise.

    ``patches`` scatters square blobs and suits the pattern-forming regimes,
    which reach the same attractor from almost any seed.

    ``front`` lays down a stripe of v with its right half cut away. The
    resulting free wave end is what curls into a spiral tip -- scattered blobs
    never produce one, because a spiral needs an unterminated front to wrap
    around. This is the only regime whose appearance depends on the seed.
    """
    u = np.ones((size, size), dtype=np.float64)
    v = np.zeros((size, size), dtype=np.float64)

    if mode == "front":
        mid = size // 2
        half = max(4, size // 40)
        u[mid - half : mid + half, :mid] = 0.25
        v[mid - half : mid + half, :mid] = 0.50
        u += 0.01 * rng.standard_normal((size, size))
        return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)

    n_patches = max(4, size // 32)
    patch = max(4, size // 24)
    for _ in range(n_patches):
        r0 = rng.integers(0, size - patch)
        c0 = rng.integers(0, size - patch)
        sl = (slice(r0, r0 + patch), slice(c0, c0 + patch))
        u[sl] = 0.50
        v[sl] = 0.25

    u += 0.02 * rng.standard_normal((size, size))
    v += 0.02 * rng.standard_normal((size, size))
    return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)


def simulate(
    regime: Regime,
    size: int = 384,
    steps: int | None = None,
    n_frames: int = 40,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one regime forward, capturing `n_frames` evenly-spaced frames.

    Returns (frames, times) where frames has shape (n_frames, size, size) and
    holds the v field -- v is what carries the visible pattern; u is close to
    its complement.
    """
    steps = regime.steps if steps is None else steps
    rng = np.random.default_rng(seed)
    u, v = seed_fields(size, rng, mode=regime.init)

    # Capture on a square-root schedule: pattern formation is fast early and
    # nearly static late, so uniform sampling wastes most of the frames.
    capture = np.unique(
        (np.linspace(0.0, 1.0, n_frames) ** 2 * steps).astype(int)
    )
    capture_set = set(int(c) for c in capture)

    frames: list[np.ndarray] = []
    times: list[int] = []
    for step in range(steps + 1):
        if step in capture_set:
            frames.append(v.copy())
            times.append(step)
        uvv = u * v * v
        u += DT * (DU * laplacian(u) - uvv + regime.F * (1.0 - u))
        v += DT * (DV * laplacian(v) + uvv - (regime.F + regime.k) * v)

    return np.stack(frames), np.asarray(times)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def make_cmap() -> mcolors.LinearSegmentedColormap:
    """Deep-ink to ember ramp.

    Perceptually monotone in lightness so it survives greyscale printing, which
    viridis-style ramps do too, but this one does not look like every other
    PDE paper.
    """
    stops = [
        (0.00, "#05070f"),
        (0.22, "#12234a"),
        (0.45, "#1d5b8f"),
        (0.65, "#37a0a5"),
        (0.82, "#c2c96b"),
        (1.00, "#fdf3c8"),
    ]
    return mcolors.LinearSegmentedColormap.from_list("ember", stops)


CMAP = make_cmap()


def levels(data: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> tuple[float, float]:
    """Contrast limits from percentiles rather than min/max.

    Several regimes put almost all of the domain at v ~ 0 and reach their
    maximum on a handful of pixels, so a min/max stretch renders them nearly
    black. Clipping the tails is what makes gliders and spirals legible.
    """
    lo, hi = np.percentile(data, [lo_pct, hi_pct])
    return float(lo), float(hi)


def normalise(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi - lo < 1e-9:
        return np.zeros_like(frame)
    return np.clip((frame - lo) / (hi - lo), 0.0, 1.0)


def to_rgb(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (CMAP(normalise(frame, lo, hi))[..., :3] * 255).astype(np.uint8)


def save_still(frames: np.ndarray, regime: Regime, out: Path) -> None:
    lo, hi = levels(frames[-1])
    rgb = to_rgb(frames[-1], lo, hi)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(rgb, interpolation="lanczos")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_strip(frames: np.ndarray, times: np.ndarray, regime: Regime, out: Path) -> None:
    """Filmstrip: five frames spanning formation, shared colour scale."""
    picks = np.linspace(0, len(frames) - 1, 5).astype(int)
    lo, hi = levels(frames[picks])

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.35), dpi=170)
    fig.patch.set_facecolor("#0b0d14")
    for ax, idx in zip(axes, picks):
        ax.imshow(to_rgb(frames[idx], lo, hi), interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(f"t = {times[idx]:,}", color="#9fb0c8", fontsize=10, pad=6)
    fig.suptitle(
        f"{regime.name}   (F = {regime.F}, k = {regime.k})",
        color="#e8edf5",
        fontsize=13,
        y=0.99,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, wspace=0.04)
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_gif(frames: np.ndarray, out: Path, size: int = 260, fps: int = 12,
             colors: int = 64) -> None:
    """Looping animation. Falls back silently if Pillow is unavailable.

    Kept deliberately small: six of these ship together, and a colormapped
    field quantises to 64 colours with no visible loss.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - optional dependency
        print(f"  (skipped {out.name}: Pillow not installed)")
        return

    lo, hi = levels(frames[len(frames) // 3 :])
    images = []
    for frame in frames:
        img = Image.fromarray(to_rgb(frame, lo, hi))
        if img.size[0] != size:
            img = img.resize((size, size), Image.LANCZOS)
        images.append(img.convert("P", palette=Image.ADAPTIVE, colors=colors))

    # Hold the final pattern before looping, otherwise the reset reads as a glitch.
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:] + [images[-1]] * 8,
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


def save_atlas(finals: dict[str, np.ndarray], out: Path) -> None:
    """All six regimes in one panel -- the single figure for a paper."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.4), dpi=180)
    fig.patch.set_facecolor("#0b0d14")
    for ax, regime in zip(axes.ravel(), REGIMES):
        frame = finals[regime.name]
        lo, hi = levels(frame)
        ax.imshow(to_rgb(frame, lo, hi), interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(regime.name, color="#e8edf5", fontsize=15, pad=9)
        ax.text(
            0.5,
            -0.045,
            f"F = {regime.F}   k = {regime.k}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="#8296b0",
            fontsize=10,
        )
    fig.suptitle(
        "Gray-Scott regimes in The Well, re-simulated at full resolution",
        color="#f2f5fa",
        fontsize=17,
        y=0.975,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04, wspace=0.06, hspace=0.20)
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def optimise_pngs(out_dir: Path) -> None:
    """Palette-quantise the written PNGs in place.

    These are colormapped scalar fields, so they carry far fewer than 256
    distinct colours to begin with; quantising is visually free and cuts the
    directory by roughly 4x. Six regimes of un-optimised 200-dpi PNG is 40 MB,
    which is not a thing to put in a git repository.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - optional dependency
        return

    before = sum(p.stat().st_size for p in out_dir.glob("*.png"))
    for path in sorted(out_dir.glob("*.png")):
        img = Image.open(path).convert("RGB")
        img.convert("P", palette=Image.ADAPTIVE, colors=256).save(path, optimize=True)
    after = sum(p.stat().st_size for p in out_dir.glob("*.png"))
    print(f"optimised PNGs: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("figures/simulations"))
    parser.add_argument("--size", type=int, default=384, help="grid resolution")
    parser.add_argument("--regime", type=str, default=None, help="render only this regime")
    parser.add_argument("--frames", type=int, default=40, help="frames captured per run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small grid and short integration, for checking the plumbing",
    )
    args = parser.parse_args()

    size = 128 if args.quick else args.size
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        [REGIMES_BY_NAME[args.regime]] if args.regime else list(REGIMES)
    )

    finals: dict[str, np.ndarray] = {}
    for regime in selected:
        steps = 2000 if args.quick else regime.steps
        print(f"simulating {regime.name:8s} F={regime.F:<6} k={regime.k:<6} "
              f"{size}x{size} x {steps:,} steps")
        frames, times = simulate(
            regime, size=size, steps=steps, n_frames=args.frames, seed=args.seed
        )
        finals[regime.name] = frames[-1]

        save_still(frames, regime, args.out_dir / f"gs_{regime.name}.png")
        save_strip(frames, times, regime, args.out_dir / f"gs_{regime.name}_strip.png")
        if not args.no_gif:
            save_gif(frames, args.out_dir / f"gs_{regime.name}.gif")

    if len(finals) == len(REGIMES):
        save_atlas(finals, args.out_dir / "gs_atlas.png")

    optimise_pngs(args.out_dir)
    print(f"wrote {len(list(args.out_dir.glob('*')))} files to {args.out_dir}")


if __name__ == "__main__":
    main()
