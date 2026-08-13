"""Emit the data-driven diagrams as standalone SVG.

Most diagrams in figures/diagrams/ are hand-authored, because a drawing of an
argument is an editorial object and generating it from code buys nothing. This
one is different: the harmonic shell mask is computed by the model at runtime,
so drawing it by hand would let the picture drift away from the behaviour.

The radius rule below mirrors ``litefno.models.harmonic.harmonic_mask``. It is
reimplemented in numpy so the diagram can be rebuilt without a torch install;
``tests/test_diagrams.py`` pins the two against each other.

Usage:
    python scripts/make_diagrams.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Layout constants, in SVG user units.
CELL = 26
ORIGIN_X = 96
ORIGIN_Y = 232
ACCENT = "#d2673a"
COOL = "#2f7d63"


def shell_radii(fundamental: float, n_harmonics: int, max_mode: int) -> list[float]:
    """Mirror of ``harmonic_shells``: a fundamental and its integer multiples."""
    shells = [fundamental * n for n in range(1, n_harmonics + 1)]
    return [s for s in shells if s <= max_mode]


def shell_mask(modes1: int, modes2: int, shells, width: float = 0.5) -> np.ndarray:
    """Mirror of ``harmonic_mask``, in numpy, returning the signed-ky grid.

    Returned in *centred* ky order (most negative first) rather than the folded
    storage order the layer uses, because a picture of the spectrum should look
    like the spectrum.
    """
    pos = modes1 // 2 + 1
    neg = modes1 - pos
    ky = np.concatenate([np.arange(-neg, 0), np.arange(0, pos)]).astype(float)
    kx = np.arange(modes2).astype(float)
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    mask = np.zeros_like(radius, dtype=bool)
    for shell in shells:
        mask |= np.abs(radius - shell) <= width
    return mask, ky, kx


def render_mode_shells(out: Path, modes1: int = 12, modes2: int = 12,
                       fundamental: float = 3.5, n_harmonics: int = 3) -> None:
    shells = shell_radii(fundamental, n_harmonics, max_mode=max(modes1, modes2))
    mask, ky, kx = shell_mask(modes1, modes2, shells)

    parts: list[str] = []
    add = parts.append

    width_px = ORIGIN_X + modes2 * CELL + 300
    height_px = 470

    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}" '
        f'color="#12151c" role="img" aria-label="The retained Fourier mode grid, with '
        f'the {len(shells)} radial harmonic shells that receive a learnable bias '
        f'highlighted against the modes that do not.">'
    )
    add('<g font-family="ui-sans-serif, system-ui, sans-serif" fill="currentColor">')

    add(f'<text x="30" y="34" font-size="13" font-weight="600" opacity="0.55" '
        f'letter-spacing="0.08em">RETAINED rfft2 BLOCK — {modes1} × {modes2} MODES</text>')

    # Shell arcs, drawn behind the dots.
    for radius in shells:
        r = radius * CELL
        add(f'<path d="M {ORIGIN_X} {ORIGIN_Y - r} A {r} {r} 0 0 1 {ORIGIN_X} {ORIGIN_Y + r}" '
            f'fill="none" stroke="{ACCENT}" stroke-width="1.1" stroke-dasharray="4 4" opacity="0.5"/>')
        add(f'<text x="{ORIGIN_X + r + 6:.0f}" y="{ORIGIN_Y - 6}" font-size="10.5" '
            f'fill="{ACCENT}" opacity="0.9">|k| = {radius:g}</text>')

    # Mode dots.
    for i, ky_val in enumerate(ky):
        for j, kx_val in enumerate(kx):
            cx = ORIGIN_X + kx_val * CELL
            cy = ORIGIN_Y - ky_val * CELL
            if mask[i, j]:
                add(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="6.5" fill="{ACCENT}"/>')
            else:
                add(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="3.2" fill="currentColor" opacity="0.28"/>')

    # Axes.
    axis_bottom = ORIGIN_Y + (abs(ky.min()) + 0.8) * CELL
    axis_right = ORIGIN_X + (kx.max() + 0.8) * CELL
    add(f'<line x1="{ORIGIN_X}" y1="{ORIGIN_Y - (ky.max() + 0.8) * CELL:.0f}" '
        f'x2="{ORIGIN_X}" y2="{axis_bottom:.0f}" stroke="currentColor" stroke-width="1" opacity="0.45"/>')
    add(f'<line x1="{ORIGIN_X - 0.8 * CELL:.0f}" y1="{ORIGIN_Y}" x2="{axis_right:.0f}" '
        f'y2="{ORIGIN_Y}" stroke="currentColor" stroke-width="1" opacity="0.45"/>')
    add(f'<text x="{axis_right + 8:.0f}" y="{ORIGIN_Y + 4}" font-size="12" opacity="0.7">kₓ</text>')
    add(f'<text x="{ORIGIN_X - 6}" y="{ORIGIN_Y - (ky.max() + 1.1) * CELL:.0f}" '
        f'font-size="12" opacity="0.7" text-anchor="middle">k_y</text>')

    # Legend and the claim.
    legend_x = ORIGIN_X + modes2 * CELL + 44
    add(f'<circle cx="{legend_x}" cy="118" r="6.5" fill="{ACCENT}"/>')
    add(f'<text x="{legend_x + 16}" y="123" font-size="12.5">gets a learnable bias b</text>')
    add(f'<text x="{legend_x + 16}" y="141" font-size="11" opacity="0.65">'
        f'{int(mask.sum())} of {mask.size} modes</text>')
    add(f'<circle cx="{legend_x}" cy="170" r="3.2" fill="currentColor" opacity="0.28"/>')
    add(f'<text x="{legend_x + 16}" y="175" font-size="12.5" opacity="0.8">weight only</text>')

    add(f'<text x="{legend_x}" y="228" font-size="12.5" font-weight="600">Why a ring, '
        f'not a point</text>')
    for n, line in enumerate([
        "A Turing pattern selects a wavelength,",
        "not an orientation, so its energy spreads",
        "around the whole shell. Selecting on |k|",
        "follows the physics; selecting on a single",
        "(kₓ, k_y) would catch one slice of it.",
    ]):
        add(f'<text x="{legend_x}" y="{250 + n * 17}" font-size="11.5" opacity="0.75">{line}</text>')

    add(f'<text x="{legend_x}" y="368" font-size="12.5" font-weight="600" fill="{COOL}">'
        f'Where ext10 says the energy is</text>')
    for n, line in enumerate([
        "maze and spots hold ~99% of their spatial",
        "variance in the mid-spectrum Turing band,",
        "not at low wavenumber — which is why the",
        "fundamental sits at 3.5 and not at 1.",
    ]):
        add(f'<text x="{legend_x}" y="{390 + n * 17}" font-size="11.5" opacity="0.75">{line}</text>')

    add(f'<text x="30" y="{height_px - 12}" font-size="11.5" opacity="0.6">'
        f'Shells at a fundamental of {fundamental:g} and its multiples, annulus half-width 0.5. '
        f'Drawn in centred k_y order; the layer stores the negative half folded to the end.</text>')

    add("</g></svg>")
    out.write_text("\n".join(parts) + "\n")
    print(f"wrote {out}  ({int(mask.sum())}/{mask.size} modes in {len(shells)} shells)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("figures/diagrams"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render_mode_shells(args.out_dir / "mode_shells.svg")


if __name__ == "__main__":
    main()
