# Visuals

Everything in `figures/` before this was a matplotlib plot of a result. That is
the right form for a result and the wrong form for two other jobs: showing what
the systems being modelled actually look like, and showing how the parts of the
method fit together. This adds both.

Produced by [`scripts/render_simulations.py`](../scripts/render_simulations.py)
and [`scripts/make_diagrams.py`](../scripts/make_diagrams.py).

```bash
python scripts/render_simulations.py     # figures/simulations/
python scripts/make_diagrams.py          # figures/diagrams/mode_shells.svg
```

## Simulation renders

The models here train on 32x32 fields — Gray-Scott downsampled 4x from The
Well's native 128x128. At that size a maze and a set of spots are a few dozen
pixels each and essentially indistinguishable by eye, which makes it hard to
say what any of the regime names mean.

`render_simulations.py` integrates the Gray-Scott equations directly at 384x384
and renders each regime as a still, a filmstrip, and a looping animation, plus
a six-panel atlas.

| Output | What it is |
| --- | --- |
| `gs_atlas.png` | all six regimes, one panel — the figure for a paper |
| `gs_<regime>.png` | single high-resolution final frame |
| `gs_<regime>_strip.png` | five frames spanning pattern formation |
| `gs_<regime>.gif` | looping animation, 260 px |

### These are illustrations, not the data

This matters enough to state plainly, and it is repeated in the script's
docstring:

- The renders are **independently simulated**, not read from the Zenodo files.
  They will not match a training trajectory frame-for-frame.
- The (F, k) values were **chosen empirically**, not taken from The Well's
  metadata. Several published pairs for these regime names — worms at
  (0.078, 0.061), bubbles at (0.098, 0.057) — collapse to the trivial `u = 1`
  state under this file's `Du/Dv = 0.16/0.08` and forward-Euler step, so the
  nearest living point in the same pattern class is used instead.
- **spirals is the weakest likeness.** Gray-Scott at this diffusion scaling does
  not sustain large rotating spirals; wave fronts close into rings rather than
  winding. What the render shows is spiral-*tip* turbulence — many small curling
  free ends — which is the honest appearance of the regime here, not the clean
  textbook spiral. It is also the only regime whose appearance depends on the
  initial condition: it needs a broken wave front (`init="front"`), because a
  spiral has to wrap around an unterminated edge and scattered blobs never
  provide one.

So: fine for a figure captioned "the regimes look like this", not fine as
evidence about the trained model. Anything quantitative uses the real data.

Two rendering choices are load-bearing. Contrast comes from the 1st and 99.5th
percentiles rather than min/max, because gliders and spirals put nearly the
whole domain at `v ~ 0` and reach their maximum on a handful of pixels — a
min/max stretch renders them almost black. And the PNGs are palette-quantised
on the way out; these are colormapped scalar fields with far fewer than 256
distinct colours, so it is visually free and takes the directory from 27 MB to
10 MB.

## Diagrams

Four SVGs in `figures/diagrams/`, sized by `viewBox` and drawn in
`currentColor` so they invert cleanly on a dark background.

| Diagram | The claim it carries |
| --- | --- |
| `spectral_layer.svg` | where the parameter saving comes from — one shared FFT path, two ways to store `W`: dense at 589,824 params, CP rank 32 at 4,896 |
| `mode_shells.svg` | which Fourier modes get the harmonic bias, and why the selection is a radial ring rather than a point |
| `alpha_fairness.svg` | that error amplification and manipulation gain are the *same* curve, `|1-α|/α`, zero only at `α = 1` |
| `pipeline.svg` | how ext10–ext23 connect: one spine, three readouts of the same reconstructed state |

`mode_shells.svg` is generated rather than hand-drawn, because it depicts a mask
the model computes at runtime. `make_diagrams.py` reimplements the radius rule
in numpy so the figure can be rebuilt without a torch install, and
[`tests/test_diagrams.py`](../tests/test_diagrams.py) pins that reimplementation
against `litefno.models.harmonic.harmonic_mask` across several grid shapes —
including the deliberate row reordering, since the figure lays `k_y` out centred
while the layer stores the negative half folded to the end. A further test fails
if the committed SVG drifts from what the script emits. The other three are
hand-authored: a drawing of an argument is an editorial object, and generating
it from code buys nothing.

## Reusing these

The SVGs carry no external references, no `<script>`, and no embedded raster —
they can be dropped into a LaTeX build (via `svg` or converted), a README, or an
HTML page under a strict CSP. A test asserts that.

The colormap in `render_simulations.py` (`make_cmap`) is monotone in lightness,
so the renders survive greyscale printing.
