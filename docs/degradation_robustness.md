# Accuracy under input degradation (H8 / ext27)

ext27 sweeps a synthetic capture chain from 0% to 100% severity and asks whether
training on that chain buys anything back. It does -- a median **81%** of the
degradation-induced error rise is removed -- but the three things that qualify
that number are the reason this document is longer than one table.

## The chain

"Smartphone degradation" is clinical-imaging vocabulary: a photograph re-shot on
a phone picks up optical blur, sensor noise, an exposure shift and encoding
artifacts. This repo predicts Gray-Scott fields on a periodic 32x32 grid, so the
chain is reproduced as a synthetic analog applied to the model's *input state*,
with one severity knob `s`:

| stage | operation | at s = 1 |
| --- | --- | --- |
| illumination | per-sample gain and offset | +/- 30% gain, +/- 0.20 sigma offset |
| optics | periodic Gaussian blur | sigma = 1.2 px |
| sensor | additive Gaussian noise | 0.30 sigma (~10.5 dB) |
| encoding | uniform quantisation | step = 0.40 sigma |

This is a synthetic analog, not a photograph of anything. The claim it supports
is about a model's response to a parameterised input corruption.

Two properties are load-bearing and tested rather than assumed. Every component
is **exactly** the identity at `s = 0`, so the clean column is genuinely clean
and every reported excess is measured against the untouched field -- the run
confirms it independently, since both corruptions return bit-identical VRMSE
(0.03729) at zero severity. And the blur is done in Fourier space with
wraparound, because ext24 established that on this grid the Fourier modes *are*
the lattice Laplacian's eigenvectors, so a periodic blur is the physically right
one and is exact at sigma = 0 rather than approximate.

Scales are relative to the field's own standard deviation. ext10 measured the
six regimes differing by two orders of magnitude in where they keep their
variance, so a fixed absolute noise floor would mean something different on each.

## The arms

`baseline` trains on clean inputs. `robust` is identical except that every batch
is degraded at a severity drawn uniformly on [0, 1] -- the model must handle the
whole range, not one operating point.

## What it bought

Matched corruption, the chain `robust` trained on:

| severity | baseline | robust | rise closed | robust better? |
| --- | --- | --- | --- | --- |
| 0% | 0.03729 | 0.10328 | n/a | no |
| 10% | 0.05499 | 0.10467 | 92% | no |
| 25% | 0.12107 | 0.11366 | 88% | yes |
| 50% | 0.28518 | 0.15121 | 81% | yes |
| 75% | 0.45584 | 0.20419 | 76% | yes |
| 100% | 0.60570 | 0.26517 | 72% | yes |

The effect is large. From clean to full severity the baseline degrades 16x; the
robust arm degrades 2.6x.

## Three things that qualify it

### The clean-input tax is 177%

At 0% artifacts the robust arm scores 0.10328 against the baseline's 0.03729 --
2.8x worse on exactly the inputs the surrogate was built for. This is why each
arm's excess is measured against *its own* clean error. Charging the robust arm
for the tax twice, once as a worse clean number and again as a smaller closed
fraction, would double-count it; reporting the fraction without the tax would
hide it entirely. The metric separates them so both are visible.

### A closed gap and a worse model are not exclusive

At 10% severity the robust arm closes 92% of the rise and is still *worse* in
absolute terms, 0.10467 against 0.05499. The tax dominates until roughly 20%
severity, which is where the curves cross.

So "92% of the gap closed" and "use the baseline instead" are both true at that
operating point. A table of closed-percentages alone would recommend the wrong
model for the entire low-severity half of the sweep, which is why the
`robust better?` column is reported beside the fraction and not derived from it.

### Most of it was familiarity, not robustness

`robust` trains on the corruption it is tested on, so an in-family win cannot by
itself distinguish a model that got steadier from one that learned this
particular augmentation. The sweep therefore also runs a corruption training
never showed -- pixel dropout, which is neither smooth, nor additive, nor a
monotone map of the value, and so shares no structure with the training chain.

| severity | baseline | robust | rise closed | robust better? |
| --- | --- | --- | --- | --- |
| 10% | 0.17631 | 0.18146 | 44% | no |
| 25% | 0.27475 | 0.26120 | 33% | yes |
| 50% | 0.38669 | 0.35832 | 27% | yes |
| 75% | 0.47186 | 0.43548 | 24% | yes |
| 100% | 0.54215 | 0.50016 | 21% | yes |

Median closure falls from **81% to 27%**, and at 10% dropout the robust arm is
outright worse than the baseline. Roughly two thirds of the apparent benefit
does not survive a change of corruption.

That gap between 81% and 27% is the actual finding. It is only visible because
of the held-out control, and reporting the matched column alone would have
overstated the case by about a factor of three.

## Scope

Pilot scale: 24 training trajectories, 60 epochs, one seed per arm, so no error
bars and no claim that a single fold's movement is real. The uniform-severity
augmentation is deliberately aggressive -- a schedule concentrated on mild
severities would almost certainly trade less clean accuracy for less robustness,
and where on that curve a deployment should sit is not something this sweep
answers.

Reproduce with:

    python3 scripts/degradation_sweep.py --data-dir data/processed/gs_pilot \
        --epochs 60 --out-dir results/pilot --fig-dir figures/pilot

Outputs `ext27_sweep.csv`, `ext27_gap_closed.csv`, `ext27_degradation.png`.
