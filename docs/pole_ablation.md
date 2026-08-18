# Ablating the instrument: does the pole readout need a trained operator? (H14 / ext33)

ext19 extracts a per-mode transfer function from a trained model, fits poles to
it, and scores those poles against an answer known in closed form. It reports a
rank correlation of 0.987 against the exact magnitudes and a frequency error of
1.5e-4. ext20 and ext21 are built on that readout.

None of that establishes the poles come from **learning**. A spectral operator
has mode-indexed structure before it is trained at all -- the truncation, the CP
factorization and the initialization are all functions of wavenumber. The repo
had two specific reasons to worry:

- ext20 found a **wavenumber-only baseline** reproduced the resonance-risk
  correlation (+0.78 against the readout's -0.78, leaving -0.14 after
  partialling). Structure that looks like physics can be structure that is
  merely smooth in `|k|`.
- ext21 found the CP mode basis is **set by initialization rather than physics**:
  two models trained on the *same* regime share no more spectral basis (0.258)
  than models trained on different ones (0.230-0.283).

So the readout could plausibly have been reading its own construction. This runs
the identical extraction on weights that never learned anything.

## The arms

Every arm goes through the same probes, the same pole fit and the same scoring.
Only the weights differ.

| arm | what it is |
| --- | --- |
| `trained` | the ext19 arm: initialise, fit, extract |
| `untrained` | identical architecture and seed, **zero** optimizer steps |
| `shuffled` | the **trained** weights, entries permuted within each tensor |
| `resampled` | fresh Gaussians matched to each trained tensor's mean and std |

`shuffled` is the sharpest control: it preserves each tensor's exact multiset of
values -- mean, variance, extremes, sparsity -- and destroys only the
arrangement, so it matches the trained model's weight distribution to the last
element.

## Result: the readout needs training

| system | weights | VRMSE | rho(extracted, exact) | freq error |
| --- | --- | --- | --- | --- |
| rotating | **trained** | 0.0183 | **+0.9876** | **1.86e-04** |
| rotating | untrained | 1.0085 | +0.1166 | 2.38e-02 |
| rotating | shuffled | 1.0249 | -0.0613 | 4.75e-01 |
| rotating | resampled | 1.0272 | +0.0756 | 2.30e-02 |
| advection | **trained** | 0.1348 | **+0.7611** | **1.33e-03** |
| advection | untrained | 1.0202 | -0.2177 | 4.69e-01 |
| advection | shuffled | 1.0181 | +0.0225 | 2.22e-02 |
| advection | resampled | 1.0236 | -0.2413 | 2.71e-02 |

On the rotating system the trained arm reproduces ext19's headline exactly
(+0.9876 against the documented 0.987; frequency error 1.86e-4 against 1.5e-4),
and **no control gets past +0.12**. The gap is **0.871**. On advection the gap is
**0.520**.

Frequency separates the arms even more cleanly than magnitude: the trained arms
are two to three orders of magnitude more accurate (1.86e-4 and 1.33e-3) than
every control (2.2e-2 to 4.7e-1).

**The pole structure does not survive removal of training.** ext19's instrument
measures the operator, not the architecture, and the claims downstream of it do
not inherit an artifact.

Worth stating plainly because it could have gone the other way: given ext20's
wavenumber baseline and ext21's initialization result, a null here was a live
possibility. It did not happen.

## The control this study could not use, and why

The obvious control would be ext20's: partial the wavenumber out of both sides.
**It is undefined here.** In both closed-form systems the exact pole magnitude is
a function of `|k|` alone -- measured `rho(exact, radius)` is **-1.0000** for
rotating and **-0.9996** for advection, and every radius maps to a single
magnitude. Partialling radius out of the ground truth removes the ground truth,
leaving an identically-zero residual.

That is why the control here is an ablation of the *weights* rather than a
partial correlation: the ablation arms hold the entire wavenumber structure
fixed -- same grid, same truncation, same architecture -- and vary only what the
weights contain.

The advection partials that the script does compute (-0.11 to +0.13) should be
read as noise rather than signal, since they sit on a residual worth about 0.04%
of the variance. They are recorded in the CSV and deliberately not used in the
verdict.

## An asymmetry worth noticing

Advection's trained arm scores lower on magnitude (+0.76) than rotating's
(+0.99). That is consistent with ext19's own framing: advection is the
**negative control system**, chosen because it has no oscillation anywhere, so
"which mode is most resonant" has less to rank. Its frequency correlation is the
one that is high (+0.95) -- the trained model correctly reports near-zero
frequency almost everywhere, and the controls do not.

## Scope

Two closed-form systems, one seed per cell, 60 epochs, 32x32, modes 10, rank 8,
the ext19 defaults. Single-seed means the *gap* is the robust quantity, not the
third digit of any single correlation; a gap of 0.87 and 0.52 against controls
that cluster near zero is not a seed effect, but a smaller gap would need
replication.

The nonlinear `lambda` system is not ablated here because only its frequency is
known, so the magnitude comparison that carries this result does not exist for it.

Reproduce with:

    python3 scripts/pole_ablation.py --systems rotating advection --epochs 60

Outputs `ext33_ablation_summary.csv`, `ext33_ablation_modes.csv`.
