# Does the pole readout predict failure? (ext20, H1)

`scripts/resonance_risk.py`

H1: modes classified as near-resonant are where autoregressive rollout error
grows fastest, so pole proximity to instability, measured from the weights with
no extra data, predicts deployment failure.

The hypothesis splits into two claims that turn out to have different answers.

| form | claim | result |
|---|---|---|
| strong | pole margin predicts which *modes* go wrong, inside one model | **not supported** beyond wavenumber |
| weak | the risk score orders whole *scenarios* by how badly the model will do | **supported**, AUC 0.983 |

Three seeds, mean +/- sd:

```
strong: margin vs error growth, raw    -0.7844 +/- 0.0262
strong: ... controlling for |k|        -0.1440 +/- 0.0172
strong: |k|-only baseline              +0.7796 +/- 0.0258
extracted vs exact margin              +0.9902 +/- 0.0017
weak:   risk vs rollout error          +0.7739 +/- 0.0373
weak:   AUC, flagging the worse half   +0.9833 +/- 0.0047
```

## Why the raw correlation is not the result

Pole margin correlates with wavenumber, and rollout error correlates with
wavenumber, because high modes are both more damped and harder to predict. The
raw -0.78 is therefore mostly both quantities tracking `|k|` -- and the
wavenumber-only baseline, at +0.78, is exactly as good. Partialling `|k|` out
leaves -0.14.

So the strong form fails in the way that matters: **the pole readout tells you
almost nothing per mode that "high wavenumber is worse" does not already tell
you.** A version of this experiment without the control would have reported
-0.78 and claimed H1.

The -0.14 residual is consistent across seeds rather than noise, so it is a
small real effect, not zero. It is not a basis for a per-mode diagnostic.

## Why the scenario-level result is the useful one

The risk score is the mode-energy-weighted pole margin: each mode's margin
weighted by how much of *that scenario's* energy sits in it. A mode the operator
would amplify does not matter if the scenario never excites it, which is why the
score is per-regime and not a property of the weights alone.

Over 20 scenarios spanning damping and frequency independently, that score
orders scenarios by rollout error at rank correlation +0.77 and separates the
worse half at AUC 0.983 (permutation p < 5e-5). It is computed from the weights
and one input frame, before any long rollout.

The unweighted variant (`risk_max`, the single least-damped mode) is unstable
across seeds -- +0.08, +0.24, -0.68 -- so the energy weighting is not a
refinement, it is what makes the score work at all.

## Two things that were nearly wrong

**Unseeded model construction.** The initialization drew from the global RNG
while only `fit` was seeded, so identical invocations disagreed on the partial
correlation by -0.257 (p = 0.046) versus -0.083 (p = 0.56) -- opposite
conclusions from the same command. The seed is now set before construction, and
the study runs over three seeds and reports the spread rather than trusting one,
because a quantity that moves by more than its own effect size between runs
should not be reported as a single number.

**Scenario energy read from the wrong frame.** The mode energy was originally
taken from each scenario's first step. Every scenario is seeded identically and
its initial condition is generated before any physics is applied, so step 0 is
the *same field* in all of them and all 20 scenarios received an identical risk
score. That looked like a clean null result and was an artifact of where the
frame was taken. Energy is now taken over the whole trajectory and the probe
runs at a mid-trajectory state.

## Reproducing

```bash
python3 scripts/resonance_risk.py            # ~15 minutes on CPU, 3 seeds
python3 scripts/resonance_risk.py --quick    # plumbing check only
```

Outputs: `results/extensions/ext20_mode_risk.csv`, `ext20_scenarios.csv`,
`ext20_summary.csv`, `ext20_across_seeds.csv`, and
`figures/extensions/ext20_resonance_risk.png`.

The significance tests are permutations rather than parametric: the statistics
are rank-based, the scenario sample is 20, and no convenient null distribution
applies. Reported p-values floor at `1 / (draws + 1)` and print as `<5e-05`
rather than `0.0`, because no finite number of shuffles shows an arrangement is
impossible.
