# Reading the poles out of a trained operator (ext19)

`scripts/operator_poles.py`

SpecScope steps 1-3. Everyone treats a trained FNO's spectral weights as opaque
parameters. This treats them as an empirical transfer function of the system,
extracts pole structure from it, and then does the part that makes the first two
steps worth anything: scores the extracted poles against an answer known in
closed form.

## What is extracted

An FNO layer is diagonal in Fourier space by construction. The spectral
convolution multiplies mode `k` by a channel-mixing matrix and touches no other
mode, and the pointwise skip is the same matrix at every mode, so with the
activation linearized the whole network collapses to one matrix per mode:

```
M_k = P (g L_k^(L)) ... (g L_k^(1)) Q,     L_k^(l) = W_k^(l)T + S^(l)
```

The model predicts the next state directly, so `M_k` is the one-step propagator
restricted to mode `k` and its eigenvalues are discrete-time poles: `sigma =
log|z|` per step, `freq = arg(z) / 2pi`. That is the same convention
[`litefno/poles.py`](../src/litefno/poles.py) uses on measured data, so a
model's poles and the data's poles are directly comparable.

Two routes to the same object, both run every time:

| route | how | assumes |
|---|---|---|
| `analytic_mode_operators` | composes the CP factors across layers | the linearization; needs the architecture |
| `empirical_mode_operators` | finite-difference probe around a real state | nothing; works on any model |

## Ground truth, which is the unusual part

Interpretability results are usually unfalsifiable because nobody knows what the
network should have learned. Here two systems have a closed form
([`litefno/systems.py`](../src/litefno/systems.py)):

- **rotating diffusion**, `A_t = D lap A + i omega A`. Every mode's pole is
  exactly `exp((-D q^2 + i omega) dt)`: complex, near-neutral at low wavenumber,
  strongly damped at high. Nothing is linearized because the system is linear.
- **advection-diffusion**, exact and with no oscillation at all. The negative
  control: an extractor reporting resonant modes here is inventing them.

The nonlinear **lambda-omega** limit cycle is also run, with a weaker claim
attached, because only its frequency is known in closed form. Linearizing about
a limit cycle is a Floquet problem, so the lab-frame linearization is
time-periodic and has no fixed matrix to compare a network's fixed matrix
against. Claiming a pole ground truth there would be manufacturing agreement out
of a frame choice.

## Results

| arm | test VRMSE | \|z\| mean abs error | \|z\| rank corr | freq mean error |
|---|---|---|---|---|
| rotating | 0.0176 | 0.0061 | **+0.987** | 0.00015 |
| advection | 0.135 | 0.0062 | +0.825 | 0.0020 |
| lambda | 0.0046 | no closed form | | |

**The poles are recovered.** On the system where the answer is known exactly,
extracted pole magnitudes track the analytic ones at rank correlation 0.987 and
the frequencies to 1.5e-4 cycles per step. The extractor is reading real
dynamics out of the weights, not producing plausible numbers.

### Three things that do not work, reported as findings

**The neutral/damped label is below the method's resolution.** Pole
*magnitudes* are recovered to a mean absolute error of 0.0061, but the neutral
band is +/- 0.005 wide, so a mode the system makes near-neutral lands on the
wrong side of the line about as often as not: 1 of 4 genuinely near-neutral
modes is labelled neutral at the default tolerance. The labels only agree with
the truth from a tolerance of 0.0075 upward. The ranking is trustworthy; the
binary classification at a tight threshold is not, and downstream steps should
use the ordering rather than the label.

**The composed route disagrees with the probe by about 25%,** which is expected
and mostly harmless: the log-magnitude gap is -0.258 with a spread of only 0.012
and a rank correlation of +0.966 between the two routes. The disagreement is
almost entirely a single constant scale factor, the linearization gain, so the
composed route ranks modes correctly and cannot be used for an absolute
stability call. Backing the observed gap out gives an effective per-layer gain
of 0.47 against the 0.5 assumed at zero, which is why the constant is so nearly
constant.

**The nonlinear system's operator does not oscillate.** On lambda-omega the
least-damped extracted mode has frequency 0.0000 while the medium visibly cycles
at 0.0239 per step. This is not a failure of the extractor: the oscillation
lives in the base trajectory, not in the dynamics of perturbations about it, and
the instantaneous Jacobian of an oscillatory system need not have oscillatory
poles. It is a caution for anyone expecting a spectral readout of a cycling
system to show the cycle.

## Reproducing

```bash
python3 scripts/operator_poles.py                # about 3 minutes on CPU
python3 scripts/operator_poles.py --quick        # plumbing check only
python3 scripts/operator_poles.py --probe-check  # finite-difference plateau
```

The finite-difference step size is checked rather than assumed: over eps from
1e-1 to 1e-4 the probed operator moves by at most 1.4e-4 relative, so the
default sits on the plateau between curvature error and float32 cancellation.

Outputs: `results/extensions/ext19_ground_truth.csv`,
`ext19_route_agreement.csv`, `ext19_summary.csv`, and
`figures/extensions/ext19_operator_poles.png`.

## Known limitation

The composed route is exact for the linearized network except in the `kx = 0`
column, where the retained spectral block holds `+ky` and `-ky` with independent
weights, is therefore not conjugate-symmetric, and `irfft2` quietly symmetrizes
it on the way out. The composition does not model that step. The effect is under
half a percent on the models tested and is pinned by a test so it cannot grow
silently.

## The instrument has since been ablated

Everything above shows the extractor *recovers* the exact poles. It does not
show the poles come from learning rather than from the architecture -- a
spectral operator is mode-indexed before training, and ext20's wavenumber
baseline and ext21's initialization result both gave reason to worry.

ext33 runs this identical readout on weights that never learned: an untrained
model at the same seed, the trained weights shuffled within each tensor, and
weights resampled to the trained moments. The structure does not survive. On the
rotating system the trained arm scores +0.9876 against the exact magnitudes and
no control exceeds +0.12; frequency error is 1.86e-4 trained against 2.2e-2 to
4.7e-1 for the controls. See [pole_ablation.md](pole_ablation.md).

It also records why the obvious control could not be used: in both closed-form
systems the exact pole is a function of |k| alone (rho = -1.0000 and -0.9996), so
partialling out wavenumber removes the ground truth and the ablation has to be
done on the weights instead.

