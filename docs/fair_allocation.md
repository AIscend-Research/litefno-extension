# What does a surrogate's error cost a fair decision? (ext22, H3)

`scripts/fair_allocation.py`, `src/litefno/allocation.py`,
`src/litefno/models/allocator.py`

A neural operator is usually scored on field error. If its output is going to be
used for something, the number that matters is what the *decision* taken on it
loses. This extension bolts a resource-allocation layer onto the surrogate --
split a scarce resource across 16 regions of the reconstructed ecosystem, under
fairness rules ranging from pure max-efficiency to max-min -- and asks what the
surrogate's error costs in welfare rather than in VRMSE.

H3: that cost is not a property of the surrogate alone. It depends on which
fairness rule sits downstream, and the dependence is derivable in closed form
before anything is trained.

**The headline: fairness does not cost robustness. The reverse.** Sensitivity to
surrogate error is U-shaped in the fairness parameter with an exact zero at the
envy-free point, rising toward *both* max-efficiency and max-min. The
efficiency-maximising rule is among the most fragile things you can put
downstream of a surrogate.

## The setup, and what in it is a modelling choice

The ecosystem is `lambda_omega` from `src/litefno/systems.py` -- the Hopf normal
form that predator-prey cycles reduce to -- run in a defect-bearing,
incompletely relaxed regime so that region populations actually differ. A 4x4
partition gives 16 regions with a population spread (max/min) of 1.570.

Region `r` has population `g_r`, a scarce divisible resource `B` is split as
`a_r`, and the region realises `x_r = g_r a_r`: a unit of resource does more
good where there are more animals. The rule is the alpha-fair family
(Mo & Walrand 2000), whose optimum on the simplex is a single exponent:

    a_r  ∝  g_r^((1-alpha)/alpha)

| alpha | allocation | the rule it is |
|---|---|---|
| 0 | all to `argmax g` | max-efficiency (utilitarian) |
| 1/2 | `a ∝ g` | proportional to population |
| 1 | equal | proportional fairness = **envy-free** |
| 2, 4, 8 | `a ∝ g^-1/2 … g^-7/8` | increasingly egalitarian |
| inf | `a ∝ 1/g` | **max-min** (Rawlsian) |

Only the surrogate is physics. The gains are read as `1.5 + u` averaged over a
block, and nothing in the PDE says that is what a region deserves -- it is a
stylized decision problem, and the result is about *error propagation through a
composition*, which is measurable, not about what is ethically owed, which a PDE
cannot answer.

### Where envy-freeness lands, and why it is not more interesting than that

Region `r` values bundle `a_s` at `g_r a_s`, so it envies `s` exactly when
`a_s > a_r` -- the gain cancels. With one homogeneous divisible resource,
envy-freeness therefore *forces equal division* (Foley 1967, Varian 1974), which
is the rule at alpha = 1. That is a theorem, not an artefact of this setup, and
it has a consequence: **the envy-free allocation does not depend on the state at
all**, so no forecast can improve or damage it. Its row in every table below is
an exact zero.

The informal reading -- envy over *outcomes*, `x_s > x_r` -- forces equal
outcomes instead, which is the rule at alpha = inf. The two readings of envy
bracket the family rather than adding anything to it.

## Result 1: the fragility law

Expanding the welfare around its own optimum under a perturbed gain
`ghat = g e^eta` gives two statements. The allocation moves by exactly
`|1-alpha|/alpha` times the error, and the relative welfare loss is

    (1 - alpha)^2 / (2 alpha)  x  Var_w(eta)

with `Var_w` the allocation-weighted variance of the log gain errors.

Measured against numerically exact optima on injected noise:

```
the law (measured / predicted)        allocation sensitivity, sd = 0.01
 alpha   sd=0.01  sd=0.05  sd=0.2      alpha    0.25   0.5    1.0   2.0    4.0    8.0
  0.25    1.000    1.005   0.917       theory   3.000 1.000  0.000 0.500  0.750  0.875
  0.5     1.000    1.002   0.983       measured 3.000 1.000  0.000 0.500  0.750  0.875
  2.0     1.000    1.000   0.994
  4.0     1.000    1.000   0.963
  8.0     1.000    1.000   0.873
```

The sensitivity is exact to three decimals at every alpha. The welfare law holds
to 0.5% at sd 0.01-0.05 and degrades to 0.87 at sd 0.2, which is the expected
behaviour of a second-order expansion at large error and is reported rather than
cropped. At alpha = 0 the expansion does not apply at all: the optimum is a
vertex of the simplex and moves discontinuously.

### The same law against a real surrogate

The law was derived for an arbitrary perturbation and checked on i.i.d. noise.
A trained operator's error is neither. It still predicts the plug-in rule's
realised welfare loss to within 1%:

| surrogate | alpha | plug-in loss | law | ratio |
|---|---|---|---|---|
| strong | 0.5 | 2.964e-06 | 2.966e-06 | 0.999 |
| strong | 8.0 | 4.052e-05 | 4.059e-05 | 0.998 |
| weak | 0.5 | 2.468e-04 | 2.463e-04 | 1.002 |
| weak | 8.0 | 3.119e-03 | 3.090e-03 | 1.010 |

One reason it transfers: the rule is scale-free in the gains, so an error common
to every region cancels. The starved surrogate's gain error has log sd 0.0834,
of which only 0.0316 survives that cancellation -- 86% of its error in variance
terms never reaches the decision.

## Result 2: the auxiliary network earns its place only when the surrogate is bad

The network (`RegionAllocator`, 2,913 parameters against the surrogate's 7,106)
takes the reconstructed field
and emits a simplex-constrained allocation, trained on realised welfare with no
allocation labels anywhere -- decision-focused learning in the sense of
Elmachtoub & Grigas (2022). Its honest control is not another network but four
lines of numpy: pool the reconstructed field into regions, apply the closed
form (`plugin`).

Relative welfare loss at horizon 8, 192 evaluation decisions, 3 seeds:

```
strong surrogate (VRMSE 0.0068, gain error 0.25%)
 alpha     plugin     shrunk   smoothed    learned  learned_r    uniform  persistence
  0.25   1.16e-05   1.16e-05   3.51e-05   6.19e-05   6.77e-05   1.75e-02     2.09e-02
  1.0    0.00e+00   0.00e+00   0.00e+00  -2.25e-09  -4.70e-09   0.00e+00     0.00e+00
  8.0    4.05e-05   4.05e-05   1.24e-04   2.44e-04   2.40e-04   5.03e-02     5.96e-02

weak surrogate (VRMSE 0.0305, gain error 6.6%)
  0.25   1.06e-03   9.84e-04   9.71e-04   8.40e-04   9.54e-05   1.75e-02     2.09e-02
  1.0    0.00e+00   0.00e+00   0.00e+00  -3.30e-09  -1.89e-09   0.00e+00     0.00e+00
  8.0    3.12e-03   2.97e-03   2.83e-03   2.37e-03   3.37e-04   5.03e-02     5.96e-02
```

On the strong surrogate the network is **4.3-5.9x worse** than pooling plus the
closed form (30.9x at alpha = 0, where a softmax cannot reach the vertex the
rule wants). Its own approximation error swamps the surrogate error it was
supposed to handle.

On the starved surrogate the same network, trained on that surrogate's own
outputs, is **9.3x better** than the plug-in rule at alpha = 8, and better at
every alpha where the rules differ. Seed spread is 4% of its mean against a 9x
gap, so this is not noise.

### Why it wins, which is not what it looks like

Two interpretable controls exist precisely so the win cannot be waved at:

- `shrunk`: the exponent shrunk toward equal division, one scalar fitted on
  validation -- the hedge a network *could* be learning. It closes 5% of the gap.
- `smoothed`: a box blur before pooling -- the softer pooling operator a
  convolutional encoder *could* amount to. It closes 9%.

Neither is the mechanism, and the allocation spread rules out a third
explanation: the winning network's allocations are as aggressive as the plug-in
rule's (bundle envy 0.382 against 0.390), so it is not hedging by retreating
toward uniform, which would in any case cost it two orders of magnitude.

Inverting the rule recovers the gains an allocation implies, which scores the
network as an *estimator* it was never trained to be:

| implied gain error, alpha = 8 | strong | weak |
|---|---|---|
| `plugin` (block-pooled prediction) | 0.00212 | 0.02255 |
| `learned_robust` (the network) | 0.00521 | 0.00666 |

The network is reading the true region populations out of the reconstructed
field **better than the block mean of that field does**, using spatial structure
the pooling discards. Its estimator error sits near 0.005-0.007 regardless of
surrogate quality, while the plug-in rule's tracks the surrogate exactly. That
gives the crossover directly rather than by inference: the learned allocator is
worth having once the surrogate's pooled region-gain error exceeds roughly
0.005, and is a liability below it.

## Result 3: the forecast earns its place, and a stale observation does not

`persistence` -- allocate from the last state actually observed, no forecast --
is the control that could have killed the pipeline. It does not: over 8 steps
the medium's phase advances far enough that the observed populations are 0.533
off in relative terms, against the strong surrogate's 0.0025, and the welfare
loss is ~1500x larger.

The sharper finding is that **a stale observation is worse than no observation**.
At every alpha except 0, `persistence` loses more than `uniform`, which ignores
the state entirely (5.96e-2 against 5.03e-2 at alpha = 8). Acting confidently on
an out-of-date state is worse than declining to use the state at all, and the
surrogate is what makes the difference between those two options.

## What fairness costs and what it buys

| alpha | price of fairness | min/max outcome |
|---|---|---|
| 0.25 | 0.121 | 0.267 |
| 1 (envy-free) | 0.157 | 0.671 |
| 8 | 0.168 | 0.945 |

Measured in the efficiency objective, following Bertsimas, Farias & Trichakis
(2011). Moving from near-utilitarian to near-max-min takes the worst-off
region's outcome from 27% of the best-off region's to 95%, for 4.7 percentage
points of total efficiency -- and, by Result 1, at a cost in surrogate-error
sensitivity that is *lower* in the middle of that range than at either end.

## Honest limits

**The decision layer is stylized.** The gains are `1.5 + u` block-averaged, and
that is a units choice, not physics. Results are about how error propagates
through the composition; they say nothing about what any region is owed.

**Envy-freeness is degenerate here by construction.** One homogeneous divisible
resource makes it equal division. A genuine envy-free study needs heterogeneous
valuations over several resource types, where the Eisenberg-Gale/CEEI solution
is envy-free *and* Pareto efficient without collapsing. That is the natural
follow-up and it is a different experiment, not a parameter change.

**The law is second order.** It is 13% off at sd 0.2 and does not cover
alpha = 0 at all.

**Regions are a fixed square partition** chosen for the grid, not for anything
ecological or administrative.

**The evaluation sample is 192 decisions from 12 trajectories x 16 start times**,
which overlap. That buys resolution, not independent degrees of freedom; the
seed spread reported alongside is the honest error bar.

**This system is generated here, not downloaded.** Following the boundary
`systems.py` already draws: no accuracy number on it is comparable to a
published one, and none is averaged with results on The Well's data.

## Reproducing

```bash
python3 scripts/fair_allocation.py           # ~7 minutes on CPU
python3 scripts/fair_allocation.py --quick   # plumbing check only
```

Outputs: `results/extensions/ext22_fragility.csv`, `ext22_arms.csv`,
`ext22_horizon.csv`, `ext22_summary.csv`, and
`figures/extensions/ext22_fair_allocation.png`.
