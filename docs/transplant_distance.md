# H2 as a dose-response, on distance and on dose (ext34)

ext21 asked whether resonant CP components transplant across regimes, compared
them to a size-matched damped control, and read a null: 0 of 8 cells put the
resonant arm more than one standard deviation ahead. This was meant to turn that
yes/no into a curve by varying how far the target regime sits from the source.

It does produce a curve, but not the one requested, and on the way it **overturns
ext21's null**. Both of those need saying carefully, because the effect that
survives is small and the practical conclusion barely moves.

## What blocked this experiment, and the number that unblocked it

The first attempt compared a 3-component transplant against a full fine-tune and
found the transplant arms landing within 2e-5 of each other while fine-tuning
moved the error several-fold. That looks like "the subspace carries nothing"
until you ask how much of the network a 3-component transplant actually writes:

| dose | params | of the model | of the spectral layers |
| --- | --- | --- | --- |
| 1 component | 84 | 1.18% | 3.09% |
| 3 components (ext21's dose) | 252 | **3.55%** | 9.26% |
| all 8 components | 672 | 9.46% | 24.71% |
| fine-tune | 7,106 | **100%** | 100% |

ext21's transplant moved **3.55%** of the weights and was implicitly read against
a ceiling that moves 100%. A small effect from a small dose is not evidence of no
effect. So dose became the second axis, and the null had to be re-tested at
several of them.

The matched sets cap at 3 -- the classifier finds 3 resonant and 5 damped
components and ext21 trims to the smaller -- so the matched ladder runs k = 1, 2,
3, and the ceiling is supplied by `transplant_all`, which freezes every
component's mode structure and by construction has no size-matched control.

## The dose axis carries the signal

| dose | % of model | scratch | resonant | damped | gap | wins |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.18% | 0.07968 | 0.07784 | 0.07915 | **+1.7%** | 30/30 |
| 2 | 2.36% | 0.07968 | 0.07715 | 0.07852 | **+1.8%** | 30/30 |
| 3 | 3.55% | 0.07968 | 0.07628 | 0.07816 | **+2.5%** | 30/30 |

**Resonant beat its size-matched damped control in 90 of 90 paired runs**, every
seed 10/10, every individual gap positive (smallest +0.53%). The gap grows with
dose, Spearman(dose, gap) = **+0.764**.

Ceilings, over every cell: scratch 0.07968, `transplant_all` 0.07329 (+8.4% over
scratch), fine-tune 0.01578 (+80.4%).

## ext21's null was a power failure, not a null

This is the part that changes an existing conclusion in the repo, so it is worth
being precise about *why* the two runs disagree.

Re-reading ext21's own published table as a relative gap rather than as a
multiple of seed spread, **6 of its 8 cells are already positive**:

| target | budget | ext21's gap |
| --- | --- | --- |
| slow_diffuse | 2 | +0.62% |
| slow_diffuse | 4 | +2.23% |
| slow_diffuse | 8 | -0.97% |
| slow_diffuse | 16 | +0.17% |
| fast_sharp | 2 | +0.10% |
| fast_sharp | 4 | +1.60% |
| fast_sharp | 8 | +5.85% |
| fast_sharp | 16 | -0.19% |

The two negatives sit at the two **largest** budgets, where scratch has already
converged to fine-tune and there is nothing left to transfer.

ext21 tested each cell against its own seed standard deviation. The effect is
about 2%; **seed spread on the scratch arm here runs 14% to 49%.** An unpaired
one-standard-deviation-per-cell test cannot resolve a 2% effect against 20%
noise, and its `/sd` column (+0.02, +0.40, -0.09, +0.05, +0.00, +0.08, +0.46,
-0.03) is what that looks like from the inside. The test had almost no power and
the null it returned was the expected output of an underpowered test.

What this run changes is the design, not the phenomenon: **pairing per seed**, so
that resonant and damped share an initialisation and a target draw and the seed
noise cancels, then 90 paired comparisons instead of 8 unpaired cells. A 2%
effect is invisible against 20% seed spread and obvious when the spread is
differenced away.

## What this does not rescue

The corrected reading is that the mode basis carries **something**, not that H2's
practical claim works. Freezing *every* component -- the entire claimed shared
physics, 9.5% of the weights -- buys **+8.4%** over scratch, while simply
fine-tuning buys **+80.4%**. Transfer across these regimes is real and large, and
still does not meaningfully decompose into the mode-classified pieces.

So ext21's headline conclusion survives in its practical form and fails in its
mechanistic one. The classifier is selecting real structure -- 90/90 with a
monotone dose-response is not an arbitrary basis -- but the structure it selects
accounts for a few percent of what transfer is worth. ext21's overlap-matrix
argument (same-regime different-seed pairs share 0.258, cross-regime 0.230-0.283)
is separate evidence and is not contradicted here; what this shows is that the
overlap metric is not sensitive enough to see an effect that a paired error
comparison can.

## The requested axis -- distance -- is flat

| ray | d | scratch | finetune | resonant | damped | gap |
| --- | --- | --- | --- | --- | --- | --- |
| diffuse | 0.00 | 0.07231 | 0.01349 | 0.06850 | 0.07031 | +2.6% |
| diffuse | 0.25 | 0.08049 | 0.01519 | 0.07635 | 0.07830 | +2.5% |
| diffuse | 0.50 | 0.08960 | 0.01717 | 0.08538 | 0.08738 | +2.3% |
| diffuse | 1.00 | 0.11085 | 0.02208 | 0.10659 | 0.10860 | +1.8% |
| diffuse | 1.75 | 0.14842 | 0.03307 | 0.14500 | 0.14885 | +2.6% |
| sharp | 0.25 | 0.06494 | 0.01204 | 0.06144 | 0.06316 | +2.8% |
| sharp | 0.50 | 0.05941 | 0.01079 | 0.05627 | 0.05785 | +2.8% |
| sharp | 1.00 | 0.05158 | 0.00929 | 0.04920 | 0.05049 | +2.7% |
| sharp | 1.75 | 0.04685 | 0.01120 | 0.04553 | 0.04634 | +2.0% |

Spearman(distance, gap) = **-0.319** -- the predicted sign, and weak. The gap
ranges +1.8% to +2.8% with no clean decay: the diffuse ray falls from +2.6% to
+1.8% and then rises back to +2.6% at the farthest rung, and the sharp ray is
essentially flat until its last point.

**H2-as-a-curve on regime distance is not supported.** The transplanted subspace
helps about equally at every distance tested, including a regime whose diffusion
and rotation are each moved by a factor of ~3.5. That is a real answer to the
question asked -- the benefit is not regime-local -- but it is a flat line, not a
dose-response, and the dose-response that exists is on dose.

## Scope and the things that would change this

Effect sizes are ~2% against 14-49% seed spread, so **the paired design is
load-bearing**; none of this is visible unpaired, which is exactly the lesson
about ext21. Three seeds, 4 target trajectories, 40 epochs, one source model
reused across every cell so that source-training noise cannot masquerade as a
distance effect.

`d = 0` is the same regime on both rays, so those runs are identical by
construction and 9 of the 90 pairs are duplicates -- **81 independent pairs**, which
does not change a unanimous sign test.

The source model is built at seed 0, so the seed-0 target shares its
initialisation. If the effect were an initialisation artifact it would appear
only there; it is 10/10 at every seed separately.

Reproduce with:

    python3 scripts/transplant_distance.py

Outputs `ext34_distance_cells.csv`, `ext34_distance_summary.csv`.
