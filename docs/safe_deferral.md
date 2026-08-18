# Safe Deferral Rate (H9 / ext28)

ext26 measured the price of leaving the training distribution. ext28 asks whether
the surrogate can *avoid* paying it: if the model knew which of its own
predictions were the bad ones, it could hand those back to the real solver and
keep the rest.

The answer is no, and the reason is more interesting than a weak signal. The
confidence signal turns out to be nearly perfect, and the gap still does not
close.

## Why this is a rate and not a score

Deferring here is not free abstention. A deferred step means running the
ground-truth simulation, which is the expense the surrogate existed to avoid. So
the headline is the **Safe Deferral Rate**: the smallest fraction of held-out
steps that must be deferred before the retained ones match in-distribution error.
A method that reaches in-distribution accuracy by deferring 80% of the work has
not saved anything.

## Sensitivity and specificity from a regression

Both need a binary event and this repo predicts fields. A held-out step is
*unsafe* when its per-sample error exceeds a tolerance tau, and tau is derived
rather than chosen: the 95th percentile of the *in-distribution* per-sample
errors, so "unacceptable" means "worse than in-distribution work almost ever is".

    sensitivity = P(deferred | unsafe)    the bad steps caught
    specificity = P(kept     | safe)      the good steps not wasted

Both are reported because either alone is trivially winnable -- defer everything
for perfect sensitivity, defer nothing for perfect specificity.

## The signal, and the two controls

Confidence is **ensemble disagreement**: the spread of the members' predictions,
which ext26's `robust+unc` arm already builds. It uses no labels, so it is
available at deployment time. A test asserts it never touches the targets.

The curve is meaningless without both bounds:

- **random** -- defer a uniformly random subset. Flat in expectation, so any
  descent in the real curve is the signal doing work rather than deferral itself.
- **oracle** -- defer the genuinely worst steps, ranked by true error.
  Unachievable, but it separates "our signal is weak" from "this is impossible".

That second control is what makes the result below a finding instead of a
disappointment.

## A normalisation that would otherwise fake it

VRMSE divides by the variance of the target. Recomputing that per retained subset
would let the curve improve by dropping high-variance targets rather than by
keeping good predictions. Every retained score is therefore normalised by one
fixed variance computed before any deferral, so `q = 0` is exactly ext26's number
and the curve stays comparable along its whole length.

Deferring everything reports NaN, not zero: an empty retained set has no error,
and scoring it as perfect would make 100% deferral the best point on the plot.

## Result: bubbles (a real 1.95x gap)

In-distribution 0.04600, held-out 0.08958, 53% of steps unsafe.

| defer | retained | vs in-dist | sens | spec |
| --- | --- | --- | --- | --- |
| 0% | 0.08958 | 1.95x | 0.00 | 1.00 |
| 10% | 0.08174 | 1.78x | 0.19 | 1.00 |
| 25% | 0.07303 | 1.59x | 0.48 | 1.00 |
| 50% | 0.05635 | 1.22x | 0.94 | 1.00 |
| 65% | 0.05111 | 1.11x | 1.00 | 0.75 |
| 80% | 0.04816 | 1.05x | 1.00 | 0.44 |

**The Safe Deferral Rate is never reached** -- and neither is it reached by the
oracle. Even deferring 80% of steps leaves retained error at 1.05x
in-distribution.

That is the finding. The gap is not a weakness of the confidence signal, because
the signal is essentially optimal:

| defer | confidence | oracle | random | gain captured |
| --- | --- | --- | --- | --- |
| 10% | 0.08174 | 0.08174 | 0.08878 | 100% |
| 25% | 0.07303 | 0.07294 | 0.08996 | 99% |
| 50% | 0.05635 | 0.05635 | 0.09084 | 100% |
| 80% | 0.04816 | 0.04816 | 0.09489 | 100% |

Ensemble disagreement ranks the held-out steps in almost exactly the order true
error would, capturing 100% of the achievable gain at nearly every rate, while
random deferral stays flat around 0.089-0.095. The ranking is as good as ranking
can be.

So the out-of-distribution gap on this fold is **not concentrated in a
discardable minority of ambiguous cases**. It is spread across the whole held-out
regime, and no threshold on any signal can remove it -- selective prediction is
the wrong tool for this failure, and the oracle control is what proves that
rather than leaving it as a limitation of the method.

What the signal *is* good for is triage. Specificity holds at a perfect 1.00
through 50% deferral while sensitivity climbs to 0.94, and at 55% deferral it
catches every unsafe step with specificity still 0.96. As a detector of which
predictions to distrust it is excellent; as a route back to in-distribution
accuracy it is not available at any price worth paying.

## The other fold, and why it is reported

The first run held out `maze`, chosen because ext10 put it at 1.3% of variance
below mode 8 -- an extreme spectral outlier that the standing hypothesis said
should be among the hardest folds. It came out **easier** than the regimes the
model trained on: held-out 0.04798 against 0.05105 in-distribution, a 0.94x gap.
ext26's independent baseline agrees at 0.90x.

With no gap, every arm reaches "in-distribution" at 0% deferral and the Safe
Deferral Rate is 0% for confidence, oracle and random alike -- the metric is
vacuous rather than favourable. Only 2 of 118 steps clear tau, so sensitivity
pins at 1.00 from the first nonzero rate and the detection curve is computed on
two positives.

It is kept in `ext28_deferral_maze.csv` because a metric that reports 0% on a
fold where nothing was achieved is worth showing next to one that reports "never
reached" on a fold where something real was measured.

## Scope

Pilot scale: 118 held-out steps per fold, 4 members, 30 epochs, one seed. Two
folds, not six. The claim that the gap is not concentrated in a discardable
minority rests on the oracle bound for `bubbles` and should be checked on the
remaining folds before being generalised.

Reproduce with:

    python3 scripts/safe_deferral.py --data-dir data/processed/gs_pilot \
        --held-out bubbles --members 4 --epochs 30 \
        --out-dir results/pilot --fig-dir figures/pilot
