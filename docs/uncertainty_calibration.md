# Are the confidence scores honest? (H10 / ext29)

ext28 showed that ensemble disagreement *ranks* the surrogate's errors almost
exactly as an oracle reading the true error would. This asks the other question,
and gets the opposite answer: the ranking is excellent and the numbers are not
honest. Every interval the ensemble reports is too narrow, at every confidence
level, on both splits, by a factor between 1.4 and 5.2.

Those two results are complementary rather than contradictory, and the reason is
one line: **ranking is invariant to scale.** A signal that reports every
uncertainty at a fifth of its true size ranks identically to one that reports it
correctly. ext28 measured discrimination; this measures whether the number means
what it says.

## Reliability for a regression

A classifier bins predictions by confidence and asks whether 70%-confident
predictions are right 70% of the time. This repo predicts fields, so the question
is asked two ways.

**Quantile calibration.** Treat the ensemble as a Gaussian predictive law with
mean `mu` and spread `sigma`. For nominal central coverage `p` the interval is
`mu +/- Phi^-1((1+p)/2) * sigma`, and a calibrated model has the truth land
inside it `p` of the time. ECE is the mean absolute gap between observed and
nominal coverage across levels; MCE is the largest.

**Error-vs-sigma.** Bin predictions by the sigma they claim and compare each
bin's claimed sigma to the RMSE it achieves. Reported normalised by the overall
RMSE, so it is scale-free.

Calibration is a property of individual predictions, so the unit is one pixel of
one channel of one step -- 241664 predictions on a held-out fold rather than 118.

## Overconfident everywhere

| nominal | in-distribution | out-of-distribution |
| --- | --- | --- |
| 50% | 45.8% (-4.2) | 38.3% (-11.7) |
| 80% | 69.8% (-10.2) | 62.2% (-17.8) |
| 90% | 77.0% (-13.0) | 69.2% (-20.8) |
| 95% | 81.3% (-13.7) | 73.3% (-21.7) |

| split | ECE | MCE |
| --- | --- | --- |
| in-distribution | 0.066 | 0.137 |
| out-of-distribution | **0.131** | **0.217** |

Every gap is negative at every level on both splits: the intervals are never too
wide, only too narrow. Calibration error **doubles** off-distribution. A nominal
95% interval covers 73% of the held-out regime -- a stated 1-in-20 failure rate
that is really worse than 1-in-4.

## The shape is worse than the coverage numbers admit

| bin | in-dist achieved/claimed | out-of-dist achieved/claimed |
| --- | --- | --- |
| 0 (most confident) | 4.6 | **5.2** |
| 3 | 1.7 | 2.1 |
| 6 | 1.4 | 1.8 |
| 9 | 1.6 | 3.1 |
| 11 (least confident) | 2.2 | 4.1 |

sigma-ECE 0.388 in-distribution and 0.495 out, sigma-MCE 1.370 and 1.950.

The ratio never drops below 1.4 in-distribution or 1.8 out of it -- every bin
under-reports its own error -- and the relationship is **U-shaped**. The signal is
least trustworthy exactly where it claims to be most certain: the most-confident
bin understates its error 4.6x in-distribution and 5.2x out of it.

That shape is the practical finding. A monotone under-report would be a units
problem. A U-shape means the discrepancy is a function of the claimed sigma, so
the pixels the model is surest about are the ones whose error bars are most
wrong.

## One scale factor does not fix it

A single constant, fit on in-distribution data only -- which is where a
practitioner could actually fit it -- lands at `sigma *= 1.16`.

| split | ECE before | ECE after | MCE before | MCE after |
| --- | --- | --- | --- | --- |
| in-distribution | 0.066 | 0.036 | 0.137 | 0.105 |
| out-of-distribution | 0.131 | 0.094 | 0.217 | 0.185 |

Real improvement, and not enough. Rescaled out-of-distribution ECE (0.094) is
still worse than *uncorrected* in-distribution ECE (0.066). The U-shape is why:
no constant can flatten a discrepancy that varies with the claimed sigma, and
recalibrating on the regimes you have does not reach the regime you do not.

## What to do with the signal

Use it to **order** predictions -- triage, deferral priority, which steps to
check. ext28 shows that ordering is essentially optimal.

Do not use it as an error bar. It will mislead by a factor of two to five, most
severely on the predictions it is most confident about, and more so exactly when
the model has left its training distribution -- which is when an error bar would
have been worth having.

## Scope and the direction of the bias

Two choices push toward *apparent* overconfidence and are stated rather than
buried. The Gaussian assumption is doing real work; and the sample standard
deviation of four members is biased low even where the variance estimate is
unbiased. So the effect size here is an upper bound.

The *sign* is not in doubt at this magnitude -- a four-member bias does not
manufacture a 5.2x under-report -- but a larger ensemble would shrink the
numbers. A finding of *under*confidence would have been the surprising one.

One fold (bubbles, the 1.95x gap), one seed, pilot scale: 4 training
trajectories per regime, 30 epochs, 4 members.

Reproduce with:

    python3 scripts/uncertainty_calibration.py --data-dir data/processed/gs_pilot \
        --held-out bubbles --members 4 --epochs 30 \
        --out-dir results/pilot --fig-dir figures/pilot

Outputs `ext29_reliability_bubbles.csv`, `ext29_sigma_bins_bubbles.csv`,
`ext29_summary_bubbles.csv`, `ext29_calibration_bubbles.png`.
