# Data-efficiency curve: does the harmonic prior buy data? (H11 / ext30)

Roadmap task: *"Add the data-efficiency curve: accuracy vs training-set size,
harmonic vs plain."*

ext15 built a harmonic-conditioned spectral operator and its control and was
never run. This asks the question a low-resource scientist actually cares about:
not "is the conditioned arm more accurate" but "does the prior substitute for
data". An architectural prior that supplies structure the data would otherwise
have to teach should shift the error-vs-size curve *left*, not merely down.

The answer is no. The multiplier is **1.00x** at every size where it can be
measured, and in **0 of 12** paired runs did the harmonic arm win.

## Design

    plain       CP-factorized spectral LiteFNO             7,490 params
    harmonic    the same, plus a learnable complex bias
                on the Turing shells                       7,710 params (+2.94%)

The two are bit-identical at initialisation -- the bias starts at zero -- so at a
given seed both runs share initialisation, subset and data order and the
comparison is **paired**. That matters here because the expected effect is small
and an unpaired comparison at this scale would be swamped by seed variance.

Size is varied by *trajectory*, not by frame: frames within a trajectory are
consecutive states of the same system and nowhere near independent, so
subsampling frames would shrink the nominal size far more than the information.
Subsets are balanced across the six regimes -- ext10 showed they differ by two
orders of magnitude in where they keep variance, so an unbalanced draw would
confound size with composition, and a small one could omit a regime entirely.
Which trajectories are drawn varies with the seed; both arms at a seed get the
identical draw.

## The curve

| trajectories | pairs | plain | harmonic | paired diff | seeds helped |
| --- | --- | --- | --- | --- | --- |
| 6 | 354 | 0.09221 +- 0.00186 | 0.09230 +- 0.00188 | +0.1% | 0/3 |
| 12 | 708 | 0.06874 +- 0.00428 | 0.06879 +- 0.00428 | +0.1% | 0/3 |
| 18 | 1062 | 0.05200 +- 0.00240 | 0.05203 +- 0.00241 | +0.1% | 0/3 |
| 24 | 1416 | 0.04529 +- 0.00189 | 0.04531 +- 0.00189 | +0.1% | 0/3 |

The two curves lie on top of each other. The harmonic arm is very slightly
*worse* at every size, it never wins on a paired run, and the effect is about
**2% of the seed spread** -- consistent with 220 extra parameters contributing
noise and nothing else.

## The multiplier

| at trajectories | harmonic error | plain needs | multiplier | status |
| --- | --- | --- | --- | --- |
| 6 | 0.09230 | n/a | n/a | out_of_range_low |
| 12 | 0.06879 | 12.0 | **1.00x** | ok |
| 18 | 0.05203 | 18.0 | **1.00x** | ok |
| 24 | 0.04531 | 24.0 | **1.00x** | ok |

At 6 trajectories the harmonic arm is worse than the smallest plain run measured,
so there is no size at which plain matches it inside the measured range; that is
reported as out-of-range rather than as a number. Extrapolating past the ends of
the curve would manufacture a multiplier out of its slope, which is exactly the
figure a reader would most want to trust and least be able to check.

## The differential prediction also fails

ext15's prediction was not that the conditioning helps uniformly -- it was that
it should help **maze and spots**, which hold ~99% of their spatial variance
above mode 8 in the band the bias targets, and do nothing for the low-wavenumber
regimes. Ordered by that quantity, at the largest size:

| regime | var below mode 8 | plain | harmonic | change |
| --- | --- | --- | --- | --- |
| spirals | 77.4% | 0.08796 | 0.08797 | +0.0% |
| gliders | 69.0% | 0.04397 | 0.04399 | +0.1% |
| bubbles | 58.3% | 0.04764 | 0.04771 | +0.2% |
| worms | 30.9% | 0.03564 | 0.03567 | +0.1% |
| maze | 1.3% | 0.03878 | 0.03876 | **-0.1%** |
| spots | 0.6% | 0.02835 | 0.02837 | +0.1% |

maze is the only regime that improves, by 0.1%, and spots -- the more extreme
outlier, and the one the mechanism should favour most -- gets slightly worse.
There is no differential effect to speak of.

Worth distinguishing this from the failure mode ext15 named in advance. It warned
that a *uniform improvement* would be evidence against the mechanism, since that
would mean the extra parameters were helping generically rather than supplying
structure where it was needed. What happened is neither: there is no improvement
at all, uniform or targeted.

## This is the outcome the repo predicted

ext15's own docstring recorded a low prior on the effect size before any run,
citing two committed measurements (the claim's full record across all three
forms is in [harmonic_verdict.md](harmonic_verdict.md)): ext9 killed the spatial
harmonic prior in its low-wavenumber form, and ext12 found that even with documented, exactly periodic
forcing, harmonics carry only 5.4% of temporal variance globally. The narrower
claim -- a mid-spectrum band two regimes genuinely concentrate in -- does not
rescue it at this scale.

## What the curve does show

The plain arm's own scaling is clean and is the more useful number here:

    VRMSE ~ 0.239 * n^-0.521      (residuals within 4.7% across all four sizes)

An exponent near -0.5 is what independent samples would give. Quadrupling the
data from 6 to 24 trajectories cuts error 2.04x, almost exactly the 2x that
exponent predicts. So on this testbed data buys accuracy at the classical rate,
and this particular architectural prior buys none of it.

## Scope

Pilot scale: the training pool is 24 trajectories (4 per regime), so the curve
spans 6 to 24 and a quarter-decade of size. 60 epochs, 3 seeds, width 32,
fundamental 4.0, 3 harmonics. Twelve paired comparisons.

What the design can and cannot support: with the effect at 2% of the seed spread
and 0/12 paired wins, "no benefit at this scale" is well supported. A benefit
that only appears at much larger training sets, or at a different fundamental,
is not excluded -- the band was inherited from ext10 rather than swept, and
sweeping it would be a different experiment.

Reproduce with:

    python3 scripts/data_efficiency.py --data-dir data/processed/gs_pilot \
        --per-regime 1 2 3 4 --seeds 0 1 2 --epochs 60 \
        --out-dir results/pilot --fig-dir figures/pilot

Outputs `ext30_curve.csv`, `ext30_summary.csv`, `ext30_multiplier.csv`,
`ext30_per_regime.csv`, `ext30_data_efficiency.png`.
