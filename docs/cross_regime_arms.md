# Can the cross-regime gap be bought down? (H7 / ext26)

ext14 measured the leave-one-regime-out gap. ext26 adds the two standard answers
to "the held-out domain looks different" and finds that neither works -- and
that the gap itself is not what the repo predicted it would be.

Three results, in increasing order of how much they change the picture:
robustness training moves the gap 2%, ensembling makes it worse, and the
pre-registered spectral hypothesis for *which* regimes are hard is refuted with
the correlation running strongly the wrong way.

## The arms

| arm | training | evaluation |
| --- | --- | --- |
| `baseline` | MSE on clean inputs | single model |
| `robust` | inputs perturbed at 30 dB every batch | single model |
| `robust+unc` | identical to `robust` | mean of 4 members |

`robust+unc` differs from `robust` in exactly one field, the member count, so
the difference between them is the ensemble's contribution with the
augmentation held fixed. The 30 dB scale is ext3's committed definition,
`std * 10 ** (-snr / 20)` on the input only, reused rather than tuned.

Members train in lockstep so the gap-stability curve exists for the ensemble arm
too; evaluating a partially trained ensemble needs every member at the same
epoch.

## Most regimes have no gap at all

| fold | held-out | seen | gap |
| --- | --- | --- | --- |
| spirals | 0.09209 | 0.04443 | 2.07x |
| bubbles | 0.08779 | 0.04685 | 1.87x |
| gliders | 0.05085 | 0.05192 | 0.98x |
| worms | 0.04782 | 0.04940 | 0.97x |
| maze | 0.04454 | 0.04945 | 0.90x |
| spots | 0.02828 | 0.05034 | 0.56x |

Four of the six regimes are *no harder* held out than the five the model trained
on, and spots is nearly twice as easy. "The cross-regime gap" is therefore not a
property of this setup; it is a property of spirals and bubbles specifically.

That distinction is load-bearing for the next table. A fold with `g <= 1` has no
excess to remove, so its "fraction closed" divides by zero or by a small
negative number, and a trivial movement becomes a huge signed percentage. Those
folds report NaN and are excluded from the medians rather than being allowed to
dominate them.

## Neither arm closes it

| arm | median closed | held-out error actually fell |
| --- | --- | --- |
| `robust` | 2% (range -3% to 8%) | 3/6 |
| `robust+unc` | -17% (range -21% to -14%) | 1/6 |

Medians are over the two folds that have a gap. Robustness training moves it by
2%; the only genuine closure is bubbles at 8%, and spirals goes slightly the
wrong way.

The ensemble is worse than useless here. It widened the gap on **both** folds
that had one, and degraded held-out error in **5 of 6** folds -- while improving
seen error. spirals shows it in its purest form: `robust+unc` posted the best
seen-regime error of any arm (0.04372) and the worst held-out error (0.10045).

Ensembling reliably buys in-distribution accuracy and reliably fails to transfer
it. That is consistent with ext28, where the same ensemble's *disagreement* was a
near-perfect failure detector while averaging its members bought nothing
out-of-distribution.

This is also why the `real` flag exists. Reading the ratio alone would credit
`robust` on maze (0.90x to 0.95x looks like movement) when its held-out error
rose; the flag separates a numerator falling from a denominator moving.

## The prediction that did not survive

`cross_regime.py` states a hypothesis written before any of these folds ran.
ext10 measured what fraction of spatial variance each regime keeps below mode 8
-- spirals 77%, gliders 69%, bubbles 58%, worms 31%, maze 1.3%, spots 0.6% --
and the argument was that if what a model transfers is spectral content, then
holding out a regime whose spectrum is unlike the training set should hurt most.
maze and spots, outliers by two orders of magnitude, should be the two worst
folds.

    prediction: maze and spots are the two worst folds
    outcome:    worst two are [bubbles, spirals]  ->  0/2 correct
    rho(variance below mode 8, gap) = +0.943   (mechanism predicts negative)
    rho(|distance from training mean|, gap) = -0.486   (symmetric variant)

Not merely unsupported -- inverted, with near-maximal rank agreement in the
opposite direction. maze and spots are the two *easiest* folds. What predicts
the gap is how much energy a regime keeps at low wavenumbers, not how far its
spectrum sits from the training mean. The symmetric variant, offered as the
weaker hypothesis, also carries the wrong sign for its own claim.

The alternative the docstring named -- that the gap is governed by something
other than spectral distance -- is what the data supports. A mechanism for the
positive correlation is not established here, and rho over six points is a
direction, not a law.

One confound worth stating rather than burying: VRMSE divides by target
variance, and the gap is a ratio of two VRMSEs computed on different sets. That
cancels scale within a fold but not the possibility that held-out and seen sets
are differently normalised, which is a live worry precisely for the regimes ext10
found concentrate their energy differently. The inversion should be reproduced on
an unnormalised error before being treated as physics.

## Is the gap an artifact of the epoch budget?

    spirals  10:2.24 20:2.09 30:2.07
    bubbles  10:1.15 20:1.60 30:1.87
    gliders  10:0.99 20:1.01 30:0.98
      worms  10:0.89 20:0.88 30:0.97
       maze  10:0.86 20:0.91 30:0.90
      spots  10:0.60 20:0.70 30:0.56

Five folds are flat. bubbles is the exception -- still climbing at the end, so
its 1.87x is a lower bound and the true gap under a longer schedule is larger.
Since bubbles is one of only two folds with a gap, that caveat attaches directly
to the headline.

## Scope

Pilot scale: 4 training trajectories per regime rather than the reference 12,
30 epochs rather than 100, one seed per arm. Absolute VRMSE is therefore not
comparable to ext13's or to the reference runs -- only the ratio is
self-comparable, which is why the ratio is what gets reported. With one seed,
a single fold's movement of a few percent is not distinguishable from noise; the
claims that survive that are the ones resting on direction across folds (the
ensemble's 5/6) or on a rank correlation (rho = +0.943).

Reproduce with:

    python3 scripts/cross_regime.py --data-dir data/processed/gs_pilot \
        --epochs 30 --out-dir results/pilot --fig-dir figures/pilot

Outputs `ext26_arms.csv`, `ext26_gap_closed.csv`, `ext26_prediction_check.csv`,
`ext26_curves.json`, `ext26_cross_regime.png`.
