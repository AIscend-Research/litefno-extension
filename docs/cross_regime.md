# Leave-one-regime-out generalization (ext14)

Board task: *"Test LITEFNO generalization: train on subset of disaster types,
test on held-out type -- measure cross-disaster gap."*

The Well's Gray-Scott dataset ships as six named regimes, each a distinct (F, k)
pair producing a qualitatively different pattern -- bubbles, gliders, maze,
spirals, spots, worms. They are this repo's "types": a model trained on five and
tested on the sixth has to extrapolate to a pattern class it has never seen.

Six folds. Each holds out one regime, trains on the other five, and evaluates the
same trained model twice -- on the held-out regime's test trajectories and on the
five seen regimes' test trajectories. The gap is the ratio.

## Why the ratio and not the absolute error

Both columns come from one model, one training run, one epoch budget. Anything
that shifts the absolute error -- a shorter schedule, fewer trajectories,
different hardware -- shifts both, so the ratio survives choices the absolute
number does not. That matters here: each fold trains on 20 trajectories against
the reference run's 72, so its absolute VRMSE is not comparable to ext13's. The
ratio is comparable to itself across folds, which is what the question needs.

## Result

| held out | held-out VRMSE | seen VRMSE | gap |
| --- | --- | --- | --- |
| spirals | 0.09209 | 0.04443 | **2.07x** |
| bubbles | 0.08779 | 0.04685 | **1.87x** |
| gliders | 0.05085 | 0.05192 | 0.98x |
| worms | 0.04782 | 0.04940 | 0.97x |
| maze | 0.04454 | 0.04945 | 0.90x |
| spots | 0.02828 | 0.05034 | 0.56x |

Median 0.97x, range 0.56x to 2.07x.

**Only two of six regimes cost anything to hold out.** Four are no harder than
the five the model trained on, and spots is nearly twice as *easy*. So there is
no single "cross-regime gap" for this dataset -- there is a gap for spirals and
bubbles, and its absence everywhere else. A study that held out one regime and
reported the number would have drawn an almost arbitrary conclusion depending on
which one it picked.

## Is the gap stable, or an artifact of the epoch budget?

Gap ratio at each checkpoint:

    spirals  10:2.24  20:2.09  30:2.07
    bubbles  10:1.15  20:1.60  30:1.87
    gliders  10:0.99  20:1.01  30:0.98
      worms  10:0.89  20:0.88  30:0.97
       maze  10:0.86  20:0.91  30:0.90
      spots  10:0.60  20:0.70  30:0.56

Five folds are flat, so the budget is not driving them. bubbles is the exception
-- still climbing at the end -- so its 1.87x is a lower bound and a longer
schedule would report a larger gap. Since bubbles is one of only two folds with a
gap at all, that caveat attaches directly to the headline rather than to a
footnote.

## The prediction this was written to test, and its outcome

ext10 measured what fraction of spatial variance each regime keeps below mode 8:
spirals 77%, gliders 69%, bubbles 58%, worms 31%, maze 1.3%, spots 0.6%. Maze and
spots are outliers by two orders of magnitude -- their energy sits in a narrow
ring at the Turing wavelength while the others are low-wavenumber.

The hypothesis, written before these folds ran: if what a model transfers is
spectral content, holding out a regime whose spectrum is unlike the training set
should hurt most, so **maze and spots should be the two worst folds**.

They are the two *best*. The check reports 0/2 correct, with

    rho(variance below mode 8, gap) = +0.943   (the mechanism requires negative)
    rho(|distance from training mean|, gap) = -0.486   (symmetric variant)

Not merely unsupported -- inverted, with near-maximal rank agreement in the
opposite direction. What tracks the gap is how much energy a regime keeps at
*low* wavenumbers, not how far its spectrum sits from the training mean. The
symmetric variant, offered as the weaker hypothesis, also carries the wrong sign
for its own claim.

The alternative named at the time -- that the gap is governed by something other
than spectral distance -- is what the data supports. No mechanism for the
positive correlation is established here, and a rank correlation over six points
is a direction, not a law.

One confound worth stating rather than burying: VRMSE divides by target variance,
and the gap is a ratio of two VRMSEs computed on different sets. That cancels
scale within a fold but not the possibility that held-out and seen sets are
differently normalised -- a live worry precisely for the regimes ext10 found
concentrate their energy differently. The inversion should be reproduced on an
unnormalised error before being treated as physics.

## Provenance of these numbers

`scripts/cross_regime.py` grew a three-arm structure in ext26 (baseline,
robustness-trained, robustness+ensemble). Its `baseline` arm is unchanged from
what ext14 specified, so the table above is that arm, sliced out of the ext26 run
into `results/extensions/ext14_cross_regime.csv`. No separate ext14 run exists;
the two extensions share one measurement rather than duplicating it, and
[cross_regime_arms.md](cross_regime_arms.md) covers what the other two arms do
to these same folds.

Pilot scale: 4 training trajectories per regime against the reference 12, 30
epochs rather than 100, one seed. See [cross_regime_arms.md](cross_regime_arms.md)
for the full scope discussion.

Reproduce with:

    python3 scripts/cross_regime.py --data-dir data/processed/gs_pilot \
        --epochs 30 --arms baseline \
        --out-dir results/pilot --fig-dir figures/pilot
