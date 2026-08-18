# Does the Fourier advantage appear above 32x32? (H15 / ext36)

The reproduction's negative result is explicitly scoped: at 32x32 a
parameter-matched low-rank CNN matches or outperforms the spectral arms, so
there is no consistent evidence for a Fourier inductive-bias advantage **at that
scale**. This sweeps 32, 64 and 128 to close the scope.

The headline is not the one the experiment was set up to find, and the reason is
in the first section rather than the last.

## This run does not reproduce the reproduction at 32x32

| size | cnn | litefno | fno_s | best spectral | gap |
| --- | --- | --- | --- | --- | --- |
| 32 | 0.04341 | **0.03786** | 0.05951 | litefno | **+12.5%** |
| 64 | 0.04255 | **0.03938** | 0.06327 | litefno | +6.4% |
| 128 | 0.05046 | **0.04121** | 0.06642 | litefno | **+18.3%** |

The CP spectral arm beats the CNN at **every** resolution, including 32x32 --
where the reproduction reports the opposite. That discrepancy has to be settled
before any of this says anything about resolution, because a resolution trend
measured against a 32x32 baseline that disagrees with the reference is not a
trend in resolution.

The likely cause is training budget. This runs 24 trajectories for 30 epochs
across 2 seeds; the reproduction runs 72 trajectories for 200 epochs across 3.
An under-trained comparison plausibly favours the smaller-capacity arm, and
`litefno` here has 43,162 parameters against the CNN's 45,602. **This document
therefore does not claim to overturn the reproduction.** It reports that under a
substantially smaller budget the ordering is reversed, which is a fact about this
protocol until someone runs the reference protocol at 128.

## What is solid: the 128x128 result is seed-stable

Per-seed gaps tell a different story from the means:

| size | seed 0 | seed 1 | mean | CNN seed spread |
| --- | --- | --- | --- | --- |
| 32 | +16.6% | +8.4% | +12.5% | 12.3% |
| 64 | +17.0% | **-4.1%** | +6.4% | 19.5% |
| 128 | +19.1% | +17.5% | **+18.3%** | **0.1%** |

At 64x64 one seed **reverses the sign**. At 32x32 the two seeds disagree by a
factor of two. At 128x128 both seeds agree to within 1.6 points and the CNN's own
seed spread collapses to 0.1%.

So the only cell measured precisely enough to carry a claim is 128x128, and there
the spectral arm is ahead by 18.3% with essentially no seed noise.

## The trend claim the script prints, and why it is not supported

`resolution_scaling.py` prints "opens with resolution" because it compares the
endpoints, 12.5% at 32 against 18.3% at 128. That verdict is **too strong** and
is contradicted by its own middle point: the sequence is 12.5, 6.4, 18.3, which
is not monotone.

The defensible statement is weaker and about noise rather than about scale: the
CNN's seed variance falls from ~12-20% at 32 and 64 to 0.1% at 128, so the
low-resolution gaps are imprecise estimates and the 64x64 dip is well within what
one lucky CNN seed produces. There is no evidence here that the *advantage*
grows with resolution -- only that it becomes **measurable** at 128, where the
noise that swamps it at lower resolution disappears.

Two seeds cannot distinguish "the gap opens" from "the gap was always there and
the measurement got quieter". Saying which would need more seeds.

## The confound this study designed for and did not execute

The repo trains with `MODES = min(16, H//2)`. At 32x32 that is 16 of 16, so the
spectral arms truncate **nothing**; at 128x128 it is 16 of 64, so they discard
three quarters of the available modes. Resolution therefore changes two things at
once, and the script provides a `--modes-policy proportional` arm that holds the
retained *fraction* fixed to separate them.

**Only the `repo` policy was run.** The proportional arm is implemented and not
executed, so scale and truncation remain confounded in every number above. That
is the single most important missing piece: if the 128x128 advantage is really
about truncation -- a spectral arm benefiting from being forced to discard
high-wavenumber noise that the CNN must fit -- then it is a regularisation
result, not a Fourier-inductive-bias result, and those are different claims.

## The dense arm loses everywhere, and that is informative

`fno_s` carries 2,101,538 parameters against the CNN's 45,602 and is the **worst**
arm at every resolution (0.0595, 0.0633, 0.0664). With 24 training trajectories
that is the expected shape of overfitting, and it matters for reading the rest:
whatever `litefno` is doing, it is not "spectral layers are better", because the
dense spectral arm is worst. It is specific to the CP-factorized, low-parameter
form.

## Scope

24 trajectories, 30 epochs, 2 seeds, width 32, 4 layers, one-step VRMSE on the
Gray-Scott test split. Data streamed **once** at native 128x128 and downsampled to
64 and 32 with the repo's own `downsample_spatial`, so the three rows are the same
fields at different resolutions rather than different draws.

What would settle it: the reference protocol (72 trajectories, 200 epochs, 3
seeds) at 128x128, plus the proportional-modes arm. Either could overturn the
reading above.

Reproduce with:

    python3 scripts/stream_preprocess.py --out-dir data/processed/gs_native \
        --downsample 1 --train-per-scenario 4
    python3 scripts/resolution_scaling.py --data-dir data/processed/gs_native \
        --sizes 32 64 128 --modes-policy repo --seeds 0 1 --epochs 30 --batch 16

Outputs `ext36_resolution_cells.csv`, `ext36_resolution_summary.csv`.
