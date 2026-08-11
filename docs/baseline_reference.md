# In-distribution reference number for LiteFNO

The number every extension in this repo is implicitly compared against, made
into a command you can run rather than a notebook someone ran once.

Produced by [`scripts/baseline_reference.py`](../scripts/baseline_reference.py).

```bash
python scripts/baseline_reference.py --data-dir data/processed/gray_scott_proxy
```

## Why this needed doing

The repo contains two different things called LiteFNO, and the distinction
decides what the existing numbers mean.

`src/litefno/models/litefno.py` is a **low-rank CNN placeholder**. This is not a
discovery — [`docs/reproducibility_findings.md`](reproducibility_findings.md)
states it plainly, and `results/extensions/freebieA_repro_audit.json` records
`repo_litefno_class_is_spectral: false`. It has no FFT, no complex weights, and
no CP factorization.

The consequence is easy to miss. Everything in
`results/extensions/logs_reproduction_table.csv` — the eight-dataset table that
looks like the repo's headline reproduction — was produced by that placeholder
and by FNO-S. Its columns are honestly named `cnn_test_vrmse` and
`fnos_test_vrmse`, but the file sits under a name that reads as a LiteFNO
reproduction and **contains no LiteFNO number at all**.

The real CP-factorized spectral model exists only inside
`notebooks/headline_3seed.ipynb`, was run once on Kaggle, and covers one
dataset. So the in-distribution reference number for the architecture the paper
is about rested on a single un-rerun notebook. This script makes it reproducible
and adds a replication check against that original run.

## Protocol

Taken unchanged from `notebooks/headline_3seed.ipynb`:

```
modes 16, width 64, layers 8, CP factorization at rank 0.02
200 epochs, batch 64, Adam lr 1e-3, StepLR(step=100, gamma=0.5)
one-step training, seeds 0/1/2
```

Three arms train under identical conditions, because a single model's error is
not interpretable on its own:

| arm | what it is | params |
|---|---|---|
| `litefno` | neuralop FNO, CP-factorized complex spectral weights — the paper's architecture | 179,666 |
| `fno_s` | the repo's own dense spectral FNO (`src/litefno/models/fno_s.py`) | — |
| `cnn` | the repo's low-rank CNN (`src/litefno/models/litefno.py`), under its honest name | 107,842 |

Metrics are `litefno.metrics.vrmse`, unchanged: one-step on the test split, plus
the autoregressive rollout windows the repo's config already specifies
(`eval_windows: [[6, 12], [13, 30]]`).

## Results

See `results/baseline/ext13_baseline_summary.csv` for the table and
`ext13_replication_check.csv` for the comparison against the committed Kaggle
run. The headline is the **one-step test VRMSE**, reported as mean ± sample
standard deviation over three seeds.

## Reading the rollout numbers

The rollout columns carry a caveat the one-step column does not. A one-step
VRMSE is a stable quantity. A 30-step autoregressive rollout of a model trained
one step at a time is not, and the committed table shows it: its seed spread is
30–40% of its mean, and one CNN seed in this run's smoke test scored a rollout
VRMSE of 26 while another scored 0.6.

ext10 gives the mechanism. The 60-step training window (`max_steps: 60`)
resolves no temporal structure in this data — every Gray-Scott scenario scores a
peak-to-AR(1)-null ratio below 1.8 in that window, with the peak pinned to the
lowest resolvable bin. Nothing in training constrains what happens over a long
rollout, so nothing should be expected to. Treat rollout as an order of
magnitude, not a number.

## What "in-distribution" means here, precisely

The test split holds trajectories the model never saw, from the same six
Gray-Scott regimes, at the same resolution, over the same time window. So this
measures generalisation across initial conditions — not across regimes, not
across resolutions, not beyond the training horizon.

Two limits are worth stating rather than leaving implicit:

- **The training window is the spin-up phase.** `max_steps: 60` keeps the first
  60 steps of a 1001-step trajectory, and ext10 showed those are dominated by
  the one-time pattern-formation transient rather than settled dynamics. The
  reference number therefore describes accuracy on the transient. That is the
  repo's documented protocol, and it is what the committed numbers measure too,
  so the comparison is like-for-like — but it is not a measurement of the
  settled regime.
- **The regimes are not interchangeable.** ext10 found maze and spots hold ~1%
  of their spatial variance below mode 8 against spirals' 77%. A single VRMSE
  averaged over all six hides a two-order-of-magnitude difference in how hard
  they are. A per-regime breakdown would be a strictly better reference number
  and is the obvious next step.

## One benchmark, not eight

The board task says "ecosystem benchmarks", plural. Only Gray-Scott is
reported, for a reason worth recording rather than quietly dropping:

Of The Well datasets this repo configures, Gray-Scott is the only one with
enough trajectories per file to train on at this scale. `turbulent_radiative_
layer_2D` stores **one** trajectory per file; `viscoelastic_instability` has 20
timesteps total. The larger families (`euler_multi_quadrants`,
`acoustic_scattering`) are 5–8 GB per file before preprocessing. Extending the
reference number to them is a data-acquisition problem, not a modelling one,
and `scripts/stream_preprocess.py` (added here) is the piece that makes it
tractable on a machine without 44 GB to spare.

## Streaming preprocessor

`litefno download` fetches an entire Well dataset before `litefno preprocess`
reduces it — 44 GB for Gray-Scott. But preprocessing caps time at 60 steps and
downsamples space by 4×, discarding over 99% of what was downloaded. That is the
one step of the pipeline a low-resource machine cannot do, which is awkward for a
repo whose stated emphasis is low-resource deployment.

[`scripts/stream_preprocess.py`](../scripts/stream_preprocess.py) reads only the
part that survives: a contiguous 60-step prefix per trajectory, one ~4 MB range
request against a 21 GB file, reduced on arrival. It imports
`downsample_spatial` from `litefno.preprocess` rather than reimplementing it, so
the output is the same function of the raw data — only the bytes moved differ.

Building the Gray-Scott train split this way costs about 31 MB of output from
under 1 GB of transfer. Verified against the shipped `gray_scott_proxy`: field
statistics and per-step change ratio agree (A mean 0.81 vs 0.71, per-step change
/ field std 0.096 vs 0.113), i.e. the streamed data has the same character.

The reference run below uses the shipped `gray_scott_proxy`, deliberately: it is
the data the committed `seed_table.csv` was produced on, so the replication
check is like-for-like. The streamed builder is the path to the datasets the
repo does not yet have locally.

## Correctness

`tests/test_baseline_reference.py` pins the thing that went wrong before: the
arm labelled `litefno` must actually be the CP-factorized spectral model. It is
checked on the parameters — complex spectral weights, a CP rank vector plus one
complex factor matrix per tensor mode, and a rank that responds to the
configured fraction — rather than by intercepting `torch.fft`, because neuralop
does not reach the FFT through an attribute a test can patch and a spy there
reports a false alarm. The `cnn` arm is checked to have *no* complex parameters
and to be labelled `n/a`, and the protocol constants are asserted against the
notebook's values so they cannot drift.

Around training: one-step VRMSE is checked to equal `litefno.metrics.vrmse` and
to be invariant to evaluation batch size (VRMSE normalises by target variance,
so a mean of per-batch ratios would be a different and wrong quantity); rollout
windows are checked to score different steps, which is what catches an
off-by-one in the window slice.
