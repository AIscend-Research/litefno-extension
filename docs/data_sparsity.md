# Field recovery under thin sensor coverage

What survives when the simulation is observed the way a low-resource deployment
would observe it — a handful of stations, a coarse sensor lattice, a survey with
swath-shaped holes — rather than on a dense grid.

Produced by [`scripts/data_sparsity.py`](../scripts/data_sparsity.py).

```bash
python scripts/data_sparsity.py --family gray_scott
```

The first run fetches settled-segment frames from The Well over HTTP range
requests and caches them to `data/processed/sparsity_cache.npz` (17 MB,
git-ignored); later runs are offline.

## Headline

**Sampling geometry matters more than sampling density.** At 5% coverage the
same field, same reconstruction, same mode budget gives:

| regime | VRMSE at 5% coverage (field A, best mode budget) |
|---|---|
| grid (regular lattice) | **0.15 – 0.36** |
| random (i.i.d. pixels) | 0.68 – 1.05 |
| stations (3×3 sites) | 1.50 – 2.00 ✗ |
| blocks (8×8 patches) | 2.32 – 5.61 ✗ |

✗ = worse than VRMSE 1.0, the score you get by predicting the mean everywhere.

A regular lattice at 5% coverage lands within 0.15 VRMSE of what the same mode
budget achieves with *every* pixel observed. Random sampling at the same density
is 3–5× worse, and clustered coverage is worse than not modelling the field at
all. Buying more sensors helps far less than placing the ones you have on a grid.

**And ext10's spectral measurement predicts which scenarios survive, exactly.**
ext10 measured what fraction of each scenario's spatial variance sits below mode
8. Ranking the scenarios by that number reproduces their reconstruction ranking
at thin coverage with Spearman ρ = **−1.000**:

| scenario | variance below mode 8 (ext10) | VRMSE at 1% grid coverage |
|---|---|---|
| spirals | 77.4% | 0.811 |
| gliders | 69.0% | 0.929 |
| bubbles | 58.3% | 1.017 ✗ |
| worms | 30.9% | 1.132 ✗ |
| maze | 1.3% | 1.264 ✗ |
| spots | 0.6% | 1.267 ✗ |

This was a prediction, not a pattern read off afterwards: ext10 ran first and
concluded that maze and spots keep their energy in a narrow band at the Turing
wavelength near mode 13–16 while spirals keeps its energy low, so maze and spots
should collapse under thin coverage. They do — and at 1% coverage the split is
exactly the line between usable and worse-than-mean.

The correlation holds where the reconstruction is well-determined (ρ = −1.000
for `grid`, −0.83 for `random`) and breaks down for the clustered regimes
(ρ ≈ −0.37), which is consistent rather than contradictory: there the outcome is
governed by geometry, not by the spectrum, because nothing determines the field
in the unobserved region.

## Why the scenario difference shows up in the mode budget

At 10% coverage or more the sparsity penalty is nearly identical across
scenarios (0.008–0.013 for `grid`). The scenarios do not differ in how well
their modes are recovered; they differ in **which mode budget they need**, and
therefore in how much coverage buys a usable budget:

- a box of M modes has (2M+1)² real degrees of freedom — 81 at M=4, 289 at M=8,
  1089 at M=16
- maze and spots need M≥12 to capture anything (they hold 1.3% and 0.6% below
  mode 8), so they need ≥625 observations before reconstruction is even
  well-posed
- spirals and gliders reach 77% and 69% at M=8, which needs 289

At 1% coverage of a 128×128 grid you have ~164 observations. That affords M=4
and nothing more. spirals still scores 0.81 there because 39% of its variance is
below mode 4; spots scores 1.27 because 0.1% of its is.

## What breaks and how

**Clustered coverage cannot be rescued.** `blocks` exceeds VRMSE 1.0 at every
coverage below 50%, and `stations` at every coverage below 25% (at 25% it clears
1.0 for three of six scenarios, 0.74–0.96). The estimator used here is
deliberately far too generous to be a real method: for each scenario it picks
both the mode budget *and* the SVD truncation level in hindsight, tuned against
the very field being reconstructed. Failures that survive that are structural.
Reconstructing a band-limited field outside a contiguous observed patch is
extrapolation, and the information is not there at any regularisation.

**A lattice below its Nyquist is optimal; above it, it is worse than useless.**
The lattice wins because its design matrix is orthogonal for every mode it can
resolve — perfect conditioning, no coherence. But a stride-s lattice samples at
frequency H/s, so a mode above H/2s folds down onto a lower one. In the test
suite a mode-12 field sampled by a stride-4 lattice lands its entire energy at
mode (±4, ±4) and scores VRMSE √2: a confident, well-conditioned answer at the
wrong wavenumber. An unresolved mode can be fixed with more sensors; an aliased
one is destroyed at acquisition.

This cuts against the compressed-sensing intuition that incoherent random
sampling beats a regular grid. That result is about *sparse* recovery with an
L1 prior. Least-squares fitting of a known band has the opposite preference, and
the estimator here is least squares because that is what an FNO's mode
truncation is.

## Two traps this run walked into

**Min-norm least squares is not sparsity-promoting.** An early version assumed
that random sampling would recover a single-mode field from far fewer
observations than degrees of freedom. It does not: at 256 observations against
1089 unknowns, lattice and random sampling fail identically (VRMSE 0.866 vs
0.871). Only the observations-per-degree-of-freedom ratio matters in that
regime, and `obs_per_dof` is reported on every row for this reason.

**Default `rcond` produced VRMSE around 10⁸.** With a clustered mask the design
matrix is near-singular, and `lstsq` at machine-precision cutoff keeps the null
directions and returns enormous coefficients. Those numbers were a property of
the solver, not of the sensor network. The fix is truncated-SVD least squares,
but the truncation level is not innocuous:

| regime, 25% coverage | rcond 1e-6 | 1e-4 | 1e-3 | 1e-2 | 1e-1 |
|---|---|---|---|---|---|
| random | 0.153 | 0.153 | 0.153 | 0.153 | 0.153 |
| grid | 0.131 | 0.131 | 0.131 | 0.131 | 0.131 |
| blocks | 1188.9 | 25.6 | 4.67 | 1.50 | 1.79 |

`random` and `grid` are invariant across five orders of magnitude — their
results are a property of the data. `blocks` moves by three orders, which means
the data is not determining it and picking one constant would be picking an
answer. Hence the sweep-and-take-the-best design, and the `rcond_spread` column,
which records how far each result moved across truncation levels. Over the whole
sweep that column reads: `grid` exactly 0 everywhere; `random` median 0 with
occasional excursions near the underdetermined boundary (max 18); `blocks`
median 704; `stations` median 5.5. Read it as a per-row warning label — where it
is large, the number beside it is not determined by the observations.

## Outputs

| file | contents |
|---|---|
| `results/extensions/ext11_sparsity_gray_scott.csv` | one row per scenario × field × regime × coverage × mode budget: VRMSE, oracle floor, sparsity penalty, obs/dof, constrained modes, rcond spread, seed spread |
| `results/extensions/ext11_ext10_prediction_check_gray_scott.csv` | Spearman ρ between ext10's below-mode-8 variance and reconstruction error, per regime and coverage |
| `figures/extensions/ext11_sparsity_gray_scott.png` | VRMSE vs coverage per regime, with oracle floors and the coverage each scenario needs |

Coverage is reported as realised, not nominal: a square lattice quantises to
integer strides (nominal 5% becomes 6.25%), blocks quantise to whole 8×8 tiles,
and station footprints overlap. Every configuration is run with 3 mask seeds and
the spread is in the CSV.

## Caveats

- Reconstruction here is a static, single-frame inverse problem. It bounds what
  a model could learn from such data; it is not a measurement of LiteFNO's
  performance under sparse input, which would need the trained checkpoints and a
  retraining run.
- 24 frames per scenario/field, drawn from the settled segment of 3 trajectories
  (ext10 established that the spin-up transient is not representative). Frames
  within a trajectory are consecutive; independent realisations come from using
  separate trajectories.
- Only Gray-Scott. The turbulent-radiative-layer family is not periodic in y, so
  a global Fourier basis is the wrong model class for it.
