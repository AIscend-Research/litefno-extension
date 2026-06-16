# Deviations from paper & bugs found during reproduction

## Preprocessing (`src/litefno/preprocess.py`)

### Bug 1: Naive striding in spatial downsampling
**Location:** `downsample_spatial()`, line uses `array[:, :, ::factor, ::factor, :]`

**Issue:** Takes every Nth pixel without anti-aliasing. High-frequency content
folds into low frequencies (aliasing). Since FNO learns by manipulating Fourier
modes, aliased input directly corrupts the input distribution the model is
trained on.

**Expected behavior:** Block-mean averaging — reshape into (factor x factor)
blocks and take the mean. Standard practice in FNO literature for downsampling
discrete fields before spectral methods.

**Impact:** Likely worse VRMSE than paper. Magnitude unknown until measured.

**Fix applied:** Replace striding with block-mean downsampling to avoid aliasing.

### Bug 2: Non-random trajectory selection
**Location:** `cap_trajectories()`, line uses `array[:max_trajectories]`

**Issue:** Takes the first N trajectories rather than a random sample.

**Paper claim:** Section 3 (Dataset) states "randomly sampling 1000 trajectories"
for datasets with more than 1000 trajectories.

**Impact (Gray-Scott-specific):** Gray-Scott has 1200 trajectories grouped by
6 parameter regimes (gliders, bubbles, maze, worms, spirals, plus a 6th)
with 200 trajectories per regime. Taking the first 1000 likely excludes one
entire regime. Training distribution will be missing a chunk of the parameter
space the paper trained on.

**Impact (other datasets):** TBD per dataset — depends on whether trajectories
in the source HDF5s are ordered by any meaningful property.

**Fix applied:** Randomly sample trajectories (optionally seeded for reproducibility).

## Status
- [x] Reported to team
- [x] Decision made on whether to fix before or after first reproduction run
- [x] Fixed
- [ ] Re-run with fix and compared to buggy baseline

## Similarities with paper (confirmed in configs)
- Base model settings match the paper defaults: `layers=8`, `width=64`, and
  `rank=32` for LiteFNO or `modes=12` for FNO-S (`configs/experiments/base_*.yaml`).
- Optimizer settings match the paper: AdamW with `lr=1e-3` and a step LR schedule
  every 100 epochs (`gamma=0.5`).
- Training length matches the paper default: `epochs=500`.
- Evaluation windows align with reported VRMSE windows: `6–12` and `13–30`.
