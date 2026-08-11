# Data & Preprocessing

## Dataset source

All datasets are from The Well benchmark (Polymathic AI). Do **not** download the full collection (≈15TB). Start with the smallest dataset and only pull the splits you need.

Recommended order (smallest to largest):

1. `gray_scott_reaction_diffusion`
2. `euler_multi_quadrants_openBC`
3. `euler_multi_quadrants_periodicBC`
4. `acoustic_scattering_discontinuous`
5. `active_matter`
6. `rayleigh_benard`
7. `turbulent_radiative_layer_2D`
8. `viscoelastic_instability`

## Expected format

The raw data are HDF5 arrays with shape:

```
(n_traj, n_steps, H, W, fields)
```

The pipeline down-samples spatially, caps the number of trajectories, and caps the time dimension to match the paper's preprocessing.

## Download

Download a dataset split using the YAML config:

```bash
litefno download --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

Repeat for other datasets by swapping the config file.

## Preprocess

```bash
litefno preprocess --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

Preprocessing applies:

- `downsample_factor`: spatial block-mean downsampling (e.g., 4×). If the spatial
  dimensions are not divisible by the factor, the excess rows/columns are
  dropped before averaging.
- `max_trajectories`: random sample cap on the number of trajectories (paper:
  ≤1000).
- `random_seed`: optional seed to make the trajectory sampling reproducible.
- `max_steps`: cap on time steps (paper: 30-60, dataset-dependent).

## Common download pattern

For each dataset, download and preprocess train/valid/test splits:

```bash
litefno download --config configs/datasets/gray_scott_reaction_diffusion.yaml
litefno preprocess --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

Repeat with the dataset-specific YAML files in `configs/datasets/`.

## Dataset config fields

Each dataset config lives under `configs/datasets/` and uses:

- `name`: The Well dataset name.
- `input_dir`: directory for raw splits.
- `output_dir`: directory for processed splits.
- `splits`: list of splits to download/preprocess (`train`, `valid`, `test`).
- `dataset_key`: HDF5 dataset key (default `data`).
- `downsample_factor`: spatial stride factor.
- `max_trajectories`: maximum trajectories to keep.
- `random_seed`: seed for trajectory sampling (optional).
- `max_steps`: maximum time steps to keep.
- `input_steps`: number of input frames.
- `output_steps`: number of target frames.
- `fields`: number of physical fields (channels).
- `stride`: temporal stride between windows.
- `cache`: `none` or `memory`.
