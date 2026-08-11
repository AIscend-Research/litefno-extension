# litefno-repro

Reproduction and extensions for the Lightweight Fourier Neural Operator (LITEFNO) paper, with an emphasis on low-resource deployment.

## Repository layout

```
src/litefno/      Python package (models, training, metrics, data, preprocessing)
scripts/          CLI helpers + Kaggle notebook builders (build_*_notebook.py)
notebooks/        Generated Kaggle notebooks (phase2, phase3, headline_3seed,
                  mechinterp_3seed, spectral_regularizer)
configs/          YAML configs — datasets/ and experiments/
data/             Gray-Scott data (processed/ + raw/; not in git → Zenodo)
figures/          ALL figures — extensions/ mechinterp/ headline/ reproduction/
results/          Numeric outputs — checkpoints/ seeds/ mechinterp/ extensions/ logs/
tests/            pytest suite
docs/             Documentation + reproducibility notes
```

The authoritative seed-robust results are in `results/seeds/` (3-seed headline) and
`results/mechinterp/` (3-seed dead-mode / CP-rank / ablation); their figures are in
`figures/headline/` and `figures/mechinterp/`. Checkpoints (`results/checkpoints/`)
and data (`data/`) are git-ignored and distributed via Zenodo.

## Documentation

- [Project overview](docs/overview.md)
- [Setup](docs/setup.md)
- [Data & preprocessing](docs/data.md)
- [Training & evaluation](docs/training.md)
- [Reproduction guide](docs/reproduction.md)
- [Experiments](docs/experiments.md)
- [Configuration reference](docs/configs.md)
- [Metrics](docs/metrics.md)
- [Extensions roadmap](docs/extensions.md)
- [Harmonic content by scenario](docs/harmonic_content.md)
- [Field recovery under thin sensor coverage](docs/data_sparsity.md)

## Setup (quickstart)

```bash
conda create -n fno python=3.10
conda activate fno
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

For development tools (tests):

```bash
pip install -e .[dev]
```

### Install `the-well-download`

Dataset downloads use the official Polymathic `the_well` package, which provides
the `the-well-download` CLI:

```bash
pip install the_well
```

This installs `the-well-download` on your `PATH`. Verify with:

```bash
which the-well-download
the-well-download --help
```

The downloader pulls files from HuggingFace; if a dataset is gated, log in once
with `huggingface-cli login` before running `litefno download`.

## Data & checkpoints (Zenodo)

Preprocessed Gray-Scott data and trained checkpoints (matched CNN + 3-seed
CP-factorized spectral LiteFNO) are archived on Zenodo:

**DOI: [10.5281/zenodo.20718092](https://doi.org/10.5281/zenodo.20718092)** (CC BY 4.0)

This is the fast path to reproduce the results without re-downloading the 44 GB
raw dataset from The Well. Download `litefno-repro-data.zip`, then:

```bash
unzip litefno-repro-data.zip
cp -R litefno-repro-data/data/processed/*        data/processed/
cp    litefno-repro-data/checkpoints/*.pt        results/checkpoints/
```

You can then run the notebooks / `litefno test` directly. To regenerate the
processed data from scratch instead, use the download + preprocess steps below.

## Data (quickstart)

The project expects The Well datasets as HDF5 with shape `(n_traj, n_steps, H, W, fields)`.

Download (uses `the-well-download` under the hood):

```bash
litefno download --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

Preprocess (downsampling, trajectory/time caps):

```bash
litefno preprocess --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

## Training & evaluation (quickstart)

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml
```

Override config values on the CLI:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml --set training.epochs=10 --set training.device=cuda
```

Metrics are logged to the JSONL path in the config under `logging.metrics_path`.

### Checkpoints

When `training.checkpoint_every > 0` or `training.checkpoint_best_metric` is
set, checkpoints are written to `training.checkpoint_dir` (defaults to
`outputs/checkpoints/<dataset>/<model>/`):

- `last.pt` — overwritten every `checkpoint_every` epochs
- `best.pt` — overwritten whenever `checkpoint_best_metric` (e.g. `valid_vrmse`)
  improves

Resume training from a checkpoint by setting `training.resume_from` in the
config, or via `--set`:

```bash
litefno train \
  --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml \
  --set training.resume_from=outputs/checkpoints/gray_scott_reaction_diffusion/litefno/last.pt
```

### Evaluate a checkpoint on the test split

`litefno test` loads a checkpoint and evaluates it on the requested split,
printing the metrics and appending them to the experiment's metrics JSONL with
`step: -1`:

```bash
litefno test \
  --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml \
  --checkpoint outputs/checkpoints/gray_scott_reaction_diffusion/litefno/best.pt
```

`--split` accepts `train`, `valid`, or `test` (default `test`). Config values
can be overridden with `--set` just like during training, e.g.
`--set training.batch_size=128`.

## Tests

```bash
python -m pytest
```

GitHub Actions runs the same test command on pushes and pull requests via
`.github/workflows/tests.yml`.
