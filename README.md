# litefno-repro

A from-scratch reproduction of the Lightweight Fourier Neural Operator (LiteFNO,
Ahn et al., 2025), plus a parameter-matched low-rank CNN ablation that the
original paper does not include.

Headline result: on Gray-Scott at 32x32 across three seeds, the CNN baseline
matches or outperforms LiteFNO on one-step VRMSE and on autoregressive rollout,
so we find no consistent evidence for a Fourier inductive-bias advantage at this
scale. See [docs/reproducibility_findings.md](docs/reproducibility_findings.md)
for the full statement of what is and is not reproduced.

## Repository layout

```
src/litefno/   Python package (models, training, metrics, data, preprocessing)
configs/       YAML configs: datasets/ and experiments/
scripts/       CLI helpers and Kaggle notebook builders (build_*_notebook.py)
notebooks/     Generated Kaggle notebooks (built by scripts/build_*.py)
results/       Numeric outputs: seeds/ mechinterp/ extensions/ logs/ checkpoints/
figures/       Plots: headline/ mechinterp/ extensions/ reproduction/
data/          Gray-Scott data (raw/ and processed/; not in git, see Zenodo)
tests/         pytest suite
docs/          Setup, reproduction guide, and reproducibility notes
```

`metadata.yaml` and `bibliography.bib` are submission metadata for the
accompanying write-up; they are not needed to run the code.

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
- [Forced harmonics: is a temporal prior worth it?](docs/forced_harmonics.md)
The authoritative seed-robust numbers are in `results/seeds/` (3-seed headline)
and `results/mechinterp/` (3-seed dead-mode, CP-rank, mode ablation), with
figures in `figures/headline/` and `figures/mechinterp/`. Checkpoints and data
are git-ignored and distributed via Zenodo.

## Install

```bash
conda create -n fno python=3.10
conda activate fno
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .          # add [dev] for the test dependencies
```

Dataset downloads use the official Polymathic `the_well` package, which provides
the `the-well-download` CLI:

```bash
pip install the_well
the-well-download --help   # verify it is on your PATH
```

The downloader pulls from HuggingFace; if a dataset is gated, run
`huggingface-cli login` once before `litefno download`.

## Fast path: data and checkpoints from Zenodo

Preprocessed Gray-Scott data and trained checkpoints (matched CNN plus 3-seed
CP-factorized spectral LiteFNO) are archived on Zenodo under CC BY 4.0:

**DOI: [10.5281/zenodo.20718092](https://doi.org/10.5281/zenodo.20718092)**

```bash
unzip litefno-repro-data.zip
cp -R litefno-repro-data/data/processed/* data/processed/
cp    litefno-repro-data/checkpoints/*.pt results/checkpoints/
```

That is enough to run the notebooks and `litefno test` without re-downloading
the 44 GB raw dataset from The Well.

## Slow path: build the data yourself

The project expects The Well datasets as HDF5 with shape
`(n_traj, n_steps, H, W, fields)`.

```bash
litefno download   --config configs/datasets/gray_scott_reaction_diffusion.yaml
litefno preprocess --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

## Train and evaluate

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml
```

Override any config value on the CLI:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml \
  --set training.epochs=10 --set training.device=cuda
```

Metrics are appended to the JSONL path in the config under
`logging.metrics_path`.

Evaluate a checkpoint on a split (`train`, `valid`, or `test`; default `test`).
Results are printed and appended to the metrics JSONL with `step: -1`:

```bash
litefno test \
  --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml \
  --checkpoint outputs/checkpoints/gray_scott_reaction_diffusion/litefno/best.pt
```

### Checkpoints

When `training.checkpoint_every > 0` or `training.checkpoint_best_metric` is
set, checkpoints go to `training.checkpoint_dir` (default
`outputs/checkpoints/<dataset>/<model>/`):

- `last.pt`: overwritten every `checkpoint_every` epochs
- `best.pt`: overwritten whenever `checkpoint_best_metric` (e.g. `valid_vrmse`)
  improves

Resume with `training.resume_from`, either in the config or via
`--set training.resume_from=<path>`.

### Run everything

```bash
scripts/run_all.sh                       # both models, all 8 datasets
scripts/run_all.sh --model litefno       # one model
scripts/run_all.sh --dataset gray_scott_reaction_diffusion
```

## Tests

```bash
python -m pytest
```

GitHub Actions runs the same command on pushes and pull requests via
[.github/workflows/tests.yml](.github/workflows/tests.yml).

## Documentation

- [Project overview](docs/overview.md) and [TL;DR](docs/tldr.md)
- [Setup](docs/setup.md)
- [Data and preprocessing](docs/data.md)
- [Training and evaluation](docs/training.md)
- [Reproduction guide](docs/reproduction.md)
- [Experiments](docs/experiments.md)
- [Configuration reference](docs/configs.md)
- [Metrics](docs/metrics.md)
- [Reproducibility findings](docs/reproducibility_findings.md)
- [Deviations from the paper](docs/notes_deviations.md)
- [Extensions roadmap](docs/extensions.md)
