# Reproduction guide

This guide outlines the steps used to reproduce the LITEFNO paper results.

## 1. Prepare data

Pick a dataset and run download + preprocessing:

```bash
litefno download --config configs/datasets/gray_scott_reaction_diffusion.yaml
litefno preprocess --config configs/datasets/gray_scott_reaction_diffusion.yaml
```

Repeat for each dataset you plan to reproduce.

## 2. Run baselines

FNO-S baseline:

```bash
litefno train --config configs/experiments/fno_s_gray_scott_reaction_diffusion.yaml
```

LITEFNO:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml
```

## 3. Hyperparameters (paper defaults)

The base configs match the paper's defaults:

- `layers`: 8
- `width`: 64/128/160
- `rank`: 32/48 (LITEFNO)
- `modes`: 12 (FNO-S)
- `optimizer`: AdamW, `lr=1e-3`, step LR every 100 epochs (`gamma=0.5`)
- `epochs`: 500

Use CLI overrides to sweep widths/ranks:

```bash
litefno train --config configs/experiments/litefno_gray_scott_reaction_diffusion.yaml \
  --set model.width=128 \
  --set model.rank=48
```

## 4. Metrics to report

The logger writes JSONL records that include:

- `rmse`, `vrmse`
- `vrmse_6_12`, `vrmse_13_30` (when windows are covered)
- training loss (`loss`)
- parameter count (`params`)

Report one-step and multi-step VRMSE for the time windows used in the paper.

## 5. Issues, discrepancies, and similarities

Keep a running log of reproduction issues and confirmed matches in
[`notes_deviations.md`](notes_deviations.md). Update it whenever you discover
new discrepancies, fixes, or configuration matches with the paper.
