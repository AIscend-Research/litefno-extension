# Reproducibility notes

This document records the status and findings of our attempt to reproduce
*"Lightweight Fourier Neural Operator for Time-Dependent Partial Differential
Equations"* (LiteFNO, NeurIPS ML4PS 2025).

## Important framing: this is a from-scratch reimplementation

This repository is **our own from-scratch implementation** built from the paper
description (initial commit by the team, scaffolding/stubs added with a coding
assistant). It is **not** a fork of, and does not run, the authors' released
code. Therefore we make **no claims** about the correctness or contents of the
authors' official implementation (`anonymous.4open.science/r/LFNO`); any
statements here concern only our own reimplementation.

## Current implementation status

Two model classes exist in the repo:

- **FNO-S** ([../src/litefno/models/fno_s.py](../src/litefno/models/fno_s.py)): a
  genuine spectral FNO (uses `torch.fft.rfft2`, complex spectral weights, mode
  truncation). This baseline is implemented.

- **`LiteFNO`** ([../src/litefno/models/litefno.py](../src/litefno/models/litefno.py)):
  currently a **low-rank CNN placeholder** (1x1/3x3 `Conv2d` + GELU bottlenecks).
  It does **not** yet implement the paper's three core components:
  1. spectral convolution (FFT),
  2. CP low-rank factorization of the spectral weights,
  3. transduction (spatial -> spatio-temporal fine-tuning).

  This is an **incomplete part of our reimplementation**, not a finding about
  anyone else's code. All metrics in [../results/logs/](../results/logs/) and the
  committed checkpoints were produced by this CNN placeholder, so they do **not**
  yet constitute a reproduction of LiteFNO's architecture or claims.

## What a valid reproduction requires (our plan)

To make this a genuine reproduction (or a documented partial/negative result),
we must implement the real architecture and compare to the paper's claims:

- Implement the real **CP-factorized spectral LiteFNO** (planned via the
  `neuraloperator` library): `notebooks/phase2_train_real_litefno.ipynb`.
- Optionally add the **transduction** stage (LiteFNO's central novelty).
- Compare, under a documented protocol, against:
  - the **paper's reported numbers** (Tables 6/7), and
  - our **FNO-S** baseline and the **CNN** (as an ablation arm).

From-scratch reimplementation is a legitimate (and common) reproducibility
methodology; the key requirement is that we actually build the spectral
architecture, since the CNN placeholder reproduces none of the paper's
mechanisms.

## Protocol notes / planned deviations

For a fair internal head-to-head we plan a matched protocol (e.g. 200 epochs,
MSE loss, single-step) rather than the paper's full 500 + 100-epoch transduction
recipe with relative-L2 loss and super-resolution evaluation. Such deviations,
and the fact that our preprocessing (GS downsample factor 4 -> 32x32, single
step) differs from the paper's setup, must be stated explicitly; they mean our
numbers are not directly comparable to Tables 6/7 unless we also run the
paper-faithful configuration.

## Honest current limitation

As of now the project has **not** reproduced LiteFNO: only the FNO-S baseline and
a CNN placeholder are trained. A reproduction paper (including a partial/negative
result) requires completing the spectral LiteFNO implementation first.
