# Reproducibility findings

This document records reproducibility findings discovered while attempting to
reproduce *"Lightweight Fourier Neural Operator for Time-Dependent Partial
Differential Equations"* (LiteFNO, NeurIPS ML4PS 2025). It is intended to feed
directly into the paper (a documented reproducibility insight is a first-class
MLRC contribution).

## Finding 1: the reference implementation does not match the paper's architecture

**What the paper describes.** LiteFNO is a *Fourier* neural operator with three
core components (Sec. 2 of the paper):
1. **Spectral convolution** — `L(z) = σ(Wz + b + F⁻¹(W·F(z)))` (Eqs. 2–3).
2. **Low-rank CP factorization** of the spectral weights, reducing complexity
   from `O(mnk^d)` to `O((m+n+kd)R)` (Eq. 4).
3. **Transduction** — extending the spatial model to spatio-temporal learning by
   adding a temporal low-rank factor `U(t)` and fine-tuning over time windows.

**What the code implements.** The `LiteFNO` class in
[litefno/models/litefno.py](../litefno/models/litefno.py) is a **low-rank CNN**:
1×1/3×3 `Conv2d` layers with a GELU bottleneck (`reduce → conv → expand`). It
contains **no Fourier transform, no CP factorization of spectral weights, and no
transduction**. The `rank` argument is a convolutional bottleneck channel count,
not a spectral factorization rank.

By contrast, the **baseline** `FNO-S`
([litefno/models/fno_s.py](../litefno/models/fno_s.py)) *is* a genuine spectral
FNO (it uses `torch.fft.rfft2`, complex spectral weights, and mode truncation).

**Verification.** Reproducible via the audit cell in
`notebooks/phase3_extensions.ipynb` ("Freebie A"), which inspects each model's
source for spectral operations and emits `freebieA_repro_audit.json`.

**Git history.** The model has been a CNN since the commit that introduced it
(`af743ad "Add model and training stubs"`); a spectral version never existed in
the repository (`git log -S "rfft" -- litefno/models/litefno.py` returns
nothing). It was an unfilled placeholder.

**Implication.** All metrics in [outputs/logs/](../outputs/logs/) and the
committed Gray-Scott checkpoint correspond to the **CNN**, not the paper's
spectral LiteFNO. They therefore cannot be read as a reproduction of the paper's
architectural claims (parameter reduction via CP factorization, the
spatio-temporal transduction benefit, or resolution-invariant
super-resolution).

## Finding 2: evaluation task and protocol deviations

The repo runs differ from the paper in ways that make the numbers not directly
comparable to Tables 6/7:

| Aspect | Paper | Repo runs |
| --- | --- | --- |
| Architecture (LiteFNO) | spectral + CP + transduction | low-rank CNN |
| Epochs | 500 (+100 transduction) | 200 |
| Loss | relative L2 | MSE |
| Batch size | 16 / 32 | 512 |
| Prediction | multi-step (4 frames → step 33) | single-step |
| Evaluation | super-resolution (coarse → original) | same (downsampled) resolution |
| GS spatial downsample | factor giving 64×64 | factor 4 giving 32×32 |

For example, one-step VRMSE on GS: paper reports FNO-S 0.299 / LiteFNO 0.0098;
our repo runs give FNO-S ≈ 0.0245 / CNN-"LiteFNO" ≈ 0.0227 — the FNO-S baseline
is ~12× "better" than the paper's, indicating an easier (same-resolution,
single-step) task rather than a faithful reproduction.

## How this shapes the study

Because a faithful reproduction is not possible from the released code as-is, we
reframe the work as a **generalization / ablation study**: we implement the real
CP-factorized spectral LiteFNO (`notebooks/phase2_train_real_litefno.ipynb`,
via `neuraloperator`) and compare it head-to-head against the released low-rank
CNN and the FNO-S baseline under a matched protocol, asking: *does LiteFNO's
spectral machinery earn its complexity, or does a plain low-rank CNN reach the
same lightweight-efficiency frontier?*

**Open limitation:** our real-LiteFNO arm tests the spectral CP factorization
but (in the matched-protocol setting) not the **transduction** stage, which is
LiteFNO's central novelty. This should be stated explicitly in the paper, and
adding a transduction fine-tuning stage is the highest-value next step.
