# TL;DR

## The paper in simple terms
- The paper shows how to predict how physical systems change over time (fluids, reaction-diffusion, etc.).
- It uses a Fourier Neural Operator (FNO), which learns patterns in the frequency domain.
- LiteFNO makes FNO cheaper by using low‑rank approximations, so it runs faster and uses fewer parameters.
- The goal is to keep accuracy close to the full model while reducing cost.

## This project in simple terms
- Reproduces the LiteFNO paper results on The Well datasets.
- Uses configs for download, preprocessing, training, and evaluation so runs are repeatable.
- Records metrics (RMSE/VRMSE) and notes any deviations or fixes from the paper.

## What we are trying to achieve
- Match the paper’s reported accuracy for both LiteFNO and the FNO‑S baseline.
- Ensure preprocessing and hyperparameters align with the paper and document any differences.
- Extend the work to low‑resource settings (smaller ranks, quantization, robustness tests, explainability).
- Produce clear, reproducible documentation for every experiment and change.
