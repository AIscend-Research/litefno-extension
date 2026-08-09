# Extensions roadmap

The extension phase focuses on low-resource deployment and accessibility.

## Lightweight models

- Sweep smaller rank values (e.g., 4, 8, 16) vs the paper's 32-48.
- Plot parameter count vs VRMSE trade-offs.
- Evaluate INT8 quantization and measure accuracy loss.

## Robustness testing

- Add Gaussian noise at varying SNR levels.
- Evaluate at resolutions not seen during training (zero-shot super-resolution).
- Cross-dataset generalization across PDE families.

## Explainability

- Analyze Fourier mode importance to identify which frequencies drive predictions.
