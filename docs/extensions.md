# Extensions roadmap

The extension phase focuses on low-resource deployment and accessibility.

## Lightweight models

- Sweep smaller rank values (e.g., 4, 8, 16) vs the paper’s 32–48.
- Plot parameter count vs VRMSE trade-offs.
- Evaluate INT8 quantization and measure accuracy loss.

## Robustness testing

- Add Gaussian noise at varying SNR levels.
- Evaluate at resolutions not seen during training (zero-shot super-resolution).
- Cross-dataset generalization across PDE families.

## Explainability

- Analyze Fourier mode importance to identify which frequencies drive predictions.

## Spectral characterisation of the data

- [Harmonic content by scenario](harmonic_content.md) — temporal and spatial
  variance decomposition for all six Gray-Scott scenarios and the eight
  turbulent-radiative-layer cooling times. Finds that only spirals and gliders
  contain a real temporal harmonic (the other four are frozen patterns with a
  red-noise drift), that the 60-step training window is too short to resolve
  either of them, and that maze/spots hold under 2% of their spatial variance
  below mode 8 while spirals holds 77%.
