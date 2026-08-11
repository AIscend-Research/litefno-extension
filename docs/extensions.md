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
- [Field recovery under thin sensor coverage](data_sparsity.md) — the same six
  scenarios masked to four low-resource sampling regimes (random pixels, a
  regular lattice, swath-shaped block gaps, a few point stations). Sampling
  geometry dominates density: a lattice at 5% coverage beats i.i.d. random
  sampling at the same density by 3-5x and clustered coverage by more than 10x,
  which is worse than predicting the mean at any density below 25%. Confirms the
  harmonic-content prediction at Spearman rho = -1.000.
- [Forced harmonics: is a temporal prior worth it?](forced_harmonics.md) — the
  temporal version of the harmonic claim, tested on planetswe, whose daily
  (24-step) and annual (1008-step) forcing periods are documented rather than
  inferred. The forcing is unambiguous (phase-locked at 0.995-1.000 across four
  independent trajectories, enriched >700x over chance) but accounts for only
  5.4% of temporal variance globally and 11.5% in the best latitude band. The
  temporal prior survives where the spatial one did not, as a real but minor
  effect.
