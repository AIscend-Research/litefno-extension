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

## SpecScope: interrogating the trained operator

The extension's main line. Everything above characterises the *data*; these ask
what the trained network learned, by treating its spectral weights as an
empirical transfer function and extracting pole structure from them.

- [Reading the poles out of a trained operator](operator_poles.md) (ext19) --
  steps 1-3. Extracts a per-mode propagator from a checkpoint by two independent
  routes and scores it against systems whose poles are known in closed form. The
  poles are recovered (rank correlation 0.987, frequency error 1.5e-4), but the
  near-neutral *label* is below the method's resolution at a tight tolerance, and
  the linearized composition route is off by a near-constant factor and so is
  usable for ranking modes but not for an absolute stability call.
- [Does the pole readout predict failure?](resonance_risk.md) (ext20, H1) --
  step 4. The strong per-mode form does not survive its control: the raw
  correlation of -0.78 between pole margin and rollout error growth is matched by
  a wavenumber-only baseline at +0.78, leaving -0.14 after partialling. The weak
  per-scenario form does hold: an energy-weighted risk score computed from the
  weights and one frame separates the worse half of 20 held-out regimes at
  AUC 0.983.
- [Do the resonant factors carry across regimes?](mode_transplant.md) (ext21,
  H2) -- step 5. No: 0 of 8 (target, budget) cells put the resonant transplant
  ahead of the size-matched damped control. The overlap matrix says why, and it
  needed the same-regime different-seed pair to say it: two models trained on the
  *same* regime share no more spectral basis (0.258) than models trained on
  different ones (0.230-0.283), so the CP mode basis is set by initialization
  rather than physics and there is no identifiable subspace to move. Transfer
  itself is real and large -- full fine-tuning beats scratch by a factor of six
  at the smallest budget -- but it does not decompose.

## Downstream: what the surrogate's error costs a decision

- [What does a surrogate's error cost a fair decision?](fair_allocation.md)
  (ext22, H3) -- a fairness-aware resource allocation layer over the
  reconstructed ecosystem state, scored on realised welfare rather than on field
  error. Sensitivity to surrogate error is U-shaped in the fairness parameter
  with an exact zero at the envy-free point, so max-efficiency and max-min are
  both fragile and the fair middle is not; the closed-form law predicts a
  trained operator's realised decision cost to within 1%. The auxiliary network
  is 4-6x worse than pooling the field and applying the closed form when the
  surrogate is good, and 9.3x better when it is starved -- because it reads the
  region populations out of the field better than a block mean does, which two
  interpretable controls (a fitted shrinkage, a box blur) fail to reproduce.
  Acting on an 8-step-stale observation is worse than ignoring the state
  entirely.
- [Is the allocation robust to manipulation?](strategic_allocation.md)
  (ext23, H4) -- the same allocation layer read as a mechanism, when the regions
  being allocated to can influence what the allocator sees about them. The
  incentive to misreport and the sensitivity to surrogate error turn out to be
  the same derivative, `|1-alpha|/alpha`, so the two cannot be bought
  separately: the only strategy-proof member of the family is the envy-free one
  at alpha = 1, and it is strategy-proof because it ignores the state. Max
  efficiency is unboundedly manipulable on 100% of states -- a 20% lie flips the
  argmax and takes the whole budget. Adds a leximin implementation by
  progressive filling, where a per-region capacity bounds any lie without
  payments or verification: against a 10x misreport it cuts the winnings from
  6.93x to 1.64x for 4.2% of the worst-off region's welfare. The learned
  allocator is *not* a softer target than the closed form under a matched
  attack, and a no-regret learner, though it holds its guarantee, loses to a
  one-step forecast by three to four orders of magnitude -- its comparator is a
  constant allocation, which in an oscillating ecosystem it beats outright.
- [Does scarcity travel on a network the operator cannot see?](network_scarcity.md)
  (ext24, H5) -- regions that trade are not independent given the field, so
  scarcity is propagated between them by an SIS cascade borrowed from
  epidemiology and a graph-convolutional head is added on top of LiteFNO. What
  makes the result interpretable is a closed form: on a periodic grid the Fourier
  modes are exactly the lattice Laplacian's eigenvectors (residual 1.5e-14), so
  the spectral convolution is already a graph convolution and only non-lattice
  edges can add capacity. Four arms at 4913 parameters each, differing in one
  matrix, confirm the topological half of H5 -- the true network's advantage over
  a lattice graph is 0% at zero shortcuts and 21-28% once a fifth or more of the
  edges leave the lattice, and it survives a degree-preserving rewired control by
  12-23%. The null half is refuted: hard-wiring even the *redundant* lattice
  graph buys 7.3%, so representable is not learned. The simulated cascade matches
  its `1/lambda_1` epidemic threshold to 0.3% on four graph families. Sentinel
  placement by centrality beats random by only 0.22 steps of warning, and on the
  regular lattice the centrality scores are constant, which the table labels
  rather than reports as a result.

## Baseline

- [In-distribution reference number for LiteFNO](baseline_reference.md) — the
  number the extensions are implicitly compared against, as a runnable command
  rather than a one-off notebook. Trains the real CP-factorized spectral model
  alongside FNO-S and the repo's low-rank CNN under one protocol, and checks the
  result against the committed Kaggle run. Also adds
  `scripts/stream_preprocess.py`, which builds the processed splits from The
  Well over HTTP range requests instead of downloading 44 GB to throw 99% of it
  away.
