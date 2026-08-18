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

- [Verdict on the harmonic claim](harmonic_verdict.md) -- consolidates ext9,
  ext12, ext15 and ext30 into one negative result rather than leaving it
  mentioned in passing across four documents. The claim was tested in three
  forms: the low-wavenumber spatial prior (**dead** -- the spectrum peaks at k=5,
  needs 15 shells for 95% of variance, and only 29% sits at or below k=4), the temporal
  prior under documented forcing (**survives, small** -- phase-locked at
  0.995-0.998 but worth 5.4% of temporal variance), and the mid-spectrum bias on
  the Turing shells (**dead** -- 0/12 paired wins, 1.00x data multiplier, no
  differential effect on maze and spots). Written up beside the negative
  reproduction in [reproducibility_findings.md](reproducibility_findings.md),
  which it mirrors from the opposite direction: that page asks whether a generic
  Fourier bias earns its keep, this asks whether sharpening it does.

- [The harmonic prior on data that actually oscillates](seasonal_real_data.md)
  (ext31, H12) -- swaps Gray-Scott for NOAA CPC global monthly soil moisture
  (0.5 degree, 943 months, 1948-2026) through the same 5-D HDF5 contract, to
  answer the strongest objection to the harmonic verdict: that the prior was
  only ever tested where no seasonal cycle existed. Six regions replace six
  regimes and span a 6x seasonality range, with the monsoon belt at **65%** of
  temporal variance in the annual cycle against planetswe's 5.4%. The prior
  still does not help -- data multiplier **0.96x** at the full training set,
  3 of 6 paired runs favouring it -- and the differential prediction inverts:
  the three most seasonal regions get worse (mean +6.1%), two of the three least
  seasonal improve, Spearman +0.314 where the mechanism needs negative (n=6,
  p=0.54, single seed per region, so suggestive rather than established). The
  harmonic arm's seed spread is 3.0x the plain arm's at the largest size. Also
  records the leakage fix the seasonality measurement needs: 943 months is not a
  whole number of years, and the raw annual line reads 7.4% where the truth is
  13.4%.

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

- [Ablating the instrument](pole_ablation.md) (ext33, H14) -- runs ext19's pole
  readout on weights that never learned anything, to test whether the extracted
  pole structure is a property of the trained operator or of the architecture.
  Four arms share one readout and differ only in weights: trained, untrained at
  the same seed, the trained weights shuffled within each tensor (preserving
  each tensor's exact multiset of values), and weights resampled to the trained
  moments. **The structure does not survive.** On the rotating system trained
  scores **+0.9876** against the exact magnitudes -- reproducing ext19's 0.987 --
  and no control exceeds **+0.12**, a gap of 0.871; on advection the gap is
  0.520. Frequency separates them harder still: 1.86e-4 trained against 2.2e-2
  to 4.7e-1. The readout measures the operator, not its construction, and ext20
  and ext21 do not inherit an artifact. Also records a methodological finding:
  ext20's partial-correlation control is **undefined** on these systems, because
  the exact pole is a function of |k| alone (rho = -1.0000 rotating, -0.9996
  advection), so partialling removes the ground truth -- which is why the
  control has to be an ablation of the weights.

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

- [Fairness checklist](fairness_checklist.md) — a structured statement of what
  ext22-ext24 license, written because those extensions use the vocabulary of
  distributive justice (envy-freeness, leximin, price of fairness) on a decision
  problem that is entirely synthetic. Records the proxy standing behind every
  quantity, the notions that are *not* measured (group, individual and
  counterfactual fairness — there are no protected attributes to define them
  over), the single-deviation assumption that all the manipulation numbers rest
  on, and what would have to change before any of it is an equity claim.

- [Where do real disaster allocations sit in the alpha-family?](fema_svi_equity.md)
  (ext32, H13) -- turns ext22's fragility law and ext23's manipulation result
  from statements about a family of rules into a measurement of one real
  allocation, using FEMA Public Assistance obligations against CDC/ATSDR social
  vulnerability. The bridge is exact and tested across the family rather than
  asserted: the family allocates `a ∝ g^beta` with `beta=(1-alpha)/alpha`, so
  ext23's manipulation incentive `|1-alpha|/alpha` **is** the fitted log-log
  elasticity. Every fit is within one disaster, across only the counties FEMA
  declared for it, because Public Assistance tracks damage and a pooled
  cross-section would measure where storms land. Across 12 hurricanes since
  FY2015 there is **no vulnerability gradient**: sign splits 6/6, median R2 is
  **0.025**, the inverse-variance pooled elasticity is **-0.088 +- 0.266**
  (z = -0.33), and only 2 of 12 slopes beat their own permutation null -- in
  opposite directions. Pooled `alpha_hat = 1.10` puts observed allocation
  statistically at the envy-free point, which ext23 proved is the only
  strategy-proof rule *because it ignores the state*, and ext22 puts at exactly
  zero sensitivity to forecast error. The doc states plainly that "sits at
  alpha=1" and "has no detectable signal" are the same observation here. The
  household-facing Individual Assistance arm, where a gradient would actually be
  expected, is **underpowered and not reported as a result** -- registration
  volumes skipped every large hurricane and exactly one disaster survived.

## Cost: is any of this deployable?

- [Is the low-rank operator actually deployable?](deployability.md) (ext25, H6)
  -- size, FLOPs and latency across nine arms spanning 383x in parameters, with
  the FLOP model derived in closed form and checked against torch's own tracer
  (exact on the CNN and dense-spectral arms, 0.4% on CP). H6 -- that parameter
  count predicts deployability -- is refuted: the rank correlation between
  parameters and batch-1 latency is 0.067, and the CP-factorized arm has 383x
  fewer parameters than dense FNO-S while running 37% slower. The cause is closed
  form: CP rebuilds its dense spectral weight every forward pass at
  `8*rank*in*out*m1*m2` flops, which does not scale with batch size and is
  74-92% of the batch-1 budget. Folding that reconstruction once at eval time is
  worth 1.4-1.8x for bitwise-identical outputs, an unchanged checkpoint and
  +8-64 MB of RAM. FLOPs rank models much better (0.57-0.88) but price them
  badly -- the removed work ran at 115 GFLOP/s and what remains runs at 8, so
  the closed form over-predicts the speedup 12.35x against 1.79x measured.
  Latency scaling with grid size inverts the ranking between 32x32 and 128x128,
  and every width-64 configuration in the repo -- including the two its own
  protocol trains -- misses a 12-hour CPU session by a factor of two.

## Generalization: off the training distribution

- [Leave-one-regime-out generalization](cross_regime.md) (ext14) -- six folds,
  each holding out one Gray-Scott regime and scoring the same model on the
  held-out regime and on the five it trained on. Only **two of six** regimes cost
  anything to hold out: spirals 2.07x and bubbles 1.87x, against gliders 0.98x,
  worms 0.97x, maze 0.90x and spots 0.56x. There is no single "cross-regime gap"
  for this dataset -- a study holding out one regime would have concluded almost
  anything depending on which. Five folds are flat across the epoch budget;
  bubbles was still climbing, so its 1.87x is a lower bound. The pre-registered
  ext10 prediction that the spectral outliers maze and spots should be the two
  *worst* folds scores 0/2 -- they are the two best, with
  rho(variance below mode 8, gap) = +0.943 where the mechanism requires negative.

- [Harmonic conditioning on the Turing shells](harmonic_conditioning.md) (ext15)
  -- a CP-factorized spectral convolution plus a learnable complex bias on a
  fundamental wavenumber and its multiples, added on radial shells because a
  Turing pattern selects a wavelength rather than an orientation. The band is
  inherited from ext10 (maze and spots keep ~99% of spatial variance above mode
  8, near mode 3-4 after downsampling) rather than tuned on the data it is tested
  on. Control and conditioned arms are bit-identical at initialisation -- the bias
  starts at zero, pinned by a test -- and the bias adds 1-2% of parameters, so a
  difference cannot be attributed to size. The prediction is differential: it
  should help maze and spots and do nothing for spirals and gliders, and a
  *uniform* gain would be evidence against the mechanism. **Not yet run** -- the
  model, script and tests exist; no results do.

- [Data-efficiency curve: does the harmonic prior buy data?](data_efficiency.md)
  (ext30, H11) -- the ext15 arms run across four training-set sizes at three
  seeds, asking not whether the conditioned arm is more accurate but whether the
  prior *substitutes for data*, which would show as a leftward shift of the
  error-vs-size curve. It does not: the data multiplier is **1.00x** at every
  size where it is measurable, the harmonic arm is very slightly worse at all
  four sizes, and it wins **0 of 12** paired runs -- an effect about 2% of the
  seed spread, consistent with its 220 extra parameters adding noise and nothing
  else. The differential prediction fails too: maze improves 0.1% and spots, the
  more extreme outlier the mechanism should favour most, gets slightly worse.
  This is neither a benefit nor the uniform-gain failure ext15 named in advance;
  it is no effect. The plain arm's own scaling is the useful number --
  `VRMSE ~ 0.239 * n^-0.521`, residuals within 4.7%, so quadrupling the data
  cuts error 2.04x. Data buys accuracy at the classical rate here; this prior
  buys none of it, which is what ext15's own pre-registered low prior expected.

- [Can the cross-regime gap be bought down?](cross_regime_arms.md) (ext26, H7)
  -- three arms on the leave-one-regime-out fold: baseline, noise-augmented
  (`robust`), and a 4-member ensemble on top (`robust+unc`), the last differing
  from the second in exactly one field so the ensemble's contribution is
  separable. First finding is about the gap itself: four of six regimes are no
  harder held out than the five trained on, and spots is nearly twice as easy,
  so the gap belongs to spirals (2.07x) and bubbles (1.87x) rather than to the
  setup. Over those two folds `robust` closes a median 2% and `robust+unc`
  closes -17%, widening both while degrading held-out error in 5 of 6 folds and
  improving in-distribution error -- spirals shows it purest, the ensemble
  posting the best seen error of any arm and the worst held-out. Folds with no
  gap report NaN rather than dividing by a small negative excess, which would
  turn trivial movements into 249% and -368%. The pre-registered ext10
  prediction that the spectral outliers maze and spots should be hardest scores
  0/2: rho(variance below mode 8, gap) = **+0.943** where the mechanism requires
  negative, and the symmetric variant is -0.486, also the wrong sign for its own
  claim. bubbles was still climbing at the epoch budget, so its 1.87x is a lower
  bound.

- [Accuracy under input degradation](degradation_robustness.md) (ext27, H8) --
  a synthetic capture chain (illumination, periodic blur, sensor noise,
  quantisation) swept 0% to 100%, every component exactly the identity at zero
  so the clean reference is genuinely clean; both corruptions return
  bit-identical VRMSE there, confirming it in the run. Training on the chain
  closes a median **81%** of the degradation-induced error rise (baseline
  degrades 16x from clean to full severity, the robust arm 2.6x). Three
  qualifications: a **177%** clean-input tax, curves that do not cross until
  ~20% severity so the robust arm is worse below that despite closing 92% of
  the rise at 10%, and a held-out corruption (pixel dropout) where median
  closure falls to **27%** and the robust arm is worse than baseline at 10%.
  Two thirds of the apparent benefit is familiarity with the augmentation,
  visible only because of the held-out control.

- [Safe Deferral Rate](safe_deferral.md) (ext28, H9) -- whether the surrogate
  can abstain its way back to in-distribution accuracy, where deferring costs
  the real solver so the metric is a rate. On bubbles (a real 1.95x gap) the
  Safe Deferral Rate is **never reached**: 80% deferral still leaves 1.05x, and
  the **oracle never reaches it either**, so the gap is not concentrated in a
  discardable minority and no threshold on any signal removes it. The signal is
  not the limitation -- ensemble disagreement captures 100% of the oracle's
  achievable gain at nearly every rate while random deferral stays flat. What it
  is good for is triage: specificity 1.00 through 50% deferral with sensitivity
  0.94, and every unsafe step caught at 55% with specificity still 0.96. The
  maze fold is kept alongside because it fails differently -- no gap (0.94x), so
  every signal including random reports 0%, vacuous rather than favourable, on
  a sensitivity curve resting on 2 positives.

- [Are the confidence scores honest?](uncertainty_calibration.md) (ext29, H10)
  -- reliability diagrams, ECE and MCE for the same ensemble ext28 found ranks
  errors as well as an oracle. Ranking is invariant to scale, so this asks the
  independent question and gets the opposite answer. Every coverage gap is
  negative at every level on both splits -- the intervals are never too wide,
  only too narrow -- and ECE **doubles** off-distribution, 0.066 to 0.131
  (MCE 0.137 to 0.217). A nominal 95% interval covers 73% of the held-out
  regime. The error-vs-sigma diagram is worse than the coverage numbers admit:
  achieved/claimed never drops below 1.4 in-distribution or 1.8 out of it, and
  it is **U-shaped**, peaking at 4.6x and **5.2x** in the *most confident* bin,
  so the signal is least trustworthy exactly where it claims certainty. A single
  scale fit in-distribution (sigma *= 1.16) helps but cannot flatten a
  discrepancy that varies with the claimed sigma -- rescaled out-of-distribution
  ECE (0.094) remains worse than uncorrected in-distribution ECE (0.066). Both
  the Gaussian assumption and the low bias of a 4-member sample standard
  deviation push toward apparent overconfidence, so the magnitude is an upper
  bound; the sign is not in doubt at 5.2x.

- [Does the Fourier advantage appear above 32x32?](resolution_scaling.md)
  (ext36, H15) -- the reproduction's negative result is explicitly scoped to
  32x32, so this sweeps 32, 64 and 128 from one native-resolution stream. The
  honest headline is not the intended one: **this run does not reproduce the
  reproduction at 32x32**. The CP spectral arm beats the parameter-matched CNN
  at *every* resolution (+12.5%, +6.4%, +18.3%), including the size where the
  reference reports the opposite -- under a much thinner protocol (24
  trajectories / 30 epochs / 2 seeds against 72 / 200 / 3), so it is a fact
  about this budget rather than an overturning. Only the **128x128** cell is
  measured precisely enough to carry a claim: both seeds agree (+19.1%, +17.5%)
  and the CNN's seed spread collapses to **0.1%**, against 12.3% at 32 and
  19.5% at 64 where one seed **reverses the sign** (-4.1%). So the script's
  printed "opens with resolution" verdict is *not supported* -- the sequence is
  non-monotone and two seeds cannot separate "the gap opens" from "the gap was
  always there and the measurement got quieter". Two limitations are load-
  bearing: only the `repo` modes policy ran, so scale and truncation are
  confounded (16 of 16 modes at 32, 16 of 64 at 128) and the implemented
  `proportional` arm that separates them was not executed; and dense `fno_s`
  (2.1M params) is worst everywhere, so whatever helps is specific to the
  CP-factorized low-parameter form rather than to spectral layers.

- [Does the manipulation law survive coalitions?](coalition_manipulation.md)
  (ext35, H16) -- ext23's `|1-alpha|/alpha` was derived and checked against
  *single-region* deviations, the weakest threat model available. Run against
  joint deviations it survives **exactly**: the capture ratio is the same closed
  form with the region's share replaced by the coalition's pooled share,
  matching to 4.4e-16 over 96 cells. Two consequences are sharp. alpha = 1 is
  **group** strategy-proof, not merely strategy-proof (ratio exactly 1 at every
  coalition size), and a **grand coalition captures exactly nothing** -- if
  everyone inflates by the same factor the normalisation cancels. The third is
  the one that could have gone the other way: **collusion dilutes itself**. Joint
  capture is **subadditive in 90 of 90 cells**, a cartel of eight winning about
  half what its members would have won deviating one at a time, so the lone
  deviator is the worst case and ext23's numbers were the bound rather than an
  optimistic reading. The ratio hides the harm, though -- members do worse per
  head while the budget moved and the welfare lost both roughly quadruple from
  |C|=1 to |C|=8, and both are reported. The assumed corner lie is verified by
  exhaustive search (45,216 alternative joint reports, 0 beat it). Finally,
  ext23's capacity cap is shown to be **aimed at the wrong regions**: under
  leximin the regions with most to gain by lying are the least-served ones and a
  uniform cap constrains the most-served, so at ext23's own kappa = 1.5 the cap
  never binds; it does defend the tail at kappa = 16 (9.2x held to 1.89) but only
  for small coalitions, being redundant against large ones where dilution has
  already done the work.

- [H2 as a dose-response, on distance and on dose](transplant_distance.md)
  (ext34) -- turns ext21's yes/no into a curve, and **overturns its null on the
  way**. The blocker was an accounting one: ext21's 3-component transplant writes
  252 of 7,106 parameters, **3.55%** of the model, and was implicitly read
  against a fine-tune ceiling that writes 100%, so a small effect from a small
  dose looked like no effect. Adding dose as a second axis, the resonant arm
  beats its size-matched damped control in **90 of 90 paired runs** (every seed
  10/10, smallest gap +0.53%) with the gap growing monotonically in dose (+1.7%,
  +1.8%, +2.5% at 1, 2, 3 components; Spearman(dose, gap) = +0.764). ext21's null
  was a **power failure**: re-read as relative gaps, 6 of its own 8 cells are
  already positive with the two negatives at the largest budgets, and its
  one-standard-deviation-per-cell test was asking a ~2% effect to beat a 14-49%
  seed spread. Pairing within a seed cancels that noise. What does **not** change
  is the practical conclusion -- freezing every component buys +8.4% over scratch
  against fine-tuning's +80.4% -- so transfer is real, large, and still does not
  usefully decompose; what fails is only the stronger claim that the basis is
  arbitrary. The **requested axis is flat**: Spearman(distance, gap) = -0.319,
  gaps +1.8% to +2.8% with no clean decay over a 3.5x change in both parameters,
  so H2-as-a-curve on regime *distance* is not supported and the dose-response
  that exists is on dose.

## Baseline

- [In-distribution reference number for LiteFNO](baseline_reference.md) — the
  number the extensions are implicitly compared against, as a runnable command
  rather than a one-off notebook. Trains the real CP-factorized spectral model
  alongside FNO-S and the repo's low-rank CNN under one protocol, and checks the
  result against the committed Kaggle run. Also adds
  `scripts/stream_preprocess.py`, which builds the processed splits from The
  Well over HTTP range requests instead of downloading 44 GB to throw 99% of it
  away.
