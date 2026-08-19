# Paper kit

Everything needed to write the paper, in one place: the story, every headline
number with the file that backs it, a figure manifest with suggested captions,
and the caveats that must be stated. Nothing here is new — every number is
copied from the extension docs, which remain the authoritative source; each
section links to the doc it condenses.

Companion archive: `paper_figures.zip` at the repo root holds every paper-usable
figure (all PNGs and SVGs under `figures/`, folder structure preserved; GIFs
excluded as non-print).

---

## 1. The story in one paragraph

A from-scratch reproduction of LiteFNO (a CP-factorized low-rank Fourier neural
operator) on The Well's Gray-Scott data finds **no consistent evidence for a
Fourier inductive-bias advantage at 32×32**: a parameter-matched low-rank CNN
matches or outperforms the spectral arm on one-step VRMSE and rollout across
three seeds. The extensions then interrogate that operator from every side. The
**harmonic claim** — that naming the Fourier modes that matter should help —
fails in four forms, including on real seasonal data where the annual mode holds
65% of the variance. **SpecScope** shows the trained weights nonetheless contain
real, extractable physics: pole structure recovered at rank correlation 0.987,
validated by an ablation in which no untrained control exceeds +0.12, and
learned factors that *do* transplant across regimes once the test is properly
powered (90/90 paired wins). A **decision layer** on top of the surrogate yields
closed-form results — sensitivity to surrogate error and incentive to misreport
are the same derivative, the envy-free rule is group strategy-proof, collusion
dilutes itself — and a measurement of 12 real FEMA hurricane allocations lands
statistically at exactly that envy-free point. Throughout, measurements of the
*data* (ext9/ext10 spectra) predicted the fate of every prior placed on the
*model*.

## 2. Framing that must be stated (scope caveats)

These are documented in [reproducibility_findings.md](reproducibility_findings.md),
[notes_deviations.md](notes_deviations.md), and [visuals.md](visuals.md):

1. **From-scratch reimplementation.** The repo is not a fork of and does not run
   the authors' released code; no claims are made about the official
   implementation.
2. **The `litefno.py` placeholder.** `src/litefno/models/litefno.py` is a
   low-rank CNN with no FFT. Everything in `results/logs/` and the committed
   checkpoints comes from it. The genuine CP-factorized spectral convolution is
   `src/litefno/models/harmonic.py` (added in ext15) and the neuralop model in
   the phase-2 notebook; the 3-seed headline numbers use the real spectral arm.
3. **Protocol deviations.** Matched internal protocol (200 epochs, MSE,
   single-step, Gray-Scott downsampled 4× to 32×32) rather than the paper's
   500 + 100-epoch transduction recipe — so numbers are not directly comparable
   to the paper's Tables 6/7.
4. **The simulation renders are illustrations, not the data.** Independently
   simulated at 384×384 with empirically chosen (F, k); spirals is the weakest
   likeness. Fine for "the regimes look like this", not as evidence about the
   trained model.
5. **Synthetic decision problem.** ext22–ext24 use fairness vocabulary on a
   fully synthetic allocation task; the [fairness checklist](fairness_checklist.md)
   states what is and is not licensed. ext32 is the one measurement on real data.

## 3. Headline results by storyline

### A. The reproduction: a negative result about the Fourier bias

- On Gray-Scott at 32×32, 3 seeds: a parameter-matched low-rank CNN matches or
  outperforms the CP-factorized spectral LiteFNO on one-step VRMSE and
  autoregressive rollout. ([reproducibility_findings.md](reproducibility_findings.md);
  `results/seeds/`, `figures/headline/headline_3seed.png`)
- Spectral drift: LiteFNO's high-frequency energy over/undershoots ground truth
  during rollout while the CNN decays smoothly (`figures/headline/headline_3seed.png`,
  right panel; `figures/mechinterp/spectral_drift.png`).
- **ext36 scopes it**: sweeping 32/64/128 under a thinner budget (24 traj / 30
  epochs / 2 seeds), the spectral arm wins at every resolution (+12.5%, +6.4%,
  +18.3%) — *including* 32×32, so the disagreement with the reference is about
  training budget, not resolution. Only 128×128 is measured precisely enough to
  carry a claim (both seeds agree, CNN seed spread 0.1%); the sequence is
  non-monotone and "the gap opens with resolution" is **not supported**.
  ([resolution_scaling.md](resolution_scaling.md), `ext36_resolution_*.csv`)

### B. The harmonic claim: dead in four forms ([harmonic_verdict.md](harmonic_verdict.md))

| form | tested in | outcome | key numbers |
| --- | --- | --- | --- |
| Low-wavenumber spatial prior | ext9 | **dead** | spectrum peaks at k=5; 58% of variance at k≥6; k≤4 holds 29.2% |
| Temporal prior under documented forcing | ext12 | **survives, small** | phase-locked 0.995–0.998, enriched >700×, but 5.4% of temporal variance |
| Mid-spectrum bias on the Turing shells | ext15 / ext30 | **dead** | worse in 3/3 seeds, ρ = −0.200 (prediction inverts); 0/12 paired wins; data multiplier 1.00×; gap ≈ 2% of seed spread; 3.5 sweep also null |
| Annual prior on data that actually oscillates | ext31 | **dead** | NOAA CPC soil moisture, monsoon belt 65% annual variance; multiplier 0.96×; three most seasonal regions get *worse* (mean +6.1%) |

Supporting characterisation: ext10 (maze/spots hold ~99% of spatial variance
above mode 8, in a ring at the Turing wavelength — the measured basis for the
ext15 band), ext11 (sampling geometry dominates density; lattice at 5% beats
i.i.d. by 3–5×; Spearman ρ = −1.000 against the harmonic-content prediction).

The scaling law that *does* hold: plain-arm error follows
`VRMSE ≈ 0.239·n^−0.521` (residuals within 4.7%) — quadrupling data cuts error
2.04×; the prior buys none of it. ([data_efficiency.md](data_efficiency.md))

### C. SpecScope: the operator contains real, extractable structure

- **ext19** — poles extracted from a checkpoint match closed-form ground truth:
  rank correlation **0.987**, frequency error **1.5e-4**. Near-neutral *labels*
  are below the method's resolution; the composition route ranks but cannot make
  absolute stability calls. ([operator_poles.md](operator_poles.md))
- **ext33** — the ablation that makes ext19 believable: identical readout on
  untrained / shuffled / moment-matched weights. Trained scores **+0.9876**; no
  control exceeds **+0.12**. The readout measures the operator, not the
  architecture. Also: ext20's partial-correlation control is undefined here
  because the exact pole is a function of |k| alone (ρ = −1.0000).
  ([pole_ablation.md](pole_ablation.md))
- **ext20 (H1)** — strong per-mode form fails its control (−0.78 raw is matched
  by a wavenumber-only baseline; −0.14 after partialling); weak per-scenario
  form holds: risk score separates the worse half of 20 held-out regimes at
  **AUC 0.983**. ([resonance_risk.md](resonance_risk.md))
- **ext21 → ext34 (H2)** — ext21's null (0/8 cells) is **overturned as a power
  failure**: the transplant wrote 3.55% of parameters and was read against a
  100% ceiling. Paired within seed with dose as an axis, resonant beats damped
  in **90 of 90 runs**, gap growing monotonically in dose (+1.7/+1.8/+2.5%,
  Spearman +0.764). Unchanged: transfer is large (+8.4% frozen vs +80.4%
  fine-tuned) and still does not usefully decompose. Distance axis is **flat**
  (Spearman −0.319). CP mode basis is set by initialization, not physics
  (same-regime overlap 0.258 vs cross-regime 0.230–0.283).
  ([mode_transplant.md](mode_transplant.md), [transplant_distance.md](transplant_distance.md))

### D. The decision layer: closed forms, stress-tested, then measured on reality

- **ext22 (H3)** — sensitivity of realised welfare to surrogate error is
  U-shaped in the fairness parameter with an **exact zero at the envy-free
  point**; the law predicts a trained operator's decision cost within 1%. The
  learned allocator is 4–6× worse than the closed form when the surrogate is
  good, 9.3× better when starved. Acting on an 8-step-stale observation is worse
  than ignoring the state. ([fair_allocation.md](fair_allocation.md))
- **ext23 (H4)** — manipulation incentive and error sensitivity are the **same
  derivative** `|1−α|/α`; the only strategy-proof rule is α = 1 (envy-free),
  *because it ignores the state*. Max-efficiency is unboundedly manipulable on
  100% of states. Leximin with per-region caps holds a 10× lie to 1.64× (from
  6.93×) for 4.2% of worst-off welfare. ([strategic_allocation.md](strategic_allocation.md))
- **ext35 (H16)** — the law survives coalitions **exactly** (share → pooled
  share, 4.4e-16 over 96 cells): α = 1 is **group** strategy-proof, a grand
  coalition captures nothing, and **collusion dilutes itself** (subadditive in
  90/90 cells — a cartel of 8 wins about half what lone deviators would).
  Corner lie verified against 45,216 alternatives. ext23's cap is aimed at the
  wrong regions under leximin. ([coalition_manipulation.md](coalition_manipulation.md))
- **ext32 (H13)** — the elasticity is measurable on real data: 12 hurricanes of
  FEMA Public Assistance vs CDC/ATSDR SVI, fitted within disaster. **No
  vulnerability gradient**: sign splits 6/6, median R² 0.025, pooled elasticity
  −0.088 ± 0.266 → **α̂ ≈ 1.1, statistically at the envy-free point** — the rule
  ext23 proved strategy-proof because it ignores the state. The
  household-facing IA arm is underpowered and not reported as a result.
  ([fema_svi_equity.md](fema_svi_equity.md))
- **ext24 (H5)** — on a periodic grid the Fourier modes *are* the lattice
  Laplacian's eigenvectors (residual 1.5e-14), so only non-lattice edges add
  capacity: true network beats degree-preserving rewiring by 12–23%, beats the
  lattice by 0% at zero shortcuts rising to 21–28%; hard-wiring even the
  redundant lattice buys 7.3% (representable ≠ learned). Cascade matches the
  1/λ₁ epidemic threshold to 0.3%. ([network_scarcity.md](network_scarcity.md))

### E. Off-distribution behaviour and honesty of uncertainty

- **ext14** — there is no single "cross-regime gap": only spirals (2.07×) and
  bubbles (1.87×, a lower bound) cost anything to hold out; spots is *easier*
  (0.56×). The pre-registered spectral-distance prediction scores 0/2 —
  ρ = **+0.943** where the mechanism requires negative. ([cross_regime.md](cross_regime.md))
- **ext26 (H7)** — noise augmentation closes a median 2% of the gap; a 4-member
  ensemble closes **−17%** (widens it) while improving in-distribution error.
  ([cross_regime_arms.md](cross_regime_arms.md))
- **ext27 (H8)** — training on a synthetic capture chain closes a median **81%**
  of degradation-induced error rise, but at a **177% clean-input tax**, and only
  **27%** on a held-out corruption — two-thirds of the apparent benefit is
  familiarity with the augmentation. ([degradation_robustness.md](degradation_robustness.md))
- **ext28 (H9)** — the Safe Deferral Rate is **never reached**, even by the
  oracle: the error is not concentrated in a discardable minority. The ensemble
  signal captures 100% of the oracle's achievable gain and triages unsafe steps
  at specificity 1.00 through 50% deferral. ([safe_deferral.md](safe_deferral.md))
- **ext29 (H10)** — the same signal is dishonest as an error bar: overconfident
  at every level on both splits, ECE doubles off-distribution (0.066 → 0.131), a
  nominal 95% interval covers 73%, and achieved/claimed error is U-shaped,
  peaking at **5.2×** in the *most confident* bin. ([uncertainty_calibration.md](uncertainty_calibration.md))

### F. Cost and deployability

- **ext25 (H6)** — parameter count predicts latency at rank correlation
  **0.067**: the CP arm has 383× fewer parameters than dense FNO-S and runs
  **37% slower**, because CP rebuilds its dense spectral weight every forward
  pass (74–92% of the batch-1 budget). Folding it once at eval is worth
  1.4–1.8× for bitwise-identical outputs. FLOPs rank models (0.57–0.88) but
  misprice them 12.35× vs 1.79× measured. ([deployability.md](deployability.md))
- Early robustness/cost sweeps from the reproduction phase: rank–VRMSE Pareto
  (`gray_scott_rank_pareto.png`), INT8 quantization (ext2), noise robustness
  (ext3), rollout error growth (ext4), spectral sensitivity (ext5), energy
  spectrum (ext6), error maps (ext7), CPU throughput benchmark (ext8; CNN
  peaks at batch 16). Files: `results/extensions/ext{2,3,4,5,6,8,9}_*.csv`,
  figures of the same names.

## 4. Figure manifest with suggested captions

Paths relative to `figures/`; all included in `paper_figures.zip`.

### Main-text candidates

| file | suggested caption / role |
| --- | --- |
| `simulations/gs_atlas.png` | The six Gray-Scott regimes in The Well, re-simulated at 384×384 for legibility (training uses 32×32). Labelled with the (F, k) used for the render; illustration, not training data. |
| `headline/headline_3seed.png` | Headline reproduction: rollout VRMSE and spectral drift, mean ± std over 3 seeds; the parameter-matched CNN matches or beats the spectral arm and drifts less. |
| `diagrams/spectral_layer.svg` | Method: the shared spectral path with W stored dense (FNO-S, 589,824 params) or CP rank-R (LiteFNO, 4,896 params, 120× fewer); the dashed harmonic-bias block is optional and input-independent. |
| `diagrams/mode_shells.svg` | The harmonic shells: which of the 12×12 retained modes receive a learnable bias, and why a ring rather than a point (a Turing pattern selects a wavelength, not an orientation). |
| `extensions/ext15_harmonic_ab.png` | The title-claim A/B: control vs harmonic-conditioned arms, bit-identical at init; per-regime effect vs spectral profile; the differential prediction inverts. |
| `extensions/ext19_operator_poles.png` | Pole extraction vs closed-form ground truth on the exactly solvable testbed. |
| `extensions/ext20_resonance_risk.png` | Pole margin vs rollout error growth, and the per-scenario risk score (AUC 0.983). |
| `extensions/ext21_mode_transplant.png` | Transplant vs damped control and the overlap matrix: the CP basis is set by initialization. (Pair with ext34's correction in the text.) |
| `extensions/ext22_fair_allocation.png` | Decision cost of surrogate error is U-shaped in α with an exact zero at the envy-free point. |
| `extensions/ext23_strategic.png` | Manipulation: capture ratio across the α-family; leximin capacity cap vs a 10× misreport. |
| `extensions/ext35_coalition.png` | Coalitions: the capture law holds exactly with pooled share; collusion is subadditive — the lone deviator is the worst case. |
| `extensions/ext24_network_scarcity.png` | Scarcity on an unseen trade network: four arms at matched parameters; only non-lattice edges add capacity. |
| `extensions/ext25_deployability.png` | Parameters vs latency across nine arms spanning 383×: parameter count does not predict deployability (ρ = 0.067). |

### Supplementary / appendix candidates

| file | role |
| --- | --- |
| `simulations/gs_<regime>.png`, `gs_<regime>_strip.png` (×6) | Per-regime high-res still and five-frame formation filmstrip. |
| `diagrams/pipeline.svg`, `diagrams/alpha_fairness.svg` | Data pipeline; the α-fairness family. |
| `reproduction/logs_convergence.png`, `logs_frontier.png`, `logs_improvement.png` | Training-log convergence and frontier. |
| `mechinterp/cprank.png`, `deadmode.png`, `mode_ablation.png`, `rollout_curve.png`, `spectral_drift.png` | 3-seed mech-interp: CP rank usage, dead modes, mode ablation, rollout, drift. |
| `extensions/ext9_variance_decomposition.png`, `ext10_harmonic_content_*.png`, `ext11_sparsity_gray_scott.png`, `ext12_planetswe_forced.png` | The data characterisation behind the harmonic verdict. |
| `extensions/ext2_quantization.png` … `ext8_benchmark.png`, `gray_scott_rank_pareto.png`, `freebieB_flops.png` | Reproduction-phase robustness and cost sweeps. |
| `pilot/ext26_cross_regime.png` … `ext30_data_efficiency.png`, `pilot_cpc/ext30_data_efficiency.png` | OOD arms, deferral, calibration, data-efficiency (Gray-Scott and CPC). |

## 5. Honest limitations (state these; they are already documented)

- The reproduction's negative result is a statement about **this testbed at this
  scale** (32×32 Gray-Scott, spectrum peaking at k=5) — a field with no narrow
  band to exploit. Neither it nor the harmonic verdict says Fourier structure is
  useless in general.
- ext36's disagreement with the reproduction at 32×32 is attributed to budget,
  and only the `repo` modes policy ran — scale and truncation are confounded;
  the implemented `proportional` arm was not executed.
- ext15/ext30 nulls are at pilot scale; larger training sets are not excluded.
  The fundamental was tested at 4.0 (pre-registered) and 3.5 (one point), not
  swept.
- ext31 is single-seed per region (n=6, p=0.54 on the differential test) —
  suggestive, not established.
- ext32's household-facing (IA) arm is underpowered; "sits at α=1" and "has no
  detectable signal" are the same observation there.
- Fairness results are on a synthetic decision problem with no protected
  attributes; see the [fairness checklist](fairness_checklist.md) for what would
  have to change before any of it is an equity claim.
- ext14's bubbles fold was still climbing at the epoch budget: 1.87× is a lower
  bound. ext28's maze sensitivity curve rests on 2 positives.
- No timing instrumentation exists in the training logs; wall-clock training
  cost is unmeasured (ext25 measures inference only).

## 6. Infrastructure facts for the paper

- **Data**: The Well Gray-Scott (6 named regimes), preprocessed to 32×32 via 4×
  downsample; archive with checkpoints on Zenodo, CC BY 4.0, DOI
  [10.5281/zenodo.20718092](https://doi.org/10.5281/zenodo.20718092). NOAA CPC
  soil moisture for ext31; FEMA PA + CDC/ATSDR SVI for ext32.
- **Tests**: 658 passing (`python -m pytest`), run in CI on every push.
  Load-bearing pins include: control and conditioned ext15 arms bit-identical at
  init; banded/uniform arms parameter-matched; ext27's corruptions exactly
  identity at zero severity.
- **Metrics**: VRMSE (variance-normalised RMSE) throughout; definitions in
  [metrics.md](metrics.md).
- **Submission metadata**: `metadata.yaml` and `bibliography.bib` at the repo
  root.
- Headline seed-robust numbers: `results/seeds/` and `results/mechinterp/`;
  per-extension numbers: `results/extensions/`, `results/pilot/`,
  `results/pilot_cpc/`, `results/baseline/`.
