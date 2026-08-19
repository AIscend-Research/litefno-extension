# Results for the paper

All numeric results, as paper-ready tables. Every table names the CSV it is
built from (paths relative to `results/`); the linked extension doc holds the
protocol and interpretation. Values are rounded for print — the CSVs are
authoritative. Metric is VRMSE (variance-normalised RMSE) unless stated.

---

## 1. Headline reproduction — spectral LiteFNO vs parameter-matched CNN

Gray-Scott 32×32, 3 seeds, matched protocol. Source: `seeds/seed_table.csv`.

| model | seed | one-step | rollout 6–12 | rollout 13–30 | HF energy ratio (final) | train (s) |
| --- | --- | --- | --- | --- | --- | --- |
| CNN | 0 | 0.01055 | 0.2618 | 0.4579 | 0.483 | 1803 |
| CNN | 1 | 0.01033 | 0.2152 | 0.3122 | 0.909 | 1802 |
| CNN | 2 | 0.01052 | 0.2802 | 0.4238 | 0.780 | 1802 |
| LiteFNO (CP) | 0 | 0.01376 | 0.2041 | 0.3700 | 1.754 | 5028 |
| LiteFNO (CP) | 1 | 0.01286 | 0.3179 | 0.5302 | 0.642 | 5027 |
| LiteFNO (CP) | 2 | 0.01294 | 0.2949 | 0.5043 | 0.789 | 5023 |
| **CNN mean ± sd** | | **0.01047 ± 0.00012** | **0.252 ± 0.033** | **0.398 ± 0.076** | 0.724 ± 0.218 | 1802 |
| **LiteFNO mean ± sd** | | **0.01319 ± 0.00049** | **0.272 ± 0.060** | **0.468 ± 0.086** | 1.062 ± 0.601 | 5026 |

The CNN is better on one-step in 3/3 seeds and on late rollout in 3/3 seeds,
at 2.8× less training time. HF energy ratio = high-frequency energy / ground
truth at rollout end (1.0 is faithful; the spectral arm overshoots).

In-distribution reference under the ext-phase protocol (200 epochs, seed 0):
CP LiteFNO at 179,666 params reaches one-step **0.00953** test VRMSE
(`baseline/ext13_baseline_seeds.csv`).

## 2. Resolution scaling (ext36, H15)

One native-resolution stream; 24 traj / 30 epochs / 2 seeds ("thin" budget —
not comparable to §1). `repo` modes policy (16 modes at every size). Source:
`extensions/ext36_resolution_summary.csv`.

| grid | CNN | CNN seed sd | CP LiteFNO | FNO-S (dense) | spectral gap vs CNN |
| --- | --- | --- | --- | --- | --- |
| 32×32 | 0.0434 | 0.00268 (6.2%) | **0.0379** | 0.0595 | +12.8% |
| 64×64 | 0.0426 | 0.00415 (9.8%) | **0.0394** | 0.0633 | +7.4% |
| 128×128 | 0.0505 | 0.00002 (0.05%) | **0.0412** | 0.0664 | +18.3% |

Per-seed gaps at 64×64 include one sign reversal (−4.1%); at 128×128 both
seeds agree (+19.1%, +17.5%). The sequence is non-monotone; only the 128×128
cell is measured precisely enough to carry a claim. Dense FNO-S (2.1M params)
is worst at every size.

## 3. The harmonic claim

### 3.1 Where the variance lives (ext9)

Cumulative spatial variance by wavenumber shell, ground truth. Source:
`extensions/ext9_variance_decomposition.csv`.

| k ≤ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | … 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cum. share | 3.5% | 10.9% | 19.3% | 29.2% | 41.8% | 50.6% | 59.8% | 67.6% | 72.8% | 95.8% |

Peak shell is k = 5; 58% of variance sits at k ≥ 6.

### 3.2 Per-regime spectral profile (ext10) and the ext15 A/B

Share of spatial variance below mode 8, and the harmonic-conditioning A/B
(3 seeds, 100 epochs, fundamental 4.0, shells [4, 8, 12]; arms bit-identical
at init; +220 params = +2.94%). Sources:
`extensions/ext15_harmonic_ab_per_regime.csv`, `_summary.csv`, `_seeds.csv`.

| regime | var below mode 8 | control | conditioned | Δ (%) |
| --- | --- | --- | --- | --- |
| spots | 0.6% | 0.01806 | 0.01805 | −0.05 |
| maze | 1.3% | 0.03031 | 0.03030 | −0.02 |
| worms | 30.9% | 0.03536 | 0.03536 | −0.01 |
| bubbles | 58.3% | 0.04184 | 0.04200 | +0.37 |
| gliders | 69.0% | 0.02315 | 0.02314 | −0.02 |
| spirals | 77.4% | 0.04004 | 0.04002 | −0.05 |
| **all** | | **0.03215 ± 0.00546** | **0.03218 ± 0.00545** | **+0.09** |

Conditioned is worse in 3/3 seeds. Differential test: Spearman ρ(var-below-8,
Δ) = **−0.200** where the mechanism requires strongly negative — the largest
improvement goes to spirals, the most low-wavenumber regime. Sweep at
fundamental 3.5: also null (ρ = +0.086, conditioned wins 2/3 seeds; see
[harmonic_conditioning.md](harmonic_conditioning.md)). The mean gap is ~345×
smaller than the control's own seed spread.

### 3.3 Does the prior substitute for data? (ext30, H11)

Same two arms, four training-set sizes, 3 seeds, paired. Sources:
`pilot/ext30_summary.csv`, `pilot/ext30_multiplier.csv`.

| n traj | plain | harmonic | rel. change | seeds helped | data multiplier |
| --- | --- | --- | --- | --- | --- |
| 6 | 0.09221 ± 0.00186 | 0.09230 ± 0.00188 | +0.09% | 0/3 | — (out of range) |
| 12 | 0.06874 ± 0.00428 | 0.06879 ± 0.00428 | +0.08% | 0/3 | 0.998 |
| 18 | 0.05200 ± 0.00240 | 0.05203 ± 0.00241 | +0.07% | 0/3 | 0.999 |
| 24 | 0.04529 ± 0.00189 | 0.04531 ± 0.00189 | +0.06% | 0/3 | 0.999 |

0 of 12 paired wins; multiplier 1.00× throughout. Plain-arm scaling:
VRMSE ≈ 0.239·n^−0.521 (residuals < 4.7%) — 4× data cuts error 2.04×.

### 3.4 Temporal forcing (ext12) and real seasonal data (ext31)

- planetswe (documented daily/annual forcing): phase-locked **0.995–0.998**
  across 4 trajectories, enriched **>700×** over chance, but **5.4–5.5%** of
  temporal variance globally (11.5% best latitude band).
  (`extensions/ext12_planetswe_*.csv`)
- NOAA CPC soil moisture (943 months; monsoon belt **65%** annual variance):
  data multiplier **0.96×** at full size (1.003 at the smaller size), 3/6
  paired runs favour the prior, and the three most seasonal regions get
  *worse* (mean +6.1%); ρ = +0.314 where the mechanism needs negative (n = 6,
  p = 0.54, single seed). (`pilot_cpc/ext30_multiplier.csv`;
  [seasonal_real_data.md](seasonal_real_data.md))

## 4. SpecScope: reading the trained operator

### 4.1 Pole extraction against closed form (ext19)

Source: `extensions/ext19_summary.csv`.

| system | test VRMSE | magnitude ρ | freq MAE | label acc. (tol 0.005) | route rel. diff (median) |
| --- | --- | --- | --- | --- | --- |
| rotating | 0.0176 | **0.987** | 1.5e-4 | 0.952 | 0.256 |
| advection | 0.1351 | 0.825 | 2.0e-3 | 0.810 | 0.475 |

### 4.2 The instrument ablation (ext33, H14)

Identical readout, four weight conditions. Source:
`extensions/ext33_ablation_summary.csv`.

| system | weights | magnitude ρ | median abs freq error |
| --- | --- | --- | --- |
| rotating | **trained** | **+0.9876** | **1.9e-4** |
| rotating | untrained | +0.117 | 2.4e-2 |
| rotating | shuffled | −0.061 | 4.8e-1 |
| rotating | resampled | +0.076 | 2.3e-2 |
| advection | **trained** | **+0.761** | **1.3e-3** |
| advection | untrained | −0.218 | 4.7e-1 |
| advection | shuffled | +0.023 | 2.2e-2 |
| advection | resampled | −0.241 | 2.7e-2 |

No control exceeds +0.12 (rotating) / +0.24 (advection, wrong sign). The exact
pole is a function of |k| alone (ρ vs radius = −1.0000 rotating, −0.9996
advection), which is why the control must be an ablation, not a partial
correlation.

### 4.3 Does the readout predict failure? (ext20, H1)

3 seeds, 20 held-out scenarios. Source: `extensions/ext20_summary.csv`.

| seed | per-mode ρ raw | wavenumber-only baseline | partial (mode-level) | per-scenario ρ | AUC |
| --- | --- | --- | --- | --- | --- |
| 0 | −0.811 | +0.805 | −0.165 | 0.812 | **0.98** |
| 1 | −0.794 | +0.789 | −0.145 | 0.787 | **0.99** |
| 2 | −0.749 | +0.744 | −0.123 | 0.723 | **0.98** |

Strong per-mode form fails its control; weak per-scenario risk score holds
(AUC 0.98–0.99, p ≈ 5e-5).

### 4.4 Transplantation: ext21's null and ext34's correction (H2)

ext21 (unpaired, 3-component dose = 3.55% of parameters): resonant ahead of
damped in 0/8 (target, budget) cells at ≥1 sd; same-regime spectral overlap
0.258 vs cross-regime 0.230–0.283 (basis set by initialization). Fine-tune
beats scratch by ~6× at the smallest budget. (`extensions/ext21_summary.csv`)

ext34 (paired within seed, dose as an axis; 3 seeds × 10 configurations per
dose): source `extensions/ext34_distance_summary.csv`.

| dose (components) | % of model written | resonant−damped gap | paired wins |
| --- | --- | --- | --- |
| 1 | 1.2% | +1.7% | 30/30 |
| 2 | 2.4% | +1.8% | 30/30 |
| 3 | 3.5% | +2.5% | 30/30 |

**90/90 paired wins** (every seed 10/10, smallest gap +0.53%);
Spearman(dose, gap) = +0.764. The distance axis is flat: Spearman(distance,
gap) = −0.319, gaps +1.8% to +2.8% over a 3.5× parameter-ratio range.
Unchanged: transplant-all buys +8.4–10.2% over scratch vs fine-tune's
+80.1–81.3%, so transfer still does not usefully decompose.

## 5. The decision layer

### 5.1 Fragility of fair allocation (ext22, H3)

Sensitivity of realised welfare to surrogate error is U-shaped in α with an
exact zero at α = 1 (envy-free). Closed-form law vs trained operator:
law_ratio 0.996–1.000 across α (i.e. within ~1%). Fragility coefficient by α:
∞ (α=0), 1.125 (α=0.25), 0.25 (α=0.5), **0** (α=1), 0.25 (α=2), 1.125 (α=4),
3.06 (α=8). Price of fairness at α=1: 15.7%. Learned allocator vs pooled
closed form: 4–6× worse with a strong surrogate, 9.3× better when starved.
Acting on an 8-step-stale observation is worse than ignoring the state.
(`extensions/ext22_summary.csv`)

### 5.2 Manipulation (ext23, H4)

Incentive to misreport = |1−α|/α = the error-sensitivity derivative. α = 1 is
the unique strategy-proof member. Max-efficiency (α→0): unbounded capture on
100% of states. Leximin with per-region capacity κ: a 10× misreport is held to
**1.64×** (from 6.93×) at 4.2% worst-off welfare cost; cap-multiplier sweep in
`extensions/ext23_leximin.csv` (e.g. cap 1.02: ratio 1.13, welfare −23.6%;
cap 1.15: ratio 1.42, welfare −13.9%).

### 5.3 Coalitions (ext35, H16)

Capture ratio under joint deviation equals the single-region closed form with
the coalition's pooled share — matched to **4.4e-16 over 96 cells**. Source:
`extensions/ext35_coalition_law.csv`; table as in
[coalition_manipulation.md](coalition_manipulation.md):

| α | \|C\|=1 | \|C\|=2 | \|C\|=4 | \|C\|=8 | \|C\|=16 |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 3.109 | 2.872 | 2.444 | 1.806 | 1.000 |
| 0.5 | 1.462 | 1.425 | 1.353 | 1.222 | 1.000 |
| **1.0** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| 2 | 1.209 | 1.194 | 1.164 | 1.106 | 1.000 |
| 8 | 1.394 | 1.363 | 1.303 | 1.192 | 1.000 |

Subadditive in 90/90 cells (collusion dilutes itself); grand coalition
captures nothing; corner lie verified against 45,216 alternative joint
reports (0 beat it). Leximin cap at ext23's κ = 1.5 never binds (max capture
1.46 uncapped); at κ = 16 it holds 9.2× to 1.89× for small coalitions only.
(`extensions/ext35_leximin_cap.csv`, `ext35_best_response.csv`)

### 5.4 Real allocations: FEMA PA vs SVI (ext32, H13)

Within-disaster log-log elasticity of obligations vs vulnerability, 12
hurricanes FY2015+ (Public Assistance). Source:
`extensions/ext32_disaster_fits_pa.csv` (per-state fits shown for the
best-powered arm; IA is underpowered and not reported).

| disaster (state, FY) | n counties | β (elasticity) | R² | perm. p |
| --- | --- | --- | --- | --- |
| Milton (FL, 25) | 42 | −0.71 | 0.034 | 0.27 |
| Helene (GA, 24) | 80 | +2.11 | 0.143 | 0.00 |
| Helene (SC, 24) | 32 | +0.54 | 0.010 | 0.54 |
| Helene (FL, 24) | 43 | −0.52 | 0.006 | 0.65 |
| Francine (LA, 24) | 22 | +0.96 | 0.045 | 0.37 |
| Beryl (TX, 24) | 37 | −2.66 | 0.059 | 0.14 |
| Idalia (GA, 23) | 30 | +0.43 | 0.004 | 0.76 |
| Idalia (FL, 23) | 45 | +0.31 | 0.003 | 0.74 |
| Nicole (FL, 23) | 31 | −3.32 | 0.343 | 0.005 |
| Ian (SC, 23) | 17 | −2.31 | 0.120 | 0.21 |
| Ian (FL, 22) | 48 | −0.77 | 0.015 | 0.43 |
| Ida (MS, 22) | 20 | +0.06 | 0.000 | 0.98 |

Signs split 6/6; median R² **0.025**; 2/12 beat their permutation null, in
opposite directions. Inverse-variance pooled elasticity **−0.088 ± 0.266**
(z = −0.33) → pooled **α̂ = 1.10**: statistically at the envy-free point.

### 5.5 Scarcity on an unseen network (ext24, H5)

Fourier modes = lattice Laplacian eigenvectors (residual 1.5e-14). Four arms
at 4,913 params: true network beats degree-preserving rewiring by **12–23%**;
advantage over the lattice graph is 0% at zero shortcuts → **21–28%** at ≥20%
non-lattice edges; hard-wiring the redundant lattice buys **7.3%**. Cascade
matches the 1/λ₁ epidemic threshold to **0.3%** on four graph families.
Sentinel placement by centrality: +0.22 steps of warning over random.
(`extensions/ext24_*.csv`)

## 6. Off-distribution and uncertainty

### 6.1 Leave-one-regime-out (ext14)

30 epochs, seed 0, CP arm. Source: `extensions/ext14_cross_regime.csv`.

| held out | held-out VRMSE | seen VRMSE | gap ratio |
| --- | --- | --- | --- |
| spirals | 0.0921 | 0.0444 | **2.07×** |
| bubbles | 0.0878 | 0.0468 | **1.87×** (lower bound; still improving) |
| gliders | 0.0509 | 0.0519 | 0.98× |
| worms | 0.0478 | 0.0494 | 0.97× |
| maze | 0.0445 | 0.0495 | 0.90× |
| spots | 0.0283 | 0.0503 | 0.56× |

Pre-registered spectral prediction (maze/spots hardest): 0/2;
ρ(var-below-8, gap) = **+0.943** where the mechanism requires negative.

### 6.2 Buying the gap down (ext26, H7)

Fraction of gap closed on the two folds that have one. Source:
`pilot/ext26_gap_closed.csv`.

| held out | robust (noise-aug) | robust + 4-ensemble |
| --- | --- | --- |
| bubbles | +8.1% | −13.7% |
| spirals | −3.5% | −20.6% |

Median across the two: robust +2%, ensemble **−17%** (widens both); the
ensemble degrades held-out error in 5/6 folds while improving in-distribution.

### 6.3 Degradation robustness (ext27, H8)

Synthetic capture chain, severity 0–100%. Training on the chain closes a
median **81%** of the error rise (baseline degrades 16× clean→full, robust
2.6×), at a **177% clean-input tax**; curves cross at ~20% severity. Held-out
corruption (dropout): median closure **27%** (e.g. 44% at severity 0.1, 21% at
1.0), robust worse than baseline below ~10%. (`pilot/ext27_gap_closed.csv`)

### 6.4 Safe deferral (ext28, H9)

bubbles fold (gap 1.95×): Safe Deferral Rate **not reached by any signal —
including the oracle** (NaN at every τ; 80% deferral still leaves 1.05×).
Ensemble disagreement ≈ oracle for ranking; triage: specificity 1.00 through
50% deferral at sensitivity 0.94; all unsafe steps caught at 55% deferral with
specificity 0.96. maze fold (gap 0.94×): vacuous (2 positives).
(`pilot/ext28_*.csv`)

### 6.5 Calibration (ext29, H10)

4-member ensemble, Gaussian intervals. Source: `pilot/ext29_summary_bubbles.csv`.

| split | ECE | MCE | ECE after single σ-rescale (fit in-dist) |
| --- | --- | --- | --- |
| seen | 0.066 | 0.137 | 0.036 |
| held-out | **0.131** | 0.217 | 0.094 |

Every coverage gap negative at every level (intervals only too narrow);
nominal 95% covers 73% held-out; achieved/claimed error is U-shaped, worst —
**5.2×** — in the most confident bin. Rescaled OOD ECE (0.094) remains worse
than raw in-distribution ECE (0.066).

## 7. Deployability (ext25, H6)

Envelope: ≤1 GB disk, interactive batch-1 latency, trains in a 12 h CPU
session. Source: `extensions/ext25_envelope.csv` (Apple-silicon host,
`ext25_host.json`).

| arm | params | batch-1 ms | train (h) | deployable |
| --- | --- | --- | --- | --- |
| cnn-w32-r16 | 13,730 | 0.40 | 4.3 | ✓ |
| cnn-w64-r32 | 107,842 | 1.17 | 20.2 | ✗ (session) |
| fno_s-w32-m8 | 528,674 | 1.45 | 5.0 | ✓ |
| fno_s-w32-m16 | 2,101,538 | 4.04 | 5.6 | ✓ |
| fno_s-w64-m16 | 16,810,818 | 22.4 | 26.0 | ✗ (session) |
| cp-w32-r8 | 7,490 | 4.03 | 5.5 | ✓ |
| cp-w32-r8 **fused** | 7,490 | 3.20 | — | ✓ |
| cp-w32-r32 | 16,802 | 5.46 | 5.7 | ✓ |
| cp-w32-r32 **fused** | 16,802 | 2.88 | — | ✓ |
| cp-w64-r8 | 43,906 | 30.9 | 25.4 | ✗ (session) |

Headline: rank correlation(params, batch-1 latency) = **0.067**; the CP arm
has 383× fewer parameters than dense FNO-S and is **37% slower** at batch 1
(weight reconstruction = 74–92% of the budget). Eval-time fusing: **1.4–1.8×**
speedup, bitwise-identical outputs. FLOPs rank models (ρ 0.57–0.88) but
over-predict the fusing speedup 12.35× vs 1.79× measured. Every width-64
configuration misses the 12 h session by ~2×.

## 8. Reporting footnotes

- Seeds: 3 unless stated (ext36: 2; ext14/ext26–29: 1 training seed with
  4-member ensembles where noted; ext31: 1 per region).
- Test suite: 658 tests passing; CI on every push. Pinned invariants include
  bit-identical ext15 arms at init and exact parameter matching in A/Bs.
- Data/checkpoints: Zenodo DOI
  [10.5281/zenodo.20718092](https://doi.org/10.5281/zenodo.20718092) (CC BY 4.0).
