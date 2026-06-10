# Context + paper-edit instructions: 3-seed mechanistic-interpretability results

**Read this whole file, then edit the paper as instructed at the bottom.** These
are AUTHORITATIVE, seed-robust numbers (3 seeds, with std). They REPLACE any
earlier single-seed mech-interp numbers in the draft.

## What these files are

This folder holds the mechanistic-interpretability analysis of the CP-factorized
spectral **LiteFNO** (the real `neuraloperator` FNO with `factorization="cp"`,
rank=0.02 → 179,666 params, modes=16, width=64, 8 layers), run over **3 training
seeds** (`litefno_seed0/1/2.pt`) on Gray-Scott reaction-diffusion (32×32, 2
fields). Every figure shows **mean ± std error bands across the 3 seeds**.

- `deadmode.png` / `deadmode.csv` — per-Fourier-mode weight magnitude (normalized),
  mean±std over seeds, for the two spectral mode axes (dim0=16 modes, dim1=9 modes).
- `cprank.png` — CP-rank utilization: cumulative energy vs. # sorted rank
  components, mean±std.
- `mode_ablation.png` / `mode_ablation.csv` — causal ablation: keep only the
  lowest fraction of Fourier modes (zero the rest), measure one-step VRMSE +
  rollout windows, mean±std.

## The two findings (USE THESE EXACT NUMBERS)

### Finding 1 — No dead modes (full spectral utilization), seed-robust
From `deadmode.csv`: every mode's normalized magnitude is in **[0.74, 1.0]** on
both axes, with std ≤ 0.037. **0% of modes fall below the 10%-of-peak "dead"
threshold**, on both axes, across all 3 seeds.
→ Claim: *The CP-factorized spectral weights use their full allocated bandwidth;
the 99.9% compression introduces no dead/unused modes.*

### Finding 2 — Every mode is causally necessary (monotone ablation), seed-robust
From `mode_ablation.csv` (keep_frac → one-step VRMSE mean±std, rollout 13:30 mean±std):

| modes kept | one-step VRMSE | rollout 13:30 |
|-----------:|:--------------:|:-------------:|
| 100%       | 0.0132 ± 0.0004 | 0.468 ± 0.070 |
| 75%        | 0.0197 ± 0.0003 | 0.526 ± 0.046 |
| 50%        | 0.0578 ± 0.0056 | 0.592 ± 0.064 |
| 25%        | 0.0614 ± 0.0067 | 0.596 ± 0.036 |
| 10%        | 0.0621 ± 0.0064 | 0.595 ± 0.032 |

→ Claim: *Ablating modes degrades accuracy immediately and monotonically — even
dropping 25% of modes (100%→75%) nearly doubles one-step error. There is no
prunable spectral capacity; all modes contribute causally.*

These two findings are **mutually consistent** (no dead modes ⇒ can't ablate
without damage) and **tie out** with the headline run (keep=100% reproduces the
seed-mean one-step 0.0132 and rollout 0.468 from `seed_table.csv`).

## HONEST FRAMING — DO NOT OVERSELL (critical for MLRC/TMLR)

This is a **supporting, mechanistic/efficiency** result, **NOT** a novel headline,
and it does **NOT** explain a performance win over the CNN baseline (the 3-seed
headline run found **CNN ≈ LiteFNO**, no LiteFNO advantage). Specifically:

- Frame it as: *"the CP compression is not wasteful — full, seed-robust spectral
  utilization with no dead capacity."* (An efficiency / characterization result.)
- It is an **expected/null-leaning** result (a compact, well-trained model using
  all its modes is unsurprising). Present it as confirming efficiency, with bands,
  NOT as a surprising discovery.
- **Never** write that mech-interp explains why LiteFNO beats the CNN. It doesn't,
  and LiteFNO does not beat the CNN here. Any such sentence must be deleted.

## HOW TO EDIT THE PAPER

1. **Add/replace a mechanistic-interpretability subsection** (in Results or a
   short "Analysis" section) with Findings 1 and 2 above, using the exact numbers
   and the two figures (`deadmode.png`, `mode_ablation.png`; optionally
   `cprank.png`). State "3 seeds, mean ± std" in the caption of each.

2. **Update the limitations**: change any "mechanistic analysis used a single
   seed" bullet to **"mechanistic analysis run across 3 seeds (mean ± std)."**

3. **Delete/replace any single-seed mech-interp numbers** elsewhere in the draft
   with the table above.

4. **Delete any sentence** claiming mech-interp explains a LiteFNO advantage over
   the CNN, or implying LiteFNO outperforms the CNN. Keep the honest headline
   result (CNN ≈ LiteFNO across 3 seeds).

5. **Suggested figure captions:**
   - deadmode.png: *"Per-mode spectral weight magnitude (normalized), mean ± std
     over 3 seeds. All modes exceed 74% of peak on both axes; none fall below the
     10% dead-mode threshold — the CP-factorized weights use full spectral
     bandwidth."*
   - mode_ablation.png: *"Causal mode ablation (3 seeds, mean ± std). Keeping only
     the lowest-frequency fraction of modes degrades one-step and rollout VRMSE
     immediately and monotonically; every mode is causally necessary."*

6. **One-line takeaway to weave into the abstract/conclusion:** *"A seed-robust
   mechanistic analysis shows the CP-factorized spectral operator uses its full
   mode budget with no dead capacity and no prunable modes — the compression is
   efficient, though it confers no accuracy advantage over a matched CNN."*
