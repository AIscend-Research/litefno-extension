# Do the resonant factors carry across regimes? (ext21, H2)

`scripts/mode_transplant.py`

H2: the learned low-rank spectral factors corresponding to stable and resonant
modes are shared physics, so freezing them into a model for a different regime
recovers much of full training at a fraction of the data, while transplanting
the damped modes recovers nothing. The asymmetry between those two is what would
make the mode classification physically meaningful.

**The answer is no, and the reason is more interesting than the result.**

## What was run

A source model is trained on one regime of the exactly solvable oscillatory
system (one-step VRMSE 0.0183). Its rank components are split into resonant and
damped by the ext19 pole readout, matched to equal size (4 and 4 of rank 8,
split margin 0.374). Four arms then train on each of two held-out regimes at
four data budgets, three seeds each:

| arm | what carries over |
|---|---|
| `scratch` | nothing |
| `finetune` | the entire source model, all weights trainable |
| `transplant_resonant` | 4 resonant components, frozen |
| `transplant_damped` | 4 damped components, frozen (the control) |

Freezing is per component and enforced by masking the gradient, not by an
initialization that training is then free to undo -- otherwise the arm measures
a warm start, which is a much weaker claim.

## Results

```
target: slow_diffuse
 budget    scratch   finetune   resonant     damped   asymmetry     /sd
      2    0.26286    0.04410    0.25988    0.26150    +0.00162   +0.02
      4    0.13726    0.03042    0.13114    0.13413    +0.00299   +0.40
      8    0.03559    0.02861    0.03744    0.03708    -0.00036   -0.09
     16    0.02943    0.02804    0.02954    0.02959    +0.00005   +0.05

target: fast_sharp
      2    0.24190    0.02259    0.24143    0.24166    +0.00024   +0.00
      4    0.04574    0.00993    0.04425    0.04497    +0.00071   +0.08
      8    0.02147    0.00705    0.01900    0.02018    +0.00118   +0.46
     16    0.01018    0.00558    0.01065    0.01063    -0.00002   -0.03
```

0 of 8 cells put the resonant transplant more than one standard deviation ahead
of the size-matched damped control. The asymmetries are in the third decimal
place, two of the eight are negative, and neither transplant is meaningfully
better than starting from scratch.

Full fine-tuning, meanwhile, is dramatically better than everything: 0.044
against 0.263 from scratch at two trajectories. Transfer across these regimes is
real and large. It just does not decompose into the mode-classified pieces H2
proposed.

## Why: the spectral basis is not identifiable

The overlap matrix is the control that turns a null result into an explanation.
It measures the principal angles between the mode-axis bases two models learned,
and it includes **same-regime pairs trained from different seeds** as the
reference:

| pair | overlap | median angle |
|---|---|---|
| source s0 vs source **s1** (same regime) | 0.258 | 64.0 deg |
| source vs slow_diffuse | 0.283 | 65.2 deg |
| source vs fast_sharp | 0.230 | 72.1 deg |
| slow_diffuse vs fast_sharp | 0.226 | 69.4 deg |

Two models trained on the *same* regime, differing only in seed, share no more
spectral structure than models trained on different regimes -- 0.258 against
0.230 to 0.283, all at median principal angles around 65 degrees. The learned CP
mode basis is dominated by initialization, not by physics.

That single row disposes of the transplant question. There is no regime-specific
signal to fail to transfer, because there is no seed-stable signal to begin
with. It also explains why fine-tuning works while partial transplants do not:
what transfers is the *whole* operator acting coherently, not any identifiable
subspace of it, and moving four of eight components into a model whose other
four are random noise contributes nothing to either.

Reading the cross-regime overlaps without the same-seed reference would have
suggested the opposite: 0.23 to 0.28 looks like meaningful shared structure
until the same-regime pair turns out to sit at the same value.

## What this rules out, and what it does not

Ruled out: that CP rank components of a trained spectral operator are
individually interpretable units of physics that can be selected by a pole
classifier and moved between regimes. On this architecture and these systems,
they are not.

Not ruled out: that neural operators learn transferable physics. They evidently
do -- the fine-tune arm is a factor of six better than scratch at the smallest
budget. The claim that fails is specifically about *decomposability*, and the
mechanism is the well-known non-identifiability of a CP factorization, which is
unique only up to permutation and scaling of its components and, in a trained
network, not even approximately pinned down.

A follow-up with a chance of working would have to make the basis identifiable
first -- an orthogonality or alignment penalty during training, or matching
components across models before comparing them rather than assuming a shared
index order.

## An honest limit of the method

CP factorization stores no per-mode weights, so "transplant the resonant modes"
is not directly expressible: the rank components each spread over the entire
mode grid. The selection assigns each component to whichever half of the
spectrum holds more of its footprint, which is weaker than identifying a
component *as* the resonant physics.

The first implementation thresholded each set independently and returned all
eight components for both sets, silently collapsing the two transplant arms into
the same arm. The script now reports the component counts and the split margin
so a degenerate selection is visible in the output rather than hidden behind
suspiciously equal numbers.

## Reproducing

```bash
python3 scripts/mode_transplant.py           # ~25 minutes on CPU
python3 scripts/mode_transplant.py --quick   # plumbing check only
```

Outputs: `results/extensions/ext21_transplant.csv`, `ext21_overlap.csv`,
`ext21_summary.csv`, and `figures/extensions/ext21_mode_transplant.png`.

## Status update: this null does not survive a paired re-test (ext34)

ext34 re-ran this comparison with a dose axis and a paired design and found the
resonant arm ahead of its size-matched damped control in **90 of 90 paired
runs**, with the gap growing monotonically in the number of components
transplanted. The conclusion above should be read with that correction.

The disagreement is a design difference, not a data one. Read as a relative gap
rather than as a multiple of seed spread, **6 of the 8 cells in the table above
are already positive**, and the two negatives are at the two largest budgets
where scratch has already converged to fine-tune. The effect is about 2% and the
seed spread here is 14-49%, so the one-standard-deviation-per-cell test used
above could not have resolved it. Pairing resonant against damped within a seed
cancels that noise; ext34 does, this does not.

What survives unchanged is the practical claim. Freezing *every* component buys
+8.4% over scratch against fine-tuning's +80.4%, so transfer across these regimes
is still real, large, and still does not usefully decompose into the
mode-classified pieces. What fails is the stronger mechanistic reading -- that
the basis is arbitrary and carries nothing -- since an arbitrary basis does not
win 90 of 90 with a dose-response. See [transplant_distance.md](transplant_distance.md).
