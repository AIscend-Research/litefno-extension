# Is the allocation robust to manipulation? (ext23, H4)

`scripts/strategic_allocation.py`, `src/litefno/mechanism.py`

[ext22](fair_allocation.md) asked what a fair allocation loses when the
surrogate feeding it is wrong by accident. H4 asks what it loses when the input
is wrong *on purpose* — when the regions being allocated to also influence what
the allocator sees about them.

**The answer is that the two questions have the same answer, and that is not a
coincidence.**

## The result that organises everything

The α-fair rule gives region `r` a share `a_r ∝ ĝ_r^((1-α)/α)`, where `ĝ_r` is
whatever the allocator believes about region `r`. Nothing in that expression
knows whether the belief is wrong because a surrogate erred or because a region
lied. So the elasticity of the allocation to a *reported* gain and to an
*erroneous* gain are one derivative, `|1-α|/α`:

```
   alpha    exponent    manipulation elasticity    error amplification
    0.25       3.000                      2.866                  3.000
    0.5        1.000                      0.933                  1.000
    1.0        0.000                      0.000                  0.000
    2.0       -0.500                      0.468                  0.500
    4.0       -0.750                      0.700                  0.750
    8.0       -0.875                      0.823                  0.875
```

(The measured elasticity sits just under `|1-α|/α` because the rule renormalises
over regions: a region already holding a large share has less left to take. The
exact closed form, `κ^|β| / (1 + (κ^|β| - 1) s_r)`, reproduces every measured
ratio to the digits printed below.)

**Manipulation-robustness and error-robustness cannot be bought separately.**
Both are bought by the same thing — ignoring the state — and at α = 1 the rule
ignores it completely. That single point is simultaneously the envy-free rule,
the error-free rule of ext22, and the only strategy-proof member of the family.
It is also the rule that makes no use of the neural operator this repository is
about.

This is a local instance of a general impossibility (Hurwicz 1972): efficiency,
envy-freeness and strategy-proofness do not coexist. Nothing here evades it, and
the useful question is not how to be strategy-proof but how to be *boundedly*
manipulable while still using the data.

## 1. Who can lie, and by how much

A region distorts its own reported gain by up to a factor 1.2, everyone else is
truthful, and the incentive ratio is what it multiplies its own allocation by.

```
   alpha  exponent  incentive  closed form  group loss  verdict
     0.0       inf        inf            -           -  manipulable (unbounded on 100% of states)
    0.25     3.000     1.6860       1.6860    4.00e-03  manipulable
     0.5     1.000     1.1879       1.1879    5.98e-04  manipulable
     1.0     0.000     1.0000       1.0000    0.00e+00  strategy-proof
     2.0    -0.500     1.0894       1.0894    4.56e-04  manipulable
     4.0    -0.750     1.1372       1.1372    1.84e-03  manipulable
     8.0    -0.875     1.1619       1.1619    4.09e-03  manipulable
```

**Max-efficiency is unboundedly manipulable, on every single state.** At α = 0
the rule hands the entire budget to the largest region, so a 20% overstatement
that flips the argmax takes a region from nothing to everything. The ratio is
not large, it is infinite, and it is infinite on 100% of the states tested. The
efficiency-maximising rule is the one with the most to steal.

The rest of the family is manipulable by 9–69%, and the group pays for it: one
region's lie costs everyone else between 4.6e-4 and 4.1e-3 of welfare. The
damage is U-shaped in α for the same reason the fragility was, with its minimum
at the strategy-proof point.

## 2. Leximin, and the capacity cap as a mechanism

`leximin_allocation` is the requested lightweight implementation: lexicographic
max-min over region outcomes by progressive filling, subject to a budget and
per-region capacities, exact rather than iterative.

The capacities are the interesting part rather than a detail. **Without them
leximin is not a new object** — equalising outcomes is always feasible, so it
collapses onto ext22's α = ∞ rule and the lexicographic refinement never gets
past its first level. With them, a region can never receive more than its cap,
so however it lies its winnings are bounded by `c_r / a_r` — no payments, no
verification, no reports.

```
           cap  welfare given up   min/max   worst lie, kappa=1.2   worst lie, kappa=10
   1.02x equal            0.2359    0.6701                 1.1272                1.1300
   1.05x equal            0.2134    0.7258                 1.1733                1.2254
    1.1x equal            0.1760    0.7883                 1.1816                1.3332
   1.15x equal            0.1385    0.8389                 1.1845                1.4204
    1.2x equal            0.1018    0.8838                 1.1857                1.4987
    1.3x equal            0.0421    0.9528                 1.1873                1.6427
      uncapped            0.0000    1.0000                 1.1883                6.9256
```

**The cap is insurance against the tail, not a marginal improvement.** Against a
20% misreport it does almost nothing (1.127 against 1.188 — the uncapped rule is
already boundedly manipulable at that scale). Against a 10× misreport the
uncapped rule lets a liar take 6.93× its honest share while a 1.3× cap holds it
to 1.64× for 4.2% of the worst-off region's welfare. That is the whole case for
the mechanism, and it is only visible because the misreport bound was swept
rather than fixed — at either bound alone the table tells a misleading story.

The dial is monotone in both directions, which is what makes it a dial: tighter
caps buy robustness and give up the equalisation that leximin exists to
produce. At a 1.02× cap the allocation is nearly uniform, the min/max outcome
ratio falls to 0.67, and 23.6% of the leximin objective is gone.

## 3. Is the learned allocator easier to fool?

ext22 concluded that the auxiliary network beats the closed form when the
surrogate is weak. That conclusion would be worth little if the network were
easier to manipulate, so both face one threat model: a region may perturb the
field **inside its own block only**, by at most 0.05 per pixel. The closed form
sees only the block mean, so its best response is exactly ±ε and needs no
search; the network is attacked with 40 steps of projected gradient ascent on
its own output share.

```
   alpha        closed form        leximin capped       learned network
            mean     worst      mean     worst      mean     worst   honest loss
     0.5   1.0311   1.0317    1.0188   1.0241    1.0277   1.0286      1.87e-05
     2.0   1.0160   1.0163    1.0188   1.0241    1.0142   1.0148      2.24e-05
     8.0   1.0282   1.0287    1.0188   1.0241    1.0248   1.0259      3.29e-04
```

**The network is not more manipulable than the rule it replaces — it is slightly
less**, by about 10% of the incentive ratio at every α tested, while remaining
accurate when honest. The adversarial-fragility worry does not materialise here.

The honest-loss column is in the table for a reason: an allocator that ignores
its input cannot be manipulated *and* is worthless, which is the α = 1 lesson,
so robustness is only meaningful next to responsiveness. The network is
robust while still being accurate to 1.9e-5.

This is a limited claim and should be read as one. Single deviating region, an
L∞ ball, one ε, one architecture, no adversarial training. It rules out the
worry at this scale; it does not establish that learned allocators are safe.

## 4. What a no-regret guarantee is worth here

An exponentiated-gradient learner over the simplex needs no surrogate, no model
and no forecast, and provably has vanishing regret against the best fixed
allocation in hindsight. Both halves of that sentence were measured.

```
   alpha      forecast    persistence    online_no_regret    best_fixed      uniform
     0.5     7.743e-07      1.200e-04           3.689e-03     3.986e-03    4.034e-03
     2.0     8.071e-07      1.206e-04           1.022e-03     4.006e-03    4.018e-03
     8.0     9.981e-06      1.507e-03           2.943e-03     4.291e-02    4.347e-02
```

The guarantee holds and is nearly worthless. **A one-step forecast beats the
no-regret learner by three to four orders of magnitude** (7.7e-7 against 3.7e-3
at α = 0.5), and even naive persistence beats it by 30×.

The reason is the comparator. "No regret" here means *no regret against a
constant allocation*, and the learner beats that comparator outright on 16 of 24
runs — its average regret goes negative, which is only possible because the
benchmark is constrained to be constant while the ecosystem oscillates. Being
certifiably as good as the best constant allocation is a weak thing to be
certified as when no constant allocation is any good: `best_fixed` scores
4.0e-3, barely distinguishable from `uniform` at 4.0e-3.

A regret bound against a fixed comparator is not evidence that a model-free
allocator is competitive with a forecast, and in this system it is three orders
of magnitude away from being so.

## What this rules out, and what it does not

Ruled out: that the α-fair family can be made strategy-proof by choosing α.
Only α = 1 is, it is strategy-proof because it ignores its input, and every
informative member of the family is manipulable by an amount fixed by the same
exponent that governs its error sensitivity.

Ruled out at this scale: that a learned allocator is a soft target relative to
the closed form it replaces.

Not ruled out: that a mechanism with payments, verification, or repeated-game
enforcement could do better. None is attempted here. Money is the standard route
to truthfulness and it is not available for a divisible public budget with no
transfers, which is the setting this whole extension is in.

## Honest limits

- **Single deviations only** -- *closed by ext35*. Every incentive ratio here
  assumes one region lies while the rest are truthful, and for egalitarian rules
  a coalition is the natural threat: several regions understating together move
  the common outcome level in a way no single deviation can.
  [coalition_manipulation.md](coalition_manipulation.md) ran the law against
  joint deviations and it survives exactly -- the single-region formula with the
  region's share replaced by the coalition's pooled share, to 4.4e-16 over 96
  cells -- so alpha = 1 is group strategy-proof, not merely strategy-proof, and
  collusion makes manipulation *worse* for the colluders.
- **The cap is exogenous.** It has to be. A cap set from reported gains would be
  circular — under an egalitarian rule a region that understates raises both its
  allocation *and* its cap, and the bound evaporates. Tying caps to a trusted
  history would fix this and is not implemented.
- **A uniform cap binds unevenly.** It bounds each region at `c_r / a_r`, which
  is loosest exactly for the region with the smallest honest share — the one
  with the most to gain. The worst-case ratio is therefore set by the region the
  cap constrains least, which is why the κ = 1.2 column barely moves.
- **The threat model is the allocator's input**, not the surrogate's weights or
  its training data. A region that can poison the surrogate has a longer lever
  than anything measured here.
- **The decision layer is stylized**, exactly as in ext22: nothing in the PDE
  says what a region deserves. What transfers beyond this setup is the
  elasticity result, which is a property of the rule and not of the ecosystem.

## Reproducing

```bash
python3 scripts/strategic_allocation.py           # ~2 minutes on CPU
python3 scripts/strategic_allocation.py --quick   # plumbing check only
```

Outputs: `results/extensions/ext23_manipulation.csv`, `ext23_leximin.csv`,
`ext23_attack.csv`, `ext23_online.csv`, and
`figures/extensions/ext23_strategic.png`.
