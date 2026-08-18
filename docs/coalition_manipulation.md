# Does the manipulation law survive coalitions? (H16 / ext35)

ext23 derived the incentive to misreport, `|1-alpha|/alpha`, and checked it
against **single-region** deviations. That is what strategy-proofness is defined
by, and it is also the weakest threat model available: it assumes the regions
being allocated to never coordinate. Regions that share a border, a budget line
or a political interest plainly do.

Running the derived law against joint deviations, it survives **exactly** -- to
4.4e-16 over 96 cells -- and the way it survives is the interesting part.
Collusion makes manipulation *worse for the colluders*.

## The law does not change; its argument does

With `w_r = g_r^beta` and coalition `C` holding truthful share `s_C`, if every
member reports its profitable corner then every `w_r` in `C` is multiplied by the
same `lambda = kappa^|beta|`, so what the cartel captures is

    rho_C = lambda / (1 + (lambda - 1) s_C)

which is **the single-region formula with `s_r` replaced by `s_C`**. Measured
against 48 states x 16 regions x 7 alphas x every coalition size 1..16:

| alpha | \|C\|=1 | \|C\|=2 | \|C\|=4 | \|C\|=8 | \|C\|=16 |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 3.109 | 2.872 | 2.444 | 1.806 | **1.000** |
| 0.5 | 1.462 | 1.425 | 1.353 | 1.222 | **1.000** |
| **1.0** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| 2 | 1.209 | 1.194 | 1.164 | 1.106 | **1.000** |
| 8 | 1.394 | 1.363 | 1.303 | 1.192 | **1.000** |

Every cell matches the closed form to machine precision.

## Three consequences, two of them sharp

**alpha = 1 is group strategy-proof, not merely strategy-proof.** `beta = 0`
gives `rho_C = 1` for every coalition of every size, at `|ratio - 1| = 0` exactly.
That is the strictly stronger property, and it comes for the same reason ext23
gave for the weaker one: the envy-free rule ignores the state, so there is
nothing to lie about. Nothing was gained by strengthening the threat model
because nothing was being defended.

**A universal cartel achieves exactly nothing.** At `|C| = 16` the ratio is 1.0
at every alpha, again exactly. If everyone inflates by the same factor the
normalisation cancels. The rule is scale-invariant in the reports, so a lie that
everyone tells is not a lie the mechanism can see.

**Collusion dilutes itself.** `rho_C` is decreasing in `s_C` and `s_C` grows with
the coalition, so the capture ratio falls monotonically in coalition size at
every alpha. Measured against the same regions deviating **one at a time**:

| alpha | \|C\|=2 | \|C\|=4 | \|C\|=8 | \|C\|=16 |
| --- | --- | --- | --- | --- |
| 0.25 | 0.88 | 0.68 | 0.38 | **0.00** |
| 0.5 | 0.92 | 0.77 | 0.48 | **0.00** |
| 2 | 0.93 | 0.78 | 0.51 | **0.00** |
| 8 | 0.92 | 0.77 | 0.49 | **0.00** |

joint capture as a fraction of the sum of solo captures. **Subadditive in 90 of
90 cells.** A cartel of eight wins about half what its members would have won
acting alone, and a cartel of everyone wins nothing at all.

This is the result that could have gone the other way, and it is why the
single-deviation law was worth stress-testing rather than merely extending: the
intuitive expectation is that collusion is the harder threat, and here **the lone
deviator is the worst case**. ext23's numbers were not optimistic. They were the
bound.

## But the damage still grows, and the ratio hides it

Reporting only the ratio would understate the harm, because the two move in
opposite directions:

| alpha=8 | \|C\|=1 | \|C\|=2 | \|C\|=4 | \|C\|=8 |
| --- | --- | --- | --- | --- |
| capture ratio | 1.394 | 1.363 | 1.303 | 1.192 |
| budget captured | 0.021 | 0.040 | 0.067 | **0.088** |
| welfare loss | 0.015 | 0.030 | 0.056 | **0.096** |

Members do worse per head while outsiders lose four times as much. "Collusion is
self-defeating" is true from inside the cartel and false from outside it, and
both are in the CSV.

## The assumed lie is verified, not assumed

The formula presumes every member goes to its own corner. That follows from each
`a_r` being monotone in `w_r` over a denominator common to all members, so the
corner is dominant *within* the coalition too -- but a coalition maximising a
total could in principle want a member to sacrifice, and the single-agent
argument does not rule that out by itself.

Searched exhaustively over all `2^|C|` corner profiles plus 200 random interior
profiles per cell: **45,216 alternative joint reports, 0 beat the corner
profile, largest margin 0.0e+00.** The greedy coalition (the smallest-share
members) is also checked against exhaustive enumeration at sizes 2 and 3, so the
attack reported is the strongest one available and not a convenient one.

## ext23's capacity defence is aimed at the wrong regions

ext23 proposed a per-region capacity cap as insurance: whatever a region
reports, it cannot receive more than `c_r`. Under a coalition attack on capped
leximin, at ext23's own `kappa = 1.5` the cap **never binds** -- the capture
ratio is 1.464 at every cap setting including uncapped, identical to four
decimals.

The reason is a mismatch this experiment exposes rather than fixes. Under
leximin the regions with the most to gain by lying are the **least-served** ones,
and a uniform cap constrains the **most-served**. The attackers are not near
their capacity, so the cap is not in the way.

| cap | kappa | \|C\|=1 | \|C\|=2 | \|C\|=4 | \|C\|=8 |
| --- | --- | --- | --- | --- | --- |
| none | 4 | 3.480 | 3.076 | 2.487 | 1.740 |
| none | 16 | 9.167 | 6.401 | 3.966 | 2.137 |
| 1.5x | 4 | **1.892** | **1.879** | **1.856** | 1.707 |
| 1.5x | 16 | **1.892** | **1.879** | **1.856** | **1.735** |

Bold is where the cap binds. It does its job against the tail exactly as ext23
claimed -- at `kappa = 16` uncapped leximin gives up a 9.2x capture and the 1.5x
cap holds it to 1.89 -- but only for **small** coalitions. By `|C| = 8` the cap
has largely stopped binding, because dilution has already pushed the ratio below
it. The cap and the coalition structure defend against the same tail, so against
a large cartel the cap is redundant rather than protective.

Leximin is not a power rule and has no exponent to read the direction off, so
its best response is searched over corners rather than derived. It dilutes
anyway, which says the dilution result is a property of budget normalisation and
not of the alpha-fair functional form.

## alpha = 0 is excluded, and that is not a gap being papered over

The argmax rule is discontinuous, so no elasticity describes it and the formula
returns NaN rather than a number that would compare wrongly. Measured directly, a
coalition that truthfully holds nothing and then captures the budget has an
**unbounded** ratio, not a large one -- 60% of states at `|C| = 1`. Averaging a
floored denominator would have turned "infinitely profitable" into a finite
number that averages like any other, so the unbounded fraction is reported as its
own column.

## Scope

48 states x 16 regions from the same lambda-omega ecosystem as ext22 and ext23,
so the three are comparable. `kappa = 1.5` throughout except the cap sweep, which
adds 4 and 16. No surrogate and no training: this is a property of the allocation
rule, and putting a learned model in front of it would confound a mechanism
result with a forecasting one. The threat model is a bounded multiplicative
misreport with truthful outsiders; side payments, sequential deviation, and
coalitions that lie about *which* regions they are are all out of scope.

Reproduce with:

    python3 scripts/coalition_manipulation.py

Outputs `ext35_coalition_law.csv`, `ext35_best_response.csv`,
`ext35_leximin_cap.csv`, `ext35_coalition.png`.
