# Fairness checklist

A structured statement of what the allocation layer's fairness results do and do
not license. It exists because ext22-ext24 use the vocabulary of distributive
justice — envy-freeness, leximin, the price of fairness, worst-off outcomes — on
a decision problem that is entirely synthetic, and that vocabulary carries
implications the experiments do not earn.

Everything here is a pointer into a measured result or an explicit statement
that no measurement exists. Where a row says "not measured", that is the finding
for that row, not an omission to be filled in later by inference.

Covers [ext22](fair_allocation.md), [ext23](strategic_allocation.md) and
[ext24](network_scarcity.md).

---

## 1. What is being allocated, to whom

| Question | Answer |
|---|---|
| What is the resource? | One scarce, homogeneous, divisible budget `B`. Not a bundle, not typed, not indivisible. |
| Who are the recipients? | 16 "regions": a fixed 4x4 square partition of the simulation grid. |
| Where do the regions come from? | The grid geometry. **Not** from any ecological, administrative, census or jurisdictional boundary. |
| What is a region's claim based on? | Its population `g_r`, read as `1.5 + u` block-averaged over the reconstructed field. |
| Is `1.5 + u` a population? | No. It is a units choice that makes the block mean positive. Nothing in the PDE says a region's entitlement is an affine function of a chemical concentration. |
| Are there people in this experiment? | **No.** No human, demographic, protected-attribute or socio-economic data is used anywhere in this repository. |

**Consequence.** No result here is a claim about what any real region, community
or population is owed. The measured object is *how error propagates through the
composition* `field -> pooled gains -> allocation -> realised welfare`, which is
a property of the rule and the surrogate, not of the world.

## 2. Which fairness notion, and is it the right one

| Question | Answer |
|---|---|
| Fairness family | alpha-fair (Mo & Walrand 2000), optimum `a_r ∝ g_r^((1-alpha)/alpha)`. |
| Range swept | alpha = 0 (utilitarian), 0.25, 0.5, 1, 2, 4, 8, inf (max-min). |
| Is a single alpha privileged? | No. alpha is a free parameter and the results are reported across it. The repo does not recommend an alpha. |
| Envy-freeness | Holds at alpha = 1, where it **forces equal division** (Foley 1967, Varian 1974) because the resource is homogeneous and divisible and the gain cancels from the comparison. |
| Is that an interesting envy-freeness result? | No, and it is labelled as degenerate in [ext22](fair_allocation.md). Equal division is envy-free by construction here; a non-trivial study needs heterogeneous valuations over several resource types (Eisenberg-Gale / CEEI). |
| Group fairness / disparate impact | **Not measured.** There are no groups and no protected attributes to define them over. |
| Individual fairness | **Not measured.** No similarity metric over regions is defined. |
| Counterfactual / causal fairness | **Not measured.** |

**Consequence.** The only fairness notions tested are welfare-aggregation rules
over a one-dimensional claim. Every fairness notion that requires knowing
something about *who* a recipient is, is out of scope, because the recipients
have no attributes.

## 3. What the fairness results actually say

Reported so the direction and the size are both visible.

| Claim | Status | Number |
|---|---|---|
| Fairness costs robustness | **Refuted** in this setup | Sensitivity is U-shaped in alpha with an exact zero at alpha = 1; both ends are more fragile than the middle. |
| Sensitivity of the allocation to gain error | Derived and confirmed to 3 decimals | `\|1-alpha\|/alpha` |
| Relative welfare loss | Derived; 0.5% accurate at sd 0.01-0.05, 13% off at sd 0.2 | `(1-alpha)^2 / (2 alpha) * Var_w(eta)` |
| The law transfers to a real surrogate's error | Confirmed | Predicts realised plug-in loss to within 1% |
| Price of fairness | Measured | 0.121 -> 0.168 moving alpha 0.25 -> 8 |
| What that buys the worst-off region | Measured | min/max outcome 0.267 -> 0.945 |
| Manipulation-robustness and error-robustness are separable | **Refuted** | Both are the same derivative `\|1-alpha\|/alpha` |
| A strategy-proof member exists | Yes, exactly one | alpha = 1 — and it is strategy-proof *because it ignores the state* |
| Max-efficiency manipulability | Measured | Unbounded, on 100% of states; a 20% misreport flips the argmax and takes the whole budget |
| Leximin with per-region capacity bounds a lie | Measured | Against a 10x misreport, winnings cut 6.93x -> 1.64x, for 4.2% of the worst-off region's welfare |

**The load-bearing caveat on the last block, since closed:** every incentive
number above assumes **one region lies and the rest are truthful** -- and for
egalitarian rules a coalition is the natural threat, because several regions
understating together move the common outcome level in a way no single deviation
can. ext35 ([coalition_manipulation.md](coalition_manipulation.md)) ran the law
against joint deviations: it survives exactly, with the region's share replaced
by the coalition's pooled share, so alpha = 1 is **group** strategy-proof and
collusion makes manipulation worse for the colluders.

## 4. Proxies, and what each stands in for

Nothing in this repository measures a fairness-relevant quantity directly.

| Proxy used | Stands in for | Why it is a proxy and not the thing |
|---|---|---|
| `1.5 + u` block mean | A region's population / need | A concentration field rescaled to be positive. |
| 4x4 grid partition | Administrative or ecological regions | Chosen to divide the grid evenly. |
| `lambda_omega` Hopf normal form | An ecosystem | A generated 2-field oscillatory PDE, not ecological data. |
| Gray-Scott regimes | "Disaster types" / distinct hazard classes | Six (F, k) parameter pairs in a reaction-diffusion benchmark. |
| Watts-Strogatz graph | A trade network | Stipulated topology; nothing in the physics says which regions trade. |
| SIS cascade | Scarcity propagation | Borrowed from epidemiology; only the dynamics and the `1/lambda_1` threshold are pinned to closed form. |
| Realised welfare `sum_r g_r a_r` under alpha-fair aggregation | Social welfare | A stylized objective with no normative standing. |

## 5. Data provenance and consent

| Question | Answer |
|---|---|
| Source of the field data | The Well (Polymathic AI), Gray-Scott reaction-diffusion and planetswe. Simulation output. |
| Source of the allocation testbed | Generated in-repo by `src/litefno/systems.py`. Never downloaded, never averaged with results on The Well's data. |
| Personal data | None. |
| Consent / IRB | Not applicable — no human subjects, no human-derived data. |
| Licence | Data CC BY 4.0 via Zenodo, DOI [10.5281/zenodo.20718092](https://doi.org/10.5281/zenodo.20718092). |

## 6. Statistical honesty

| Question | Answer |
|---|---|
| Seeds | 3 for the allocation arms and the network arms. |
| Error bars | Standard deviations across paired differences, **not** confidence intervals. |
| Evaluation sample | 192 decisions from 12 trajectories x 16 start times — these **overlap**, so they buy resolution, not independent degrees of freedom. |
| Is the aggregate the verdict? | No. Per-regime and per-alpha breakdowns are the reported quantity; aggregates are labelled context. |
| Are controls size-matched? | Yes. `shrunk` and `smoothed` exist so the learned allocator's win cannot be attributed to hedging; the network is 2,913 parameters against the surrogate's 7,106. |
| Negative results reported? | Yes — H2, H6 and the low-wavenumber spatial prior are all reported as refuted. |

## 7. Deployment

| Question | Answer |
|---|---|
| Has any of this been deployed? | **No.** |
| Has it been tested against a real allocation decision? | **No.** |
| Is there a human-in-the-loop design? | Not specified. The layer emits an allocation; nothing models review, appeal or override. |
| Failure mode if deployed as-is | At low alpha the rule is both maximally fragile to surrogate error and unboundedly manipulable — the two failures coincide, and the rule that removes them (alpha = 1) does so by discarding the forecast entirely. |
| Known dangerous configuration | Acting on a stale observation. At every alpha except 0, allocating from an 8-step-old state loses more welfare than ignoring the state entirely. |

## 8. What would have to change for any of this to be an equity claim

In rough order of leverage:

1. **Real populations.** Replace the block-mean gain with actual population or
   need data. OpenFEMA award amounts against CDC SVI is the concrete route
   already identified.
2. **Real regions.** Replace the 4x4 partition with administrative boundaries,
   which are neither square nor equal-area, and whose unequal sizes interact
   with the fairness rule.
3. **Protected attributes.** Without them, no group-fairness notion is even
   definable, and disparate impact cannot be measured.
4. **Heterogeneous resources.** Envy-freeness is degenerate until there is more
   than one resource type.
5. **A deployment test.** Nothing here has been run against a decision anyone
   acted on.

Until at least (1) and (2) hold, the correct reading of ext22-ext24 is: *these
are results about error propagation through a composition of a surrogate and an
aggregation rule, expressed in fairness vocabulary because the aggregation rule
is drawn from the fairness literature.* They are not findings about equity.

## References

- Bertsimas, Farias & Trichakis (2011), *The Price of Fairness*.
- Elmachtoub & Grigas (2022), *Smart "Predict, then Optimize"*.
- Foley (1967), *Resource Allocation and the Public Sector*.
- Mo & Walrand (2000), *Fair End-to-End Window-Based Congestion Control*.
- Varian (1974), *Equity, Envy, and Efficiency*.
