# Where do real disaster allocations sit in the alpha-family? (H13 / ext32)

ext22 derived, in closed form, how much a surrogate's error costs an alpha-fair
decision: sensitivity is U-shaped in alpha with an exact zero at the envy-free
point. ext23 read the same layer as a mechanism and found the incentive to
misreport is the *same derivative*, so the only strategy-proof member of the
family is the one that ignores the state.

Both describe a family of rules. Neither says where any real allocation sits in
it. This locates one, using FEMA Public Assistance obligations against
CDC/ATSDR social vulnerability.

## The bridge is exact

The family allocates `a ∝ g^beta` with `beta = (1 - alpha) / alpha`
(`litefno.allocation.allocation_exponent`). So a log-log regression of award on
need estimates `beta` directly, and

    alpha_hat      = 1 / (1 + beta_hat)
    manipulability = |1 - alpha| / alpha = |beta|          (ext23)
    fragility      = (1 - alpha)^2 / (2 alpha)             (ext22)

The middle identity is checked numerically for every alpha, not asserted:
ext23's manipulation incentive **is** the fitted elasticity. Fitting the slope of
award against vulnerability therefore measures, in ext23's own units, what a
county could gain by overstating its need.

## The confound the design exists to defeat

Public Assistance obligations track *damage*, not vulnerability. A high-SVI
county with no hurricane receives nothing, so a pooled cross-section of all
counties would mostly measure where storms make landfall.

Every fit is therefore **within one disaster**, across only the counties FEMA
declared for that disaster. All counties in a fit were hit by the same event
under the same rules, so what remains is allocation rather than exposure.
Disasters are selected by incident type, a declaration-year floor and a minimum
county count -- never by anything computed from the awards.

## Result: no vulnerability gradient

12 hurricanes, FY2015 onward, 17-80 counties each.

| DR | event | n | beta | R2 | p (permutation) |
| --- | --- | --- | --- | --- | --- |
| 4680 | Nicole | 31 | **-3.321** | 0.343 | 0.005 |
| 4798 | Beryl | 37 | -2.658 | 0.059 | 0.138 |
| 4677 | Ian | 17 | -2.313 | 0.120 | 0.210 |
| 4673 | Ian | 48 | -0.770 | 0.015 | 0.425 |
| 4834 | Milton | 42 | -0.711 | 0.034 | 0.268 |
| 4828 | Helene | 43 | -0.515 | 0.006 | 0.647 |
| 4626 | Ida | 20 | +0.058 | 0.000 | 0.978 |
| 4734 | Idalia | 45 | +0.314 | 0.003 | 0.740 |
| 4738 | Idalia | 30 | +0.430 | 0.004 | 0.762 |
| 4829 | Helene | 32 | +0.536 | 0.010 | 0.540 |
| 4817 | Francine | 22 | +0.964 | 0.045 | 0.372 |
| 4830 | Helene | 80 | **+2.112** | 0.143 | 0.000 |

- **Sign is a coin flip**: 6 positive, 6 negative.
- **Median R2 = 0.025.** Vulnerability explains about 2.5% of the variance in
  award per capita.
- **Pooled elasticity (inverse-variance weighted) = -0.088 +- 0.266, z = -0.33.**
  Indistinguishable from zero.
- Only **2 of 12** slopes beat their own permutation null, and those two point in
  *opposite* directions (Nicole -3.32, Helene DR-4830 +2.11), which is what
  heterogeneity looks like, not a gradient.

## What that means in the family's own terms

Pooled `beta_hat = -0.088` gives **`alpha_hat = 1.10`**: statistically
indistinguishable from `alpha = 1`, the envy-free point.

That is the single most interesting place it could have landed, because ext23
proved `alpha = 1` is the *only* strategy-proof member of the family -- and
proved it is strategy-proof **because it ignores the state**. ext22 puts the same
point at exactly zero sensitivity to surrogate error.

So the empirical and the theoretical line up in an uncomfortable way. Observed
Public Assistance allocation is approximately non-manipulable, and it appears to
buy that property the way the theorem says you must: by not responding to need.
By ext22's U-shape the corollary is that a better forecast would change this
allocation very little, because the allocation is not listening to the state.

**The honest caveat on that reading**: "sitting at alpha = 1" and "having no
detectable signal" are the same observation in this estimator. A pooled slope of
zero with median R2 of 0.025 is consistent with a deliberate envy-free rule and
equally consistent with allocation driven by things this regression does not
see. The data cannot separate those, and the claim here is the observational one.

Note also that the median of `|beta|` is 0.741 even though the median of `beta`
is -0.229. Individual disasters carry large elasticities of both signs. The
*average* allocation is unresponsive; any *particular* disaster is not, and by
ext23's identity a county inside one of those disasters faces a real, if
inconsistently-signed, incentive.

## Public Assistance is the program least likely to show a gradient

This is the sharpest limitation and it is not a footnote. Public Assistance
repairs public infrastructure -- roads, debris, public buildings. It is not
designed to track household vulnerability, so a null gradient there is close to
what the program's own rules would predict. **Individual Assistance**, which pays
households, is where an equity gradient would actually be expected.

That arm was attempted and is **underpowered, not reported as a result**. IHP is
one row per registration; the large hurricanes run to 300k-1.2M registrations
and were skipped by a stated cap, and most small ones had too few usable
counties. Exactly **one** disaster survived (Idalia DR-4734, n = 18,
`beta = +5.34`, R2 = 0.43, p = 0.000). A single positive, significant slope is
suggestive -- and it points the opposite way from the PA pooled estimate -- but
one disaster is not a finding, and `ext32_disaster_fits_ia.csv` is committed with
that caveat attached rather than summarised into a headline.

Closing the IA arm properly is the obvious next step and needs a bulk download
rather than paged API access.

## Other limitations

SVI is a **percentile rank**, so an elasticity computed on it is not the same
object as an elasticity on a cardinal gain. The sign and rough magnitude survive
that; the third digit does not.

Counties with zero obligation are dropped rather than floored, because a zero is
an absent observation on a log scale and any epsilon would set the answer.
Awards are per capita, so a large county is not generous merely for being large.
Hurricanes only, FY2015 onward, one incident type.

## Reproduce

    python3 scripts/fema_svi_equity.py --program pa \
        --incident-type Hurricane --since 2015
    python3 scripts/fema_svi_equity.py --program ia --max-registrations 120000

Outputs `ext32_disaster_fits_{pa,ia}.csv`, `ext32_county_panel_{pa,ia}.csv`.
Both sources are open: OpenFEMA needs no key, and CDC/ATSDR SVI 2022 is read
from the agency's own ArcGIS feature service.
