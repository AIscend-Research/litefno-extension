# The harmonic claim: a negative result in three forms

The extension phase opened with a plausible idea, inherited from the roadmap's
"Explainability" line and from LiteFNO's own framing: if a PDE's dynamics
concentrate in a few Fourier modes, then telling the model about those modes
should help. This document is the verdict on that idea after four measurements.

It failed three times, in three different forms, and survived once as a real but
minor effect. That is a negative result, and it belongs next to the repository's
other negative result rather than scattered across four docs that each mention it
in passing.

## The claim, and the three forms it was tested in

| # | form of the claim | measured in | outcome |
| --- | --- | --- | --- |
| 1 | Variance is dominated by **low spatial wavenumbers**, so a low-k prior helps | ext9 | **dead** |
| 2 | Known **temporal forcing** justifies a temporal harmonic prior | ext12 | **survives, small** |
| 3 | A **mid-spectrum** bias on the Turing shells helps the regimes that live there | ext15 / ext30 | **dead** |

### Form 1: the low-wavenumber spatial prior (ext9)

`ext9_variance_decomposition.csv` converts the ground-truth radial PSD into a
cumulative variance fraction over wavenumber, weighting each shell by how many
Fourier modes it contains -- the correction that matters, since summing mean
per-shell power would silently under-weight the high-k shells that hold most of
the modes.

The spectrum is not low-wavenumber dominated. Peak energy sits at **k = 5**, and
the cumulative share is:

    k <= 4    29.2%
    k <= 8    67.6%
    k <= 15   95.8%

**58% of the variance sits at k >= 6** -- the median wavenumber is k = 6, not
k = 1 or 2. A prior that buys its advantage by assuming energy lives in the first
few modes has nothing to work with here.

### Form 2: the temporal prior under documented forcing (ext12)

Form 1's failure is about *space*, and four of six Gray-Scott regimes contain no
temporal line at all -- but in those the forcing is absent or unknown, so they
cannot fault a temporal prior for missing a periodicity that was never there.
planetswe is the fair test: shallow water on a rotating sphere with an explicit
solar-like heating term whose periods are documented (day = 24 steps, year = 1008).

The forcing is unambiguous -- phase-locked at **0.995-0.998** across four
independent trajectories, enriched more than 700x over chance. And it accounts
for **5.4-5.5%** of temporal variance globally, rising to 11.5% in the most
favourable latitude band (18.5% if the red-noise drift is discounted entirely).

So the temporal form survives, as a real effect that is too small to build an
architecture around. This is the one place the harmonic claim is not dead, and
saying so is the difference between a verdict and a dismissal.

### Form 3: the mid-spectrum retry (ext15, measured by ext30)

Forms 1 and 2 leave a narrower version standing, and ext10 is what made it worth
trying. It measured, per regime, the share of spatial variance below mode 8:

    spirals 77.4%   gliders 69.0%   bubbles 58.3%
    worms   30.9%   maze     1.3%   spots    0.6%

maze and spots are outliers by two orders of magnitude: they hold ~99% of their
variance *above* mode 8, in a narrow ring at the Turing wavelength. A bias placed
on that ring is not the low-k prior ext9 killed. It is a targeted claim with a
differential prediction -- help maze and spots, do nothing for the low-wavenumber
regimes -- and a *uniform* gain would have been evidence against it.

ext15 built the model and the A/B and never ran it. ext30 ran the same two arms
across four training-set sizes at three seeds and found nothing:

- the harmonic arm is very slightly **worse** at every size,
- it wins **0 of 12** paired runs on identical initialisation, subset and order,
- the data multiplier is **1.00x** (0.9985-0.9991 exactly) -- the prior
  substitutes for no data at all, and if anything the plain arm reaches the
  harmonic arm's error on marginally *less* data,
- and the differential prediction does not appear: maze improves 0.1%, spots
  gets slightly worse.

The effect is about **2% of the seed spread**, consistent with the bias's 220
extra parameters (+2.94%) contributing noise and nothing else.

Two honest qualifications. This is not the ext15 A/B as specified -- that is
three seeds at 100 epochs with a per-regime verdict of its own, and it remains
unrun. And a null at pilot scale does not exclude an effect at much larger
training sets, or at a fundamental other than the inherited 4.0, which was never
swept.

## Why this sits next to the reproduction result

The repository's other negative result is that a parameter-matched low-rank CNN
matches or outperforms the spectral arm on Gray-Scott at 32x32 across three
seeds, giving no consistent evidence for a Fourier inductive-bias advantage at
that scale.

The harmonic line reaches the same place from the opposite direction. The
reproduction asks whether a *generic* Fourier bias earns its keep and finds it
does not. The harmonic experiments ask whether *sharpening* that bias into an
explicit prior on named modes earns its keep, and find that it does not either --
not at low wavenumber, not on the Turing shells, and only marginally in the one
setting where the periodicity is documented and exact.

Neither result says Fourier structure is useless in general. Both are statements
about this testbed at this scale: a 32x32 downsample of Gray-Scott whose spectrum
peaks at k = 5 and needs 15 shells to reach 95% of its variance. That is a field
with no narrow band to exploit, which is precisely the condition under which a
spectral prior has nothing to give.

The pattern worth carrying forward is that the repo's *measurements of the data*
predicted its *measurements of the models*. ext9 and ext10 characterised the
spectrum before any of the priors were built, and both times the prior's fate
followed from what those spectra already said. ext15's own docstring recorded a
low prior on its effect size before running, citing ext9 and ext12 by name. It
was right.

## The objection this verdict invited, now answered

Everything above ran on a field with almost no seasonal cycle, so the natural
rebuttal was that the prior never had a fair test. ext31 removes that defence:
NOAA CPC soil moisture, ingested through the same 5-D contract, where six
regions span a 6x range in seasonality and the monsoon belt carries **65%** of
its temporal variance in the annual cycle -- against 5.4% for planetswe and no
line at all for most of Gray-Scott.

The prior still does not help. The data multiplier is 0.96x at the full training
set, three of six paired runs favour it, and the differential prediction does not
appear: the three most seasonal regions all got *worse* (mean +6.1%) while two of
the three least seasonal improved. See
[seasonal_real_data.md](seasonal_real_data.md).

So the claim has now failed in four forms, including on data whose seasonal mode
is unmistakable. What that leaves is narrower and firmer than a testbed
complaint: hand-placing a learnable bias on a named band does not beat letting
the spectral layer learn its own weights.

## Where the evidence lives

| extension | doc | result file |
| --- | --- | --- |
| ext9 | (in [harmonic_content.md](harmonic_content.md)) | `ext9_variance_decomposition.csv` |
| ext10 | [harmonic_content.md](harmonic_content.md) | `ext10_harmonic_summary_gray_scott.csv` |
| ext12 | [forced_harmonics.md](forced_harmonics.md) | `ext12_planetswe_*.csv` |
| ext15 | [harmonic_conditioning.md](harmonic_conditioning.md) | none -- never run |
| ext30 | [data_efficiency.md](data_efficiency.md) | `ext30_*.csv` |

No compute was run for this document; it consolidates results already committed.
