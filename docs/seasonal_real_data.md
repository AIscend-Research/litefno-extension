# The harmonic prior on data that actually oscillates (H12 / ext31)

[harmonic_verdict.md](harmonic_verdict.md) closed with an obvious objection
against itself. The harmonic claim was tested three times and died twice -- but
every test ran on a field with almost no seasonal cycle. Four of six Gray-Scott
regimes contain no temporal line at all (ext10), and planetswe, the one testbed
with documented forcing, puts 5.4% of temporal variance there (ext12). A prior
on the annual mode cannot be expected to earn its keep where there is no annual
mode.

This answers that objection with data where the mode is unmistakably present.
The prior still does not help.

## The data

NOAA CPC global monthly soil moisture: 0.5 degree, 943 monthly steps, 1948-2026,
in mm. It is the same physical variable as ERA5-Land soil moisture, and needs no
credentials -- ERA5-Land requires a Copernicus CDS key and MODIS NDVI a NASA
Earthdata login, neither of which this environment has. Ingested through the
repo's existing 5-D contract by
[`scripts/ingest_cpc_soil.py`](../scripts/ingest_cpc_soil.py) and verified
against the repo's own loaders: `load_dataset`, `to_pairs`, `build_model` and
`evaluate_one_step` all work unmodified.

Six named regions replace Gray-Scott's six regimes and span a **6x range** in
exactly the quantity the prior is meant to exploit:

| region | seasonal share of temporal variance |
| --- | --- |
| india_monsoon | 65.7% |
| sahel_wafrica | 65.1% |
| amazon_brazil | 61.5% |
| la_plata | 20.7% |
| us_midwest | 11.7% |
| eurasia_boreal | 11.3% |

Against 5.4% for planetswe and no line at all for most of Gray-Scott, the
monsoon regions carry a signal **12x** stronger than the best case the repo had.

Two measurements, one leakage-free, agree per region to within a few points
(64.2/65.7, 65.6/65.1, 21.1/20.7, 11.5/11.3). That agreement is load-bearing:
943 months is not a whole number of years, so an annual line measured on the
raw record leaks across bins and reads 7.4% where the truth is 13.4% -- an 81%
understatement, the same class of artifact ext12 found understating by 2-6x.

## Result: still nothing

| trajectories | plain | harmonic | paired diff | seeds helped | multiplier |
| --- | --- | --- | --- | --- | --- |
| 18 | 0.31954 +- 0.00210 | 0.31942 +- 0.00202 | -0.0% | 2/3 | 1.00x |
| 54 | 0.27987 +- 0.00273 | 0.28119 +- 0.00830 | +0.5% | 1/3 | **0.96x** |

At the full training set the data multiplier is **0.96x**: the plain arm reaches
the harmonic arm's error on slightly *less* data. Three of six paired runs
favour the harmonic arm, which is what a coin does.

## The differential prediction inverts

ext15's prediction was never that the prior helps uniformly -- it was that it
helps where the targeted band holds the energy. Here that is directly testable,
because the regions differ 6x in seasonality. Per region at the largest size:

| region | seasonal | harmonic change |
| --- | --- | --- |
| india_monsoon | 65.7% | +1.4% |
| sahel_wafrica | 65.1% | **+16.2%** |
| amazon_brazil | 61.5% | +0.7% |
| la_plata | 20.7% | -0.3% |
| us_midwest | 11.7% | -3.3% |
| eurasia_boreal | 11.3% | +2.5% |

The three **most** seasonal regions all got worse (mean +6.1%); two of the three
least seasonal improved (mean -0.3%). Spearman(seasonality, change) = **+0.314**
where the mechanism requires a negative value.

That correlation is not significant at n = 6 (p = 0.54) and these per-region
numbers come from a **single seed**, so the ordering should not be read as an
established inversion. What it does rule out is the effect the prior was built
to produce: there is no sign of the predicted benefit concentrating where the
annual mode lives, on data where that mode is unambiguous.

## The prior also destabilises training

At the largest size the harmonic arm's seed-to-seed spread is **3.0x** the plain
arm's (0.00830 against 0.00273); at the smaller size they match (1.0x). The
sahel outlier at +16.2% is consistent with that: 220 extra parameters on a
narrow band appear to add variance rather than structure, and to do so more as
the training set grows.

## What this settles, and what it does not

It closes the strongest objection to
[harmonic_verdict.md](harmonic_verdict.md). The verdict's three failures could
have been blamed on the testbed; that explanation is now unavailable. The prior
fails on a field whose seasonal cycle carries two thirds of its temporal
variance.

It does not show that Fourier structure is useless. It shows that *hand-placing*
a learnable bias on a named band -- the annual mode here, the Turing shells in
ext15, low wavenumbers in ext9 -- does not beat letting the spectral layer learn
its own weights, in four separate settings now, including the one where the
band is unmistakably real.

The honest reading of the whole line: the repo has repeatedly found that
measuring the data predicts the model result, and the thing the data kept saying
is that these operators do not need to be told where the energy is.

## Scope

Pilot scale, two sizes (18 and 54 trajectories), 3 seeds, 50 epochs, 32x32
tiles of one field. Absolute VRMSE (0.28-0.32) is far higher than Gray-Scott's
(0.045): real soil moisture is a harder one-step prediction, dominated by
interannual and weather variability rather than by the seasonal cycle even in
the monsoon belt. Per-region figures are single-seed. The harmonic fundamental
was inherited rather than tuned to the annual period of this data, which is a
real limitation -- a fundamental matched to 12 months is the obvious next test
and is not run here.

Reproduce with:

    curl -L -o data/raw/cpc_soilw.nc \
      https://downloads.psl.noaa.gov/Datasets/cpcsoil/soilw.mon.mean.nc
    python3 scripts/ingest_cpc_soil.py
    python3 scripts/data_efficiency.py --data-dir data/processed/cpc_soil \
        --per-regime 3 9 --seeds 0 1 2 --epochs 50 \
        --out-dir results/pilot_cpc --fig-dir figures/pilot_cpc
