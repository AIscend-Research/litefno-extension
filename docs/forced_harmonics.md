# Does known periodic forcing justify a temporal harmonic prior?

The spatial form of the harmonic claim is already dead: ext9 / PR #15 showed
Gray-Scott's variance is not low-wavenumber dominated. ext10 then found that four
of six Gray-Scott scenarios contain no temporal line at all. But in all of those
the forcing is absent or unknown, so none is a fair test of a *temporal* prior —
you cannot fault a prior for missing a periodicity that was never there.

planetswe is the fair test. Shallow water on a rotating sphere with an explicit
solar-like heating term, and the forcing periods are documented rather than
inferred:

```
day  =   24 timesteps        year = 1008 timesteps
```

Produced by [`scripts/forced_harmonics.py`](../scripts/forced_harmonics.py).

```bash
python scripts/forced_harmonics.py --ics IC36 IC37 IC38 IC39
```

First run fetches three model years per initial condition and caches to
`data/processed/planetswe_cache.npz` (273 MB, git-ignored); later runs are offline.

## Headline

**The forcing is unmistakably there, and it is a small minority of the
variance.** Across all four test-split initial conditions, three model years each:

| band | forced share of temporal variance | excluding record-length drift |
|---|---|---|
| tropics | 6.7 / 9.1 / 10.8 / 11.5% | 9.7 / 13.9 / 16.4 / 18.5% |
| polar_S | 7.9 / 11.2 / 6.9 / 3.8% | 12.5 / 13.8 / 8.5 / 4.0% |
| global | 5.4 / 5.7 / 5.5 / 5.5% | 7.6 / 7.4 / 8.3 / 10.7% |
| midlat_S | 0.47 / 0.54 / 0.74 / 0.66% | 0.47 / 0.55 / 0.75 / 0.66% |

That the forcing is real is not in question. It is enriched 728–2220× over chance
(90× in the one band where it is weakest), it sits on the exact documented bins,
and it is phase-locked across independent trajectories:

| cell | harmonic | phase resultant (4 ICs) | chance |
|---|---|---|---|
| ω=3, k=0 | annual | **0.997** | 0.50 |
| ω=6…18, k=0 | annual ×2…×6 | **0.995 – 0.998** | 0.50 |
| ω=126, k=−1 | diurnal | **1.000** | 0.50 |
| ω=252, k=−2 | diurnal ×2 | 0.317 | 0.50 |
| ω=378 / 504 | diurnal ×3, ×4 | 0.537 / 0.383 | 0.50 |

The forcing is identical in every trajectory, so a forced response must repeat
its phase and internal variability must not. The annual ladder and the diurnal
fundamental are locked at essentially 1.0. The diurnal *harmonics* sit at chance
— the diurnal response is a pure fundamental, with no overtones, which is also
what their variance shares say (2×10⁻⁵ % and below).

So the temporal claim does not die the way the spatial one did. It survives as a
real but minor effect: **a prior with perfect knowledge of both forcing periods
addresses at most 11.5% of the temporal variance in the most favourable latitude
band (18.5% if the drift is discounted entirely), and a very stable 5.4–5.5%
globally.** Everything else is drift and internal variability.

## What the rest of the variance is

Tropical budget for IC36 (three model years):

| component | share |
|---|---|
| annual, all six harmonics | 5.7% |
| diurnal, all harmonics | 1.0% |
| record-length drift (ω=±1, k=0) | 31.5% |
| largest single internal mode (ω=32, k=2 — period 94.5 steps ≈ 3.9 days) | 5.0% |
| everything else | 56.8% |

Two things stand out. The drift at record-length period is the single largest
component nearly everywhere (0.4–62.9% depending on band and IC, and above 18%
in every band except the southern midlatitudes); with only three
years it is one cycle, indistinguishable from a trend, and no periodic prior
keyed to day and year captures it. This is the same trap ext10 hit on
Gray-Scott, which is why it is broken out rather than folded into "low
frequency", and why the tables also report the forced share with the drift
removed from the denominator entirely — the most generous reading available.

And a *single* internal mode routinely rivals or beats the entire forced signal.
In IC38's southern midlatitudes one internal mode at ω=15, k=1 holds 15.0% while
everything the forcing drives holds 0.74% — a factor of 20 the wrong way.

The annual harmonic ladder decays fast (IC36 tropics: 3.77%, 0.73%, 0.64%,
0.19%, 0.23%, 0.12% for n=1…6), so extending a seasonal prior to more harmonics
buys almost nothing beyond the first two.

## Why the measurement is generous by construction

The result is negative, so it was built to survive the best case:

- **Space-time, not frequency alone.** The dataset card gives `lon_center =
  time_of_day*2*pi` — the heating circles the planet once per day. The response
  therefore lives at one (frequency, zonal wavenumber) cell rather than spread
  over all wavenumbers at that frequency. Isolating the cell *removes* competing
  internal variability and raises the measured forced share. Measured at ω only,
  the diurnal share is 0.12%; isolated at (ω=126, k=−1) it is 1.0%. The annual
  forcing migrates north-south (`lat_center = sin(time_of_year*2*pi)*...`), so
  it is zonally symmetric at k=0.
- **The best latitude band, not the global mean.** The diurnal response is
  tropical and a global average dilutes it 8.6–17.7×.
- **Whole-period records.** Three chunks of one initial condition are
  consecutive years, so the record is exactly 3024 steps = 3 years = 126 days.
  Every forcing harmonic lands on an exact integer bin with no leakage.

That last point flips a methodological choice relative to ext10. There the
records were not periodic and a Hann taper was needed; **here boxcar is correct
and Hann is the worse estimator**, because it smears an exactly-on-bin line into
its neighbours. Measured with Hann, the tropical forced share reads 1.1 / 3.4 / 2.4 / 5.9%
against boxcar's 6.7 / 9.1 / 10.8 / 11.5% — a factor of 2–6 understatement,
purely from the window.
Both are in the CSV; boxcar is the headline.

## Directional specificity

The (ω, k) isolation is not cosmetic. An eastward wave at the diurnal frequency
is not the forced response, and the test suite checks that a 30%-amplitude
eastward injection at ω=126 is scored under 2% forced. The measured diurnal
response appears at (ω=+126, k=−1) with 0.503% and at (ω=+126, k=+1) with
0.00002% — a 25,000:1 preference for westward, matching the apparent motion of
the sun.

## Two indexing traps, both hit while writing this

**`theta` is colatitude in radians, not degrees of latitude.** It runs 0…π
descending (row 0 is the south pole). Read as degrees, every latitude band
selects the entire globe, because 0…3.14 lies inside any sensible degree range —
and the per-band table comes out identical everywhere without raising anything.
`colatitude_to_latitude` now asserts the input range, and a test checks the bands
tile the sphere without overlap.

**`np.fft.fftfreq(n) * n` is not exactly integral.** `int()` truncates 125.99999
to 125 and silently reads the neighbouring bin. That moved the measured diurnal
share by four orders of magnitude (0.5% → 0.00003%) between two probes, which is
the only reason it was caught. All bin lookup goes through `bin_index`, which
rounds and then asserts the recovered frequency matches what was asked for.

## Correctness

The machinery is shown able to find a forced signal that *is* there, since the
finding is that one mostly isn't. `tests/test_forced_harmonics.py` injects a
travelling wave carrying a known fraction of the variance at the diurnal cell
and checks it is recovered: 2%, 10% and 40% injections all come back within 6%
relative. Unforced backgrounds score near chance; eastward waves are not counted;
phase-locking separates a synthetic locked ensemble (resultant > 0.9) from a
random-phase one (< 0.7).

## Outputs

| file | contents |
|---|---|
| `results/extensions/ext12_planetswe_forced.csv` | per IC × latitude band × window: annual/diurnal/forced shares, ex-drift share, enrichment, drift, largest internal mode |
| `results/extensions/ext12_planetswe_per_harmonic.csv` | per (ω, k) cell: harmonic label, period, variance share |
| `results/extensions/ext12_planetswe_phase_lock.csv` | cross-IC phase resultant per forced cell |
| `figures/extensions/ext12_planetswe_forced.png` | space-time spectrum with forced cells marked, tropical variance budget, forced share by latitude |

## Caveats

- Height field only. The forcing is thermal and enters the height equation
  directly, but the velocity response is not measured here and could carry a
  different forced fraction.
- All four test-split initial conditions, three model years each. The phase-lock
  statistic has n=4, so its chance level is 0.50 — high enough that only the
  near-1.0 results are conclusive, which is what the annual ladder and diurnal
  fundamental give. Marginal values (the diurnal overtones) should be
  read as "not established", not as "confirmed absent".
- Spatially strided 4× to 64×128. Striding rather than block-averaging is
  deliberate — averaging would low-pass the field and could attenuate exactly the
  small-scale variability being weighed against the forcing — but it does alias
  zonal wavenumbers above 64, which is far above the k=0…4 cells that matter here.
- Three years cannot separate a genuine multi-year cycle from drift. The ω=±1
  component is reported as drift on that basis; a longer record could reclassify
  some of it.
