# Harmonic content by scenario

How much of each scenario's variance sits in slow, coherent temporal modes
versus fast fluctuation — i.e. how much harmonic structure there actually is to
exploit.

Produced by [`scripts/harmonic_content.py`](../scripts/harmonic_content.py).
This is the temporal companion to
[`scripts/spectral_variance_decomposition.py`](../scripts/spectral_variance_decomposition.py)
(ext9), which did the same thing over spatial wavenumber for a single field.

```bash
python scripts/harmonic_content.py --family gray_scott --n-traj 3
python scripts/harmonic_content.py --family trl --n-traj 1
```

Both read The Well's HDF5 files over HTTP range requests straight from
HuggingFace, so no part of the 140 GB Gray-Scott family is written to disk.

## Headline

**The low-frequency share does not discriminate between Gray-Scott scenarios,
and on its own it is close to meaningless.** All twelve scenario/field pairs put
96–100% of their time-varying variance below 10% of Nyquist. Read naively, that
says every scenario is drenched in exploitable harmonic structure.

It isn't. Against a red-noise (AR(1)) null fitted to each spectrum's own lag-1
autocorrelation, only two of the six scenarios contain an actual spectral line:

| scenario | static share | peak / AR(1) null | period | verdict |
|---|---|---|---|---|
| spirals | 2.8% | **18.1** | 45.5 steps | strong harmonic |
| gliders | 7.5% | **4.3** | 100.2 steps | weak harmonic |
| worms | 87.5% | 1.3 | — | red noise |
| maze | 95.8% | 1.4 | — | red noise |
| spots | 96.5% | 1.5 | — | red noise |
| bubbles | 99.7% | 1.5 | — | red noise |

(field A, settled segment, 3 trajectories per scenario.)

Two independent measurements agree on the same split. The four red-noise
scenarios are also 87–99.7% *static*: their time-averaged pattern carries nearly
all the variance, so there is barely any dynamics left to have a spectrum. The
two harmonic scenarios are 3–8% static — the pattern is genuinely in motion.

So the answer to "how much harmonic structure exists to exploit" is: **in four
of six Gray-Scott scenarios, essentially none.** They are frozen patterns with a
slow red-noise drift. Only spirals and gliders have a periodic component worth
the name.

## Why the naive number lies

A low-frequency share this high is what you get from *any* strongly correlated
process, harmonic or not. Validating on synthetic series where the answer is
known:

| synthetic series | low-freq share | excess over null | peak / null |
|---|---|---|---|
| white noise | 9.8% | 2.0% | 1.2 |
| AR(1), φ=0.9 | **79.3%** | 2.3% | 1.1 |
| AR(1), φ=0.99 | **97.3%** | 3.7% | 1.1 |
| sine (period 20) + white noise | 34.4% | 36.4% | **61.7** |
| sine + AR(1) φ=0.9 | 80.5% | 28.0% | **54.1** |
| linear ramp | 100.0% | 34.6% | 1.6 |

(Hann window throughout; 400 series of length 1000. Reproduced as a test in
[`tests/test_harmonic_content.py`](../tests/test_harmonic_content.py), which
needs no network.)

An AR(1) with φ=0.99 scores 97% low-frequency while containing no harmonic at
all, and a pure ramp scores 100%. Gray-Scott's fitted φ is 0.988–0.999 — right
in that regime. `peak / null` is what separates the two cases, and it does so by
a factor of ~40.

Note the ramp row: `excess_share` alone reads 34.6% there, comparable to a real
sine, because a ramp's 1/f spectrum is steeper than any AR(1) the null can fit
and the residual lands entirely in the lowest bins. `excess_share` is therefore
not a sufficient statistic on its own — `peak_over_null` is the discriminator,
and the two must be read together. This is exactly why the transient-bearing
`full` segment is reported separately rather than being the headline.

The script guards three traps of this kind, each of which independently
manufactures the "low-frequency dominated" conclusion:

1. **The spin-up transient.** Over a full trajectory the Gray-Scott pattern
   forms once and then sits there. That one-time ramp has a 1/f spectrum, so the
   low-frequency share is high no matter what the settled dynamics do. Every
   scenario is decomposed over three segments (`full`, `train`, `settled`) and
   only `settled` answers the question.
2. **Red noise.** Handled by the AR(1) null above.
3. **Spectral leakage, and mean-vs-total power per shell.** The series are not
   periodic in time, so every temporal metric is computed under both boxcar and
   Hann windows; the two agree on the ranking. On the spatial side the pass sums
   energy per shell rather than averaging it, since high-wavenumber shells hold
   many more modes (the correction ext9 applies).

The windows do disagree on one thing, informatively: in the 60-step `train`
segment the boxcar low-frequency share drops to 56–90% while Hann stays at
97–100%. That gap is leakage from the un-tapered transient, and it is a warning
sign rather than a result.

## The training window cannot see the structure that exists

The repo preprocesses to `max_steps: 60` (`configs/datasets/*.yaml`). In that
60-step window no scenario — spirals included — reaches a `peak / null` of 1.8,
i.e. none has a detectable harmonic:

| segment | spirals peak/null | spirals period found |
|---|---|---|
| full (T=1001) | 29.2 | 43.5 steps |
| settled (T=501) | 18.1 | 45.5 steps |
| **train (T=60)** | **1.6** | 30 steps (= bin 2, the resolution floor) |

The periods that exist are 45.5 steps (spirals) and 100.2 steps (gliders). A
60-step window resolves barely one cycle of the first and less than one cycle of
the second, and the peak it reports sits at bin 1 or 2 — the lowest frequency
the window can represent, which is what you get when there is nothing to find.
The model is never shown the temporal structure this analysis locates.

This is a statement about the data window, not about model quality. LiteFNO is
trained as a next-step operator, so it is not asked to represent a 45-step
period directly. But it does mean that any extension hoping to *exploit*
temporal harmonics — a temporal Fourier layer, a periodicity prior, a
longer-horizon rollout objective — has nothing to work with at `max_steps: 60`
and would need the cap raised to ~200+ steps first.

## Spatial side: the mode budget is not slack

Same trajectories, decomposed over the spatial wavenumber an FNO actually
truncates on (`m = max(|kx|, |ky|)`, matching `n_modes=(m, m)`), at the native
128×128 resolution:

| scenario | m ≤ 4 | m ≤ 8 | m ≤ 12 | m ≤ 16 | m for 99% |
|---|---|---|---|---|---|
| spirals | 39.3% | 77.4% | 93.8% | 98.3% | 18 |
| gliders | 21.6% | 69.0% | 94.7% | 98.9% | 17 |
| bubbles | 20.0% | 58.3% | 82.4% | 94.0% | 23 |
| worms | 9.0% | 30.9% | 75.0% | 94.3% | 23 |
| maze | **0.1%** | **1.3%** | 39.1% | 96.0% | 18 |
| spots | **0.1%** | **0.6%** | 29.5% | 96.2% | 19 |

Maze and spots are the outliers by two orders of magnitude: they hold
essentially *nothing* below m=8, because their variance is concentrated in a
narrow band at the Turing wavelength around m≈13–16. Any mode truncation below
about 12 destroys them while costing spirals relatively little. A rank/mode
sweep that reports a single averaged VRMSE across scenarios will hide this
completely.

This is consistent with the ext9 result (peak energy at k=5 on the 4×-downsampled
32×32 grid), which maps to k≈20 at native resolution.

The per-regime split above is what motivated the mid-spectrum retry of the
harmonic prior in ext15 — maze and spots hold ~99% of their variance above mode
8, so a bias on those shells is not the low-k prior ext9 killed. That retry was
measured in ext30 and came out null;
[harmonic_verdict.md](harmonic_verdict.md) records the claim's full result.

Note the trained configuration uses `MODES = min(16, H // 2)` on a 32×32 grid,
so at model resolution it keeps every available mode — the spatial truncation is
not binding there, and the numbers above describe what the 4× spatial
downsampling in preprocessing has already discarded.

## Second family: turbulent radiative layer

Eight scenarios, cooling time t_cool = 0.03 … 1.78, one trajectory each,
T=101. Included as a contrast and as a validity check on the method — here there
is a known physical control parameter, so the measurement can be checked against
something other than itself.

Unlike Gray-Scott, this family is genuinely broadband: low-frequency share
ranges 17–81% (it discriminates), the high band holds a real 5–21%, and fitted φ
is 0.51–0.88 rather than pinned near 1. No scenario has a strong line —
peak/null tops out at 3.5.

The band shares respond monotonically to t_cool, and the two fields respond in
*opposite* directions (Spearman ρ over the 8 scenarios):

| field | low share | mid share | high share | AR(1) φ | centroid |
|---|---|---|---|---|---|
| pressure | **+0.91** | −0.71 | **−0.95** | **+0.91** | **−0.91** |
| density | **−0.98** | **+0.98** | −0.62 | −0.12 | +0.57 |

Longer cooling times make **pressure** smoother and redder on every measure
(low share 0.40→0.81, high share 0.21→0.05, φ 0.51→0.88) — consistent with
cooling being what drives the fast pressure fluctuations. **Density** instead
shifts variance out of the low band into the mid band (0.53→0.17 low,
0.34→0.71 mid) while its noise floor and overall redness stay flat.

I am reporting the trends, not a mechanism for the density behaviour — that
would need more than this measurement. Caveat: The Well's TRL test files hold
one trajectory each, so scenario effect and trajectory-to-trajectory variability
cannot be separated here. The monotonicity across 8 independent scenarios is the
evidence; the per-scenario values are single draws.

The spatial pass is skipped for this family: it is not periodic in y, and an
un-windowed 2-D FFT there would manufacture high-wavenumber power at the
boundary.

## Correctness checks

Every identity the decomposition relies on is asserted at runtime, not assumed:

- **Parseval, temporal** — the summed boxcar spectrum must equal the dynamic
  variance to 1e-6 relative.
- **Parseval, spatial** — radial and box binned energy must each equal the
  summed per-frame spatial variance to 1e-9 relative. This is the check that
  catches a mean-per-shell vs total-per-shell mix-up.
- **Law of total variance** — static + dynamic must close on the total to 1e-6
  relative.

The float64 cast in `analyse_segment` is load-bearing rather than hygiene:
Gray-Scott's A field sits near 0.98 with a variance around 1e-3, and a float32
`var` loses enough significant digits to cancellation there to fail the Parseval
check by ~3e-3. It did, before the cast was added.

The AR(1) null is fitted by recovering the autocovariance from the spectrum via
Wiener-Khinchin rather than re-reading it off the series, so the null is fitted
to exactly the quantity it is compared against.

## Outputs

| file | contents |
|---|---|
| `results/extensions/ext10_harmonic_summary_{gray_scott,trl}.csv` | one row per scenario × field × segment, all band and null metrics, both windows |
| `results/extensions/ext10_temporal_spectrum_{gray_scott,trl}.csv` | per-bin share, cumulative share, and AR(1) null share |
| `results/extensions/ext10_spatial_spectrum_gray_scott.csv` | per-k share and cumulative, radial and box binning |
| `figures/extensions/ext10_harmonic_content_{gray_scott,trl}.png` | spectra vs null, low-share-vs-excess scatter, spatial retention curve |

Band edges are fractions of Nyquist (low ≤ 0.10, high > 0.50) so they mean the
same thing across families with different T. Every summary row also carries the
per-trajectory standard deviation of the two headline numbers
(`*_low_share_sd`, `*_excess_share_sd`).
