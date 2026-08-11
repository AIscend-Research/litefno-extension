r"""Does known periodic forcing make a temporal harmonic prior worth having?

Board task: "Measure planetswe temporal spectrum to test harmonic claim as a
temporal prior." The spatial version of the claim is already dead -- ext9/PR #15
showed Gray-Scott's variance is not low-wavenumber dominated -- and ext10 found
that four of six Gray-Scott scenarios contain no temporal line at all. But in
every one of those the forcing is absent or unknown, so none of them is a fair
test of a *temporal* prior: you cannot fault a prior for missing a periodicity
that was never there.

planetswe is the fair test, and about the best case The Well contains. It is
shallow water on a rotating sphere driven by an explicit solar-like heating term
with a diurnal and a seasonal cycle, and the periods are documented rather than
inferred:

    day  =   24 timesteps
    year = 1008 timesteps       (output cadence: 1 hour of simulation time)

Each HDF5 file holds exactly one model year, and the s1/s2/s3 files of one
initial condition are consecutive, so concatenating them gives exactly three
years. That is load-bearing: a record spanning a whole number of forcing periods
puts every forcing harmonic on an exact FFT bin, with no leakage to argue about.
It also makes the boxcar window correct here -- the opposite of the Gray-Scott
case in ext10, where the records were not periodic and a taper was needed. On
this dataset a Hann window is the *worse* estimator, because it smears an
exactly-on-bin line across its neighbours.

What is being measured
----------------------
A temporal harmonic prior -- a temporal Fourier layer, a periodicity feature, a
seasonal-cycle regression -- can only ever explain variance that actually sits
at the forcing frequencies. So:

    forced share = fraction of temporal variance in the forced (omega, k) cells

and if that is small, the prior cannot help however well it is implemented.

The measurement is deliberately generous, because the interesting result is a
negative one and a negative result has to survive the best case:

* **Space-time, not frequency alone.** The diurnal forcing is a travelling wave
  -- the dataset card gives ``lon_center = time_of_day*2*pi``, so the heating
  circles the planet once a day. Its response therefore lives at one specific
  (frequency, zonal wavenumber) cell, not spread over all wavenumbers at that
  frequency. Isolating the cell removes the internal variability sharing the
  frequency and *raises* the measured forced share. The annual forcing moves
  north-south (``lat_center = sin(time_of_year*2*pi)*max_declination``), so it
  is zonally symmetric: k = 0.
* **The most favourable latitude band, not the global mean.** The diurnal
  response is concentrated in the tropics and a global average would dilute it.
* **Phase-locking is verified, not assumed.** The forcing is identical across
  initial conditions, so a genuinely forced response must have the same phase in
  every trajectory while internal variability must not. The resultant length of
  the cross-IC phasors separates the two, which matters because the annual peak
  is not distinguishable from red noise by an AR(1) null alone.

Two indexing traps, both hit while writing this
-----------------------------------------------
``theta`` is colatitude in *radians*, descending (row 0 is the south pole), not
degrees of latitude. Treating it as degrees silently selects the entire globe
for every band, because 0..3.14 lies inside any sensible degree range, and the
per-band table comes out identical everywhere without erroring.

``np.fft.fftfreq(n) * n`` returns floats that are not exactly integral, so
``int()`` truncates 125.99999 to 125 and reads the neighbouring bin. That moved
the measured diurnal share by four orders of magnitude. Bin lookup goes through
:func:`bin_index`, which rounds and then asserts the recovered frequency.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_HC_PATH = Path(__file__).resolve().parent / "harmonic_content.py"


def _load_hc():
    spec = importlib.util.spec_from_file_location("harmonic_content", _HC_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["harmonic_content"] = module
    spec.loader.exec_module(module)
    return module


hc = _load_hc()

REPO = "polymathic-ai/planetswe"
DAY_STEPS = 24
YEAR_STEPS = 1008
CHUNKS = ("s1", "s2", "s3")
FIELD = "t0_fields/height"

# Latitude bands in degrees. 'global' is included for contrast, but the headline
# deliberately takes the most favourable band instead of this one.
BANDS = {
    "tropics": (-23.5, 23.5),
    "midlat_N": (23.5, 66.5),
    "midlat_S": (-66.5, -23.5),
    "polar_N": (66.5, 90.0),
    "polar_S": (-90.0, -66.5),
    "global": (-90.0, 90.0),
}


def colatitude_to_latitude(theta: np.ndarray) -> np.ndarray:
    """theta is colatitude in radians (0 = north pole); return degrees latitude."""
    theta = np.asarray(theta, dtype=np.float64)
    assert theta.min() >= -1e-6 and theta.max() <= np.pi + 1e-6, (
        f"theta looks like it is not radians: {theta.min()}..{theta.max()}")
    return 90.0 - theta * 180.0 / np.pi


def bin_index(freqs: np.ndarray, target: int) -> int:
    """Index of an exact integer frequency, with the result verified.

    ``fftfreq(n) * n`` is float, and int() truncation reads the wrong bin (126
    becomes 125). Round, then assert we landed on the frequency asked for.
    """
    idx = int(np.argmin(np.abs(freqs - target)))
    assert abs(freqs[idx] - target) < 1e-6, (
        f"no exact bin for frequency {target}; nearest is {freqs[idx]}")
    return idx


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def fetch_trajectory(ic: str, split: str, stride: int, time_chunk: int = 63,
                     tries: int = 5):
    """Concatenate one initial condition's three yearly files into one series.

    Read in time chunks: a year of this field is 512 MB and asking fsspec for
    that in one range request gets the connection dropped. Spatial subsampling is
    by stride rather than block-mean on purpose -- averaging is a low-pass and
    would attenuate exactly the small-scale variability being weighed against the
    forcing. Striding keeps each retained pixel's own time series intact.
    """
    pieces, times, lat = [], [], None
    for chunk in CHUNKS:
        name = f"data/{split}/planetswe_{ic}_{chunk}.hdf5"
        for attempt in range(tries):
            try:
                h5 = hc.open_remote(REPO, name, block_size=2 ** 22)
                dset = h5[FIELD]
                n_time = dset.shape[1]
                part = [np.asarray(dset[0, i:i + time_chunk])[:, ::stride, ::stride]
                        for i in range(0, n_time, time_chunk)]
                lat = colatitude_to_latitude(
                    np.asarray(h5["dimensions/theta"]))[::stride]
                times.append(np.asarray(h5["dimensions/time"]))
                h5.close()
                pieces.append(np.concatenate(part))
                break
            except Exception as exc:                        # transient HTTP/SSL
                print(f"    retry {attempt + 1} on {chunk}: {type(exc).__name__}",
                      flush=True)
        else:
            raise RuntimeError(f"could not read {name}")

    time = np.concatenate(times)
    steps = np.diff(time)
    # The concatenation only means anything if the chunks really are
    # consecutive at one cadence; a gap would put a discontinuity in the middle
    # of the record and manufacture broadband power across the whole spectrum.
    assert steps.min() > 0, "time axis not increasing"
    assert steps.max() / steps.min() < 1.05, (
        f"uneven cadence across chunks: {steps.min():.4f}..{steps.max():.4f}")
    return np.concatenate(pieces), lat, time


def build_cache(ics, split: str, stride: int, cache_path: Path) -> dict:
    if cache_path.exists():
        with np.load(cache_path) as z:
            cache = {k: z[k] for k in z.files}
        if all(f"{ic}|field" in cache for ic in ics):
            return cache
    cache = {}
    for ic in ics:
        print(f"  fetching {ic} ({len(CHUNKS)} yearly files)", flush=True)
        series, lat, time = fetch_trajectory(ic, split, stride)
        print(f"    {series.shape}, t {time[0]:.0f}..{time[-1]:.0f}", flush=True)
        cache[f"{ic}|field"] = series.astype(np.float32)
        cache[f"{ic}|lat"] = lat.astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)
    print(f"  cached {cache_path} ({cache_path.stat().st_size / 1e6:.0f} MB)")
    return cache


# --------------------------------------------------------------------------
# space-time spectrum
# --------------------------------------------------------------------------


def spacetime_coeffs(block: np.ndarray, window: str = "boxcar") -> np.ndarray:
    """Complex Fourier coefficients over (time, longitude), per latitude row.

    ``block`` is (T, n_lat, n_lon). The per-pixel temporal mean is removed, so
    the omega = 0 row carries nothing and the spectrum describes the fluctuation
    only. Returns (T, n_lat, n_lon).
    """
    x = np.asarray(block, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    if window == "hann":
        w = np.hanning(x.shape[0])
        x = x * (w / np.sqrt((w ** 2).mean()))[:, None, None]
    elif window != "boxcar":
        raise ValueError(window)
    return np.fft.fft2(x, axes=(0, 2))


def forced_cells(n_time: int, n_lon: int, n_annual: int, n_diurnal: int) -> dict:
    """The (omega, k) cells the forcing drives, as signed integer pairs.

    Annual forcing migrates north-south and is zonally symmetric  -> k = 0.
    Diurnal forcing circles the planet once a day, westward, so harmonic m sits
    at zonal wavenumber -m relative to +omega. Both signs of omega are listed
    because a real field's spectrum is Hermitian and the conjugate cell carries
    an equal half of the variance.
    """
    cells = {}
    for n in range(1, n_annual + 1):
        omega = n_time * n // YEAR_STEPS
        if n_time * n % YEAR_STEPS or omega > n_time // 2:
            continue
        label = f"annual x{n}" if n > 1 else "annual"
        cells[(omega, 0)] = label
        cells[(-omega, 0)] = label
    for m in range(1, n_diurnal + 1):
        omega = n_time * m // DAY_STEPS
        if n_time * m % DAY_STEPS or omega > n_time // 2 or m > n_lon // 2:
            continue
        label = f"diurnal x{m}" if m > 1 else "diurnal"
        cells[(omega, -m)] = label
        cells[(-omega, m)] = label
    return cells


def analyse_band(block: np.ndarray, window: str, n_annual: int,
                 n_diurnal: int) -> dict:
    n_time, _, n_lon = block.shape
    coeffs = spacetime_coeffs(block, window)
    power = (np.abs(coeffs) ** 2).sum(axis=1)          # sum over latitude rows
    total = power.sum() - power[0, 0]
    if total <= 0:
        return {}
    share = power / total

    omegas = np.fft.fftfreq(n_time, d=1.0) * n_time
    ks = np.fft.fftfreq(n_lon, d=1.0) * n_lon
    cells = forced_cells(n_time, n_lon, n_annual, n_diurnal)

    # the low-frequency drift: a record-length "period" is a trend, not a
    # cycle, and no periodic prior captures it -- reported separately so it
    # is not mistaken for forced signal (ext10 hit this trap on Gray-Scott)
    i1, j0 = bin_index(omegas, 1), bin_index(ks, 0)
    trend = float(share[i1, j0] + share[bin_index(omegas, -1), j0])

    per_cell, annual, diurnal = [], 0.0, 0.0
    for (omega, k), label in sorted(cells.items()):
        i, j = bin_index(omegas, omega), bin_index(ks, k)
        s = float(share[i, j])
        per_cell.append({"omega": omega, "k": k, "label": label,
                         "period_steps": n_time / abs(omega), "share": s})
        if label.startswith("annual"):
            annual += s
        else:
            diurnal += s


    masked = share.copy()
    masked[0, 0] = 0.0
    for (omega, k) in cells:
        masked[bin_index(omegas, omega), bin_index(ks, k)] = 0.0
    masked[i1, j0] = masked[bin_index(omegas, -1), j0] = 0.0
    ii, jj = np.unravel_index(int(np.argmax(masked)), masked.shape)
    top_int = {"omega": int(round(omegas[ii])), "k": int(round(ks[jj])),
               "period_steps": n_time / max(abs(omegas[ii]), 1e-9),
               "share": float(masked[ii, jj])}

    n_cells_total = share.size - 1
    return {
        "n_time": n_time, "years": n_time / YEAR_STEPS,
        "annual_share": annual, "diurnal_share": diurnal,
        "forced_share": annual + diurnal,
        # Discounting the record-length drift entirely is the most generous
        # reading available: it assumes the drift is model equilibration that a
        # longer record or a better spin-up would remove, and asks what
        # fraction of the *remaining* variance the forcing accounts for.
        "forced_share_ex_trend": (annual + diurnal) / max(1.0 - trend, 1e-12),
        "n_forced_cells": len(cells),
        "chance_share": len(cells) / n_cells_total,
        "enrichment": (annual + diurnal) / (len(cells) / n_cells_total),
        "trend_share": trend,
        "top_internal_omega": top_int["omega"], "top_internal_k": top_int["k"],
        "top_internal_period": top_int["period_steps"],
        "top_internal_share": top_int["share"],
        "per_cell": per_cell,
    }


def phase_lock(blocks: list[np.ndarray], omega: int, k: int,
               window: str = "boxcar") -> dict:
    """Resultant length of the cross-trajectory phasors at one (omega, k) cell.

    The forcing is identical in every initial condition, so a forced response
    must carry the same phase in all of them; internal variability must not.
    With ``n`` independent trajectories, unrelated phases give a resultant of
    about 1/sqrt(n), so that is the number to beat -- reported alongside.

    Computed per latitude row and then averaged, so the statistic pools many
    rows rather than resting on one number.
    """
    phasors = []
    for block in blocks:
        coeffs = spacetime_coeffs(block, window)
        n_time, _, n_lon = block.shape
        i = bin_index(np.fft.fftfreq(n_time, d=1.0) * n_time, omega)
        j = bin_index(np.fft.fftfreq(n_lon, d=1.0) * n_lon, k)
        c = coeffs[i, :, j]
        mag = np.abs(c)
        phasors.append(np.where(mag > 0, c / np.where(mag > 0, mag, 1), 0))
    stack = np.stack(phasors)                       # (n_ic, n_lat)
    resultant = np.abs(stack.mean(axis=0))
    n_ic = len(blocks)
    return {"omega": omega, "k": k, "n_ic": n_ic,
            "resultant": float(resultant.mean()),
            "chance_resultant": 1.0 / np.sqrt(n_ic)}


def band_mask(lat: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (lat >= lo) & (lat <= hi)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run(cache: dict, ics, windows, n_annual: int, n_diurnal: int) -> list[dict]:
    records = []
    for ic in ics:
        field = cache[f"{ic}|field"]
        lat = cache[f"{ic}|lat"]
        n_time = field.shape[0]
        assert n_time % YEAR_STEPS == 0, (
            f"record is {n_time} steps, not a whole number of {YEAR_STEPS}-step "
            "years; the forcing harmonics would not land on exact bins")
        for band, (lo, hi) in BANDS.items():
            rows = band_mask(lat, lo, hi)
            if not rows.any():
                continue
            for window in windows:
                res = analyse_band(field[:, rows, :], window, n_annual, n_diurnal)
                if not res:
                    continue
                omega_only = hc.temporal_power(field[:, rows, :], window)
                res.update({
                    "ic": ic, "band": band, "window": window,
                    "n_lat_rows": int(rows.sum()),
                    "omega_only_low_share": hc.band_metrics(omega_only)["low_share"],
                })
                records.append(res)
        print(f"  {ic}: T={n_time} = {n_time / YEAR_STEPS:.0f} model years, "
              f"{field.shape[1]}x{field.shape[2]} grid", flush=True)
    return records


def run_phase_lock(cache: dict, ics, n_annual: int, n_diurnal: int) -> list[dict]:
    if len(ics) < 3:
        print("\n(phase-locking needs >=3 initial conditions; skipped)")
        return []
    lat = cache[f"{ics[0]}|lat"]
    rows = band_mask(lat, *BANDS["tropics"])
    blocks = [cache[f"{ic}|field"][:, rows, :] for ic in ics]
    n_time, _, n_lon = blocks[0].shape
    out = []
    print(f"\n=== Phase-locking across {len(ics)} initial conditions (tropics) ===")
    print("    A forced response has the same phase in every trajectory.")
    print(f"    Unrelated phases give a resultant near "
          f"{1 / np.sqrt(len(ics)):.2f}; forced should approach 1.")
    print(f"    {'cell':>18s} {'harmonic':>12s} {'resultant':>10s} {'chance':>8s}")
    for (omega, k), label in sorted(forced_cells(n_time, n_lon, n_annual,
                                                 n_diurnal).items()):
        if omega <= 0:
            continue                                  # conjugate carries no news
        res = phase_lock(blocks, omega, k)
        res["label"] = label
        out.append(res)
        print(f"    {f'(w={omega}, k={k})':>18s} {label:>12s} "
              f"{res['resultant']:>10.3f} {res['chance_resultant']:>8.2f}")
    return out


def print_report(records: list[dict], window: str = "boxcar") -> None:
    rows = [r for r in records if r["window"] == window]
    if not rows:
        return
    r0 = rows[0]
    print(f"\n=== planetswe forced variance ({window}, T={r0['n_time']} = "
          f"{r0['years']:.0f} model years) ===")
    print(f"    day = {DAY_STEPS} steps, year = {YEAR_STEPS} steps -- both land "
          f"on exact FFT bins")
    print("    forced = share in the (omega, k) cells the forcing drives")
    print("    trend  = share at record-length period (drift, not a cycle)")
    hdr = (f"    {'IC':>6s} {'band':>9s} {'annual':>8s} {'diurnal':>8s} "
           f"{'forced':>8s} {'ex-drift':>9s} {'enrich':>8s} {'trend':>7s} "
           f"{'top internal':>22s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in sorted(rows, key=lambda r: (r["ic"], r["band"])):
        ti = (f"w={r['top_internal_omega']},k={r['top_internal_k']} "
              f"{r['top_internal_share']:.2%}")
        print(f"    {r['ic']:>6s} {r['band']:>9s} {r['annual_share']:>7.3%} "
              f"{r['diurnal_share']:>7.3%} {r['forced_share']:>7.3%} "
              f"{r['forced_share_ex_trend']:>8.3%} {r['enrichment']:>7.0f}x "
              f"{r['trend_share']:>6.1%} {ti:>22s}")

    trop = [r for r in rows if r["band"] == "tropics"]
    if trop:
        print(f"\n=== Per-harmonic detail ({trop[0]['ic']}, tropics, {window}) ===")
        print(f"    {'omega':>7s} {'k':>4s} {'harmonic':>12s} "
              f"{'period(steps)':>14s} {'share':>10s}")
        for e in trop[0]["per_cell"]:
            if e["omega"] <= 0:
                continue
            print(f"    {e['omega']:>7d} {e['k']:>4d} {e['label']:>12s} "
                  f"{e['period_steps']:>14.4f} {2 * e['share']:>9.5%}")
        print("    (share doubled to include the conjugate cell)")


def write_csv(path: Path, records: list[dict], drop=("per_cell",)) -> None:
    flat = [{k: v for k, v in r.items() if k not in drop} for r in records]
    if not flat:
        return
    keys, seen = [], set()
    for r in flat:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    print(f"wrote {path}")


def write_per_cell(path: Path, records: list[dict]) -> None:
    rows = [{"ic": r["ic"], "band": r["band"], "window": r["window"],
             "n_time": r["n_time"], **e}
            for r in records for e in r["per_cell"]]
    write_csv(path, rows, drop=())


def plot(cache: dict, records: list[dict], ics, out_png: Path,
         window: str = "boxcar", n_annual: int = 6, n_diurnal: int = 4) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped figure")
        return

    ic = ics[0]
    field, lat = cache[f"{ic}|field"], cache[f"{ic}|lat"]
    rows = band_mask(lat, *BANDS["tropics"])
    block = field[:, rows, :]
    n_time, _, n_lon = block.shape
    power = (np.abs(spacetime_coeffs(block, window)) ** 2).sum(axis=1)
    power[0, 0] = 0.0
    share = power / power.sum()
    omegas = np.fft.fftfreq(n_time, d=1.0) * n_time
    ks = np.fft.fftfreq(n_lon, d=1.0) * n_lon
    cells = forced_cells(n_time, n_lon, n_annual, n_diurnal)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (1) space-time spectrum with the forced cells marked
    ax = axes[0]
    kmax = 8
    sel_k = np.argsort(ks)
    keep = np.abs(ks[sel_k]) <= kmax
    cols = sel_k[keep]
    pos = omegas > 0
    img = share[np.ix_(pos, cols)]
    ax.pcolormesh(ks[cols], omegas[pos], np.log10(np.maximum(img, 1e-12)),
                  shading="auto", cmap="magma")
    for (omega, k), label in cells.items():
        if omega <= 0 or abs(k) > kmax:
            continue
        ax.plot(k, omega, "o", mfc="none", ms=9,
                mec="cyan" if label.startswith("annual") else "lime", mew=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("zonal wavenumber k")
    ax.set_ylabel("frequency bin (record = 3 model years)")
    ax.set_title("Space-time spectrum, tropics (log power)\n"
                 "cyan = annual cells, green = diurnal cells", fontsize=10)

    # (2) where the variance actually is
    ax = axes[1]
    rec = [r for r in records if r["window"] == window and r["ic"] == ic
           and r["band"] == "tropics"][0]
    labels = ["annual\n(all harmonics)", "diurnal\n(all harmonics)",
              "record-length\ntrend", "largest single\ninternal mode",
              "everything\nelse"]
    other = 1 - (rec["annual_share"] + rec["diurnal_share"]
                 + rec["trend_share"] + rec["top_internal_share"])
    vals = [rec["annual_share"], rec["diurnal_share"], rec["trend_share"],
            rec["top_internal_share"], other]
    colours = ["tab:cyan", "tab:green", "0.6", "tab:orange", "0.85"]
    bars = ax.bar(range(len(vals)), [v * 100 for v in vals], color=colours,
                  edgecolor="k", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{v:.1%}", ha="center", fontsize=8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("share of temporal variance (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Tropical variance budget\n(the forced part is what a prior "
                 "could reach)", fontsize=10)

    # (3) forced share by latitude band
    ax = axes[2]
    order = ["tropics", "midlat_N", "midlat_S", "polar_N", "polar_S", "global"]
    rs = [r for b in order for r in records
          if r["window"] == window and r["ic"] == ic and r["band"] == b]
    x = np.arange(len(rs))
    ax.bar(x - 0.2, [r["annual_share"] * 100 for r in rs], 0.4,
           label="annual", color="tab:cyan", edgecolor="k", linewidth=0.4)
    ax.bar(x + 0.2, [r["diurnal_share"] * 100 for r in rs], 0.4,
           label="diurnal", color="tab:green", edgecolor="k", linewidth=0.4)
    ax.axhline(rs[0]["chance_share"] * 100, color="crimson", ls="--", lw=1,
               label="chance level")
    ax.set_xticks(x)
    ax.set_xticklabels([r["band"] for r in rs], rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("share of temporal variance (%)")
    ax.set_title("Forced variance by latitude\n(headline uses the best band, "
                 "not the mean)", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("planetswe: documented daily/annual forcing vs the temporal "
                 "variance budget", y=1.03, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ics", nargs="*", default=["IC36", "IC37", "IC38", "IC39"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=4,
                    help="spatial subsampling stride on the 256x512 grid")
    ap.add_argument("--n-annual", type=int, default=6)
    ap.add_argument("--n-diurnal", type=int, default=4)
    ap.add_argument("--windows", nargs="*", default=["boxcar", "hann"])
    ap.add_argument("--cache", type=Path,
                    default=Path("data/processed/planetswe_cache.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/extensions"))
    args = ap.parse_args()

    ca = hc._certifi_path()
    if ca:
        os.environ.setdefault("SSL_CERT_FILE", ca)

    print(f"planetswe: {len(args.ics)} initial conditions; "
          f"day={DAY_STEPS} steps, year={YEAR_STEPS} steps")
    cache = build_cache(args.ics, args.split, args.stride, args.cache)
    records = run(cache, args.ics, args.windows, args.n_annual, args.n_diurnal)

    print_report(records, "boxcar")
    if "hann" in args.windows:
        print_report(records, "hann")
    locks = run_phase_lock(cache, args.ics, args.n_annual, args.n_diurnal)

    write_csv(args.out_dir / "ext12_planetswe_forced.csv", records)
    write_per_cell(args.out_dir / "ext12_planetswe_per_harmonic.csv", records)
    if locks:
        write_csv(args.out_dir / "ext12_planetswe_phase_lock.csv", locks)
    plot(cache, records, args.ics,
         args.fig_dir / "ext12_planetswe_forced.png",
         n_annual=args.n_annual, n_diurnal=args.n_diurnal)


if __name__ == "__main__":
    sys.exit(main())
