r"""Pole-residue analysis on planetswe, where the right answer is documented.

Board task: use pole-residue / neutral-stability reasoning to decide which
spectral modes are actually oscillatory rather than transient, as a principled
basis for which to preserve. The method is in ``src/litefno/poles.py``; this
runs it and checks it against a known answer.

planetswe is the right test because the forcing periods are documented rather
than inferred: day = 24 steps, year = 1008 (ext12). Concatenating one initial
condition's three yearly files gives a 3024-step record. A pole finder that
works has to place a near-neutral complex pole at period 24, at the zonal
wavenumber ext12 identified for it, and it has to do so without being told.

It also has to be honest about the annual, and that is the more interesting half.

Two things this reports that a power spectrum cannot
----------------------------------------------------
A spectrum puts a ringing mode and a dying one in the same bin. The pole's
magnitude separates them: ``sigma = log|z|`` is the decay rate per step, and
near-zero means the mode is still going at the end of the record. So each mode
gets a share of its energy in near-neutral complex poles -- "actually
oscillatory" -- rather than merely a share at some frequency.

And a pole carries a frequency estimated from the whole record rather than
snapped to an FFT bin, so the recovered period is continuous. On planetswe that
matters: the diurnal comes back at 24.01 against a documented 24.

The resolution limit, stated up front
-------------------------------------
Pole estimation needs cycles, not samples. Measured on synthetic sinusoids in
red noise over a 3024-step record:

    126 cycles (period   24)   0.0% error
     30 cycles (period  100)   0.5% error
     10 cycles (period  300)  13.7% error
      3 cycles (period 1008)   fails outright

planetswe's annual forcing has 3 cycles in the record. So this script does not
find it, and that is a property of the record length rather than a finding about
the physics -- ext12 measured the annual perfectly well with an FFT, which needs
only that the period divide the record. Reporting the annual as "not found"
without that context would be the wrong conclusion, so the limit is measured
here rather than asserted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from litefno.poles import (  # noqa: E402
    NEUTRAL_TOL, analyse_series, classify_poles, fit_ar_poles, pole_residues)

DAY_STEPS = 24
YEAR_STEPS = 1008
BANDS = {"tropics": (-23.5, 23.5), "midlat_N": (23.5, 66.5),
         "global": (-90.0, 90.0)}


def latitudes(n_rows: int, stride: int = 4) -> np.ndarray:
    return 90.0 - np.linspace(np.pi, 0, n_rows * stride)[::stride] * 180.0 / np.pi


def zonal_modes(field: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Complex zonal-wavenumber series, averaged over a latitude band."""
    return np.fft.rfft(field[:, rows, :].astype(np.float64), axis=2).mean(axis=1)


def resolution_limit(n_time: int, order: int, seed: int = 1) -> list[dict]:
    """How many cycles the fit needs, measured rather than asserted."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    noise = np.cumsum(rng.normal(size=n_time))
    noise = noise / noise.std()
    out = []
    for period in (24, 100, 300, 1008, 1512):
        x = np.sin(2 * np.pi * t / period) + 0.5 * noise
        got = analyse_series(x, order=order)
        periods = np.where(got["freq"] > 1e-12,
                           1 / np.maximum(got["freq"], 1e-12), np.inf)
        j = int(np.argmin(np.abs(periods - period)))
        out.append({"true_period": period, "cycles_in_record": n_time / period,
                    "recovered_period": float(periods[j]),
                    "error_pct": float(100 * abs(periods[j] - period) / period),
                    "label": str(got["labels"][j]),
                    "energy_share": float(got["energy"][j]
                                          / got["energy"].sum())})
    return out


def analyse_zonal(series: np.ndarray, order: int) -> dict:
    x = series - series.mean()
    poles = fit_ar_poles(x, order=order)
    residues, energy = pole_residues(x, poles)
    got = classify_poles(poles, energy, n_time=len(x))
    periods = np.where(got["freq"] > 1e-12,
                       1 / np.maximum(got["freq"], 1e-12), np.inf)
    total = energy.sum()

    nearest = {}
    for target, name in ((DAY_STEPS, "diurnal"), (YEAR_STEPS, "annual")):
        j = int(np.argmin(np.abs(periods - target)))
        nearest[name] = {
            "period": float(periods[j]),
            "error_pct": float(100 * abs(periods[j] - target) / target),
            "sigma": float(got["sigma"][j]),
            "energy_share": float(energy[j] / total) if total else 0.0,
            "label": str(got["labels"][j])}
    return {"oscillatory_share": got["oscillatory_share"],
            "transient_share": got["transient_share"],
            "stationary_share": got["stationary_share"],
            "unstable_share": got["unstable_share"],
            "dominant_period": got["dominant_period"],
            "dominant_sigma": got["dominant_sigma"],
            "dominant_energy_share": got["dominant_energy_share"],
            "nearest": nearest}


def build_gray_scott_cache(cache: Path, settled_from: int = 500,
                           n_steps: int = 501) -> dict:
    """Settled-segment field of each Gray-Scott regime, one trajectory each.

    Streamed from The Well as a single contiguous read per regime (~33 MB), the
    same access pattern stream_preprocess uses. Only the settled half is taken:
    ext10 showed the first half is the one-time pattern-formation transient, and
    fitting poles to a ramp measures the ramp.
    """
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "harmonic_content", Path(__file__).resolve().parent / "harmonic_content.py")
    hc = importlib.util.module_from_spec(spec)
    sys.modules["harmonic_content"] = hc
    spec.loader.exec_module(hc)

    import os
    ca = hc._certifi_path()
    if ca:
        os.environ.setdefault("SSL_CERT_FILE", ca)

    fam = hc.FAMILIES["gray_scott"]
    out = {}
    for regime, path in fam.scenarios.items():
        h5 = hc.open_remote(fam.repo, path, block_size=2 ** 22)
        try:
            block = np.asarray(
                h5["t0_fields/A"][0, settled_from:settled_from + n_steps])
        finally:
            h5.close()
        out[regime] = block.astype(np.float32)
        print(f"    fetched {regime} {block.shape}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def analyse_gray_scott(cache: dict, max_mode: int, order: int) -> list[dict]:
    """Pole analysis of each regime's most energetic spatial mode.

    Reported with ``sigma_reliable``, which is the point: these are
    self-organised patterns whose phase wanders, so the constant-frequency model
    does not hold and the fit over-damps. The flag says so instead of the table
    quietly contradicting ext10.
    """
    from litefno.poles import analyse_series, spatial_mode_series
    rows = []
    for regime, field in sorted(cache.items()):
        series, radii = spatial_mode_series(field.astype(np.float64), max_mode)
        weights = (np.abs(series) ** 2).sum(axis=0)
        i = int(np.argmax(weights[1:])) + 1              # skip the DC mode
        got = analyse_series(series[:, i], order=order)
        rows.append({
            "regime": regime, "k": int(radii[i]), "n_time": field.shape[0],
            "order": order,
            "oscillatory_share": got["oscillatory_share"],
            "transient_share": got["transient_share"],
            "dominant_period": got["dominant_period"],
            "fit_sigma": got["fit_sigma"],
            "envelope_sigma": got["envelope_sigma"],
            "sigma_reliable": got["sigma_reliable"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path("data/processed/planetswe_cache.npz"))
    ap.add_argument("--ics", nargs="*", default=None)
    ap.add_argument("--order", type=int, default=24)
    ap.add_argument("--max-zonal", type=int, default=4)
    ap.add_argument("--band", default="tropics", choices=sorted(BANDS))
    ap.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    ap.add_argument("--gray-scott", action="store_true",
                    help="also analyse Gray-Scott, where the model does not hold")
    ap.add_argument("--gs-cache", type=Path,
                    default=Path("data/processed/gs_poles_cache.npz"))
    args = ap.parse_args()

    if not args.cache.exists():
        raise SystemExit(
            f"{args.cache} not found -- build it with scripts/forced_harmonics.py")
    with np.load(args.cache) as z:
        cache = {k: z[k] for k in z.files}
    ics = args.ics or sorted({k.split("|")[0] for k in cache})

    print("=== resolution limit (synthetic, measured not assumed) ===")
    limit = resolution_limit(3024, args.order)
    print(f"    {'period':>8} {'cycles':>8} {'recovered':>10} {'error':>8} "
          f"{'label':>12}")
    for r in limit:
        print(f"    {r['true_period']:>8} {r['cycles_in_record']:>8.1f} "
              f"{r['recovered_period']:>10.1f} {r['error_pct']:>7.1f}% "
              f"{r['label']:>12}")

    print(f"\n=== planetswe, {args.band}, AR({args.order}) "
          f"(day={DAY_STEPS}, year={YEAR_STEPS} steps) ===")
    lo, hi = BANDS[args.band]
    records = []
    for ic in ics:
        field = cache[f"{ic}|field"]
        lat = cache.get(f"{ic}|lat")
        if lat is None:
            lat = latitudes(field.shape[1])
        rows = (lat >= lo) & (lat <= hi)
        zon = zonal_modes(field, rows)
        for k in range(min(args.max_zonal + 1, zon.shape[1])):
            got = analyse_zonal(zon[:, k], args.order)
            records.append({"ic": ic, "band": args.band, "zonal_k": k,
                            "n_time": field.shape[0], "order": args.order,
                            "oscillatory_share": got["oscillatory_share"],
                            "transient_share": got["transient_share"],
                            "stationary_share": got["stationary_share"],
                            "dominant_period": got["dominant_period"],
                            "dominant_sigma": got["dominant_sigma"],
                            **{f"diurnal_{a}": b for a, b
                               in got["nearest"]["diurnal"].items()},
                            **{f"annual_{a}": b for a, b
                               in got["nearest"]["annual"].items()}})

    print(f"    {'IC':>6} {'k':>3} {'osc':>7} {'dom period':>11} "
          f"{'diurnal pole':>14} {'err':>7} {'sigma':>9} {'energy':>8}")
    for r in records:
        print(f"    {r['ic']:>6} {r['zonal_k']:>3} "
              f"{r['oscillatory_share']:>6.1%} {r['dominant_period']:>11.2f} "
              f"{r['diurnal_period']:>14.2f} {r['diurnal_error_pct']:>6.2f}% "
              f"{r['diurnal_sigma']:>+9.5f} {r['diurnal_energy_share']:>7.1%}")

    best = min((r for r in records if r["zonal_k"] == 1),
               key=lambda r: r["diurnal_error_pct"], default=None)
    if best:
        print(f"\n    documented diurnal period {DAY_STEPS} steps; recovered "
              f"{best['diurnal_period']:.2f} "
              f"({best['diurnal_error_pct']:.2f}% error) at zonal k=1, "
              f"sigma={best['diurnal_sigma']:+.5f}")
        print(f"    sigma near zero means neutrally stable: e-folding "
              f"{abs(1 / best['diurnal_sigma']):.0f} steps against a "
              f"{best['n_time']}-step record")

    annual_ok = [r for r in records if r["annual_error_pct"] < 20]
    print(f"\n    annual (period {YEAR_STEPS}): recovered within 20% in "
          f"{len(annual_ok)}/{len(records)} mode fits")
    print(f"    expected: the record holds "
          f"{records[0]['n_time'] / YEAR_STEPS:.0f} cycles of it, and the "
          f"limit above shows 3 cycles is not enough.")
    print("    ext12 measured the annual with an FFT, which needs only that the")
    print("    period divide the record. This is a limit of pole fitting, not a")
    print("    statement about the physics.")

    if args.gray_scott:
        print("\n=== Gray-Scott settled segments (self-organised, not forced) ===")
        gs = build_gray_scott_cache(args.gs_cache)
        gs_rows = analyse_gray_scott(gs, args.max_zonal + 2, args.order)
        print(f"    {'regime':>9} {'k':>3} {'osc':>7} {'dom period':>11} "
              f"{'fit sigma':>10} {'env sigma':>10} {'reliable':>9}")
        for r in gs_rows:
            print(f"    {r['regime']:>9} {r['k']:>3} "
                  f"{r['oscillatory_share']:>6.1%} {r['dominant_period']:>11.2f} "
                  f"{r['fit_sigma']:>+10.5f} {r['envelope_sigma']:>+10.5f} "
                  f"{str(r['sigma_reliable']):>9}")
        n_bad = sum(1 for r in gs_rows if not r["sigma_reliable"])
        print(f"\n    {n_bad}/{len(gs_rows)} regimes flagged unreliable: the fit "
              f"reports strong damping while the")
        print("    envelope is flat, which is model mismatch rather than physics. "
              "ext10 found a")
        print("    line at period 45.5 in spirals, so a bare '0% oscillatory' "
              "here would be wrong.")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / "ext18_pole_gray_scott.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gs_rows[0]))
            w.writeheader()
            w.writerows(gs_rows)
        print(f"wrote {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows_out in (("ext17_pole_zonal", records),
                           ("ext17_pole_resolution_limit", limit)):
        path = args.out_dir / f"{name}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
