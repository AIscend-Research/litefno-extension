r"""Real seasonal data through the repo's 5-D HDF5 contract (ext31 ingest)

Every measurement in this repository so far has run on a field with no seasonal
cycle to speak of. Gray-Scott's six regimes are mostly frozen patterns -- ext10
found four of them contain no temporal line at all -- and planetswe, the one
testbed with documented periodic forcing, puts only 5.4% of its temporal
variance at the forcing frequency (ext12). The harmonic prior was then tested
three times against that, and died twice
(see ``docs/harmonic_verdict.md``).

That is a weak test of a seasonal prior, because the data was never seasonal.
This script supplies data that is.

The dataset
-----------
NOAA CPC global monthly soil moisture: a 0.5-degree global grid, 1948-2026,
943 monthly steps of model-calculated soil moisture in mm. It is the same
physical variable as ERA5-Land soil moisture and needs no credentials, which
ERA5-Land (Copernicus CDS key) and MODIS NDVI (NASA Earthdata login) both do.

    https://psl.noaa.gov/data/gridded/data.cpcsoil.html

Why regions stand in for regimes
--------------------------------
Gray-Scott's "regimes" are (F, k) parameter pairs producing different dynamics.
The analogue here is climate: the seasonal share of soil-moisture variance runs
from about 11% in the boreal and mid-latitude zones to 66% in the monsoon belt,
so six named regions span a **6x range in exactly the quantity the harmonic
prior is supposed to exploit**.

That range is the point. It makes ext15's differential prediction testable on
data where the signal actually exists: a prior on the annual mode should help
the monsoon regions and do little in the boreal ones. On Gray-Scott the same
prediction was untestable in practice, because no regime had a strong line to
begin with.

Measuring seasonality without the leakage artifact
--------------------------------------------------
The seasonal share is computed two ways, and both are recorded per region.

``fft_annual_share`` needs care. 943 months is not a whole number of years, so
the annual frequency falls between FFT bins and its power leaks into
neighbours -- on a mid-latitude test box this reads 7.4% instead of 13.4%, an
81% understatement. The series is therefore truncated to whole years (936
months = 78 years) so the annual line lands exactly on a bin. ext12 hit the same
class of artifact and reported boxcar understating by a factor of 2-6.

``climatology_r2`` is leakage-free by construction: fit the month-of-year mean
per cell and take the fraction of variance it explains. On the same test box the
two agree (13.4% and 13.1%), which is the cross-check that makes either
trustworthy.

Ocean is a sentinel, not a NaN
------------------------------
Missing cells carry ``-9.96921e36``. Averaging that in silently destroys a tile,
so cells are masked on the sentinel rather than on ``isfinite``. Masked cells are
filled with the per-frame mean of the valid cells in the same tile -- not with
zero, which would introduce an artificial step at every coastline and inject
spatial high-frequency energy into precisely the spectrum these experiments
measure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

SENTINEL_MAX = -1e30          # missing_value is -9.96921e+36
URL = "https://downloads.psl.noaa.gov/Datasets/cpcsoil/soilw.mon.mean.nc"

# Six regions spanning the seasonal range, mirroring Gray-Scott's six regimes.
# Centres are (lat, lon-east); each becomes a `tile` x `tile` cell box.
REGIONS = {
    "india_monsoon":   (20.0,  78.0),
    "sahel_wafrica":   (10.0,   5.0),
    "amazon_brazil":   (-8.0, 300.0),
    "la_plata":       (-28.0, 300.0),
    "us_midwest":      (40.0, 262.0),
    "eurasia_boreal":  (58.0,  60.0),
}


def open_source(path: Path):
    import h5py
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n\nFetch it once (255 MB, no credentials):\n"
            f"  curl -L -o {path} {URL}")
    return h5py.File(path, "r")


def box_indices(lat: np.ndarray, lon: np.ndarray, centre: tuple, tile: int):
    """Top-left index of the `tile`x`tile` box centred on (lat, lon).

    Latitude descends (89.75 -> -89.75) and longitude ascends (0.25 -> 359.75),
    so the top-left corner is the northern edge and the western edge. At 0.5
    degrees a 32-cell tile spans 16 degrees, hence the 8-degree half-width.
    """
    half_deg = tile * 0.5 / 2.0
    i = int(np.argmin(np.abs(lat - (centre[0] + half_deg))))
    j = int(np.argmin(np.abs(lon - (centre[1] - half_deg))))
    i = max(0, min(i, len(lat) - tile))
    j = max(0, min(j, len(lon) - tile))
    return i, j


def seasonality(series: np.ndarray) -> dict:
    """Seasonal share of temporal variance, measured two independent ways.

    ``series`` is (T, cells) of valid land cells. Returns both the FFT annual
    line (computed on whole years so the line sits on a bin) and the
    month-of-year climatology R2, which needs no windowing at all.
    """
    T = (series.shape[0] // 12) * 12
    x = series[:T]
    if T < 24 or x.shape[1] == 0:
        return {"fft_annual_share": float("nan"),
                "fft_annual_plus_harmonics": float("nan"),
                "climatology_r2": float("nan"), "n_years": T // 12}

    xa = x - x.mean(0)
    P = (np.abs(np.fft.rfft(xa, axis=0)) ** 2).mean(1)
    P[0] = 0.0
    tot = P.sum()
    ann = T // 12                                  # exact annual bin
    harm = sorted({ann * m for m in (1, 2, 3, 4) if ann * m < len(P)})

    clim = x.reshape(T // 12, 12, -1).mean(0)
    fit = np.tile(clim, (T // 12, 1))
    ss_tot = ((x - x.mean(0)) ** 2).sum(0)
    ss_res = ((x - fit) ** 2).sum(0)
    r2 = 1.0 - np.divide(ss_res, ss_tot, out=np.ones_like(ss_tot),
                         where=ss_tot > 0)
    return {
        "fft_annual_share": float(P[ann] / tot) if tot > 0 else float("nan"),
        "fft_annual_plus_harmonics": float(P[harm].sum() / tot) if tot > 0 else float("nan"),
        "climatology_r2": float(np.mean(r2)),
        "n_years": T // 12,
    }


def extract_region(soilw, i: int, j: int, tile: int, n_steps: int):
    """(n_windows, n_steps, tile, tile, 1) plus the region's diagnostics.

    Windows are consecutive and non-overlapping, so no frame appears in two
    trajectories and a split by window is a split in time.
    """
    raw = np.asarray(soilw[:, i:i + tile, j:j + tile], dtype=np.float64)
    valid = raw > SENTINEL_MAX
    land = valid.all(0)                              # cells with data at every step
    land_fraction = float(land.mean())
    if land.sum() == 0:
        raise ValueError("region has no land cells")

    # fill masked cells with the per-frame mean of that frame's valid cells,
    # which avoids the artificial coastline step a zero-fill would create
    filled = raw.copy()
    flat = filled.reshape(filled.shape[0], -1)
    lm = land.ravel()
    frame_mean = flat[:, lm].mean(1)
    flat[:, ~lm] = frame_mean[:, None]
    filled = flat.reshape(raw.shape)

    diag = seasonality(raw.reshape(raw.shape[0], -1)[:, lm])
    diag["land_fraction"] = land_fraction

    n_win = filled.shape[0] // n_steps
    cut = n_win * n_steps
    win = filled[:cut].reshape(n_win, n_steps, tile, tile, 1)
    return win.astype(np.float32), diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/raw/cpc_soilw.nc"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/processed/cpc_soil"))
    ap.add_argument("--tile", type=int, default=32,
                    help="cells per side; 32 at 0.5deg is a 16-degree box")
    ap.add_argument("--n-steps", type=int, default=60,
                    help="months per trajectory (60 = 5 years, matching the "
                         "Gray-Scott pipeline's max_steps)")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--regions", nargs="*", default=list(REGIONS))
    args = ap.parse_args()

    import h5py
    f = open_source(args.source)
    lat, lon, soilw = f["lat"][:], f["lon"][:], f["soilw"]
    t = f["time"][:]
    epoch = dt.date(1800, 1, 1)
    span = (epoch + dt.timedelta(days=float(t[0])),
            epoch + dt.timedelta(days=float(t[-1])))
    print(f"source {args.source}  {soilw.shape}  {span[0]} .. {span[1]}")

    per_region, splits = {}, {"train": [], "valid": [], "test": []}
    buckets = {"train": [], "valid": [], "test": []}

    for name in args.regions:
        if name not in REGIONS:
            raise SystemExit(f"unknown region {name}; known: {list(REGIONS)}")
        i, j = box_indices(lat, lon, REGIONS[name], args.tile)
        win, diag = extract_region(soilw, i, j, args.tile, args.n_steps)
        n = len(win)
        n_tr = int(round(n * args.train_frac))
        n_va = int(round(n * args.valid_frac))
        n_tr = max(1, min(n_tr, n - 2))
        n_va = max(1, min(n_va, n - n_tr - 1))
        bounds = {"train": (0, n_tr),
                  "valid": (n_tr, n_tr + n_va),
                  "test": (n_tr + n_va, n)}
        for split, (a, b) in bounds.items():
            buckets[split].append(win[a:b])
            splits[split].append({
                "regime": name,
                "file": f"{args.source.name}:lat[{i}:{i+args.tile}],lon[{j}:{j+args.tile}]",
                "trajectories": list(range(a, b)),
                "available": n,
            })
        diag.update({"lat_index": i, "lon_index": j,
                     "lat_north": float(lat[i]), "lon_west": float(lon[j]),
                     "n_windows": n})
        per_region[name] = diag
        print(f"  {name:>16s}  land {100*diag['land_fraction']:3.0f}%  "
              f"annual {100*diag['fft_annual_share']:5.1f}%  "
              f"clim R2 {100*diag['climatology_r2']:5.1f}%  "
              f"{n} windows -> {n_tr}/{n_va}/{n - n_tr - n_va}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        arr = np.concatenate(buckets[split]).astype(np.float32)
        path = args.out_dir / f"{split}.h5"
        with h5py.File(path, "w") as h:
            h.create_dataset("data", data=arr, compression="gzip",
                             compression_opts=4)
        print(f"wrote {path} {arr.shape} ({arr.nbytes/1e6:.1f} MB)")

    manifest = {
        "dataset": "cpc_soil_moisture",
        "source": URL,
        "max_steps": args.n_steps,
        "downsample_factor": 1,
        "fields": ["soilw"],
        "units": "mm",
        "grid_degrees": 0.5,
        "time_span": [str(span[0]), str(span[1])],
        "split_is_temporal": True,
        "region_diagnostics": per_region,
        "splits": {k: {"files": v} for k, v in splits.items()},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.out_dir/'manifest.json'}")


if __name__ == "__main__":
    sys.exit(main())
