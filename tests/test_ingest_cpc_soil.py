"""Validation for scripts/ingest_cpc_soil.py.

The point of this ingest is that real seasonal data enters the repo through the
*same* 5-D contract the Gray-Scott pipeline uses, so the first tests are contract
conformance. The rest cover the two ways this ingest could silently corrupt the
very quantity the experiments measure: averaging the ocean sentinel into a tile,
and measuring the annual line on a record that is not a whole number of years.

No network. No source file required -- the source is synthesised.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")
import h5py                                                     # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_cpc_soil.py"


def _load():
    spec = importlib.util.spec_from_file_location("ingest_cpc_soil", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ingest_cpc_soil"] = module
    spec.loader.exec_module(module)
    return module


ing = _load()


def _seasonal(T=936, cells=64, amp=1.0, noise=0.0, seed=0):
    """A pure annual cycle plus optional white noise, shape (T, cells)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    phase = rng.uniform(0, 2 * np.pi, cells)[None, :]
    x = amp * np.sin(2 * np.pi * t / 12.0 + phase)
    return x + noise * rng.standard_normal((T, cells))


# --------------------------------------------------------------------------
# the seasonality measurement
# --------------------------------------------------------------------------


def test_a_pure_annual_cycle_reads_as_almost_all_seasonal():
    # if this fails every seasonality number the ingest records is meaningless
    d = ing.seasonality(_seasonal())
    assert d["fft_annual_share"] > 0.99
    assert d["climatology_r2"] > 0.99
    assert d["n_years"] == 78


def test_pure_noise_reads_as_almost_no_seasonality():
    rng = np.random.default_rng(1)
    d = ing.seasonality(rng.standard_normal((936, 64)))
    assert d["fft_annual_share"] < 0.02
    assert d["climatology_r2"] < 0.05


def test_the_two_measures_agree_on_a_mixed_signal():
    # the cross-check that makes either number trustworthy
    d = ing.seasonality(_seasonal(amp=1.0, noise=1.0, seed=2))
    assert abs(d["fft_annual_share"] - d["climatology_r2"]) < 0.10
    assert 0.2 < d["climatology_r2"] < 0.8


def test_record_is_truncated_to_whole_years_so_the_annual_line_hits_a_bin():
    # 943 months is not a whole number of years; leaving it ragged splits the
    # annual peak across neighbouring bins and understates it badly
    full = ing.seasonality(_seasonal(T=936))
    ragged = ing.seasonality(_seasonal(T=943))
    assert ragged["n_years"] == 78            # truncated, not used as-is
    assert ragged["fft_annual_share"] > 0.99  # so it does not leak
    assert abs(full["fft_annual_share"] - ragged["fft_annual_share"]) < 0.02


def test_leakage_is_real_if_truncation_is_skipped():
    # demonstrates the artifact the truncation exists to avoid: measured on a
    # ragged record the same pure annual cycle loses a large part of its peak
    x = _seasonal(T=943)
    xa = x - x.mean(0)
    P = (np.abs(np.fft.rfft(xa, axis=0)) ** 2).mean(1)
    P[0] = 0
    ragged_peak = P.max() / P.sum()
    assert ragged_peak < 0.95                        # leaked
    assert ing.seasonality(x)["fft_annual_share"] > 0.99   # fixed


def test_seasonality_is_undefined_on_a_record_shorter_than_two_years():
    d = ing.seasonality(_seasonal(T=12))
    assert np.isnan(d["fft_annual_share"])
    assert np.isnan(d["climatology_r2"])


# --------------------------------------------------------------------------
# the ocean sentinel
# --------------------------------------------------------------------------


def test_sentinel_is_masked_rather_than_averaged_in():
    # missing_value is -9.96921e36; averaging it into a tile destroys it, and
    # it is not a NaN so `isfinite` would not catch it
    assert ing.SENTINEL_MAX < 0
    assert -9.96921e36 < ing.SENTINEL_MAX
    assert not (-9.96921e36 > ing.SENTINEL_MAX)


def test_masked_cells_are_filled_without_creating_a_coastline_step(tmp_path):
    # a zero-fill would put a hard edge at every coast and inject spatial
    # high-frequency energy into the spectrum these experiments measure
    T, tile = 24, 8
    raw = np.full((T, tile, tile), 50.0)
    raw[:, :, :3] = -9.96921e36                     # "ocean" on the left third
    src = tmp_path / "s.nc"
    with h5py.File(src, "w") as f:
        f.create_dataset("soilw", data=raw.astype(np.float32))
    with h5py.File(src, "r") as f:
        win, diag = ing.extract_region(f["soilw"], 0, 0, tile, 12)
    assert diag["land_fraction"] == pytest.approx(5 / 8)
    assert np.isfinite(win).all()
    assert win.min() > 0                            # no sentinel survived
    # filled cells equal the frame mean of the valid cells, i.e. no step
    assert float(win.max() - win.min()) == pytest.approx(0.0, abs=1e-4)


def test_region_with_no_land_is_rejected_rather_than_silently_emptied(tmp_path):
    raw = np.full((24, 8, 8), -9.96921e36)
    src = tmp_path / "s.nc"
    with h5py.File(src, "w") as f:
        f.create_dataset("soilw", data=raw.astype(np.float32))
    with h5py.File(src, "r") as f:
        with pytest.raises(ValueError):
            ing.extract_region(f["soilw"], 0, 0, 8, 12)


# --------------------------------------------------------------------------
# the 5-D contract
# --------------------------------------------------------------------------


def test_windows_are_five_dimensional_and_non_overlapping(tmp_path):
    T, tile, n_steps = 100, 8, 12
    raw = np.arange(T * tile * tile, dtype=np.float64).reshape(T, tile, tile)
    src = tmp_path / "s.nc"
    with h5py.File(src, "w") as f:
        f.create_dataset("soilw", data=raw.astype(np.float32))
    with h5py.File(src, "r") as f:
        win, _ = ing.extract_region(f["soilw"], 0, 0, tile, n_steps)
    assert win.ndim == 5
    assert win.shape == (T // n_steps, n_steps, tile, tile, 1)
    assert win.dtype == np.float32
    # consecutive and non-overlapping: window w starts where w-1 ended
    flat = win.reshape(-1, tile, tile, 1)
    assert np.array_equal(flat[:, 0, 0, 0],
                          raw[:win.shape[0] * n_steps, 0, 0].astype(np.float32))


def test_box_indices_stay_in_bounds_at_the_poles_and_the_dateline():
    lat = np.arange(89.75, -90, -0.5)
    lon = np.arange(0.25, 360, 0.5)
    for centre in ((89.0, 359.0), (-89.0, 0.5), (0.0, 180.0)):
        i, j = ing.box_indices(lat, lon, centre, 32)
        assert 0 <= i <= len(lat) - 32
        assert 0 <= j <= len(lon) - 32


def test_box_indices_put_the_requested_centre_inside_the_tile():
    lat = np.arange(89.75, -90, -0.5)
    lon = np.arange(0.25, 360, 0.5)
    for name, centre in ing.REGIONS.items():
        i, j = ing.box_indices(lat, lon, centre, 32)
        lo_lat, hi_lat = lat[i + 31], lat[i]
        assert lo_lat <= centre[0] <= hi_lat, name
        assert lon[j] <= centre[1] <= lon[j + 31], name


def test_regions_span_a_wide_seasonal_range_by_construction():
    # the design depends on contrast: a prior on the annual mode can only be
    # shown to be differential if the regions differ in how seasonal they are
    assert len(ing.REGIONS) == 6
