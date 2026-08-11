"""Validation for scripts/forced_harmonics.py.

The planetswe measurement is a negative result -- known, exactly periodic
forcing accounts for only a few percent of the temporal variance -- so the
machinery has to be shown capable of finding a forced signal that *is* there.
The end-to-end tests inject a travelling wave of known amplitude at the diurnal
(frequency, wavenumber) cell and check the measured share comes back equal to
the injected one.

Both indexing traps hit while writing the script are pinned here as regressions:
colatitude-in-radians read as degrees, and int() truncation of fftfreq floats.

No network access.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "forced_harmonics.py"


def _load():
    spec = importlib.util.spec_from_file_location("forced_harmonics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["forced_harmonics"] = module
    spec.loader.exec_module(module)
    return module


fh = _load()

T = 3 * fh.YEAR_STEPS          # 3024, exactly three model years
NLAT, NLON = 8, 128
DIURNAL = T // fh.DAY_STEPS    # 126
ANNUAL = T // fh.YEAR_STEPS    # 3


# --------------------------------------------------------------------------
# coordinates and bin lookup: the two traps
# --------------------------------------------------------------------------


def test_colatitude_converts_to_latitude():
    theta = np.array([0.0, np.pi / 2, np.pi])           # N pole, equator, S pole
    lat = fh.colatitude_to_latitude(theta)
    assert lat == pytest.approx([90.0, 0.0, -90.0], abs=1e-9)


def test_colatitude_rejects_degrees():
    """Radians read as degrees selects the whole globe for every band, silently."""
    with pytest.raises(AssertionError):
        fh.colatitude_to_latitude(np.linspace(0, 180, 32))


def test_band_masks_select_distinct_rows():
    lat = fh.colatitude_to_latitude(np.linspace(np.pi, 0, 256))
    counts = {b: int(fh.band_mask(lat, *lim).sum()) for b, lim in fh.BANDS.items()}
    assert counts["global"] == 256
    assert 0 < counts["tropics"] < 256
    # the bug being guarded: every band coming out equal to the globe
    assert counts["tropics"] != counts["global"]
    assert counts["midlat_N"] > 0 and counts["polar_S"] > 0
    non_global = sum(v for b, v in counts.items() if b != "global")
    assert non_global == 256          # bands tile the sphere without overlap


def test_bin_index_is_exact_not_truncated():
    """int(fftfreq(n)*n) reads the neighbouring bin; this is what broke first."""
    freqs = np.fft.fftfreq(T, d=1.0) * T
    for target in (1, ANNUAL, DIURNAL, 2 * DIURNAL, -DIURNAL):
        idx = fh.bin_index(freqs, target)
        assert freqs[idx] == pytest.approx(target, abs=1e-9)
    # the specific failure: naive truncation is off by one at 126
    naive = {int(v): i for i, v in enumerate(freqs)}
    assert naive.get(DIURNAL) != fh.bin_index(freqs, DIURNAL)


def test_bin_index_rejects_a_frequency_with_no_exact_bin():
    freqs = np.fft.fftfreq(100, d=1.0) * 100
    with pytest.raises(AssertionError):
        fh.bin_index(freqs, 3000)


# --------------------------------------------------------------------------
# forced cell layout
# --------------------------------------------------------------------------


def test_forced_cells_land_on_the_documented_periods():
    cells = fh.forced_cells(T, NLON, n_annual=3, n_diurnal=2)
    assert cells[(ANNUAL, 0)] == "annual"
    assert cells[(2 * ANNUAL, 0)] == "annual x2"
    assert cells[(DIURNAL, -1)] == "diurnal"
    assert cells[(2 * DIURNAL, -2)] == "diurnal x2"
    # periods recovered from the bins are exactly 1008 and 24 steps
    assert T / ANNUAL == pytest.approx(fh.YEAR_STEPS)
    assert T / DIURNAL == pytest.approx(fh.DAY_STEPS)


def test_forced_cells_include_the_conjugate():
    cells = fh.forced_cells(T, NLON, n_annual=2, n_diurnal=1)
    for (omega, k) in list(cells):
        assert (-omega, -k) in cells


def test_annual_is_zonally_symmetric_and_diurnal_travels_westward():
    cells = fh.forced_cells(T, NLON, n_annual=2, n_diurnal=2)
    for (omega, k), label in cells.items():
        if label.startswith("annual"):
            assert k == 0                       # migrates north-south only
        else:
            m = int(label.split("x")[-1]) if "x" in label else 1
            assert k == -m * np.sign(omega)     # follows the sun, westward


# --------------------------------------------------------------------------
# end to end: inject a known forced fraction and recover it
# --------------------------------------------------------------------------


def _background(rng, amp=1.0):
    """Red-ish noise with no forcing at the diurnal or annual cells."""
    x = rng.normal(size=(T, NLAT, NLON))
    x = np.cumsum(x, axis=0)                    # random walk: broadband, red
    return amp * (x - x.mean(axis=0, keepdims=True))


def _travelling_wave(amp, omega, k, phase=0.0):
    t = np.arange(T)[:, None, None]
    lon = np.arange(NLON)[None, None, :]
    return amp * np.cos(2 * np.pi * (k * lon / NLON + omega * t / T) + phase)


@pytest.mark.parametrize("target", [0.02, 0.10, 0.40])
def test_injected_forced_share_is_recovered(target):
    """A diurnal wave carrying a known fraction of variance measures as that."""
    rng = np.random.default_rng(0)
    bg = _background(rng)
    wave = _travelling_wave(1.0, DIURNAL, -1)
    # scale the wave so it carries `target` of the total variance
    scale = np.sqrt(target / (1 - target) * bg.var() / wave.var())
    field = bg + scale * wave

    res = fh.analyse_band(field, "boxcar", n_annual=6, n_diurnal=4)
    assert res["diurnal_share"] == pytest.approx(target, rel=0.06)


def test_unforced_background_scores_near_chance():
    """With no forcing present, the forced cells hold about what chance gives."""
    rng = np.random.default_rng(1)
    res = fh.analyse_band(_background(rng), "boxcar", n_annual=6, n_diurnal=4)
    assert res["diurnal_share"] < 20 * res["chance_share"]
    assert res["forced_share"] < 0.10


def test_wave_in_the_wrong_direction_is_not_counted_as_forced():
    """An eastward wave at the diurnal frequency is not the forced response.

    This is what the (omega, k) isolation buys over a frequency-only spectrum:
    variability that merely shares the forcing period does not qualify.
    """
    rng = np.random.default_rng(2)
    bg = _background(rng)
    east = _travelling_wave(1.0, DIURNAL, +1)
    scale = np.sqrt(0.30 / 0.70 * bg.var() / east.var())
    res = fh.analyse_band(bg + scale * east, "boxcar", n_annual=6, n_diurnal=4)
    assert res["diurnal_share"] < 0.02


def test_annual_injection_is_recovered_at_k_zero():
    rng = np.random.default_rng(3)
    bg = _background(rng)
    seasonal = _travelling_wave(1.0, ANNUAL, 0)
    scale = np.sqrt(0.25 / 0.75 * bg.var() / seasonal.var())
    res = fh.analyse_band(bg + scale * seasonal, "boxcar", n_annual=6, n_diurnal=4)
    assert res["annual_share"] == pytest.approx(0.25, rel=0.10)


def test_boxcar_beats_hann_on_an_exact_period_record():
    """Why boxcar is primary here, unlike in ext10.

    The record is a whole number of forcing periods, so the line sits exactly on
    a bin and a taper only smears it into neighbours -- losing forced power that
    is really there.
    """
    rng = np.random.default_rng(4)
    bg = _background(rng)
    wave = _travelling_wave(1.0, DIURNAL, -1)
    scale = np.sqrt(0.25 / 0.75 * bg.var() / wave.var())
    field = bg + scale * wave
    box = fh.analyse_band(field, "boxcar", 6, 4)["diurnal_share"]
    han = fh.analyse_band(field, "hann", 6, 4)["diurnal_share"]
    assert box > han


# --------------------------------------------------------------------------
# phase locking
# --------------------------------------------------------------------------


def test_phase_lock_separates_forced_from_internal():
    """Forced -> resultant near 1; independent phases -> near 1/sqrt(n)."""
    rng = np.random.default_rng(5)
    n_ic = 4

    locked = [_background(rng) + 3 * _travelling_wave(1.0, DIURNAL, -1)
              for _ in range(n_ic)]
    got = fh.phase_lock(locked, DIURNAL, -1)
    assert got["resultant"] > 0.9

    free = [_background(rng)
            + 3 * _travelling_wave(1.0, DIURNAL, -1,
                                   phase=rng.uniform(0, 2 * np.pi))
            for _ in range(n_ic)]
    got = fh.phase_lock(free, DIURNAL, -1)
    assert got["resultant"] < 0.7
    assert got["chance_resultant"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------


def test_shares_are_a_partition():
    """Reported components must not overlap or exceed the whole."""
    rng = np.random.default_rng(6)
    field = _background(rng) + 2 * _travelling_wave(1.0, ANNUAL, 0)
    res = fh.analyse_band(field, "boxcar", 6, 4)
    total = (res["annual_share"] + res["diurnal_share"] + res["trend_share"]
             + res["top_internal_share"])
    assert 0 <= total <= 1.0 + 1e-9
    assert res["forced_share"] == pytest.approx(
        res["annual_share"] + res["diurnal_share"])


def test_record_must_span_whole_forcing_periods():
    """A ragged record would put the harmonics between bins."""
    cells = fh.forced_cells(1000, NLON, n_annual=6, n_diurnal=4)
    annual = [c for c, l in cells.items() if l.startswith("annual")]
    assert not annual        # 1000 is not a multiple of 1008; nothing exact
