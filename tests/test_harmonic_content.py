"""Validation for scripts/harmonic_content.py on synthetic series.

The point of these tests is the claim the analysis rests on: a high
low-frequency share is *not* evidence of harmonic structure, and the AR(1) null
is what tells the two apart. Each case below has a known answer, so a regression
in the normalisation or the null shows up as a failed assertion rather than as a
plausible-looking number in a CSV.

No network access — these run in CI alongside the rest of the suite.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "harmonic_content.py"


def _load():
    spec = importlib.util.spec_from_file_location("harmonic_content", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["harmonic_content"] = module
    spec.loader.exec_module(module)
    return module


hc = _load()

T = 1000
N = 400


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(T, N))

    def ar1(phi):
        x = np.zeros_like(noise)
        for t in range(1, T):
            x[t] = phi * x[t - 1] + noise[t]
        return x

    phase = rng.uniform(0, 2 * np.pi, size=(1, N))
    sine = np.sin(2 * np.pi * np.arange(T)[:, None] / 20 + phase)
    return {
        "white": noise,
        "ar1_0.9": ar1(0.9),
        "ar1_0.99": ar1(0.99),
        "sine_white": sine + rng.normal(size=(T, N)),
        "sine_red": 2.0 * sine + ar1(0.9),
        "ramp": np.linspace(0, 1, T)[:, None] * rng.normal(size=(1, N)),
    }


def test_parseval_boxcar_is_exact(series):
    """The boxcar spectrum sums to the summed per-series variance, exactly.

    This is the identity analyse_segment asserts at runtime.
    """
    for name, x in series.items():
        power = hc.temporal_power(x, "boxcar")
        expected = (x - x.mean(axis=0)).var(axis=0).sum()
        assert power.sum() == pytest.approx(expected, rel=1e-9), name


def test_parseval_hann_holds_only_for_stationary_input(series):
    """Hann conserves power for a stationary process, and only for one.

    The 1/sqrt(mean(w^2)) normalisation matches the *expected* power of a
    stationary series, so it is not an exact identity and cannot be used as the
    runtime correctness check -- that is why analyse_segment asserts on boxcar.
    """
    for name in ("white", "ar1_0.9", "sine_white", "sine_red"):
        x = series[name]
        power = hc.temporal_power(x, "hann")
        expected = (x - x.mean(axis=0)).var(axis=0).sum()
        assert power.sum() == pytest.approx(expected, rel=1e-2), name


def test_hann_suppresses_a_transient(series):
    """The taper discards edge-carried variance, by design.

    A ramp keeps all of its variance at the window edges, so Hann drops most of
    its power (~76%); a stationary series loses almost none. This is the
    mechanism behind reporting both windows -- a large boxcar/Hann gap in a
    segment is a signature of transient dominance, not of harmonic content.
    """
    def loss(name):
        x = series[name]
        expected = (x - x.mean(axis=0)).var(axis=0).sum()
        return 1 - hc.temporal_power(x, "hann").sum() / expected

    assert loss("ramp") > 0.5
    assert loss("white") < 0.01
    assert loss("ramp") > 100 * loss("ar1_0.9")


def test_parseval_weights_odd_and_even():
    for n in (60, 61, 1000, 1001):
        rng = np.random.default_rng(n)
        x = rng.normal(size=(n, 5))
        power = hc.temporal_power(x, "boxcar")
        assert len(power) == n // 2 + 1
        assert power.sum() == pytest.approx(x.var(axis=0).sum(), rel=1e-9)


def test_white_noise_is_flat(series):
    """A flat spectrum puts exactly the band's width of variance in the band."""
    m = hc.band_metrics(hc.temporal_power(series["white"], "hann"))
    assert m["low_share"] == pytest.approx(hc.LOW_EDGE, abs=0.02)
    assert m["high_share"] == pytest.approx(1 - hc.HIGH_EDGE, abs=0.02)


def test_red_noise_is_low_frequency_but_not_harmonic(series):
    """The central claim: low-frequency share is not evidence of a harmonic."""
    for name, floor in [("ar1_0.9", 0.6), ("ar1_0.99", 0.9)]:
        power = hc.temporal_power(series[name], "hann")
        band = hc.band_metrics(power)
        null = hc.harmonic_excess(power, T)
        assert band["low_share"] > floor, name          # looks low-frequency
        assert null["peak_over_null"] < 2.0, name       # but has no line
        assert null["excess_share"] < 0.10, name


def test_ar1_phi_is_recovered(series):
    for name, phi in [("ar1_0.9", 0.9), ("ar1_0.99", 0.99)]:
        got = hc.harmonic_excess(hc.temporal_power(series[name], "boxcar"), T)
        assert got["ar1_phi"] == pytest.approx(phi, abs=0.03), name


def test_harmonic_is_detected_at_the_right_frequency(series):
    """A real line is found even when buried in red noise."""
    for name in ("sine_white", "sine_red"):
        power = hc.temporal_power(series[name], "hann")
        got = hc.harmonic_excess(power, T)
        assert got["peak_bin"] == T // 20, name          # period 20 -> bin 50
        assert got["peak_over_null"] > 20.0, name
        assert got["excess_share"] > 0.20, name


def test_line_beats_red_noise_by_an_order_of_magnitude(series):
    """peak_over_null separates the two families; low_share does not."""
    def peak(name):
        return hc.harmonic_excess(hc.temporal_power(series[name], "hann"), T)[
            "peak_over_null"]

    def low(name):
        return hc.band_metrics(hc.temporal_power(series[name], "hann"))["low_share"]

    assert peak("sine_red") > 10 * peak("ar1_0.9")
    # ...while the low-frequency share of the two is nearly identical
    assert abs(low("sine_red") - low("ar1_0.9")) < 0.05


def test_ramp_has_no_line(series):
    """A monotone transient is 100% low-frequency and still not harmonic.

    It does score a high excess_share, because a ramp's spectrum is steeper than
    any AR(1) the null can fit. That is why peak_over_null, not excess_share, is
    the discriminator.
    """
    power = hc.temporal_power(series["ramp"], "hann")
    assert hc.band_metrics(power)["low_share"] > 0.99
    assert hc.harmonic_excess(power, T)["peak_over_null"] < 2.0


def test_spatial_parseval_and_binning():
    """Radial and box binning must both conserve the per-frame spatial variance."""
    rng = np.random.default_rng(1)
    u = rng.normal(size=(8, 32, 32))
    radial, box = hc.spatial_power(u)
    ref = (u - u.mean(axis=(1, 2), keepdims=True)).var(axis=(1, 2)).sum()
    assert radial.sum() == pytest.approx(ref, rel=1e-9)
    assert box.sum() == pytest.approx(ref, rel=1e-9)


def test_box_binning_matches_fno_truncation():
    """box[:m+1] must equal the energy an n_modes=(m, m) truncation keeps."""
    rng = np.random.default_rng(2)
    u = rng.normal(size=(4, 32, 32))
    _, box = hc.spatial_power(u)

    frames = u - u.mean(axis=(1, 2), keepdims=True)
    spec = np.fft.fft2(frames, axes=(1, 2))
    k = np.fft.fftfreq(32, d=1 / 32)
    cheb = np.maximum(np.abs(k[:, None]), np.abs(k[None, :]))
    for m in (4, 8, 12, 16):
        kept = (np.abs(spec) ** 2).sum(axis=0)[cheb <= m].sum() / 32 ** 4
        assert box[: m + 1].sum() == pytest.approx(kept, rel=1e-9), m


def test_single_frequency_lands_in_one_bin():
    """Sanity on the frequency axis itself: an exact-period sine is one bin."""
    n = 128
    t = np.arange(n)[:, None]
    x = np.sin(2 * np.pi * 8 * t / n)          # exactly 8 cycles in the window
    power = hc.temporal_power(x, "boxcar")
    assert np.argmax(power) == 8
    assert power[8] / power.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("tag", ["gray_scott", "trl"])
def test_committed_csvs_are_self_consistent(tag):
    """The per-bin spectrum CSV must reproduce the summary CSV's band shares.

    These are written by separate code paths from the same arrays, so a
    disagreement means one of them drifted. Also exercises the --replot
    reconstruction, which is what regenerates figures without network access.
    """
    out_dir = SCRIPT.parents[1] / "results" / "extensions"
    if not (out_dir / f"ext10_harmonic_summary_{tag}.csv").exists():
        pytest.skip(f"{tag} results not present")

    fam = hc.FAMILIES["gray_scott" if tag == "gray_scott" else "trl"]
    rows, summary = hc.rows_from_csv(out_dir, tag, fam)
    assert rows and summary

    by_key = {(r["scenario"], r["field"], r["segment"]): r for r in summary}
    checked = 0
    for r in rows:
        for seg in r["segments"]:
            entry = r["per_traj"][0][seg]
            for win in ("boxcar", "hann"):
                got = hc.band_metrics(entry["spec"][win])
                want = by_key[(r["scenario"], r["field"], seg)]
                assert got["low_share"] == pytest.approx(
                    want[f"{win}_low_share"], abs=1e-9)
                assert got["high_share"] == pytest.approx(
                    want[f"{win}_high_share"], abs=1e-9)
                checked += 1
    assert checked > 0


def test_segments_cover_the_expected_windows():
    assert hc.segments_for(50) == {"full": slice(None)}
    assert set(hc.segments_for(100)) == {"full", "train"}
    assert set(hc.segments_for(1001)) == {"full", "train", "settled"}
    assert hc.segments_for(1001)["train"] == slice(0, hc.TRAIN_STEPS)
    assert hc.segments_for(1001)["settled"] == slice(500, None)
