"""Validation for litefno.poles.

The claim is that pole-residue analysis separates a sustained oscillation from a
decaying transient, which a power spectrum cannot: both deposit power in the
same bin. So the tests are built around cases where the two are deliberately
confusable, and around the resolution limit that says when the answer should be
believed at all.

Two bugs found on real data are pinned here as regressions: a fixed frequency
floor that silently excluded every period above 1000 steps, and a constant
offset taking a pole of its own and 99% of the energy.

No network access.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from litefno.poles import (  # noqa: E402
    analyse_field, analyse_series, classify_poles, fit_ar_poles, pole_residues,
    spatial_mode_series)

N = 400
T = np.arange(N)


def _dominant_period(result) -> float:
    return result["dominant_period"]


# --------------------------------------------------------------------------
# the distinction a spectrum cannot make
# --------------------------------------------------------------------------


def test_sustained_oscillation_is_oscillatory():
    got = analyse_series(np.cos(2 * np.pi * T / 20), order=12)
    assert got["oscillatory_share"] > 0.99
    assert _dominant_period(got) == pytest.approx(20.0, rel=1e-3)
    assert abs(got["dominant_sigma"]) < 1e-3          # neutrally stable


def test_decaying_oscillation_is_transient_not_oscillatory():
    """Same frequency, same spectral bin, opposite verdict.

    This is the whole point: a power spectrum puts a ringing mode and a dying
    one in the same place, and the pole's magnitude is what separates them.
    """
    decaying = np.cos(2 * np.pi * T / 20) * np.exp(-T / 40)
    got = analyse_series(decaying, order=12)
    assert got["transient_share"] > 0.9
    assert got["oscillatory_share"] < 0.1


def test_the_two_are_indistinguishable_by_spectrum():
    """Guard the premise, so the test above is not proving something trivial."""
    sustained = np.cos(2 * np.pi * T / 20)
    decaying = sustained * np.exp(-T / 40)
    peak = lambda x: int(np.argmax(np.abs(np.fft.rfft(x - x.mean()))))  # noqa: E731
    assert peak(sustained) == peak(decaying), "premise broken: peaks differ"


def test_pure_decay_is_transient():
    """Decay is transient; the residual stationary share is the removed mean.

    analyse_series subtracts the temporal mean, which turns a pure decay into a
    decay minus a constant -- so a stationary pole legitimately picks up part of
    the energy. What must not happen is it being called oscillatory.
    """
    got = analyse_series(np.exp(-T / 50.0), order=12)
    assert got["transient_share"] > 0.7
    assert got["oscillatory_share"] < 0.05


def test_a_constant_is_stationary_not_oscillatory():
    got = analyse_series(np.ones(N) * 3.0 + 1e-9 * T, order=8)
    assert got["oscillatory_share"] < 0.05


def test_white_noise_is_not_reported_as_oscillatory():
    rng = np.random.default_rng(0)
    got = analyse_series(rng.normal(size=N), order=12)
    assert got["oscillatory_share"] < 0.2


def test_two_oscillations_both_appear():
    x = np.cos(2 * np.pi * T / 20) + 0.7 * np.cos(2 * np.pi * T / 7)
    got = analyse_series(x, order=16)
    periods = np.where(got["freq"] > 1e-12,
                       1 / np.maximum(got["freq"], 1e-12), np.inf)
    osc = periods[got["labels"] == "oscillatory"]
    assert any(abs(p - 20) < 0.5 for p in osc)
    assert any(abs(p - 7) < 0.5 for p in osc)


# --------------------------------------------------------------------------
# residue weighting
# --------------------------------------------------------------------------


def test_energy_uses_the_contribution_not_the_residue():
    """Equal amplitudes, wildly unequal contributions over the record.

    A pole at 0.2 has died by the fourth step; one at 0.999 is still going at
    step 400. Ranking poles by |residue| would call these equally important.
    """
    poles = np.array([0.999, 0.2])
    x = poles[0] ** T + poles[1] ** T
    residues, energy = pole_residues(x, poles)
    assert abs(residues[0]) == pytest.approx(abs(residues[1]), rel=0.05)
    assert energy[0] > 100 * energy[1]


def test_negligible_poles_cannot_outvote_the_signal():
    """An over-ordered fit invents poles; energy weighting must ignore them."""
    x = np.cos(2 * np.pi * T / 25)
    high = analyse_series(x, order=24)
    assert high["oscillatory_share"] > 0.95
    assert _dominant_period(high) == pytest.approx(25.0, rel=1e-2)


# --------------------------------------------------------------------------
# the two bugs found on real data
# --------------------------------------------------------------------------


def test_frequency_floor_scales_with_record_length():
    """Regression: a fixed floor of 1e-3 excluded every period above 1000.

    planetswe's documented annual forcing is 1008 steps, so the old default
    labelled it "stationary" on the one dataset with a known answer.
    """
    poles = np.array([np.exp(2j * np.pi / 1008), np.exp(-2j * np.pi / 1008)])
    energy = np.ones(2)
    fixed = classify_poles(poles, energy, min_cycles_per_step=1e-3)
    assert set(fixed["labels"]) == {"stationary"}          # the old behaviour

    scaled = classify_poles(poles, energy, n_time=3024)
    assert set(scaled["labels"]) == {"oscillatory"}        # 3 cycles, resolved


def test_frequency_floor_still_rejects_an_unresolvable_period():
    """Two cycles is the floor; one is not a cycle."""
    poles = np.array([np.exp(2j * np.pi / 4000), np.exp(-2j * np.pi / 4000)])
    got = classify_poles(poles, np.ones(2), n_time=3024)
    assert set(got["labels"]) == {"stationary"}


def test_constant_offset_does_not_dominate_the_classification():
    """Regression: on planetswe's zonal mean the offset took 99% of the energy."""
    x = 1000.0 + np.cos(2 * np.pi * T / 20)
    got = analyse_series(x, order=12)
    assert got["oscillatory_share"] > 0.9, got["stationary_share"]


# --------------------------------------------------------------------------
# resolution limit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("period,cycles,tol", [(24, 126, 0.02), (100, 30, 0.05)])
def test_period_is_recovered_when_there_are_enough_cycles(period, cycles, tol):
    n = 3024
    t = np.arange(n)
    got = analyse_series(np.sin(2 * np.pi * t / period), order=24)
    assert _dominant_period(got) == pytest.approx(period, rel=tol)


def test_too_few_cycles_is_not_recovered():
    """Three cycles in the record is beyond the method, and must not be trusted.

    This is why planetswe's annual forcing is reported as not found rather than
    as a null result about the physics: the record has 3 cycles of it.
    """
    n = 3024
    t = np.arange(n)
    rng = np.random.default_rng(1)
    noise = np.cumsum(rng.normal(size=n))
    x = np.sin(2 * np.pi * t / 1008) + 0.5 * noise / noise.std()
    got = analyse_series(x, order=24)
    periods = np.where(got["freq"] > 1e-12,
                       1 / np.maximum(got["freq"], 1e-12), np.inf)
    nearest = periods[int(np.argmin(np.abs(periods - 1008)))]
    assert abs(nearest - 1008) / 1008 > 0.2, "unexpectedly resolved"


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------


def test_fit_rejects_a_series_shorter_than_the_order():
    with pytest.raises(ValueError):
        fit_ar_poles(np.arange(5.0), order=12)


def test_spatial_mode_series_orders_by_wavenumber_and_matches_the_field():
    rng = np.random.default_rng(2)
    field = rng.normal(size=(30, 16, 16))
    series, radii = spatial_mode_series(field, max_mode=4)
    assert (np.diff(radii) >= 0).all(), "modes are not ordered by |k|"
    assert radii.max() <= 4
    assert series.shape[0] == 30
    # the k=0 mode is the spatial mean, up to the fft normalisation
    dc = series[:, radii == 0][:, 0]
    assert np.allclose(dc.real / (16 * 16), field.mean(axis=(1, 2)))


def test_analyse_field_skips_negligible_modes():
    """Fitting poles to numerical noise yields confident, meaningless poles."""
    t = np.arange(200)
    field = np.zeros((200, 16, 16))
    field += np.cos(2 * np.pi * t / 20)[:, None, None]     # only a DC oscillation
    got = analyse_field(field, max_mode=4, order=12, min_energy_frac=1e-3)
    assert len(got) >= 1
    assert all(g["energy_weight"] >= 1e-3 for g in got)
    dc = [g for g in got if g["k"] == 0][0]
    assert dc["dominant_period"] == pytest.approx(20.0, rel=1e-2)


# --------------------------------------------------------------------------
# the envelope cross-check
# --------------------------------------------------------------------------


def test_envelope_is_accurate_near_neutrality_where_it_is_needed():
    """Accurate for slow decay; biased low for fast decay, which is harmless.

    The envelope exists to check whether a mode is near-neutral. Measured over
    a 400-step record: 1% error at sigma=-0.001 and 1.4% at -0.005, rising to
    34% at -0.02 where the smoothing window spans a large fraction of an
    e-folding. That bias cannot change a label -- -0.013 and -0.020 are both far
    outside any neutral band -- so it is bounded loosely here rather than tuned
    away.
    """
    from litefno.poles import envelope_sigma
    for tau, tol in ((1000.0, 0.05), (200.0, 0.05)):
        x = np.cos(2 * np.pi * T / 20) * np.exp(-T / tau)
        assert envelope_sigma(x) == pytest.approx(-1 / tau, rel=tol)
    fast = np.cos(2 * np.pi * T / 20) * np.exp(-T / 50.0)
    assert -0.03 < envelope_sigma(fast) < -0.005      # right order, still damped


def test_envelope_is_flat_for_a_sustained_oscillation():
    from litefno.poles import envelope_sigma
    assert abs(envelope_sigma(np.cos(2 * np.pi * T / 20))) < 1e-4


def test_reliable_when_the_model_fits():
    """Clean damped and undamped oscillations: both estimates agree."""
    for x in (np.cos(2 * np.pi * T / 20),
              np.cos(2 * np.pi * T / 20) * np.exp(-T / 200)):
        got = analyse_series(x, order=12)
        assert got["sigma_reliable"] is True


def test_unreliable_when_the_phase_wanders():
    """A constant-frequency model cannot represent phase wander.

    It buys the misfit with damping, so the fit reports a strongly decaying
    mode while the envelope is flat -- which is what happens on Gray-Scott's
    self-organised patterns, and why their 'not oscillatory' verdict must be
    reported as unreliable rather than as a finding about the physics.
    """
    rng = np.random.default_rng(3)
    n = 500
    t = np.arange(n)
    phase = np.cumsum(rng.normal(scale=0.25, size=n))   # random-walk phase
    x = np.cos(2 * np.pi * t / 20 + phase)              # amplitude stays 1
    got = analyse_series(x, order=16)
    from litefno.poles import envelope_sigma
    assert abs(envelope_sigma(x)) < 2e-3                # envelope really is flat
    assert got["fit_sigma"] < -5e-3                     # the fit says otherwise
    assert got["sigma_reliable"] is False


def test_reliability_compares_labels_not_magnitudes():
    """Two tiny rates of opposite sign are the same call, not a disagreement."""
    x = np.cos(2 * np.pi * T / 20) * np.exp(-T / 100000)
    got = analyse_series(x, order=12)
    assert got["sigma_reliable"] is True
