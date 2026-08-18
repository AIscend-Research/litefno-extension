"""Validation for scripts/fema_svi_equity.py.

The whole claim rests on one identity -- that ext23's manipulation incentive
equals the fitted elasticity -- so that is pinned first, across the family
rather than at a convenient point. The rest cover the regression's failure
modes: zeros on a log scale, a degenerate design, and a slope that must vanish
when the two variables are unrelated.

No network. All fixtures are synthetic.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fema_svi_equity.py"


def _load():
    spec = importlib.util.spec_from_file_location("fema_svi_equity", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["fema_svi_equity"] = m
    spec.loader.exec_module(m)
    return m


fx = _load()


# --------------------------------------------------------------------------
# the bridge to ext22 / ext23
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 4.0])
def test_manipulability_equals_the_fitted_elasticity(alpha):
    # ext23's incentive is |1-alpha|/alpha; the family's exponent is
    # beta=(1-alpha)/alpha. If these ever diverge the empirical claim is void.
    from litefno.allocation import allocation_exponent
    beta = allocation_exponent(alpha)
    assert abs(beta) == pytest.approx(abs(1 - alpha) / alpha, rel=1e-12)
    assert fx.implied(beta)["manipulability"] == pytest.approx(abs(beta), rel=1e-12)


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8, 1.5, 2.0, 4.0])
def test_alpha_round_trips_through_beta(alpha):
    from litefno.allocation import allocation_exponent
    beta = allocation_exponent(alpha)
    assert fx.implied(beta)["alpha"] == pytest.approx(alpha, rel=1e-9)


def test_envy_free_point_is_zero_slope_zero_manipulability_zero_fragility():
    # the fixed point the empirical estimate is compared against
    got = fx.implied(0.0)
    assert got["alpha"] == pytest.approx(1.0)
    assert got["manipulability"] == pytest.approx(0.0)
    assert got["fragility"] == pytest.approx(0.0)


def test_implied_is_undefined_where_the_family_is():
    # alpha = 1/(1+beta) is undefined at beta=-1 and negative below it
    for bad in (-1.0, -1.5, float("nan")):
        assert np.isnan(fx.implied(bad)["alpha"])


# --------------------------------------------------------------------------
# the regression
# --------------------------------------------------------------------------


def _power_law(beta, n=200, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    svi = rng.uniform(0.05, 0.95, n)
    pc = svi ** beta * np.exp(noise * rng.standard_normal(n))
    return svi, pc


@pytest.mark.parametrize("beta", [-2.0, -0.5, 0.5, 2.0])
def test_a_known_elasticity_is_recovered(beta):
    svi, pc = _power_law(beta)
    assert fx.fit_beta(svi, pc)["beta"] == pytest.approx(beta, abs=1e-8)
    assert fx.fit_beta(svi, pc)["r2"] == pytest.approx(1.0, abs=1e-8)


def test_exactly_flat_allocation_gives_zero_slope_and_undefined_r2():
    # beta = 0 with no noise is a perfectly constant response, so R2 is 0/0.
    # The slope is still zero, which is the envy-free point -- the quantity the
    # empirical estimate is compared against -- so it must not come back NaN.
    svi, pc = _power_law(0.0)
    got = fx.fit_beta(svi, pc)
    assert got["beta"] == pytest.approx(0.0, abs=1e-8)
    assert np.isnan(got["r2"])
    assert fx.implied(got["beta"])["alpha"] == pytest.approx(1.0)


def test_noise_lowers_r2_without_biasing_the_slope():
    svi, pc = _power_law(1.0, n=4000, noise=0.8, seed=1)
    got = fx.fit_beta(svi, pc)
    assert got["beta"] == pytest.approx(1.0, abs=0.1)
    assert got["r2"] < 0.6


def test_unrelated_variables_give_a_slope_near_zero():
    rng = np.random.default_rng(2)
    svi = rng.uniform(0.05, 0.95, 3000)
    pc = np.exp(rng.standard_normal(3000))
    assert abs(fx.fit_beta(svi, pc)["beta"]) < 0.15


def test_zero_awards_are_dropped_not_floored():
    # a zero is an absent observation on a log scale; flooring it with any
    # epsilon would let that choice set the answer
    svi, pc = _power_law(1.0, n=50)
    pc2 = pc.copy()
    pc2[:10] = 0.0
    assert fx.fit_beta(svi, pc2)["n"] == 40
    assert fx.fit_beta(svi, pc2)["beta"] == pytest.approx(1.0, abs=1e-8)


def test_too_few_counties_is_undefined_rather_than_fitted():
    svi, pc = _power_law(1.0, n=5)
    got = fx.fit_beta(svi, pc)
    assert got["n"] == 5
    assert np.isnan(got["beta"])


def test_constant_vulnerability_has_no_identifiable_slope():
    svi = np.full(40, 0.5)
    pc = np.linspace(1, 5, 40)
    assert np.isnan(fx.fit_beta(svi, pc)["beta"])


def test_standard_error_shrinks_with_sample_size():
    a = fx.fit_beta(*_power_law(1.0, n=40, noise=0.5, seed=3))["se"]
    b = fx.fit_beta(*_power_law(1.0, n=4000, noise=0.5, seed=3))["se"]
    assert b < a / 3


# --------------------------------------------------------------------------
# the permutation null
# --------------------------------------------------------------------------


def test_permutation_null_rejects_a_real_slope():
    svi, pc = _power_law(1.5, n=120, noise=0.3, seed=4)
    assert fx.permutation_null(svi, pc, 200, seed=0)["p_two_sided"] < 0.05


def test_permutation_null_does_not_reject_noise():
    rng = np.random.default_rng(5)
    svi = rng.uniform(0.05, 0.95, 120)
    pc = np.exp(rng.standard_normal(120))
    assert fx.permutation_null(svi, pc, 200, seed=0)["p_two_sided"] > 0.05


def test_permutation_null_is_centred_on_zero():
    svi, pc = _power_law(2.0, n=150, noise=0.4, seed=6)
    null = fx.permutation_null(svi, pc, 300, seed=0)
    assert abs(null["null_mean"]) < 0.2
    assert null["null_sd"] > 0
