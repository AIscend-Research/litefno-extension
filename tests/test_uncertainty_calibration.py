"""Validation for scripts/uncertainty_calibration.py.

A calibration metric is only worth reporting if it returns ~0 on a distribution
that really is calibrated and something large on one that is not. Those two
anchors are the first tests here -- without them ECE is an unfalsifiable number.
The rest cover the degenerate cases (zero sigma, one bin), the scale fit, and
the property that makes this experiment distinct from ext28: calibration must be
sensitive to a rescaling that leaves ranking untouched.

Training is not exercised. No network access.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")
import torch                                                    # noqa: E402

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts"
          / "uncertainty_calibration.py")


def _load():
    spec = importlib.util.spec_from_file_location("uncertainty_calibration",
                                                  SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["uncertainty_calibration"] = module
    spec.loader.exec_module(module)
    return module


uc = _load()


def _calibrated(n=200_000, sigma=0.3, seed=0):
    """Residuals drawn from exactly the law the model claims."""
    rng = np.random.default_rng(seed)
    s = np.full(n, sigma)
    return rng.normal(0.0, sigma, n), s


# --------------------------------------------------------------------------
# the two anchors
# --------------------------------------------------------------------------


def test_a_genuinely_calibrated_predictor_scores_near_zero():
    # if this fails, every ECE the script reports is meaningless
    r, s = _calibrated()
    ece, mce = uc.ece_mce(uc.coverage_curve(r, s))
    assert ece < 0.01
    assert mce < 0.02


def test_an_overconfident_predictor_is_caught():
    # claims sigma/4, so intervals are 4x too narrow and coverage collapses
    r, s = _calibrated()
    ece, mce = uc.ece_mce(uc.coverage_curve(r, s / 4.0))
    assert ece > 0.25
    curve = uc.coverage_curve(r, s / 4.0)
    # every level should be under-covered, not a mix
    assert all(c["signed_gap"] < 0 for c in curve)


def test_an_underconfident_predictor_is_caught_and_signed_the_other_way():
    r, s = _calibrated()
    curve = uc.coverage_curve(r, s * 4.0)
    ece, _ = uc.ece_mce(curve)
    assert ece > 0.25
    assert all(c["signed_gap"] > 0 for c in curve)


def test_heteroscedastic_but_calibrated_still_scores_near_zero():
    # a varying sigma that is nonetheless correct everywhere: calibration is
    # about matching, not about being constant
    rng = np.random.default_rng(1)
    s = rng.uniform(0.05, 1.0, 200_000)
    r = rng.normal(0.0, 1.0, 200_000) * s
    ece, _ = uc.ece_mce(uc.coverage_curve(r, s))
    assert ece < 0.01


# --------------------------------------------------------------------------
# the property that separates this from ext28
# --------------------------------------------------------------------------


def test_calibration_is_sensitive_to_scale_that_leaves_ranking_unchanged():
    # ext28's ranking metrics are invariant to a positive rescaling of the
    # confidence signal; calibration must not be, or this experiment measures
    # nothing new
    rng = np.random.default_rng(2)
    s = rng.uniform(0.05, 1.0, 100_000)
    r = rng.normal(0.0, 1.0, 100_000) * s
    honest, _ = uc.ece_mce(uc.coverage_curve(r, s))
    shrunk, _ = uc.ece_mce(uc.coverage_curve(r, s * 0.2))
    # identical ordering of sigma, wildly different calibration
    assert np.array_equal(np.argsort(s), np.argsort(s * 0.2))
    assert shrunk > honest + 0.2


# --------------------------------------------------------------------------
# degenerate cases
# --------------------------------------------------------------------------


def test_zero_sigma_counts_as_covered_only_when_the_residual_is_zero():
    # claimed certainty is right only if it was right; treating a degenerate
    # interval as always-covered would let a model score perfectly by
    # claiming sigma = 0 everywhere
    r = np.array([0.0, 0.0, 1.0, -2.0])
    s = np.zeros(4)
    curve = uc.coverage_curve(r, s, levels=np.array([0.5, 0.9]))
    for c in curve:
        assert c["observed"] == pytest.approx(0.5)


def test_a_model_claiming_zero_uncertainty_everywhere_is_maximally_miscalibrated():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 1, 10_000)
    s = np.zeros(10_000)
    ece, _ = uc.ece_mce(uc.coverage_curve(r, s))
    # observed coverage is ~0 at every level, so ECE is the mean nominal level
    assert ece == pytest.approx(float(uc.LEVELS.mean()), abs=0.01)


def test_ece_and_mce_are_undefined_for_an_empty_curve():
    e, m = uc.ece_mce([])
    assert np.isnan(e) and np.isnan(m)


def test_mce_is_at_least_ece():
    r, s = _calibrated(n=20_000, seed=4)
    e, m = uc.ece_mce(uc.coverage_curve(r, s / 2.0))
    assert m >= e


# --------------------------------------------------------------------------
# the scale fit
# --------------------------------------------------------------------------


def test_fit_scale_recovers_a_known_miscalibration():
    r, s = _calibrated(n=100_000, seed=5)
    # the model reports sigma/3; the fit should recover roughly 3
    got = uc.fit_scale(r, s / 3.0)
    assert 2.4 < got < 3.7


def test_fit_scale_leaves_an_already_calibrated_model_alone():
    r, s = _calibrated(n=100_000, seed=6)
    got = uc.fit_scale(r, s)
    assert 0.85 < got < 1.2


def test_fitted_scale_actually_lowers_ece():
    rng = np.random.default_rng(7)
    s = rng.uniform(0.05, 1.0, 60_000)
    r = rng.normal(0, 1, 60_000) * s
    bad = s * 0.25
    before, _ = uc.ece_mce(uc.coverage_curve(r, bad))
    k = uc.fit_scale(r, bad)
    after, _ = uc.ece_mce(uc.coverage_curve(r, bad, k))
    assert after < before / 2


# --------------------------------------------------------------------------
# the sigma-vs-error diagram
# --------------------------------------------------------------------------


def test_sigma_bins_land_on_the_diagonal_when_calibrated():
    rng = np.random.default_rng(8)
    s = rng.uniform(0.05, 1.0, 200_000)
    r = rng.normal(0, 1, 200_000) * s
    bins = uc.sigma_bins(r, s, n_bins=10)
    assert len(bins) == 10
    for b in bins:
        assert b["ratio"] == pytest.approx(1.0, rel=0.15)
    ece, _ = uc.sigma_ece_mce(bins)
    assert ece < 0.1


def test_sigma_bins_expose_a_constant_underestimate():
    rng = np.random.default_rng(9)
    s = rng.uniform(0.05, 1.0, 100_000)
    r = rng.normal(0, 1, 100_000) * s
    bins = uc.sigma_bins(r, s * 0.2, n_bins=8)
    # every bin achieves ~5x the error it claimed
    for b in bins:
        assert b["ratio"] == pytest.approx(5.0, rel=0.25)


def test_sigma_bins_are_equal_count_and_cover_every_prediction():
    rng = np.random.default_rng(10)
    s = rng.uniform(0.05, 1.0, 1001)
    r = rng.normal(0, 1, 1001) * s
    bins = uc.sigma_bins(r, s, n_bins=7)
    assert sum(b["n"] for b in bins) == 1001
    assert max(b["n"] for b in bins) - min(b["n"] for b in bins) <= 1


def test_sigma_bins_are_ordered_by_claimed_uncertainty():
    rng = np.random.default_rng(11)
    s = rng.uniform(0.05, 1.0, 50_000)
    r = rng.normal(0, 1, 50_000) * s
    claimed = [b["claimed_sigma"] for b in uc.sigma_bins(r, s, n_bins=9)]
    assert all(a < b for a, b in zip(claimed, claimed[1:]))


# --------------------------------------------------------------------------
# the predictive distribution
# --------------------------------------------------------------------------


def test_predictive_returns_one_entry_per_pixel_and_ignores_nothing():
    x = torch.randn(5, 2, 8, 8)
    y = torch.randn(5, 2, 8, 8)
    m1, m2 = torch.nn.Conv2d(2, 2, 1), torch.nn.Conv2d(2, 2, 1)
    r, s = uc.predictive([m1, m2], x, y, "cpu")
    assert r.shape == (5 * 2 * 8 * 8,)
    assert s.shape == r.shape
    assert np.all(s >= 0)


def test_identical_members_claim_zero_uncertainty():
    # the pathological case the calibration test is meant to catch: an ensemble
    # that agrees with itself claims certainty regardless of its error
    class Const(torch.nn.Module):
        def forward(self, t):
            return torch.zeros_like(t)

    x = torch.randn(4, 1, 8, 8)
    y = torch.ones(4, 1, 8, 8)
    r, s = uc.predictive([Const(), Const()], x, y, "cpu")
    assert float(np.abs(s).max()) == pytest.approx(0.0, abs=1e-6)
    assert float(np.abs(r).min()) == pytest.approx(1.0)
