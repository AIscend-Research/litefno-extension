"""Validation for scripts/safe_deferral.py.

A deferral curve is easy to draw and easy to fake. The three ways it goes wrong
silently are all tested here: a moving normalisation that makes retained error
fall for the wrong reason, an empty retained set scoring as perfect, and a
"safe deferral rate" reported from a sweep that never actually reached the
target. The detection arithmetic follows.

Training is not exercised. No network access.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch                                                    # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "safe_deferral.py"


def _load():
    spec = importlib.util.spec_from_file_location("safe_deferral", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["safe_deferral"] = module
    spec.loader.exec_module(module)
    return module


sd = _load()

FRACTIONS = np.arange(0.0, 0.85, 0.05)


# --------------------------------------------------------------------------
# the normalisation
# --------------------------------------------------------------------------


def test_zero_deferral_reproduces_plain_vrmse():
    # the curve has to start at the number ext26 reports, or it is measuring
    # something else from the first point on
    rng = np.random.default_rng(0)
    mse = rng.random(200) * 0.01
    var = 0.5
    keep = np.ones(200, dtype=bool)
    assert sd.retained_score(mse, keep, var) == pytest.approx(
        float(np.sqrt(mse.mean() / var)))


def test_normalisation_is_fixed_not_recomputed_per_subset():
    # a subset of low-variance targets must not score better merely for being
    # low variance. Same kept predictions, same fixed denominator -> same score.
    mse = np.full(100, 0.02)
    var = 0.4
    a = sd.retained_score(mse, np.arange(100) < 50, var)
    b = sd.retained_score(mse, np.arange(100) >= 50, var)
    assert a == pytest.approx(b)
    assert a == pytest.approx(float(np.sqrt(0.02 / 0.4)))


def test_empty_retained_set_is_undefined_not_perfect():
    # deferring everything means doing no surrogate work at all; scoring that
    # as zero error would make 100% deferral the best point on the curve
    mse = np.full(10, 0.02)
    assert np.isnan(sd.retained_score(mse, np.zeros(10, dtype=bool), 0.4))


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


def test_deferral_rate_matches_the_requested_fraction():
    rng = np.random.default_rng(1)
    mse = rng.random(200) * 0.01
    conf = rng.random(200)
    rows = sd.deferral_curve(mse, conf, 0.5, 0.05, 0.008, FRACTIONS)
    for r, q in zip(rows, FRACTIONS):
        assert r["deferral_rate"] == pytest.approx(q, abs=0.011)


def test_tied_confidences_still_defer_the_requested_fraction():
    # a constant signal makes every quantile threshold identical; a naive
    # "confidence > T" would then defer none of them (or all)
    mse = np.linspace(0.001, 0.02, 100)
    conf = np.ones(100)
    rows = sd.deferral_curve(mse, conf, 0.5, 0.05, 0.01, np.array([0.0, 0.25, 0.5]))
    assert [r["n_deferred"] for r in rows] == [0, 25, 50]


def test_oracle_ranking_beats_random_ranking():
    # ranking by true error must reduce retained error faster than noise does;
    # if it does not, the sweep is not ordering by the signal it claims to
    rng = np.random.default_rng(2)
    mse = rng.random(300) ** 3 * 0.05
    var = 0.5
    oracle = sd.deferral_curve(mse, mse, var, 0.05, 0.01, FRACTIONS)
    random = sd.deferral_curve(mse, rng.random(300), var, 0.05, 0.01, FRACTIONS)
    for o, r in zip(oracle[1:], random[1:]):
        assert o["retained_vrmse"] <= r["retained_vrmse"]


def test_random_deferral_is_flat_in_expectation():
    # the null control must not drift, or a descending confidence curve proves
    # nothing. Averaged over draws, random deferral leaves retained error alone.
    rng = np.random.default_rng(3)
    mse = rng.random(400) * 0.02
    var = 0.5
    full = float(np.sqrt(mse.mean() / var))
    got = []
    for s in range(40):
        rows = sd.deferral_curve(mse, np.random.default_rng(s).random(400),
                                 var, full, 0.01, np.array([0.4]))
        got.append(rows[0]["retained_vrmse"])
    assert float(np.mean(got)) == pytest.approx(full, rel=0.02)


def test_oracle_retained_error_is_monotone_non_increasing():
    rng = np.random.default_rng(4)
    mse = rng.random(200) * 0.03
    rows = sd.deferral_curve(mse, mse, 0.5, 0.05, 0.01, FRACTIONS)
    vals = [r["retained_vrmse"] for r in rows]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def test_sensitivity_and_specificity_on_a_hand_checked_case():
    # 10 samples, the 3 worst are unsafe (tau = 0.0035). Deferring the 3 worst
    # by a perfect signal should catch all of them and waste none.
    mse = np.array([0.001, 0.002, 0.0025, 0.003, 0.0032, 0.0033,
                    0.0034, 0.004, 0.005, 0.006])
    rows = sd.deferral_curve(mse, mse, 0.5, 0.05, 0.0035, np.array([0.3]))
    r = rows[0]
    assert r["n_unsafe"] == 3
    assert r["sensitivity"] == pytest.approx(1.0)
    assert r["specificity"] == pytest.approx(1.0)


def test_deferring_everything_maximises_sensitivity_and_kills_specificity():
    # the degenerate corner both metrics exist to expose
    mse = np.linspace(0.001, 0.01, 20)
    rows = sd.deferral_curve(mse, mse, 0.5, 0.05, 0.005, np.array([1.0]))
    r = rows[0]
    assert r["sensitivity"] == pytest.approx(1.0)
    assert r["specificity"] == pytest.approx(0.0)
    # and retained accuracy is undefined, not perfect
    assert np.isnan(r["retained_vrmse"])


def test_deferring_nothing_is_the_opposite_corner():
    mse = np.linspace(0.001, 0.01, 20)
    r = sd.deferral_curve(mse, mse, 0.5, 0.05, 0.005, np.array([0.0]))[0]
    assert r["sensitivity"] == pytest.approx(0.0)
    assert r["specificity"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the headline number
# --------------------------------------------------------------------------


def test_safe_deferral_rate_is_the_first_rate_that_reaches_the_reference():
    rows = [
        {"deferral_rate": 0.0, "retained_vrmse": 0.10, "reference_vrmse": 0.05},
        {"deferral_rate": 0.1, "retained_vrmse": 0.07, "reference_vrmse": 0.05},
        {"deferral_rate": 0.2, "retained_vrmse": 0.05, "reference_vrmse": 0.05},
        {"deferral_rate": 0.3, "retained_vrmse": 0.03, "reference_vrmse": 0.05},
    ]
    assert sd.safe_deferral_rate(rows) == pytest.approx(0.2)


def test_safe_deferral_rate_is_undefined_when_the_sweep_never_gets_there():
    # reporting the best achieved rate would read as success at whatever
    # deferral the sweep happened to stop at
    rows = [
        {"deferral_rate": 0.0, "retained_vrmse": 0.10, "reference_vrmse": 0.05},
        {"deferral_rate": 0.5, "retained_vrmse": 0.08, "reference_vrmse": 0.05},
    ]
    assert np.isnan(sd.safe_deferral_rate(rows))


def test_safe_deferral_rate_ignores_an_undefined_retained_score():
    rows = [
        {"deferral_rate": 0.0, "retained_vrmse": 0.10, "reference_vrmse": 0.05},
        {"deferral_rate": 1.0, "retained_vrmse": float("nan"),
         "reference_vrmse": 0.05},
    ]
    assert np.isnan(sd.safe_deferral_rate(rows))


# --------------------------------------------------------------------------
# the confidence signal
# --------------------------------------------------------------------------


def test_disagreement_is_zero_for_identical_members_and_grows_with_spread():
    x = torch.randn(8, 2, 8, 8)
    y = torch.randn(8, 2, 8, 8)

    class Const(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, t):
            return torch.full_like(t, self.v)

    _, same, _ = sd.ensemble_stats([Const(1.0), Const(1.0)], x, y, "cpu")
    _, diff, _ = sd.ensemble_stats([Const(0.0), Const(2.0)], x, y, "cpu")
    assert float(np.abs(same).max()) == pytest.approx(0.0, abs=1e-6)
    assert float(diff.min()) > 0.5


def test_disagreement_never_looks_at_the_targets():
    # it has to be computable at deployment time; if it moved with the labels
    # it would be an oracle wearing a confidence signal's clothes
    x = torch.randn(8, 2, 8, 8)
    lin = torch.nn.Conv2d(2, 2, 1)
    lin2 = torch.nn.Conv2d(2, 2, 1)
    _, d1, _ = sd.ensemble_stats([lin, lin2], x, torch.randn(8, 2, 8, 8), "cpu")
    _, d2, _ = sd.ensemble_stats([lin, lin2], x, torch.randn(8, 2, 8, 8) * 99,
                                 "cpu")
    assert np.allclose(d1, d2)
