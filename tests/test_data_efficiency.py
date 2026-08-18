"""Validation for scripts/data_efficiency.py.

The headline is a data multiplier obtained by interpolation, so the thing that
would quietly invalidate it is extrapolation dressed up as measurement. That is
the first group of tests. The rest cover the balanced subsampling that keeps
"size" from confounding with "composition", and the paired arithmetic that makes
a small effect legible at three seeds.

Training is not exercised. No network access.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_efficiency.py"


def _load():
    spec = importlib.util.spec_from_file_location("data_efficiency", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_efficiency"] = module
    spec.loader.exec_module(module)
    return module


de = _load()

REGIMES = np.array(["bubbles"] * 4 + ["gliders"] * 4 + ["maze"] * 4
                   + ["spirals"] * 4 + ["spots"] * 4 + ["worms"] * 4)


# --------------------------------------------------------------------------
# the multiplier
# --------------------------------------------------------------------------


def _curve(pairs):
    return [{"n_traj": n, "vrmse": v} for n, v in pairs]


def test_multiplier_on_a_hand_checked_interpolation():
    # plain hits 0.10 at 12 traj and 0.05 at 24; harmonic's 0.0707 sits at the
    # geometric midpoint in log-size, so plain needs ~17 traj -> ~1.4x of 12
    curve = _curve([(12, 0.10), (24, 0.05)])
    got = de.data_multiplier(curve, target_error=0.075, at_size=12)
    assert got["status"] == "ok"
    assert got["equivalent_n"] == pytest.approx(16.97, rel=0.02)
    assert got["multiplier"] == pytest.approx(1.41, rel=0.02)


def test_multiplier_is_one_when_the_arms_match():
    curve = _curve([(6, 0.20), (12, 0.10), (24, 0.05)])
    got = de.data_multiplier(curve, target_error=0.10, at_size=12)
    assert got["status"] == "ok"
    assert got["equivalent_n"] == pytest.approx(12.0, rel=1e-6)
    assert got["multiplier"] == pytest.approx(1.0, rel=1e-6)


def test_target_the_plain_arm_never_reaches_is_not_extrapolated():
    # the number a reader would most want and least be able to check
    curve = _curve([(6, 0.20), (12, 0.10), (24, 0.05)])
    got = de.data_multiplier(curve, target_error=0.01, at_size=24)
    assert got["status"] == "out_of_range_high"
    assert np.isnan(got["multiplier"])
    assert np.isnan(got["equivalent_n"])


def test_target_worse_than_the_smallest_run_is_also_out_of_range():
    curve = _curve([(6, 0.20), (12, 0.10)])
    got = de.data_multiplier(curve, target_error=0.50, at_size=6)
    assert got["status"] == "out_of_range_low"
    assert np.isnan(got["multiplier"])


def test_multiplier_needs_at_least_two_points():
    got = de.data_multiplier(_curve([(12, 0.1)]), 0.05, 12)
    assert got["status"] == "degenerate"
    assert np.isnan(got["multiplier"])


def test_multiplier_ignores_non_finite_points():
    curve = _curve([(6, 0.20), (12, float("nan")), (24, 0.05)])
    got = de.data_multiplier(curve, target_error=0.10, at_size=6)
    assert got["status"] == "ok"
    assert 6 < got["equivalent_n"] < 24


def test_multiplier_greater_than_one_means_the_prior_bought_data():
    # harmonic at 6 traj matches what plain needs ~18 traj for -> 3x
    curve = _curve([(6, 0.30), (18, 0.15), (24, 0.12)])
    got = de.data_multiplier(curve, target_error=0.15, at_size=6)
    assert got["status"] == "ok"
    assert got["multiplier"] == pytest.approx(3.0, rel=0.02)


# --------------------------------------------------------------------------
# balanced subsampling
# --------------------------------------------------------------------------


def test_subset_is_balanced_across_regimes():
    # an unbalanced draw would confound size with composition, and ext10 showed
    # the regimes differ by two orders of magnitude in where they keep variance
    for k in (1, 2, 3, 4):
        idx = de.balanced_subset(REGIMES, k, seed=0)
        counts = {r: int((REGIMES[idx] == r).sum())
                  for r in dict.fromkeys(REGIMES)}
        assert set(counts.values()) == {k}
        assert len(idx) == k * 6


def test_subset_never_omits_a_regime_even_at_the_smallest_size():
    idx = de.balanced_subset(REGIMES, 1, seed=3)
    assert len(set(REGIMES[idx])) == 6


def test_subset_is_deterministic_per_seed_so_arms_are_paired():
    a = de.balanced_subset(REGIMES, 2, seed=7)
    b = de.balanced_subset(REGIMES, 2, seed=7)
    assert np.array_equal(a, b)


def test_subset_varies_with_seed():
    a = de.balanced_subset(REGIMES, 2, seed=0)
    b = de.balanced_subset(REGIMES, 2, seed=1)
    assert not np.array_equal(a, b)


def test_subset_has_no_duplicates_and_caps_at_availability():
    idx = de.balanced_subset(REGIMES, 99, seed=0)
    assert len(idx) == len(set(idx.tolist()))
    assert len(idx) == len(REGIMES)


# --------------------------------------------------------------------------
# the paired comparison
# --------------------------------------------------------------------------


def _rows(spec):
    return [{"arm": a, "n_traj": n, "seed": s, "vrmse": v}
            for (n, s, a, v) in spec]


def test_paired_difference_is_computed_per_seed_not_between_means():
    # a large shared seed effect must cancel; only the within-seed difference
    # is the signal
    spec = []
    for s, base in ((0, 0.10), (1, 0.30), (2, 0.50)):
        spec.append((12, s, "plain", base))
        spec.append((12, s, "harmonic", base - 0.01))
    out = de.paired_summary(_rows(spec))[0]
    assert out["paired_diff_mean"] == pytest.approx(-0.01)
    assert out["paired_diff_std"] == pytest.approx(0.0, abs=1e-12)
    # the arm means differ hugely in spread, which is exactly what pairing removes
    assert out["plain_std"] > 0.15


def test_sign_consistency_is_reported():
    spec = [(12, 0, "plain", 0.10), (12, 0, "harmonic", 0.09),
            (12, 1, "plain", 0.10), (12, 1, "harmonic", 0.11),
            (12, 2, "plain", 0.10), (12, 2, "harmonic", 0.09)]
    out = de.paired_summary(_rows(spec))[0]
    assert out["n_seeds_helped"] == 2
    assert out["n_seeds"] == 3


def test_effect_smaller_than_seed_spread_is_flagged():
    # the repo's own prior says the effect should be small; a difference buried
    # in seed variance must not read as a result
    spec = [(12, 0, "plain", 0.10), (12, 0, "harmonic", 0.0999),
            (12, 1, "plain", 0.30), (12, 1, "harmonic", 0.2999),
            (12, 2, "plain", 0.50), (12, 2, "harmonic", 0.4999)]
    out = de.paired_summary(_rows(spec))[0]
    assert out["exceeds_seed_spread"] is False


def test_large_consistent_effect_is_not_flagged_as_noise():
    spec = [(12, 0, "plain", 0.10), (12, 0, "harmonic", 0.05),
            (12, 1, "plain", 0.11), (12, 1, "harmonic", 0.06),
            (12, 2, "plain", 0.10), (12, 2, "harmonic", 0.05)]
    out = de.paired_summary(_rows(spec))[0]
    assert out["exceeds_seed_spread"] is True
    assert out["rel_change"] == pytest.approx(-0.5, rel=0.05)


def test_sizes_missing_one_arm_are_skipped():
    spec = [(6, 0, "plain", 0.2), (12, 0, "plain", 0.1),
            (12, 0, "harmonic", 0.09)]
    out = de.paired_summary(_rows(spec))
    assert [o["n_traj"] for o in out] == [12]


def test_summary_is_ordered_by_size():
    spec = []
    for n in (24, 6, 12):
        spec += [(n, 0, "plain", 0.1), (n, 0, "harmonic", 0.09)]
    assert [o["n_traj"] for o in de.paired_summary(_rows(spec))] == [6, 12, 24]


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------


def test_the_two_arms_are_bit_identical_at_initialisation():
    # the whole comparison rests on this: the bias starts at zero, so any
    # divergence is training, not a different starting model
    import torch
    cfg = {"seed": 0, "width": 16, "modes": 8, "layers": 2, "rank": 4,
           "fundamental": 4.0, "n_harmonics": 3}
    p = de.build("plain", 2, cfg).eval()
    h = de.build("harmonic", 2, cfg).eval()
    x = torch.randn(2, 2, 16, 16)
    with torch.no_grad():
        assert torch.equal(p(x), h(x))


def test_the_harmonic_arm_carries_more_parameters():
    cfg = {"seed": 0, "width": 16, "modes": 8, "layers": 2, "rank": 4,
           "fundamental": 4.0, "n_harmonics": 3}
    np_ = sum(q.numel() for q in de.build("plain", 2, cfg).parameters())
    nh = sum(q.numel() for q in de.build("harmonic", 2, cfg).parameters())
    assert nh > np_
