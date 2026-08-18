"""Validation for scripts/coalition_manipulation.py.

The experiment's whole content is that a derived single-agent law survives a
threat model it was not derived under. That is only worth reporting if the
measurement could have caught it failing, so the tests here are built around
the ways this could be vacuously true: a capture measurement that cannot
report a gain, a "best response" that is not searched, a coalition selection
that quietly picks the weakest attacker.

The two headline claims -- group strategy-proofness at alpha = 1 and dilution
with coalition size -- are pinned against brute force rather than against the
formula they came from.

No network access, no training.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts"
          / "coalition_manipulation.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litefno.allocation import alpha_fair_allocation                # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("coalition_manipulation",
                                                  SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["coalition_manipulation"] = module
    spec.loader.exec_module(module)
    return module


cm = _load()
KAPPA = 1.5
ALPHAS = [0.25, 0.5, 2.0, 8.0]


def _gains(n=8, seed=0):
    return np.random.default_rng(seed).uniform(0.4, 3.0, n)


# --------------------------------------------------------------------------
# the measurement can detect a gain at all
# --------------------------------------------------------------------------


def test_a_lone_deviator_gains_under_every_rule_except_the_envy_free_one():
    # if this failed, every "coalitions gain nothing" result below would be
    # vacuous -- the instrument would simply be blind
    g = _gains()
    for alpha in ALPHAS:
        assert cm.coalition_capture(g, [0], alpha, KAPPA)["ratio"] > 1.0 + 1e-6


def test_capture_is_one_when_the_report_bound_forbids_lying():
    g = _gains()
    for alpha in ALPHAS:
        got = cm.coalition_capture(g, [0, 1], alpha, kappa=1.0)
        assert got["ratio"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# claim 1: alpha = 1 is group strategy-proof, not merely strategy-proof
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8])
def test_no_coalition_of_any_size_gains_at_alpha_one(size):
    g = _gains()
    for C in itertools.islice(itertools.combinations(range(8), size), 20):
        assert cm.coalition_capture(g, list(C), 1.0, 4.0)["ratio"] == \
            pytest.approx(1.0)


def test_alpha_one_resists_arbitrary_joint_reports_not_just_corner_ones():
    # group strategy-proofness has to hold against every joint deviation, not
    # only the one the corner argument predicts
    rng = np.random.default_rng(3)
    g = _gains()
    honest = alpha_fair_allocation(g, alpha=1.0)
    for _ in range(200):
        rep = rng.uniform(0.01, 50.0, len(g))
        rep[6:] = g[6:]                          # regions 6,7 stay truthful
        got = alpha_fair_allocation(rep, alpha=1.0)
        assert np.allclose(got, honest)


# --------------------------------------------------------------------------
# claim 2: the closed form, at the pooled share
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("size", [1, 2, 4, 7])
def test_capture_matches_the_closed_form(alpha, size):
    g = _gains(seed=1)
    C = list(range(size))
    assert cm.coalition_capture(g, C, alpha, KAPPA)["ratio"] == \
        pytest.approx(cm.capture_formula(g, C, alpha, KAPPA), rel=1e-12)


def test_the_grand_coalition_captures_exactly_nothing():
    g = _gains(seed=2)
    for alpha in ALPHAS:
        got = cm.coalition_capture(g, list(range(len(g))), alpha, 3.0)
        assert got["ratio"] == pytest.approx(1.0, abs=1e-12)
        assert got["captured"] == pytest.approx(0.0, abs=1e-12)


def test_the_formula_refuses_the_discontinuous_rule():
    # alpha = 0 has no elasticity; returning a number there would be a silently
    # wrong comparison rather than a missing one
    assert np.isnan(cm.capture_formula(_gains(), [0], 0.0, KAPPA))


# --------------------------------------------------------------------------
# claim 3: dilution -- the surprising direction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_capture_ratio_is_non_increasing_in_coalition_size(alpha):
    g = _gains(seed=4, n=10)
    ratios = [cm.coalition_capture(g, list(range(k)), alpha, KAPPA)["ratio"]
              for k in range(1, 11)]
    assert all(b <= a + 1e-12 for a, b in zip(ratios, ratios[1:]))
    assert ratios[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_collusion_is_subadditive_against_the_same_regions_acting_alone(alpha):
    # the claim that could have gone the other way, and the reason the single-
    # deviation law is the worst case rather than the optimistic one
    g = _gains(seed=5, n=10)
    for size in (2, 4, 8):
        C = list(range(size))
        joint = cm.coalition_capture(g, C, alpha, KAPPA)["captured"]
        solo = sum(cm.coalition_capture(g, [r], alpha, KAPPA)["captured"]
                   for r in C)
        assert joint < solo


def test_absolute_capture_rises_even_though_the_ratio_falls():
    # both are true and reporting only the ratio would understate the harm
    g = _gains(seed=6, n=10)
    for alpha in ALPHAS:
        got = [cm.coalition_capture(g, list(range(k)), alpha, KAPPA)
               for k in (1, 2, 4)]
        assert got[0]["ratio"] > got[2]["ratio"]
        assert got[0]["captured"] < got[2]["captured"]


# --------------------------------------------------------------------------
# the corner profile really is the best response
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_no_joint_report_beats_the_all_corner_profile(alpha):
    rng = np.random.default_rng(7)
    g = _gains(seed=7)
    C = [0, 1, 2]
    low, high = cm.report_bounds(g, kappa=KAPPA)
    base = alpha_fair_allocation(cm.corner_report(g, C, alpha, KAPPA),
                                 alpha=alpha)[C].sum()
    for bits in itertools.product((0, 1), repeat=len(C)):
        rep = g.copy()
        for r, b in zip(C, bits):
            rep[r] = high[r] if b else low[r]
        assert alpha_fair_allocation(rep, alpha=alpha)[C].sum() <= base + 1e-12
    for _ in range(500):
        rep = g.copy()
        for r in C:
            rep[r] = rng.uniform(low[r], high[r])
        assert alpha_fair_allocation(rep, alpha=alpha)[C].sum() <= base + 1e-12


def test_corner_direction_flips_with_the_sign_of_the_exponent():
    g = _gains(seed=8)
    over = cm.corner_report(g, [0, 1], 0.5, KAPPA)      # beta > 0: look big
    under = cm.corner_report(g, [0, 1], 4.0, KAPPA)     # beta < 0: look needy
    assert np.all(over[:2] > g[:2])
    assert np.all(under[:2] < g[:2])
    assert np.allclose(over[2:], g[2:]) and np.allclose(under[2:], g[2:])


def test_the_corner_report_is_the_truth_at_the_envy_free_point():
    g = _gains(seed=9)
    assert np.allclose(cm.corner_report(g, [0, 1, 2], 1.0, KAPPA), g)


# --------------------------------------------------------------------------
# coalition selection is the strongest attacker, not a convenient one
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("size", [2, 3])
def test_the_greedy_coalition_is_the_exhaustive_best(alpha, size):
    # the greedy rule follows from the formula being decreasing in pooled
    # share; if that reasoning were wrong the experiment would be reporting a
    # weak attack as the worst case
    g = _gains(seed=10)
    best = max(itertools.combinations(range(len(g)), size),
               key=lambda C: cm.coalition_capture(g, list(C), alpha,
                                                  KAPPA)["ratio"])
    chosen = cm.greedy_coalition(g, alpha, size)
    assert cm.coalition_capture(g, chosen, alpha, KAPPA)["ratio"] == \
        pytest.approx(cm.coalition_capture(g, list(best), alpha,
                                           KAPPA)["ratio"], rel=1e-12)


# --------------------------------------------------------------------------
# the discontinuous rule
# --------------------------------------------------------------------------


def test_a_coalition_holding_nothing_under_argmax_has_an_unbounded_ratio():
    # flooring the denominator would report this as a large finite number that
    # then averages like an ordinary one
    g = np.array([3.0, 1.0, 1.0, 1.0])
    got = cm.coalition_capture(g, [1], 0.0, kappa=4.0)
    assert np.isinf(got["ratio"])
    assert got["captured"] == pytest.approx(1.0)


def test_a_coalition_that_cannot_win_the_argmax_gains_exactly_nothing():
    g = np.array([10.0, 1.0, 1.0, 1.0])
    got = cm.coalition_capture(g, [1], 0.0, kappa=1.5)   # 1.5 < 10 even lied
    assert got["ratio"] == pytest.approx(1.0)
    assert got["captured"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# leximin: no exponent to read a direction off, so the claim must be re-checked
# --------------------------------------------------------------------------


def test_leximin_also_dilutes_although_it_is_not_a_power_rule():
    g = _gains(seed=11, n=8)
    rows = cm.leximin_cap_check(g[None, :], [np.inf], [8.0],
                                sizes=(1, 2, 4, 8), sample=1)
    ratios = [r["max_capture_ratio"] for r in rows]
    assert all(b <= a + 1e-9 for a, b in zip(ratios, ratios[1:]))


def test_the_capacity_cap_is_a_feasibility_bound_that_is_never_violated():
    g = _gains(seed=12, n=8)
    rows = cm.leximin_cap_check(g[None, :], [2.0, 1.5], [4.0, 16.0],
                                sizes=(1, 2, 4), sample=1)
    for r in rows:
        assert r["max_capture_ratio"] <= r["structural_bound"] + 1e-9
