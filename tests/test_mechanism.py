"""Validation for litefno.mechanism.

Three claims are pinned here, and each fails in its own way.

Strategy-proofness is a property that a wrong implementation *reports* rather
than violates: if the best response were computed on the wrong side of the
corner, every rule would look strategy-proof and the extension's headline would
invert. So the measured incentive ratio is checked against a closed form, and
the closed form against a direct search.

Leximin is checked against the definition -- lexicographically dominating other
feasible allocations -- and not against itself. Progressive filling is easy to
write in a way that produces a plausible, budget-respecting, cap-respecting
allocation that is simply not the leximin one.

No-regret is checked by the thing that makes it a guarantee: the average regret
against the best fixed allocation has to go to zero, and the comparator it is
measured against has to actually be the best fixed allocation.

No network access, no data files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litefno.allocation import (  # noqa: E402
    allocation_exponent, alpha_fair_allocation, outcomes, welfare_ce)
from litefno.mechanism import (  # noqa: E402
    best_fixed_allocation, best_response, incentive_ratio,
    incentive_ratio_formula, leximin_allocation, leximin_incentive_ratio,
    manipulation_damage, online_allocation, regret_curve, report_bounds)

ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def _gains(n=12, seed=0, spread=0.3):
    return np.exp(np.random.default_rng(seed).normal(0, spread, n))


def _lex_ge(x, y, tol=1e-9):
    """Is the sorted-ascending outcome vector x lexicographically >= y?"""
    for a, b in zip(np.sort(x), np.sort(y)):
        if a > b + tol:
            return True
        if a < b - tol:
            return False
    return True


def _random_feasible(rng, budget, caps):
    """A feasible point of {sum a = budget, 0 <= a <= caps}, for comparison."""
    a = rng.dirichlet(np.ones(len(caps))) * budget
    for _ in range(64):
        over = a > caps
        if not over.any():
            break
        excess = float((a[over] - caps[over]).sum())
        a[over] = caps[over]
        free = ~over & (a < caps - 1e-12)
        if not free.any():
            break
        a[free] += excess * a[free] / max(a[free].sum(), 1e-30)
    return a


# --------------------------------------------------------------------------
# strategy-proofness
# --------------------------------------------------------------------------


def test_only_the_envy_free_rule_is_strategy_proof():
    """alpha = 1 ignores the gains, so no report can move it. Nothing else."""
    gains = _gains()
    assert np.allclose(incentive_ratio(gains, 1.0, kappa=2.0), 1.0)
    for alpha in [a for a in ALPHAS if a != 1.0]:
        ratios = incentive_ratio(gains, alpha, kappa=1.2)
        assert np.all(ratios > 1.0 + 1e-9), \
            f"alpha={alpha} came out strategy-proof, which it is not"


def test_best_response_changes_direction_at_alpha_one():
    """Efficiency-seeking rules reward looking big; egalitarian ones reward
    looking needy. A sign error here silently inverts the whole study."""
    gains = _gains()
    assert np.all(best_response(gains, 0.5, kappa=1.5) > gains)
    assert np.all(best_response(gains, 4.0, kappa=1.5) < gains)
    assert np.allclose(best_response(gains, 1.0, kappa=1.5), gains)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 2.0, 4.0, 8.0])
@pytest.mark.parametrize("kappa", [1.05, 1.5])
def test_incentive_ratio_matches_its_closed_form(alpha, kappa):
    gains = _gains(seed=3)
    beta = allocation_exponent(alpha)
    weights = gains ** beta
    shares = weights / weights.sum()
    measured = incentive_ratio(gains, alpha, kappa=kappa)
    expected = [incentive_ratio_formula(s, alpha, kappa) for s in shares]
    assert np.allclose(measured, expected, rtol=1e-9)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 2.0, 4.0, 8.0])
def test_manipulation_incentive_is_the_fragility_exponent(alpha):
    """The unification: the incentive to lie and the sensitivity to accidental
    error are the same derivative, ``|1-alpha|/alpha``.

    Checked in the small-distortion limit, where the exact ratio reduces to
    ``|beta| (1 - s_r) log kappa``. The ``(1 - s_r)`` is the normaliser pushing
    back -- a region already holding a large share has less left to take -- and
    it is kept in the comparison rather than tolerated as slack, because at
    alpha = 1/4 the exponent of 3 makes the shares uneven enough that the
    correction reaches 11%.
    """
    gains = _gains(n=64, seed=5)
    kappa = 1.001
    beta = allocation_exponent(alpha)
    weights = gains ** beta
    shares = weights / weights.sum()
    ratios = incentive_ratio(gains, alpha, kappa=kappa)
    slope = np.log(ratios) / np.log(kappa)
    assert np.allclose(slope, abs(beta) * (1 - shares), rtol=1e-3)


def test_manipulation_damage_is_zero_only_for_the_strategy_proof_rule():
    gains = _gains(seed=7)
    at_one = manipulation_damage(gains, 1.0, region=0, kappa=2.0)
    assert at_one["incentive_ratio"] == pytest.approx(1.0)
    assert at_one["welfare_loss"] == pytest.approx(0.0, abs=1e-12)
    for alpha in (0.5, 4.0):
        got = manipulation_damage(gains, alpha, region=0, kappa=1.5)
        assert got["incentive_ratio"] > 1.0
        assert got["welfare_loss"] > 0.0


def test_report_bounds_reject_an_ambiguous_request():
    gains = _gains(4)
    with pytest.raises(ValueError, match="exactly one"):
        report_bounds(gains, kappa=1.5, epsilon=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        report_bounds(gains)


def test_additive_bound_cannot_report_a_dead_region():
    """An additive attack must not be able to claim a gain of zero: under an
    egalitarian rule that would demand an unbounded share."""
    gains = np.array([0.05, 1.0, 2.0])
    low, _ = report_bounds(gains, epsilon=0.5)
    assert np.all(low > 0)


# --------------------------------------------------------------------------
# leximin
# --------------------------------------------------------------------------


def test_leximin_without_caps_is_the_max_min_rule():
    """Uncapped, equalising outcomes is always feasible, so leximin degenerates
    onto the alpha = inf rule and the lexicographic refinement does nothing."""
    gains = _gains(seed=1)
    assert np.allclose(leximin_allocation(gains),
                       alpha_fair_allocation(gains, alpha=np.inf))


def test_leximin_respects_budget_and_caps():
    gains = _gains(n=10, seed=2)
    caps = np.full(10, 0.25)
    alloc = leximin_allocation(gains, budget=1.0, caps=caps)
    assert alloc.sum() == pytest.approx(1.0)
    assert np.all(alloc >= -1e-12)
    assert np.all(alloc <= caps + 1e-12)


def test_leximin_equalises_the_uncapped_and_freezes_the_capped():
    gains = np.array([0.5, 1.0, 2.0, 4.0])
    caps = np.array([0.4, 0.4, 0.4, 0.4])
    alloc = leximin_allocation(gains, budget=1.0, caps=caps)
    x = outcomes(gains, alloc)
    capped = alloc >= caps - 1e-12
    assert capped.any() and not capped.all(), "this case must exercise both"
    assert np.allclose(x[~capped], x[~capped][0])
    assert np.all(x[capped] <= x[~capped][0] + 1e-12)


def test_leximin_lexicographically_dominates_other_feasible_allocations():
    """The definition, checked against the definition."""
    rng = np.random.default_rng(4)
    for trial in range(5):
        gains = _gains(n=6, seed=10 + trial)
        caps = rng.uniform(0.15, 0.5, 6)
        caps = caps * 1.5 / caps.sum() if caps.sum() < 1.0 else caps
        best = leximin_allocation(gains, budget=1.0, caps=caps)
        x_best = outcomes(gains, best)
        for _ in range(200):
            other = _random_feasible(rng, 1.0, caps)
            assert _lex_ge(x_best, outcomes(gains, other)), \
                f"a random feasible allocation beat leximin: {other}"


def test_leximin_maximises_the_minimum_outcome():
    """The first lexicographic level, checked on its own against a fine search."""
    gains = np.array([0.6, 1.0, 1.7, 3.0])
    caps = np.full(4, 0.35)
    best = leximin_allocation(gains, budget=1.0, caps=caps)
    floor = outcomes(gains, best).min()
    rng = np.random.default_rng(0)
    for _ in range(3000):
        other = _random_feasible(rng, 1.0, caps)
        assert outcomes(gains, other).min() <= floor + 1e-9


def test_leximin_refuses_an_infeasible_budget():
    gains = _gains(n=5, seed=6)
    with pytest.raises(ValueError, match="infeasible"):
        leximin_allocation(gains, budget=1.0, caps=np.full(5, 0.1))


def test_leximin_with_exactly_sufficient_capacity_has_one_answer():
    gains = _gains(n=5, seed=6)
    caps = np.full(5, 0.2)
    assert np.allclose(leximin_allocation(gains, budget=1.0, caps=caps), caps)


def test_the_cap_bounds_what_a_liar_can_win():
    """The mechanism claim: leximin is not strategy-proof, but the cap turns an
    unbounded incentive into a bounded one, with no truthfulness machinery."""
    gains = _gains(n=12, seed=8)
    caps = np.full(12, 2.0 / 12)
    truthful = leximin_allocation(gains, 1.0, caps)
    ratios = leximin_incentive_ratio(gains, 1.0, caps, kappa=100.0)
    assert np.all(ratios > 1.0), "leximin is not strategy-proof; it should not"
    assert np.all(ratios <= caps / truthful + 1e-9), \
        "a region won more than its cap, so the cap is not binding the lie"


def test_a_tighter_cap_is_a_less_manipulable_mechanism():
    """The dial: the cap trades responsiveness against manipulability, and it
    has to be monotone or it is not a dial."""
    gains = _gains(n=12, seed=9)
    worst = []
    for multiplier in (1.2, 2.0, 4.0):
        caps = np.full(12, multiplier / 12)
        worst.append(float(np.max(
            leximin_incentive_ratio(gains, 1.0, caps, kappa=10.0))))
    assert worst[0] < worst[1] < worst[2]


# --------------------------------------------------------------------------
# no regret
# --------------------------------------------------------------------------


def _oscillating_stream(steps=400, n=8, seed=0):
    """Gains that cycle, as an oscillating ecosystem's regions do."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, n)
    t = np.arange(steps)[:, None]
    return 1.5 + 0.4 * np.sin(2 * np.pi * t / 50 + phase)


def test_average_regret_vanishes_where_the_guarantee_bites():
    """The no-regret property, on a stream where the comparator is the right one.

    With stationary gains the best fixed allocation is also the best per-round
    allocation, so regret is non-negative and has to decay. That is the case
    where "no regret" means what it sounds like.
    """
    gains = _gains(n=6, seed=11)
    stream = np.repeat(gains[None], 400, axis=0)
    got = regret_curve(stream, alpha=2.0)
    early = got["average_regret"][len(stream) // 10]
    late = got["average_regret"][-1]
    assert late >= -1e-9, "regret against a per-round optimum cannot be negative"
    assert late < early, "average regret is not decreasing"
    assert late < 1e-4, f"average regret did not vanish: {late:.2e}"


def test_the_fixed_comparator_is_weak_when_the_ecosystem_oscillates():
    """Negative regret, and it is the point rather than a bug.

    The learner *beats* the best fixed allocation on an oscillating stream,
    which is only possible because the comparator is constrained to be constant
    while the ecosystem is not. It is the sharpest available statement that a
    no-regret guarantee in this setting certifies very little: being no worse
    than the best constant allocation is a low bar when no constant allocation
    is any good, and it is why the extension measures the learner against a
    forecast rather than against its own regret bound.
    """
    stream = _oscillating_stream()
    got = regret_curve(stream, alpha=2.0)
    assert got["average_regret"][-1] < 0
    assert got["online_welfare"].mean() > got["fixed_welfare"].mean()


def test_the_comparator_really_is_the_best_fixed_allocation():
    """A regret curve is only meaningful if what it is measured against is
    actually the best fixed allocation. Checked against random ones."""
    stream = _oscillating_stream(steps=120, n=6, seed=2)
    fixed = best_fixed_allocation(stream, alpha=2.0)
    mine = float(np.mean([welfare_ce(outcomes(g, fixed), 2.0) for g in stream]))
    rng = np.random.default_rng(0)
    for _ in range(300):
        other = rng.dirichlet(np.ones(6))
        theirs = float(np.mean(
            [welfare_ce(outcomes(g, other), 2.0) for g in stream]))
        assert theirs <= mine + 1e-6


@pytest.mark.parametrize("alpha", [0.5, 2.0, 8.0])
def test_the_comparator_is_never_worse_than_a_constant_split(alpha):
    """Uniform is itself a fixed allocation, so the best fixed allocation cannot
    be worse than it. It was, at alpha = 8, before the comparator was seeded and
    tracked -- exponentiated gradient does not converge from a uniform start
    there within any sane iteration count, and the un-converged comparator made
    the regret look better than it was."""
    stream = _oscillating_stream(steps=60, n=8, seed=3)
    fixed = best_fixed_allocation(stream, alpha=alpha)
    uniform = np.full(8, 1.0 / 8)
    mine = np.mean([welfare_ce(outcomes(g, fixed), alpha) for g in stream])
    theirs = np.mean([welfare_ce(outcomes(g, uniform), alpha) for g in stream])
    assert mine >= theirs - 1e-12


def test_online_allocation_stays_on_the_simplex():
    stream = _oscillating_stream(steps=50, n=5)
    got = online_allocation(stream, alpha=4.0, budget=2.0)
    assert np.allclose(got["allocations"].sum(axis=1), 2.0)
    assert np.all(got["allocations"] >= 0)


def test_online_allocation_rejects_a_malformed_stream():
    with pytest.raises(ValueError, match="gain stream"):
        online_allocation(np.ones(10), alpha=2.0)


def test_on_a_stationary_stream_it_finds_the_right_answer():
    """The sanity case. With gains that never move, the best fixed allocation is
    the alpha-fair optimum, and the learner should get there."""
    gains = _gains(n=6, seed=11)
    stream = np.repeat(gains[None], 600, axis=0)
    got = online_allocation(stream, alpha=2.0)
    target = alpha_fair_allocation(gains, alpha=2.0)
    assert np.allclose(got["final"], target, atol=0.02)
