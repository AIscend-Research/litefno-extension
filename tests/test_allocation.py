"""Validation for litefno.allocation.

The allocation rules are closed forms, and a closed form that is subtly wrong
does not fail -- it returns a plausible allocation that quietly optimises
something else, and every welfare number downstream is then internally
consistent and meaningless. So the rules are checked against numerically
maximising the objective they claim to solve, not against themselves.

The fragility law gets the same treatment. It is the extension's headline claim
and it is a second-order expansion, so it is pinned here against exact optima at
a noise level where the expansion should hold, and its U-shape in alpha -- the
part that is actually surprising -- is checked separately from its magnitude.

No network access, no data files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litefno.allocation import (  # noqa: E402
    allocation_exponent, alpha_fair_allocation, fragility_coefficient,
    implied_gains, max_envy, max_min_ratio, outcome_envy, outcomes,
    predicted_welfare_loss, price_of_fairness, region_gains,
    relative_welfare_loss, tune_shrinkage, welfare_ce)

ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def _numeric_optimum(gains, alpha, budget=1.0, restarts=3):
    """Maximise the alpha-fair welfare directly, with no closed form involved.

    Optimised over unconstrained logits pushed through a softmax, so the simplex
    constraint holds exactly at every iterate and the comparison is against the
    same feasible set the closed form lives on. Several restarts because the
    objective is concave in the allocation but not in the logits.
    """
    from scipy.optimize import minimize

    gains = np.asarray(gains, dtype=float)

    def negative(logits):
        weights = np.exp(logits - logits.max())
        alloc = budget * weights / weights.sum()
        return -float(welfare_ce(outcomes(gains, alloc), alpha))

    best = None
    rng = np.random.default_rng(0)
    for i in range(restarts):
        start = np.zeros(len(gains)) if i == 0 else rng.normal(0, 1, len(gains))
        got = minimize(negative, start, method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-10,
                                "fatol": 1e-12})
        if best is None or got.fun < best.fun:
            best = got
    weights = np.exp(best.x - best.x.max())
    return budget * weights / weights.sum()


# --------------------------------------------------------------------------
# the closed form is the optimum it claims to be
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_closed_form_matches_numerical_optimum(alpha):
    """``a ∝ g^((1-alpha)/alpha)`` maximises the alpha-fair welfare."""
    rng = np.random.default_rng(3)
    gains = np.exp(rng.normal(0.0, 0.4, 8))
    closed = alpha_fair_allocation(gains, alpha=alpha)
    numeric = _numeric_optimum(gains, alpha)
    assert np.allclose(closed, numeric, atol=2e-3), \
        f"alpha={alpha}: closed {closed} vs numeric {numeric}"
    # and the welfare it reaches is at least as good, which is the property
    # that matters even where the argmax is flat
    ce_closed = welfare_ce(outcomes(gains, closed), alpha)
    ce_numeric = welfare_ce(outcomes(gains, numeric), alpha)
    assert ce_closed >= ce_numeric - 1e-9


def test_alpha_one_is_equal_division_and_therefore_envy_free():
    """The envy-free point. Equal division, and independent of the state."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        gains = np.exp(rng.normal(0, 0.8, 12))
        alloc = alpha_fair_allocation(gains, alpha=1.0, budget=3.0)
        assert np.allclose(alloc, 3.0 / 12)
        assert max_envy(alloc) == pytest.approx(0.0, abs=1e-12)


def test_only_equal_division_is_envy_free():
    """Envy is zero *iff* the bundles are equal, with one divisible resource."""
    assert max_envy(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(0.0)
    assert max_envy(np.array([0.3, 0.25, 0.25, 0.2])) > 0


def test_max_min_limit_equalises_outcomes():
    """alpha = inf gives ``a ∝ 1/g``, so every region realises the same x."""
    gains = np.array([0.5, 1.0, 2.0, 4.0])
    alloc = alpha_fair_allocation(gains, alpha=np.inf)
    x = outcomes(gains, alloc)
    assert np.allclose(x, x[0])
    assert max_min_ratio(x) == pytest.approx(1.0)
    assert outcome_envy(x) == pytest.approx(0.0, abs=1e-12)


def test_utilitarian_is_winner_take_all_and_splits_ties():
    gains = np.array([1.0, 3.0, 2.0])
    assert np.allclose(alpha_fair_allocation(gains, alpha=0.0),
                       [0.0, 1.0, 0.0])
    tied = np.array([2.0, 2.0, 1.0])
    assert np.allclose(alpha_fair_allocation(tied, alpha=0.0),
                       [0.5, 0.5, 0.0])


def test_allocation_is_on_the_simplex_for_every_alpha():
    rng = np.random.default_rng(7)
    gains = np.exp(rng.normal(0, 0.5, (5, 9)))
    for alpha in [0.0, *ALPHAS, np.inf]:
        alloc = alpha_fair_allocation(gains, alpha=alpha, budget=2.5)
        assert np.all(alloc >= 0)
        assert np.allclose(alloc.sum(axis=-1), 2.5)


def test_large_alpha_does_not_overflow():
    """The log-space path: g^beta at alpha = 64 over a wide spread overflows."""
    gains = np.array([1e-3, 1.0, 1e3])
    alloc = alpha_fair_allocation(gains, alpha=64.0)
    assert np.all(np.isfinite(alloc))
    assert alloc.sum() == pytest.approx(1.0)
    x = outcomes(gains, alloc)
    assert np.isfinite(welfare_ce(x, 64.0))


# --------------------------------------------------------------------------
# welfare
# --------------------------------------------------------------------------


def test_welfare_ce_special_cases():
    x = np.array([1.0, 2.0, 4.0])
    assert welfare_ce(x, 0.0) == pytest.approx(x.mean())
    assert welfare_ce(x, 1.0) == pytest.approx(float(np.exp(np.mean(np.log(x)))))
    assert welfare_ce(x, np.inf) == pytest.approx(x.min())
    # a power mean is monotone in its exponent, so more fairness aversion can
    # only lower the certainty equivalent of an unequal outcome vector
    values = [welfare_ce(x, a) for a in (0.0, 0.5, 1.0, 2.0, 8.0)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


def test_welfare_ce_is_in_outcome_units():
    """Scaling every outcome scales the certainty equivalent identically."""
    x = np.array([0.4, 1.1, 2.0])
    for alpha in [0.0, 0.5, 1.0, 4.0, np.inf]:
        assert welfare_ce(3.0 * x, alpha) == pytest.approx(
            3.0 * welfare_ce(x, alpha))


@pytest.mark.parametrize("alpha", ALPHAS)
def test_regret_is_zero_at_the_optimum_and_positive_elsewhere(alpha):
    rng = np.random.default_rng(11)
    gains = np.exp(rng.normal(0, 0.5, 10))
    best = alpha_fair_allocation(gains, alpha=alpha)
    assert relative_welfare_loss(gains, best, alpha) == pytest.approx(0.0,
                                                                     abs=1e-12)
    # perturbed off the optimum rather than shrunk: at alpha = 1 the exponent is
    # already 0, so shrinking it changes nothing and the check would pass
    # vacuously at exactly the value the rest of this file cares most about
    other = best * np.exp(rng.normal(0, 0.2, len(best)))
    other = other / other.sum()
    assert relative_welfare_loss(gains, other, alpha) > 0


def test_price_of_fairness_is_zero_only_when_gains_are_equal():
    equal = np.full(6, 1.3)
    for alpha in ALPHAS:
        assert price_of_fairness(equal, alpha) == pytest.approx(0.0, abs=1e-12)
    spread = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    costs = [price_of_fairness(spread, a) for a in (0.5, 1.0, 2.0, 8.0)]
    assert all(0 < c < 1 for c in costs)
    assert all(a <= b + 1e-12 for a, b in zip(costs, costs[1:])), \
        "more fairness aversion must not cost less efficiency"


# --------------------------------------------------------------------------
# the fragility law
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ALPHAS)
def test_allocation_sensitivity_is_the_exponent(alpha):
    """Relative allocation error amplifies gain error by ``|1-alpha|/alpha``."""
    rng = np.random.default_rng(5)
    gains = np.exp(rng.normal(0, 0.4, 24))
    eta = rng.normal(0, 1e-4, 24)                 # tiny: first order only
    a_true = alpha_fair_allocation(gains, alpha=alpha)
    a_pred = alpha_fair_allocation(gains * np.exp(eta), alpha=alpha)
    moved = np.log(a_pred) - np.log(a_true)
    expected = allocation_exponent(alpha) * (eta - eta.mean())
    # the normaliser centres on an allocation-weighted mean rather than a plain
    # one, so compare the spread rather than the offset
    assert np.std(moved) == pytest.approx(np.std(expected), rel=1e-3)
    assert np.std(moved) / max(np.std(eta), 1e-30) == pytest.approx(
        abs(allocation_exponent(alpha)), rel=1e-3)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_fragility_law_predicts_measured_welfare_loss(alpha):
    """``(1-alpha)^2/(2 alpha) Var_w(eta)`` against the exact regret.

    At alpha = 1 both sides are identically zero -- the envy-free rule ignores
    the gains, so no gain error can reach it -- and the ratio test below would
    be 0/0, so that case asserts the zero directly.
    """
    rng = np.random.default_rng(2)
    gains = np.exp(rng.normal(0, 0.3, (200, 16)))
    eta = rng.normal(0, 0.02, (200, 16))
    alloc = alpha_fair_allocation(gains * np.exp(eta), alpha=alpha)
    measured = float(np.mean(relative_welfare_loss(gains, alloc, alpha)))
    predicted = float(np.mean(predicted_welfare_loss(
        gains, gains * np.exp(eta), alpha)))
    if alpha == 1.0:
        assert measured == pytest.approx(0.0, abs=1e-12)
        assert predicted == pytest.approx(0.0, abs=1e-12)
    else:
        assert measured == pytest.approx(predicted, rel=0.05)


def test_fragility_is_u_shaped_with_its_minimum_at_the_envy_free_point():
    """The surprising half of the claim, checked on its own.

    Both ends of the family are fragile and the fair middle is not, which is the
    reverse of the usual reading that fairness is what costs you robustness.
    """
    coefficients = {a: fragility_coefficient(a)
                    for a in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)}
    assert coefficients[1.0] == 0.0
    assert min(coefficients, key=coefficients.get) == 1.0
    # rises like 1/alpha toward max-efficiency and like alpha toward max-min
    assert coefficients[0.125] > coefficients[0.25] > coefficients[0.5] > 0
    assert coefficients[8.0] > coefficients[4.0] > coefficients[2.0] > 0
    assert np.isinf(fragility_coefficient(0.0))


@pytest.mark.parametrize("alpha", [0.25, 0.5, 2.0, 4.0, 8.0])
def test_implied_gains_invert_the_rule(alpha):
    """Round trip: the gains recovered from an allocation are the ones used."""
    rng = np.random.default_rng(8)
    gains = np.exp(rng.normal(0, 0.4, (20, 12)))
    alloc = alpha_fair_allocation(gains, alpha=alpha)
    recovered = implied_gains(alloc, alpha)
    assert np.allclose(recovered, gains / gains.mean(axis=-1, keepdims=True),
                       rtol=1e-8)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_implied_gains_refuse_the_uninformative_rules(alpha):
    """At alpha = 1 the rule ignores the gains, at alpha = 0 it reports only the
    argmax. Neither allocation can be inverted, and inventing a number would
    silently score an arm on an artefact."""
    alloc = np.full((3, 8), 1.0 / 8)
    with pytest.raises(ValueError, match="no information|starved"):
        implied_gains(alloc, alpha)


def test_shrinkage_is_the_identity_when_the_gains_are_exact():
    """Nothing to hedge against: the rule itself is optimal."""
    rng = np.random.default_rng(4)
    gains = np.exp(rng.normal(0, 0.4, (100, 12)))
    got = tune_shrinkage(gains, gains, alpha=8.0)
    assert got["shrink"] == pytest.approx(1.0)
    assert got["loss"] == pytest.approx(0.0, abs=1e-12)


def test_shrinkage_hedges_a_noisy_estimate_at_large_alpha():
    """With noisy gains the plug-in rule is beatable by pulling toward equal."""
    rng = np.random.default_rng(6)
    gains = np.exp(rng.normal(0, 0.3, (300, 12)))
    noisy = gains * np.exp(rng.normal(0, 0.25, gains.shape))
    got = tune_shrinkage(noisy, gains, alpha=8.0)
    plugin = alpha_fair_allocation(noisy, alpha=8.0)
    plugin_loss = float(np.mean(relative_welfare_loss(gains, plugin, 8.0)))
    assert got["shrink"] < 1.0
    assert got["loss"] < plugin_loss


# --------------------------------------------------------------------------
# reading the gains off a field
# --------------------------------------------------------------------------


def test_region_gains_are_block_means_in_row_major_order():
    field = np.zeros((4, 4, 2))
    field[0:2, 0:2, 0] = 0.5          # top-left block
    field[2:4, 2:4, 0] = -0.5         # bottom-right block
    gains = region_gains(field, blocks=2, offset=1.0)
    assert gains.shape == (4,)
    assert np.allclose(gains, [1.5, 1.0, 1.0, 0.5])


def test_region_gains_keeps_leading_axes():
    field = np.random.default_rng(0).normal(0, 0.2, (3, 7, 8, 8, 2))
    assert region_gains(field, blocks=4).shape == (3, 7, 16)


def test_region_gains_rejects_a_partition_that_does_not_divide():
    field = np.zeros((6, 6, 2))
    with pytest.raises(ValueError, match="does not divide"):
        region_gains(field, blocks=4)


def test_region_gains_rejects_a_non_positive_population():
    """A zero gain makes the egalitarian rules divide by zero. Stop, don't inf."""
    field = np.full((4, 4, 2), -3.0)
    with pytest.raises(ValueError, match="non-positive"):
        region_gains(field, blocks=2, offset=1.5)


@pytest.mark.parametrize("regime,kwargs,ceiling,ratio", [
    ("settled", {}, 1e-3, 100.0),
    ("defect-bearing", dict(perturbation=0.8, max_mode=4, spinup=20), 3e-2, 5.0),
])
def test_amplitude_would_have_been_the_degenerate_choice(regime, kwargs,
                                                         ceiling, ratio):
    """Why the gain is ``offset + u`` and not ``|A|^2``, pinned as a test.

    On the settled lambda-omega limit cycle the amplitude relaxes to 1
    everywhere, so a gain read from ``|A|^2`` is constant across regions to four
    decimal places (measured 7e-5) and every rule in the family collapses onto
    equal division. The structure of an oscillatory medium is in its phase.

    The regime the experiment actually runs on is younger and carries amplitude
    defects, so its ``|A|^2`` is not flat -- 0.017, which is why the two cases
    are parametrised rather than asserted as one number. It is still 8x below
    the phase readout, so the choice holds there too, for a weaker reason.

    Cheap to re-derive and expensive to rediscover, which is what makes it a
    test rather than a comment.
    """
    from litefno.systems import lambda_omega

    traj = lambda_omega(n_traj=1, n_steps=4, size=32, seed=0, **kwargs)
    amplitude = (traj ** 2).sum(axis=-1)[..., None]
    flat = region_gains(np.concatenate([amplitude, amplitude], axis=-1),
                        blocks=4, offset=0.0)
    structured = region_gains(traj, blocks=4)
    cv_flat = float(np.mean(flat.std(-1) / flat.mean(-1)))
    cv_structured = float(np.mean(structured.std(-1) / structured.mean(-1)))
    assert cv_flat < ceiling
    assert cv_structured > ratio * cv_flat
