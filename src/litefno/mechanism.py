r"""Manipulation, leximin, and no-regret: the allocation rule as a mechanism.

Board task: "borrow from algorithmic game theory / mechanism design -- use
game-theoretic fairness notions (strategy-proof, truthful, no-regret) to ensure
the allocation recommendation is not just statistically fair but robust to
manipulation; implement a lightweight version (e.g. lexicographic max-min
fairness)."

:mod:`litefno.allocation` asks what a rule loses when its input is *wrong by
accident*. This module asks what it loses when its input is wrong *on purpose*,
and what can be done about it cheaply.

The threat model
----------------
The gains are read off a reconstructed field, and in any real deployment the
observations feeding that reconstruction come from somewhere -- sensors, counts,
reports -- that the regions themselves often control. So the manipulation is:
region ``r`` distorts what the allocator sees about **its own region only**,
within a bound, and everyone else is truthful. That is the standard
single-deviation question of mechanism design, and it is the one the rules here
can actually be scored on.

The result that organises everything
------------------------------------
Region ``r``'s allocation under the alpha-fair rule is ``a_r ∝ ghat_r^beta``
with ``beta = (1-alpha)/alpha``, and its utility is increasing in ``a_r``
whatever its true gain is. So the best response is always a *corner*: report the
largest gain you can when ``beta > 0``, the smallest when ``beta < 0``. There is
no interior optimum, and the incentive ratio is available in closed form
(:func:`incentive_ratio_formula`).

Its logarithm is ``|beta| log kappa`` to first order -- and ``|beta|`` is
exactly the amplification factor that :mod:`litefno.allocation` derives for
*accidental* error. **The incentive to lie and the sensitivity to error are the
same derivative.** A rule cannot be made robust to manipulation without being
made robust to error, or vice versa; both are bought by ignoring the state, and
at ``alpha = 1`` the rule ignores the state completely and is strategy-proof,
envy-free and error-free at once. Every other member of the family is
manipulable, and the strategy-proof one is the one that uses no information.

That is a local instance of a general impossibility (Hurwicz 1972): efficiency,
envy-freeness and strategy-proofness do not coexist. Nothing here evades it. The
useful question is not how to be strategy-proof but how to be *boundedly*
manipulable while still using the data, which is what the capacity cap below
does.

Leximin, and why the cap is the mechanism
------------------------------------------
:func:`leximin_allocation` is the requested lightweight implementation:
lexicographic max-min over region outcomes, subject to a budget and per-region
capacities, by progressive filling. Without capacities leximin degenerates --
equalising outcomes is always feasible, so it collapses onto the ``alpha = inf``
rule that :mod:`litefno.allocation` already has, and lexicographic refinement
never gets past its first level. With capacities the two come apart, the
refinement does work, and something useful appears: **a region can never receive
more than its cap, so however it lies, its gain is bounded by ``c_r / a_r``.**

The cap turns an unbounded incentive into a bounded one without any
truthfulness machinery -- no payments, no verification, no reports at all. It is
a dial rather than a guarantee: at a cap of one equal share the allocation is
uniform and unmanipulable and uses nothing, and as the cap grows the rule
regains its responsiveness and its exposure together.

No-regret, and what its guarantee is actually against
------------------------------------------------------
:func:`online_allocation` runs exponentiated gradient over the simplex, which is
the no-regret learner for this setting: it needs no surrogate, no model and no
forecast, and its regret against the best *fixed* allocation in hindsight
vanishes as ``O(1/sqrt(T))``.

The guarantee is worth stating precisely, because the comparator is the whole of
it. "No-regret" here means no regret *against a constant allocation*. In an
oscillating ecosystem the best constant allocation is a poor thing to be as good
as, and a learner that provably converges to it can still be far worse than a
forecast. ``scripts/strategic_allocation.py`` measures both halves: that the
regret does vanish, and what it is worth.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from litefno.allocation import (
    allocation_exponent, alpha_fair_allocation, outcomes, welfare_ce)


# --------------------------------------------------------------------------
# strategy-proofness
# --------------------------------------------------------------------------


def report_bounds(gains: np.ndarray, kappa: Optional[float] = None,
                  epsilon: Optional[float] = None) -> tuple:
    """The interval each region could report, from a multiplicative or additive
    bound.

    Two bounds because the two halves of the study need different ones and
    conflating them would be sloppy. ``kappa`` is multiplicative and is the
    natural language for theory (the incentive ratio is a power of it).
    ``epsilon`` is additive and is what a *field-level* attack gives: a region
    that biases the observations inside its own block by at most ``eps`` per
    pixel moves its block mean, and therefore its gain, by at most ``eps``.
    Passing both is a caller error rather than a silently applied composition.
    """
    g = np.asarray(gains, dtype=float)
    if (kappa is None) == (epsilon is None):
        raise ValueError("give exactly one of kappa (multiplicative) or "
                         "epsilon (additive)")
    if kappa is not None:
        if kappa < 1:
            raise ValueError(f"kappa must be >= 1, got {kappa}")
        return g / kappa, g * kappa
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    # a reported gain must stay positive; the floor is what stops an additive
    # attack from claiming a region is empty and, under alpha > 1, being handed
    # the entire budget for it
    return np.maximum(g - epsilon, 1e-9), g + epsilon


def best_response(gains: np.ndarray, alpha: float, kappa: Optional[float] = None,
                  epsilon: Optional[float] = None) -> np.ndarray:
    """What each region would report, deviating alone, to maximise its own share.

    Returns (R,) of reported gains -- one per region, each the best response
    *assuming the others are truthful*. This is not a joint deviation and not an
    equilibrium; it is the single-deviation quantity strategy-proofness is
    defined by.

    The best response is a corner because ``a_r`` is monotone in ``ghat_r^beta``
    and the region's utility is monotone in ``a_r``: overstate when ``beta > 0``
    (rules near max-efficiency reward looking big), understate when ``beta < 0``
    (egalitarian rules reward looking needy). At ``beta = 0`` the report does
    nothing and the truthful value is returned.
    """
    g = np.asarray(gains, dtype=float)
    low, high = report_bounds(g, kappa, epsilon)
    beta = allocation_exponent(alpha)
    if beta == 0:
        return g.copy()
    if not np.isfinite(beta):
        # alpha = 0 takes the argmax, so the only useful lie is to look biggest
        return high.copy()
    return high.copy() if beta > 0 else low.copy()


def incentive_ratio(gains: np.ndarray, alpha: float,
                    kappa: Optional[float] = None,
                    epsilon: Optional[float] = None,
                    budget: float = 1.0) -> np.ndarray:
    """``a_r(best lie) / a_r(truth)`` per region, others truthful.

    One number per region, ``>= 1``, equal to 1 exactly when the region cannot
    gain by lying. The maximum over regions is the mechanism's manipulability:
    a mechanism is strategy-proof iff this is 1 everywhere.

    Utility is monotone in the allocation received, so the ratio of allocations
    *is* the ratio of utilities and there is no need to pick a utility scale --
    which matters, because the alpha-fair utility changes sign at alpha = 1 and
    a ratio of it would be meaningless there.
    """
    g = np.asarray(gains, dtype=float)
    truthful = alpha_fair_allocation(g, alpha=alpha, budget=budget)
    lies = best_response(g, alpha, kappa, epsilon)

    ratios = np.empty(g.shape[-1])
    for r in range(g.shape[-1]):
        reported = g.copy()
        reported[r] = lies[r]
        got = alpha_fair_allocation(reported, alpha=alpha, budget=budget)[r]
        ratios[r] = got / max(truthful[r], 1e-300)
    return ratios


def incentive_ratio_formula(share: float, alpha: float, kappa: float) -> float:
    """Closed form for the incentive ratio, as a check on the measurement.

    With ``w_r = g_r^beta`` and ``s_r = w_r / sum_s w_s`` the region's truthful
    share, a report scaled to ``kappa^|beta| w_r`` gives

        ratio = kappa^|beta| / (1 + (kappa^|beta| - 1) s_r)

    which tends to ``kappa^|beta|`` for a region holding a small share and to 1
    for a region already holding everything -- a region with nothing to gain
    cannot gain. Its log is ``|beta| log kappa`` to first order, which is the
    same coefficient that governs sensitivity to accidental error.
    """
    beta = allocation_exponent(alpha)
    if beta == 0:
        return 1.0
    if not np.isfinite(beta):
        raise ValueError("alpha = 0 is a discontinuous argmax rule; its "
                         "incentive ratio is not given by this expansion")
    scale = kappa ** abs(beta)
    return float(scale / (1 + (scale - 1) * share))


def manipulation_damage(gains: np.ndarray, alpha: float, region: int,
                        kappa: Optional[float] = None,
                        epsilon: Optional[float] = None,
                        budget: float = 1.0) -> dict:
    """What one region's lie costs everyone, not just what it wins.

    The incentive ratio says whether a region *wants* to lie. It does not say
    whether anyone should care: a manipulation that moves a negligible amount of
    resource is a curiosity. This reports the welfare the group loses when the
    allocator acts on the lie, in the same relative certainty-equivalent units
    as :func:`litefno.allocation.relative_welfare_loss`, so the two halves of
    the mechanism question are on one scale.
    """
    g = np.asarray(gains, dtype=float)
    reported = g.copy()
    reported[region] = best_response(g, alpha, kappa, epsilon)[region]

    honest = alpha_fair_allocation(g, alpha=alpha, budget=budget)
    manipulated = alpha_fair_allocation(reported, alpha=alpha, budget=budget)
    ce_honest = welfare_ce(outcomes(g, honest), alpha)
    ce_manipulated = welfare_ce(outcomes(g, manipulated), alpha)
    return {"region": int(region),
            "incentive_ratio": float(manipulated[region] / honest[region]),
            "welfare_loss": float(1.0 - ce_manipulated / ce_honest)}


# --------------------------------------------------------------------------
# leximin
# --------------------------------------------------------------------------


def leximin_allocation(gains: np.ndarray, budget: float = 1.0,
                       caps: Optional[np.ndarray] = None) -> np.ndarray:
    """Lexicographic max-min allocation by progressive filling.

    Maximises the smallest region outcome; among the allocations that do,
    maximises the next smallest; and so on -- subject to ``sum_r a_r = budget``
    and ``0 <= a_r <= caps_r``.

    With outcomes ``x_r = g_r a_r`` the algorithm is a water-filling: raise a
    common outcome level ``t``, give each region ``a_r = min(t / g_r, c_r)``,
    and stop when the budget is exhausted. A region that hits its cap freezes at
    ``x_r = g_r c_r`` and the rest keep rising, so the final outcome vector is
    ``min(t, g_r c_r)``. Every capped region is already at the most it can
    receive and every uncapped one sits at the common level, which is exactly
    the lexicographic optimum -- no reshuffle can raise any region without
    lowering one that is no higher.

    Without caps this collapses to ``a ∝ 1/g``, the alpha = inf rule, and the
    lexicographic refinement is vacuous because equalising is always feasible.
    That is why the caps are the interesting part rather than a detail: they are
    what makes leximin a different object from max-min, and they are what bounds
    manipulation.

    Solved by sweeping the cap thresholds in order rather than by bisection, so
    the answer is exact instead of converged-to.
    """
    g = np.asarray(gains, dtype=float)
    if np.any(g <= 0):
        raise ValueError(f"gains must be positive, got min {g.min()}")
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    n = g.shape[-1]

    if caps is None:
        weights = 1.0 / g
        return budget * weights / weights.sum()

    c = np.broadcast_to(np.asarray(caps, dtype=float), g.shape).copy()
    if np.any(c < 0):
        raise ValueError("capacities must be non-negative")
    total = c.sum()
    if total < budget - 1e-12:
        raise ValueError(
            f"capacities sum to {total:.6g}, below the budget {budget:.6g}; "
            "the problem is infeasible and there is no allocation to return")
    if abs(total - budget) <= 1e-12:
        return c                                   # only one feasible point

    # outcome level at which each region hits its cap
    threshold = g * c
    order = np.argsort(threshold)
    filled, freed = 0.0, float(np.sum(1.0 / g))
    previous = 0.0
    for k in range(n):
        idx = order[k]
        level = (budget - filled) / freed if freed > 0 else np.inf
        if level <= threshold[idx] + 1e-15:
            return np.minimum(level / g, c)
        previous = threshold[idx]
        filled += c[idx]
        freed -= 1.0 / g[idx]
    # every region capped: only reachable when the capacities exactly fund the
    # budget, which was handled above, so a fall-through is a bug not an input
    raise RuntimeError(
        f"progressive filling did not terminate (budget {budget}, capacity "
        f"{total}, last threshold {previous}); this is an implementation bug")


def leximin_incentive_ratio(gains: np.ndarray, budget: float = 1.0,
                            caps: Optional[np.ndarray] = None,
                            kappa: Optional[float] = None,
                            epsilon: Optional[float] = None) -> np.ndarray:
    """``a_r(best lie) / a_r(truth)`` under leximin, by direct search.

    Leximin is not a power rule, so its best response is not available by
    inspecting an exponent. It is still monotone -- understating a gain raises
    the resource needed to reach the common outcome level -- so the corner
    argument survives and both corners are evaluated rather than assumed.

    The cap is what makes this bounded: whatever a region reports, it cannot be
    given more than ``c_r``, so the ratio is at most ``c_r / a_r``.
    """
    g = np.asarray(gains, dtype=float)
    truthful = leximin_allocation(g, budget, caps)
    low, high = report_bounds(g, kappa, epsilon)

    ratios = np.empty(g.shape[-1])
    for r in range(g.shape[-1]):
        best = truthful[r]
        for value in (low[r], high[r]):
            reported = g.copy()
            reported[r] = value
            best = max(best, leximin_allocation(reported, budget, caps)[r])
        ratios[r] = best / max(truthful[r], 1e-300)
    return ratios


# --------------------------------------------------------------------------
# no regret
# --------------------------------------------------------------------------


def _welfare_gradient(gains: np.ndarray, alloc: np.ndarray, alpha: float
                      ) -> np.ndarray:
    """d/da_r of ``sum_r U_alpha(g_r a_r)`` = ``g_r^(1-alpha) a_r^(-alpha)``."""
    safe = np.maximum(alloc, 1e-12)
    return gains ** (1 - alpha) * safe ** (-alpha)


def online_allocation(gain_stream: np.ndarray, alpha: float = 2.0,
                      budget: float = 1.0, eta: float = 0.5,
                      warm_start: Optional[np.ndarray] = None) -> dict:
    """Exponentiated gradient over the simplex: the no-regret allocator.

    Plays an allocation, *then* sees the gains that were realised, and updates.
    It never sees a forecast and holds no model of the dynamics -- which is the
    point of including it, since everything else in ext22 depends on a
    surrogate.

    The gradient is normalised by its largest component before it enters the
    exponent. Un-normalised, ``a_r^(-alpha)`` at alpha = 8 and a small ``a_r``
    is astronomically large and the update collapses onto one region; the step
    size would then have to absorb a scale that varies over the run, which is
    not a step size but a bug. Normalising leaves the *direction* the theory
    needs and puts the scale in ``eta``, decayed as ``eta / sqrt(t)`` for the
    usual ``O(1/sqrt(T))`` bound.
    """
    stream = np.asarray(gain_stream, dtype=float)
    if stream.ndim != 2:
        raise ValueError(f"expected (T, R) gain stream, got {stream.shape}")
    steps, n = stream.shape

    alloc = (np.full(n, budget / n) if warm_start is None
             else budget * np.asarray(warm_start, float)
             / np.sum(warm_start))
    played, realised = np.empty((steps, n)), np.empty(steps)
    for t in range(steps):
        played[t] = alloc
        gains = stream[t]
        realised[t] = welfare_ce(outcomes(gains, alloc), alpha)
        grad = _welfare_gradient(gains, alloc, alpha)
        grad = grad / max(np.max(np.abs(grad)), 1e-30)
        step = eta / np.sqrt(t + 1)
        alloc = alloc * np.exp(step * grad)
        alloc = budget * alloc / alloc.sum()
    return {"allocations": played, "welfare": realised,
            "final": alloc, "alpha": alpha}


def _mean_welfare(stream: np.ndarray, alloc: np.ndarray, alpha: float) -> float:
    return float(np.mean([welfare_ce(outcomes(g, alloc), alpha)
                          for g in stream]))


def best_fixed_allocation(gain_stream: np.ndarray, alpha: float = 2.0,
                          budget: float = 1.0, iterations: int = 4000,
                          eta: float = 0.5) -> np.ndarray:
    """The comparator: the single best allocation in hindsight over the stream.

    Maximises ``sum_t sum_r U_alpha(g_tr a_r)``, a concave problem on the
    simplex, by the same exponentiated-gradient step run offline. This is what
    the no-regret guarantee is *against*, so it has to actually be the best
    fixed allocation -- a comparator that has not converged makes the regret
    look smaller than it is, and at large alpha it does not converge from a
    uniform start within any reasonable number of iterations.

    Two defences, because a silently bad comparator is the failure mode that
    would invalidate every regret number downstream. The iterate is scored as it
    goes and the best one is kept rather than the last, since the decaying step
    does not guarantee monotone improvement. And the search is seeded with
    allocations that are already good -- uniform, and the alpha-fair optimum for
    the stream's mean gains -- so the returned answer is never worse than the
    obvious candidates. Uniform being one of them is what makes the result
    trustworthy: the comparator can never come out worse than a constant split,
    which is a feasible fixed allocation and would otherwise be a silent
    contradiction.
    """
    stream = np.asarray(gain_stream, dtype=float)
    n = stream.shape[1]
    uniform = np.full(n, budget / n)
    seeds = [uniform, alpha_fair_allocation(stream.mean(axis=0), alpha=alpha,
                                            budget=budget)]

    best, best_value = uniform, _mean_welfare(stream, uniform, alpha)
    for seed in seeds:
        alloc = seed.copy()
        for i in range(iterations):
            grad = np.mean([_welfare_gradient(g, alloc, alpha) for g in stream],
                           axis=0)
            grad = grad / max(np.max(np.abs(grad)), 1e-30)
            alloc = alloc * np.exp((eta / np.sqrt(i + 1)) * grad)
            alloc = budget * alloc / alloc.sum()
            if i % 50 == 0 or i == iterations - 1:
                value = _mean_welfare(stream, alloc, alpha)
                if value > best_value:
                    best, best_value = alloc.copy(), value
    return best


def regret_curve(gain_stream: np.ndarray, alpha: float = 2.0,
                 budget: float = 1.0, eta: float = 0.5) -> dict:
    """Average regret against the best fixed allocation, as a function of T.

    Reported in certainty-equivalent units per round, so it is comparable with
    every other welfare number in the extension rather than being in units of
    the raw alpha-fair sum, which changes sign at alpha = 1.
    """
    stream = np.asarray(gain_stream, dtype=float)
    run = online_allocation(stream, alpha, budget, eta)
    fixed = best_fixed_allocation(stream, alpha, budget)
    fixed_welfare = np.array(
        [welfare_ce(outcomes(g, fixed), alpha) for g in stream])

    steps = np.arange(1, len(stream) + 1)
    cumulative = np.cumsum(fixed_welfare - run["welfare"])
    return {"steps": steps, "average_regret": cumulative / steps,
            "online_welfare": run["welfare"], "fixed_welfare": fixed_welfare,
            "best_fixed": fixed, "allocations": run["allocations"]}
