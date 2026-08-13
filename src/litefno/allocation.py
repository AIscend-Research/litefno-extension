r"""Fair resource allocation over regions of a reconstructed ecosystem state.

Board task: "Add fairness-aware resource allocation layer: train an auxiliary
network that, given the reconstructed ecosystem state, predicts a fair resource
allocation across populations/regions (not just max-efficiency, but satisfying
fairness constraints like max-min or envy-free)."

This module is the decision layer's mathematics; the auxiliary network itself is
:mod:`litefno.models.allocator` and the experiment is
``scripts/fair_allocation.py``.

What is physics here and what is a modelling choice
---------------------------------------------------
The surrogate's output is physics: a predicted field from a PDE this repo
integrates. Everything downstream of :func:`region_gains` is a *stylized
decision problem* bolted onto it, and saying so plainly is the only way the
result means anything. Nothing in Gray-Scott or in the lambda-omega normal form
tells you what a region deserves. What the experiment measures is not whether
the allocation is ethically right -- that is not a question a PDE can answer --
but how error in a learned surrogate propagates into a decision taken on its
output, which is a property of the *composition* and is measurable.

The allocation problem
----------------------
``R`` regions partition the domain. Region ``r`` has a gain ``g_r > 0`` read off
the state, a scarce divisible resource of total size ``B`` is split as ``a_r >=
0`` with ``sum_r a_r = B``, and the outcome region r realises is

    x_r = g_r a_r

so a unit of resource does more good where the population is larger. The
allocation rule is the alpha-fair family (Mo & Walrand 2000), maximising

    W_alpha(a) = sum_r U_alpha(x_r),   U_alpha(x) = x^(1-alpha)/(1-alpha)
                                                   (log x at alpha = 1)

which has a closed-form optimum on the simplex,

    a_r  ∝  g_r^((1-alpha)/alpha)                                        (*)

and that single exponent is the whole family:

======  ==========================  ===================================
alpha   allocation                  the rule it is
======  ==========================  ===================================
0       all of B to argmax g_r      max-efficiency (utilitarian)
1/2     a ∝ g                       proportional to population
1       a equal                     proportional fairness = **envy-free**
2, 4    a ∝ g^-1/2, g^-3/4          increasingly egalitarian
inf     a ∝ 1/g  (x_r all equal)    **max-min** (Rawlsian, leximin)
======  ==========================  ===================================

Why envy-freeness sits at alpha = 1 and not somewhere more interesting
----------------------------------------------------------------------
Envy-freeness (Foley 1967, Varian 1974) asks that no region prefer another's
bundle valued by its own utility. Region r values bundle ``a_s`` at ``g_r a_s``,
so r envies s exactly when ``a_s > a_r`` -- the gain cancels. With a *single*
homogeneous divisible resource and utilities increasing in the amount received,
envy-freeness therefore forces equal division, which is (*) at alpha = 1. That
is a theorem, not an artefact of this setup, and it has a consequence worth
stating loudly: **the envy-free allocation does not depend on the state at
all**, so no forecast, surrogate or otherwise, can improve or damage it.

An alternative reading, "envy over outcomes" -- r envies s when ``x_s > x_r`` --
is not classical envy-freeness but is what people usually mean informally; it
forces equal *outcomes*, which is (*) at alpha = inf. So the two readings of
envy-freeness land on the two ends of the family that this module already
covers, and both are reported.

Genuine, non-degenerate envy-freeness needs *heterogeneous* valuations over
several resource types, where the Nash/CEEI solution is envy-free and Pareto
efficient without collapsing to equal split. That is a different experiment and
is named in the doc's limits rather than faked here with one good.

The fragility law
-----------------
The reason this is worth running on a surrogate at all. Feed (*) a perturbed
gain ``ghat_r = g_r e^(eta_r)`` instead of the truth. In logs the allocation
moves by ``((1-alpha)/alpha)(eta_r - mean)``, so *allocation* error is amplified
by exactly

    |1 - alpha| / alpha

which is 0 at alpha = 1, diverges as alpha -> 0, and tends to 1 as alpha ->
inf. Welfare is second order at its own optimum (envelope theorem), and
expanding the certainty-equivalent :func:`welfare_ce` around it gives the
sharper statement that :func:`fragility_coefficient` returns,

    relative welfare loss  ≈  (1-alpha)^2 / (2 alpha)  Var_w(eta)

with ``Var_w`` the allocation-weighted variance of the log gain errors. So
fragility to surrogate error is **U-shaped in alpha with an exact zero at the
envy-free point**, rising like 1/alpha toward max-efficiency and like alpha
toward max-min. Both extremes are fragile and the fair middle is not, which is
the opposite of the usual "fairness costs you something" intuition and is a
prediction that can be checked rather than asserted -- ``tests/
test_allocation.py`` checks it against numerically exact optima, and
``scripts/fair_allocation.py`` checks it against a real surrogate's errors.

At alpha = 0 the expansion does not apply: the optimum sits on a vertex of the
simplex and moves discontinuously, so its loss is first order in the gain error,
not second. That case is measured directly rather than predicted.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Population read as an offset from the Hopf normal form's fixed point. The
# lambda-omega state u oscillates in [-1, 1] about an unstable equilibrium, so a
# density is u plus a carrying level; 1.5 keeps it strictly positive (gains must
# be) with a 3:1 spread between the fullest and emptiest pixel. It is a units
# choice, and region_gains takes it as an argument so the choice is visible at
# the call site rather than buried.
POPULATION_OFFSET = 1.5


def region_gains(field: np.ndarray, blocks: int = 4,
                 offset: float = POPULATION_OFFSET, channel: int = 0
                 ) -> np.ndarray:
    """Per-region population density, as the gains of the allocation problem.

    ``field`` is (..., H, W, C) -- the layout the repo's loaders use -- and the
    return is (..., blocks*blocks) in row-major region order. Regions are a
    fixed square partition: a coarse administrative grid over the domain, not
    anything the dynamics singles out.

    The density is ``offset + u`` averaged over the block, where ``u`` is
    ``channel``. Not ``|A|^2``, which is the tempting choice and is useless
    here: on the lambda-omega limit cycle the amplitude relaxes to 1 everywhere,
    so ``|A|^2`` has a region-to-region coefficient of variation of 1e-4 and
    every allocation rule collapses onto the uniform one. The spatial structure
    of an oscillatory medium lives in its *phase*, which is what a single
    component of the field sees.

    Raises if any region comes out non-positive: a zero gain makes the
    egalitarian rules divide by zero and the log-welfare undefined, and it means
    the offset is wrong for the field being passed, which is worth stopping for
    rather than propagating as an inf.
    """
    arr = np.asarray(field, dtype=float)
    if arr.ndim < 3:
        raise ValueError(f"expected (..., H, W, C), got {arr.shape}")
    height, width = arr.shape[-3], arr.shape[-2]
    if height % blocks or width % blocks:
        raise ValueError(
            f"{height}x{width} does not divide into {blocks}x{blocks} regions")

    lead = arr.shape[:-3]
    dens = offset + arr[..., channel]
    dens = dens.reshape(*lead, blocks, height // blocks, blocks, width // blocks)
    gains = dens.mean(axis=(-3, -1)).reshape(*lead, blocks * blocks)
    if np.any(gains <= 0):
        raise ValueError(
            f"non-positive region gain (min {gains.min():.4f}); the population "
            f"offset {offset} is too small for this field")
    return gains


# --------------------------------------------------------------------------
# the allocation rules
# --------------------------------------------------------------------------


def allocation_exponent(alpha: float) -> float:
    """``(1 - alpha) / alpha``, the exponent in ``a ∝ g^beta``.

    Also the signed amplification of relative gain error into relative
    allocation error, which is why it is exposed rather than inlined: the
    experiment reports it next to the measured amplification.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if np.isinf(alpha):
        return -1.0
    if alpha == 0:
        return np.inf
    return (1.0 - alpha) / alpha


def alpha_fair_allocation(gains: np.ndarray, alpha: float = 1.0,
                          budget: float = 1.0,
                          shrink: float = 1.0) -> np.ndarray:
    """The alpha-fair optimum ``a_r ∝ g_r^((1-alpha)/alpha)``, on the simplex.

    Vectorized over leading axes: ``gains`` is (..., R) and the result is the
    same shape, each row summing to ``budget``.

    ``alpha = 0`` is winner-take-all and is handled as its own case rather than
    by letting the exponent overflow; exact ties split the budget between the
    tied regions, because breaking them by index would make the rule depend on
    the region numbering.

    ``shrink`` multiplies the exponent, interpolating between the rule
    (``1.0``) and equal division (``0.0``). It is not part of the fairness
    definition -- it is the one-parameter hedge the experiment fits when the
    gains are a surrogate's estimate rather than the truth, and it exists here
    so that the "what did the network learn" comparison has an interpretable
    thing to be compared against.
    """
    g = np.asarray(gains, dtype=float)
    if g.shape[-1] == 0:
        raise ValueError("need at least one region")
    if np.any(g <= 0):
        raise ValueError(f"gains must be positive, got min {g.min()}")

    if alpha == 0 and shrink == 1.0:
        top = g.max(axis=-1, keepdims=True)
        winners = (g >= top).astype(float)
        return budget * winners / winners.sum(axis=-1, keepdims=True)

    beta = allocation_exponent(alpha) * shrink
    if not np.isfinite(beta):
        raise ValueError(
            "alpha = 0 has no finite exponent; shrink it to a finite rule or "
            "use shrink = 1 for the exact winner-take-all case")
    # in logs: a wide gain spread with a large |beta| overflows g**beta in
    # float64 well before the normalised result would
    logw = beta * np.log(g)
    logw = logw - logw.max(axis=-1, keepdims=True)
    w = np.exp(logw)
    return budget * w / w.sum(axis=-1, keepdims=True)


def outcomes(gains: np.ndarray, allocation: np.ndarray) -> np.ndarray:
    """What each region actually realises, ``x_r = g_r a_r``."""
    return np.asarray(gains, dtype=float) * np.asarray(allocation, dtype=float)


def implied_gains(allocation: np.ndarray, alpha: float) -> np.ndarray:
    """Invert the rule: which gains would make this allocation optimal.

    ``a_r ∝ g_r^beta`` gives ``g_r ∝ a_r^(1/beta)``, normalised to mean 1 since
    the rule is scale-free in the gains and the overall level is not recoverable
    from an allocation.

    This is how an allocator that never outputs a gain estimate can still be
    scored as an *estimator*. A network trained only on realised welfare has an
    implied belief about the state, and comparing that belief to the truth
    separates two very different reasons a learned allocator might beat a
    plug-in rule: because it hedges against a bad input, or because it reads the
    input better than the rule does.

    Undefined at ``alpha = 1`` and ``alpha = 0``, and raised on rather than
    fudged. At alpha = 1 the rule ignores the gains entirely, so its output
    carries no information about them; at alpha = 0 it reports only which region
    is largest. In both cases any returned number would be an artefact.
    """
    beta = allocation_exponent(alpha)
    if beta == 0 or not np.isfinite(beta):
        raise ValueError(
            f"alpha = {alpha} has exponent {beta}; the allocation carries no "
            "information about the gains, so they cannot be recovered from it")
    a = np.asarray(allocation, dtype=float)
    if np.any(a <= 0):
        raise ValueError("cannot invert an allocation with a starved region")
    g = a ** (1.0 / beta)
    return g / g.mean(axis=-1, keepdims=True)


def _log_mean_exp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    # an all -inf row (every outcome starved) shifts by -inf and gives nan
    # instead of the log(0) it should; the shift is arbitrary, so pick a finite
    # one there
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.squeeze(
        peak + np.log(np.mean(np.exp(values - peak), axis=axis, keepdims=True)),
        axis=axis)


def welfare_ce(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Certainty-equivalent welfare: the power mean of outcomes, exponent 1-a.

    ``(mean_r x_r^(1-alpha))^(1/(1-alpha))``, with the geometric mean at
    alpha = 1 and ``min_r x_r`` at alpha = inf.

    Reported instead of ``sum_r U_alpha(x_r)`` for one reason that matters: the
    power mean is in *units of outcome* for every alpha, so a 3% welfare loss
    means the same thing at alpha = 1/2 and at alpha = 8 and the fragility
    curve across alpha is a comparison rather than a units artefact. The raw
    sum is not comparable across alpha at all -- it changes sign at alpha = 1.

    It is a strictly increasing transform of ``sum_r U_alpha``, so it has the
    same maximiser; :func:`alpha_fair_allocation` optimises either one.

    Computed through logs. ``x^(1-alpha)`` at alpha = 16 and x = 0.1 is 1e15,
    and the intermediate overflows long before anything in the result would.

    A starved region (``x_r = 0``) is allowed, because the utilitarian rule
    produces one on purpose: it gives the entire budget to one region and leaves
    the rest at zero. The power mean is still defined there -- it is the mean
    itself at alpha = 0, and exactly 0 for every alpha >= 1, where starving one
    region is unboundedly bad -- so those limits are returned rather than raised
    on. Only negative outcomes are an error, and they mean a negative gain got
    past :func:`region_gains`.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("outcomes must be non-negative for alpha-fair welfare")
    if np.isinf(alpha):
        return x.min(axis=-1)
    if alpha == 0:
        return x.mean(axis=-1)
    starved = np.any(x == 0, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logx = np.log(np.where(x > 0, x, 1.0))
        # a zero outcome contributes nothing to the mean when the exponent is
        # positive (alpha < 1) and dominates it when it is negative, so the two
        # cases are carried explicitly instead of through log(0)
        if alpha == 1:
            out = np.exp(logx.mean(axis=-1))
        elif alpha < 1:
            terms = np.where(x > 0, (1 - alpha) * logx, -np.inf)
            out = np.exp(_log_mean_exp(terms) / (1 - alpha))
        else:
            out = np.exp(_log_mean_exp((1 - alpha) * logx) / (1 - alpha))
    return np.where(starved, 0.0, out) if alpha >= 1 else out


def relative_welfare_loss(true_gains: np.ndarray, allocation: np.ndarray,
                          alpha: float = 1.0, budget: float = 1.0
                          ) -> np.ndarray:
    """``1 - CE(a) / CE(a*)``, the regret of an allocation, per row.

    The optimum ``a*`` is recomputed from ``true_gains`` for the same alpha, so
    this is regret against the best that rule could have done on the true state
    -- not against a different rule. Every arm in the experiment is scored this
    way, which is what makes the arms comparable within an alpha while the
    fairness rule itself changes across alpha.
    """
    g = np.asarray(true_gains, dtype=float)
    best = alpha_fair_allocation(g, alpha=alpha, budget=budget)
    ce_best = welfare_ce(outcomes(g, best), alpha)
    ce_got = welfare_ce(outcomes(g, allocation), alpha)
    return 1.0 - ce_got / ce_best


def price_of_fairness(gains: np.ndarray, alpha: float = 1.0,
                      budget: float = 1.0) -> np.ndarray:
    """Utilitarian welfare given up by using the alpha-fair rule, as a fraction.

    ``1 - mean(x under alpha-fair) / mean(x under alpha = 0)``, following
    Bertsimas, Farias & Trichakis (2011): the price of fairness is measured in
    the *efficiency* objective, since that is what fairness is being traded
    against. Zero when all gains are equal, because then no trade-off exists.
    """
    g = np.asarray(gains, dtype=float)
    efficient = alpha_fair_allocation(g, alpha=0.0, budget=budget)
    fair = alpha_fair_allocation(g, alpha=alpha, budget=budget)
    top = welfare_ce(outcomes(g, efficient), 0.0)
    got = welfare_ce(outcomes(g, fair), 0.0)
    return 1.0 - got / top


# --------------------------------------------------------------------------
# fairness diagnostics
# --------------------------------------------------------------------------


def max_min_ratio(x: np.ndarray) -> np.ndarray:
    """``min_r x_r / max_r x_r``: 1 when outcomes are equalised, 0 when starved.

    The direct reading of how egalitarian an allocation turned out, independent
    of which rule produced it.
    """
    x = np.asarray(x, dtype=float)
    return x.min(axis=-1) / np.maximum(x.max(axis=-1), 1e-300)


def max_envy(allocation: np.ndarray) -> np.ndarray:
    """Largest bundle-envy, ``max_{r,s} (a_s - a_r) / mean(a)``, zero iff equal.

    Classical envy compares bundles under the envier's own valuation, and with
    one homogeneous divisible resource region r values ``a_s`` at ``g_r a_s``,
    so the gain cancels and envy is decided by the raw amounts. That makes this
    a spread measure -- ``(max a - min a) / mean a`` -- and makes equal division
    the only envy-free allocation. Both facts are the point rather than a
    shortcoming of the metric: they are why the envy-free rule needs no state.
    """
    a = np.asarray(allocation, dtype=float)
    mean = np.maximum(a.mean(axis=-1), 1e-300)
    return (a.max(axis=-1) - a.min(axis=-1)) / mean


def outcome_envy(x: np.ndarray) -> np.ndarray:
    """The informal reading: envy over realised outcomes rather than bundles.

    Zero exactly when outcomes are equalised, which is the alpha = inf rule.
    Kept separate from :func:`max_envy` because they are different definitions
    that happen to be given the same name in casual use, and they are minimised
    at opposite ends of the family.
    """
    x = np.asarray(x, dtype=float)
    mean = np.maximum(x.mean(axis=-1), 1e-300)
    return (x.max(axis=-1) - x.min(axis=-1)) / mean


# --------------------------------------------------------------------------
# the fragility law
# --------------------------------------------------------------------------


def fragility_coefficient(alpha: float) -> float:
    """``(1-alpha)^2 / (2 alpha)``: predicted welfare loss per unit error variance.

    Multiply by the allocation-weighted variance of the log gain errors to get
    the predicted relative welfare loss (see the module docstring). Infinite at
    alpha = 0, where the optimum is a simplex vertex and the second-order
    expansion this comes from does not hold at all.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if alpha == 0:
        return np.inf
    if np.isinf(alpha):
        return np.inf
    return (1.0 - alpha) ** 2 / (2.0 * alpha)


def predicted_welfare_loss(true_gains: np.ndarray, pred_gains: np.ndarray,
                           alpha: float, budget: float = 1.0) -> np.ndarray:
    """The fragility law's prediction for a concrete pair of gain vectors.

    ``(1-alpha)^2/(2 alpha)`` times the allocation-weighted variance of
    ``log(pred/true)``, where the weights are the optimal allocation shares --
    an error in a region holding 1% of the budget cannot cost what the same
    error in a region holding 40% of it does, and an unweighted variance would
    score them the same.
    """
    g = np.asarray(true_gains, dtype=float)
    eta = np.log(np.asarray(pred_gains, dtype=float) / g)
    weights = alpha_fair_allocation(g, alpha=alpha, budget=1.0)
    centred = eta - np.sum(weights * eta, axis=-1, keepdims=True)
    return fragility_coefficient(alpha) * np.sum(weights * centred ** 2, axis=-1)


def tune_shrinkage(pred_gains: np.ndarray, true_gains: np.ndarray,
                   alpha: float, budget: float = 1.0,
                   grid: Optional[np.ndarray] = None) -> dict:
    """Fit the one scalar that hedges a rule against its input being wrong.

    Plugging an estimate into the optimal rule is not the optimal decision under
    uncertainty: at large alpha the rule leans hard on the smallest estimated
    gain, and if that estimate is noisy the lean is misdirected. Shrinking the
    exponent toward 0 pulls the allocation toward equal division, trading the
    rule's edge for immunity to the error.

    Returned so the learned allocator has something interpretable to be measured
    against. A network trained on the surrogate's own errors could learn an
    arbitrarily complicated hedge; if its advantage is matched by this single
    fitted number, what it learned was this single number, and the experiment
    should say so.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    losses = []
    for shrink in grid:
        alloc = alpha_fair_allocation(pred_gains, alpha=alpha, budget=budget,
                                      shrink=float(shrink))
        losses.append(float(np.mean(
            relative_welfare_loss(true_gains, alloc, alpha, budget))))
    best = int(np.argmin(losses))
    return {"shrink": float(grid[best]), "loss": losses[best],
            "grid": np.asarray(grid, dtype=float),
            "losses": np.asarray(losses, dtype=float)}
