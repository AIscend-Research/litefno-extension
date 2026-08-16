r"""Scarcity as a contagion on a trade network, and the graph the operator
is already convolving on.

Board task: "borrow from epidemiology / contact tracing -- model how resource
scarcity propagates through social/ecological networks (contact graphs, trade
networks); add a graph-convolutional layer on top of LiteFNO to capture network
effects in resource flow, not just local diffusion."

This module is the network mathematics; the layer itself is
:mod:`litefno.models.graphfno` and the experiment is
``scripts/network_scarcity.py``.

The claim that has to be checked before anything else
-----------------------------------------------------
"Add a graph-convolutional layer to capture network effects, not just local
diffusion" presumes the operator does not already have one. It does. On a
periodic grid the 4-neighbour lattice Laplacian is a circulant matrix whose
eigenvectors are *exactly* the 2-D Fourier modes,

    L e_k = (4 - 2 cos(2 pi k_y / N) - 2 cos(2 pi k_x / N)) e_k,
    e_k[y, x] = exp(2 pi i (k_y y + k_x x) / N) / N

so a spectral convolution with a free complex weight per mode -- what
``CPSpectralConv2d`` is -- *is* a spectral graph filter on the lattice graph, in
the sense of Bruna et al. (2014) and Defferrard, Bresson & Vandergheynst
(2016). :func:`fourier_eigenbasis_residual` checks that identity numerically to
machine precision, and it has a consequence that decides what this extension
can honestly claim:

**Any graph filter on a translation-invariant (circulant) graph is already in
the FNO's span.** A graph layer buys new capacity only from edges that break
translation invariance -- the long-range trade links, the shortcut edges of a
small world. So the hypothesis is not "does a graph layer help" (it must, if it
has parameters) but "does the help come from the *non-lattice* edges", which is
falsifiable and has an obvious null: on a purely spatial contact graph the gain
should be zero.

:func:`shortcut_fraction` measures how much of a given network is non-lattice,
and the experiment sweeps it.

The epidemiology this borrows
-----------------------------
Region ``r`` holds a scarcity level ``x_r in [0, 1]`` -- the share of its
population that a shortfall has reached. It spreads the way an SIS infection
does, because the mechanism is the same one: a region that cannot meet its own
demand draws on its trading partners, which pushes them toward shortfall in
turn. In discrete-time mean field (the NIMFA model of Van Mieghem, Omic &
Kooij 2009),

    x_{t+1} = x_t + beta (1 - x_t) (A x_t) - gamma x_t + kappa s_t          (*)

with ``beta`` the per-edge transmission of shortfall, ``gamma`` the rate at
which a region replenishes, and ``s_t`` an exogenous seed -- here the local
demand pressure read off the PDE state, which is what couples the network to
the physics this repo simulates.

Linearizing (*) at ``x = 0`` gives ``x_{t+1} = ((1 - gamma) I + beta A) x_t``,
so the shortfall dies out iff the spectral radius of that matrix is below 1:

    tau = beta / gamma  <  1 / lambda_1(A)                                 (**)

the epidemic threshold of Wang, Chakrabarti, Wang & Faloutsos (2003) -- the
result that the threshold of *any* graph is set by one number, its largest
adjacency eigenvalue. That is the closed form this module is carried for. It
plays the same role here that :mod:`litefno.systems` plays for the pole work:
"what should happen" is known before the simulation runs, so
:func:`cascade_threshold` can be scored rather than argued about.

(**) also says why topology is not decoration. A 4x4 torus lattice is
4-regular, so ``lambda_1 = 4`` and the threshold is ``tau_c = 0.25``. Rewire a
few of its edges into long-range trade links and ``lambda_1`` rises, because
the leading eigenvalue is bounded below by the mean degree and above by the max
degree and rewiring raises the spread; the same beta and gamma that were
subcritical on the lattice become supercritical on the small world. Nothing
about the local diffusion changed.

What is physics here and what is a modelling choice
---------------------------------------------------
The field is physics: :mod:`litefno.systems` integrates it. The regions, the
trade network, and (*) are a stylized model bolted on top, exactly as the
allocation layer in :mod:`litefno.allocation` is, and the same caveat applies --
no Gray-Scott parameter tells you who trades with whom. What is measurable is
the *composition*: given that scarcity propagates on a graph, does an operator
trained on local physics learn the non-local part, and does handing it the
graph help. Both halves of that question are about the model, not about the
economics.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# graphs
# --------------------------------------------------------------------------


def grid_graph(blocks: int = 4, periodic: bool = True) -> np.ndarray:
    """4-neighbour lattice over ``blocks x blocks`` regions, row-major nodes.

    Node ``b * blocks + a`` is the region in row ``b``, column ``a`` of the
    partition :func:`litefno.allocation.region_gains` produces, so a graph built
    here and a gain vector read off a field index the same regions. Getting that
    wrong is the silent failure mode of the whole extension -- a permuted graph
    is still a valid graph, it just answers a different question -- and
    ``tests/test_networks.py`` pins the ordering against ``region_gains``.

    ``periodic`` wraps the lattice onto a torus, matching the PDE's boundary
    conditions and making the graph circulant, which is the property the
    Fourier-basis argument in the module docstring needs. The non-periodic
    version is provided for completeness and is *not* circulant.
    """
    if blocks < 2:
        raise ValueError(f"need at least 2 blocks per side, got {blocks}")
    n = blocks * blocks
    a = np.zeros((n, n), dtype=float)
    for row in range(blocks):
        for col in range(blocks):
            node = row * blocks + col
            for drow, dcol in ((0, 1), (1, 0)):
                nrow, ncol = row + drow, col + dcol
                if periodic:
                    nrow, ncol = nrow % blocks, ncol % blocks
                elif nrow >= blocks or ncol >= blocks:
                    continue
                other = nrow * blocks + ncol
                if other != node:
                    a[node, other] = 1.0
                    a[other, node] = 1.0
    return a


def small_world(blocks: int = 4, p: float = 0.0, seed: int = 0,
                periodic: bool = True) -> np.ndarray:
    """Watts-Strogatz rewiring of the lattice: local trade plus shortcuts.

    Each lattice edge is, with probability ``p``, detached at one end and
    reattached to a uniformly chosen non-neighbour. Edge count is preserved
    exactly, so ``p`` moves *where* the edges are without changing how many
    there are -- which is what makes the sweep over ``p`` a clean test. If the
    edge count moved too, a gain at high ``p`` could be a gain from more
    coupling rather than from non-local coupling.

    ``p = 0`` is the pure contact lattice: neighbours trade with neighbours,
    the graph is circulant, and the module docstring's argument says a graph
    layer on it should buy nothing over a convolution. ``p = 1`` is a random
    trade network with the same density and no spatial meaning at all.

    Rewiring that would create a self-loop or a duplicate edge is skipped rather
    than retried forever, so the realised shortcut fraction sits slightly below
    ``p``; :func:`shortcut_fraction` reports the realised value and the
    experiment plots against that rather than against the requested one.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"rewiring probability must be in [0, 1], got {p}")
    a = grid_graph(blocks, periodic=periodic)
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(a))) ]
    for u, v in edges:
        if rng.random() >= p:
            continue
        candidates = [w for w in range(n)
                      if w != u and a[u, w] == 0.0]
        if not candidates:
            continue
        w = int(rng.choice(candidates))
        a[u, v] = a[v, u] = 0.0
        a[u, w] = a[w, u] = 1.0
    return a


def preferential_attachment(n: int = 16, m: int = 2, seed: int = 0
                            ) -> np.ndarray:
    """Barabasi-Albert trade network: a few hubs, many small partners.

    Carried because real trade networks are not degree-homogeneous and the
    epidemic threshold is a statement about ``lambda_1``, which a hub dominates:
    a star's leading eigenvalue is ``sqrt(k)`` in its degree, so a scale-free
    network is easier to make supercritical than a regular one of the same mean
    degree. :func:`epidemic_threshold` is checked on this family precisely
    because its threshold is *not* ``1 / mean degree``.
    """
    if m < 1 or n <= m:
        raise ValueError(f"need n > m >= 1, got n={n}, m={m}")
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), dtype=float)
    for i in range(1, m + 1):                       # seed clique-ish core
        a[i, i - 1] = a[i - 1, i] = 1.0
    for node in range(m + 1, n):
        degrees = a[:node].sum(axis=1)
        probs = degrees / max(degrees.sum(), 1e-12)
        targets = rng.choice(node, size=m, replace=False, p=probs)
        for t in targets:
            a[node, t] = a[t, node] = 1.0
    return a


def degree_preserving_rewire(adjacency: np.ndarray, seed: int = 0,
                             swaps: Optional[int] = None) -> np.ndarray:
    """Double-edge swap: same degree sequence, different topology.

    The control arm's graph. A graph layer given the *true* network is being
    compared against one given a wrong network, and the wrong one has to differ
    in the only thing under test. Randomising the edges freely would also change
    the degree sequence, and then a loss on the control could be a loss from a
    worse-conditioned propagator rather than from the topology being wrong. The
    swap ``(u,v),(s,t) -> (u,t),(s,v)`` leaves every degree exactly where it
    was.

    Note what this does *not* preserve: ``lambda_1``, and so the epidemic
    threshold. It cannot -- the leading eigenvalue is a function of more than
    the degree sequence -- and that is a fact about graphs rather than a defect
    of the control. The experiment reports both graphs' ``lambda_1`` so the
    comparison is not read as if they were matched on it.
    """
    a = np.array(adjacency, dtype=float, copy=True)
    rng = np.random.default_rng(seed)
    edges = [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(a)))]
    if len(edges) < 2:
        return a
    swaps = swaps if swaps is not None else 10 * len(edges)
    for _ in range(swaps):
        i, j = rng.choice(len(edges), size=2, replace=False)
        u, v = edges[i]
        s, t = edges[j]
        if len({u, v, s, t}) < 4:
            continue
        if a[u, t] or a[s, v]:
            continue
        a[u, v] = a[v, u] = 0.0
        a[s, t] = a[t, s] = 0.0
        a[u, t] = a[t, u] = 1.0
        a[s, v] = a[v, s] = 1.0
        edges[i] = [u, t]
        edges[j] = [s, v]
    return a


def shortcut_fraction(adjacency: np.ndarray, blocks: int = 4,
                      periodic: bool = True) -> float:
    """Share of edges that are *not* lattice edges: the non-local content.

    The independent variable of the whole experiment. The module docstring's
    argument says a spectral convolution already spans every filter on the
    lattice graph, so only these edges can carry capacity a convolution does not
    have. Zero on a pure contact lattice, one on a network with no lattice edge
    left in it.
    """
    a = np.asarray(adjacency, dtype=float)
    lattice = grid_graph(blocks, periodic=periodic)
    if a.shape != lattice.shape:
        raise ValueError(
            f"graph has {a.shape[0]} nodes, a {blocks}x{blocks} partition has "
            f"{lattice.shape[0]}")
    edges = np.triu(a) > 0
    total = int(edges.sum())
    if total == 0:
        return 0.0
    on_lattice = int((edges & (np.triu(lattice) > 0)).sum())
    return 1.0 - on_lattice / total


# --------------------------------------------------------------------------
# spectra: the lattice identity and the epidemic threshold
# --------------------------------------------------------------------------


def torus_laplacian_eigenvalues(n: int) -> np.ndarray:
    """``4 - 2cos(2 pi ky/n) - 2cos(2 pi kx/n)`` on the ``n x n`` mode grid.

    The closed-form spectrum of the periodic 4-neighbour lattice Laplacian,
    indexed the way ``np.fft.fftfreq`` indexes modes so it lines up with a
    spectral layer's mode grid without a reshuffle.
    """
    k = np.fft.fftfreq(n, d=1.0 / n)
    return (4.0 - 2.0 * np.cos(2 * np.pi * k[:, None] / n)
            - 2.0 * np.cos(2 * np.pi * k[None, :] / n))


def fourier_eigenbasis_residual(n: int = 8) -> float:
    """``max_k |L e_k - lambda_k e_k|`` for the periodic lattice Laplacian.

    The numerical form of the claim that a spectral convolution is a graph
    convolution: if the Fourier modes are the lattice Laplacian's eigenvectors,
    then choosing a complex weight per mode and choosing a graph filter's
    spectral response are the same act. Returns a residual that should be at
    machine precision, and a test holds it there.

    Built on the ``n^2 x n^2`` dense Laplacian of the pixel grid rather than of
    the region grid, because the layer the argument is about -- the FNO's
    spectral convolution -- acts on pixels.
    """
    lap = 4.0 * np.eye(n * n) - _dense_grid_adjacency(n)
    ky = np.fft.fftfreq(n, d=1.0 / n)
    kx = np.fft.fftfreq(n, d=1.0 / n)
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    eigenvalues = torus_laplacian_eigenvalues(n)
    worst = 0.0
    for iy in range(n):
        for ix in range(n):
            mode = np.exp(2j * np.pi * (ky[iy] * y + kx[ix] * x) / n).ravel()
            residual = lap @ mode - eigenvalues[iy, ix] * mode
            worst = max(worst, float(np.max(np.abs(residual))))
    return worst


def _dense_grid_adjacency(n: int) -> np.ndarray:
    """Adjacency of the periodic ``n x n`` pixel lattice, nodes row-major.

    Written with all four neighbours rather than two-and-symmetrise, because at
    ``n = 2`` the wrap makes the two directions land on the same node and an
    accumulating version would silently give that edge weight 2 -- a Laplacian
    that is still symmetric, still plausible, and no longer the lattice's.
    """
    a = np.zeros((n * n, n * n))
    for y in range(n):
        for x in range(n):
            node = y * n + x
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                other = ((y + dy) % n) * n + (x + dx) % n
                a[node, other] = 1.0
    return a


def spectral_radius(adjacency: np.ndarray) -> float:
    """``lambda_1``: the one number the epidemic threshold depends on."""
    a = np.asarray(adjacency, dtype=float)
    if not np.allclose(a, a.T, atol=1e-9):
        return float(np.max(np.abs(np.linalg.eigvals(a))))
    return float(np.max(np.linalg.eigvalsh(a)))


def epidemic_threshold(adjacency: np.ndarray) -> float:
    """``tau_c = 1 / lambda_1(A)``, the critical ``beta / gamma``.

    Wang, Chakrabarti, Wang & Faloutsos (2003). Below it a shortfall dies out
    from any seed; above it a seed of any size reaches a finite share of the
    network. The whole of the topology enters through ``lambda_1``, which is why
    a rewiring that leaves the degree sequence alone can still move the
    threshold.
    """
    lam = spectral_radius(adjacency)
    if lam <= 0:
        return np.inf
    return 1.0 / lam


def eigenvector_centrality(adjacency: np.ndarray) -> np.ndarray:
    """Leading eigenvector of ``A``, normalised to sum 1. Non-negative.

    The sentinel-placement score. Perron-Frobenius makes the leading eigenvector
    of a connected non-negative matrix strictly positive, and it is the direction
    the linearized cascade grows along -- so a shortfall anywhere in a
    supercritical network aligns with it within a few steps, which is the
    argument for watching the regions it weights most.
    """
    a = np.asarray(adjacency, dtype=float)
    values, vectors = np.linalg.eigh((a + a.T) / 2.0)
    vec = np.abs(vectors[:, int(np.argmax(values))])
    total = vec.sum()
    return vec / total if total > 0 else np.full(len(vec), 1.0 / len(vec))


def normalized_adjacency(adjacency: np.ndarray, self_loops: bool = True
                         ) -> np.ndarray:
    """``D^-1/2 (A + I) D^-1/2``, the GCN propagator (Kipf & Welling 2017).

    Symmetric normalisation keeps the propagator's spectrum inside [-1, 1], so
    stacking layers neither explodes nor kills the signal, and the self-loop is
    what lets a node keep its own state rather than being replaced by its
    neighbours' average. Isolated nodes are left alone instead of dividing by
    zero.
    """
    a = np.asarray(adjacency, dtype=float)
    if self_loops:
        a = a + np.eye(a.shape[0])
    degree = a.sum(axis=1)
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


# --------------------------------------------------------------------------
# the contagion
# --------------------------------------------------------------------------


def cascade_step(x: np.ndarray, adjacency: np.ndarray, beta: float,
                 gamma: float, seed_input: Optional[np.ndarray] = None,
                 kappa: float = 1.0) -> np.ndarray:
    """One step of ``x + beta (1-x) A x - gamma x + kappa s``, clipped to [0,1].

    The clip is a modelling decision with a consequence, so it is stated: it
    makes the map non-expansive on the unit cube and therefore bounded, but it
    also means the *simulated* threshold is the linearized one only near
    ``x = 0``. That is exactly where :func:`cascade_threshold` measures it, from
    a small seed, so the comparison against ``1 / lambda_1`` is a comparison of
    like with like rather than of a saturated cascade against a linear theory.
    """
    x = np.asarray(x, dtype=float)
    a = np.asarray(adjacency, dtype=float)
    pressure = x @ a.T
    nxt = x + beta * (1.0 - x) * pressure - gamma * x
    if seed_input is not None:
        nxt = nxt + kappa * np.asarray(seed_input, dtype=float)
    return np.clip(nxt, 0.0, 1.0)


def _survives(adjacency: np.ndarray, tau: float, gamma: float, steps: int,
              seed_size: float) -> tuple[bool, float]:
    x = np.full(adjacency.shape[0], seed_size)
    for _ in range(steps):
        x = cascade_step(x, adjacency, beta=tau * gamma, gamma=gamma)
    return bool(x.mean() > seed_size), float(x.mean())


def cascade_threshold(adjacency: np.ndarray, gamma: float = 0.3,
                      steps: int = 400, seed_size: float = 1e-3,
                      grid: Optional[Sequence[float]] = None,
                      refine: int = 24) -> dict:
    """Find the die-out threshold by simulation and score it against theory.

    Sweeps ``tau = beta / gamma``, runs (*) from a small uniform seed with no
    exogenous input, and calls a run supercritical if the final mean scarcity
    exceeds the seed it started from. The bracket between the last subcritical
    and first supercritical ``tau`` is then bisected ``refine`` times, so
    ``rel_error`` reports how well the dynamics match ``1 / lambda_1`` rather
    than how fine the sweep grid was -- on a coarse grid the two are easy to
    confuse, and the grid's own resolution would look like a 2-3% agreement no
    matter how wrong the theory was.

    This is the ground-truth check of the whole module. If it fails, either the
    dynamics or the eigenvalue is wrong, and every network result downstream is
    reporting on a system nobody has characterised.
    """
    a = np.asarray(adjacency, dtype=float)
    predicted = epidemic_threshold(a)
    if grid is None:
        grid = np.linspace(0.25 * predicted, 2.5 * predicted, 46)
    rows = []
    for tau in grid:
        alive, final = _survives(a, float(tau), gamma, steps, seed_size)
        rows.append({"tau": float(tau), "final_mean": final,
                     "supercritical": alive})
    sub = [r["tau"] for r in rows if not r["supercritical"]]
    sup = [r["tau"] for r in rows if r["supercritical"]]
    if not sub or not sup or max(sub) > min(sup):
        return {"predicted": predicted, "measured": np.nan,
                "lambda_1": spectral_radius(a), "rel_error": np.nan,
                "rows": rows}
    lo, hi = max(sub), min(sup)
    for _ in range(refine):
        mid = 0.5 * (lo + hi)
        alive, _ = _survives(a, mid, gamma, steps, seed_size)
        lo, hi = (lo, mid) if alive else (mid, hi)
    measured = 0.5 * (lo + hi)
    return {"predicted": predicted, "measured": float(measured),
            "lambda_1": spectral_radius(a),
            "rel_error": float(abs(measured - predicted) / predicted),
            "rows": rows}


# --------------------------------------------------------------------------
# coupling the contagion to the PDE state
# --------------------------------------------------------------------------


def demand_pressure(gains: np.ndarray, share: Optional[np.ndarray] = None
                    ) -> np.ndarray:
    """Unmet demand per region: ``relu(g_r / (share_r sum g) - 1)``.

    Where the physics enters the network model. ``gains`` are the region
    populations :func:`litefno.allocation.region_gains` reads off the field, and
    ``share`` is the fraction of the resource each region holds -- equal
    division by default, which is ext22's envy-free allocation and the one
    allocation that needs no forecast. A region whose population outgrows its
    share is in deficit, and that deficit is the seed the contagion spreads.

    Relative rather than absolute, so the seed does not inherit the arbitrary
    population offset in :func:`litefno.allocation.region_gains`, and rectified
    because a region in surplus does not emit scarcity -- it absorbs it, which
    is already in the ``(1 - x)`` factor of the dynamics rather than being a
    negative seed.
    """
    g = np.asarray(gains, dtype=float)
    n = g.shape[-1]
    if share is None:
        share = np.full(n, 1.0 / n)
    share = np.asarray(share, dtype=float)
    entitled = share * g.sum(axis=-1, keepdims=True)
    return np.maximum(g / np.maximum(entitled, 1e-12) - 1.0, 0.0)


def propagate_scarcity(gains: np.ndarray, adjacency: np.ndarray,
                       beta: float = 0.08, gamma: float = 0.3,
                       kappa: float = 0.15,
                       share: Optional[np.ndarray] = None) -> np.ndarray:
    """Roll (*) along a trajectory of region gains. Returns (..., T, R).

    ``gains`` is (n_traj, T, R) from :func:`litefno.allocation.region_gains`,
    and the returned scarcity is aligned with it in time: ``x[:, t]`` is the
    state *before* the step from ``t`` to ``t+1``, so a model predicting
    ``x[:, t+h]`` from the field at ``t`` is predicting the future rather than
    reading it off its input.

    Two things drive the result and they are separable on purpose. The seed
    :func:`demand_pressure` is a pure function of the local field, so a model
    that sees the field can compute it. The propagation is a pure function of
    the graph, which no amount of field resolution reveals. That separation is
    what makes the experiment a test of the graph layer rather than of capacity:
    the field-only arm is not starved of information about the *seeds*, only
    about where they travel.
    """
    g = np.asarray(gains, dtype=float)
    if g.ndim != 3:
        raise ValueError(f"expected (n_traj, T, R) gains, got {g.shape}")
    a = np.asarray(adjacency, dtype=float)
    if a.shape[0] != g.shape[-1]:
        raise ValueError(
            f"{g.shape[-1]} regions against a {a.shape[0]}-node graph")
    seeds = demand_pressure(g, share)
    out = np.zeros_like(g)
    x = np.zeros((g.shape[0], g.shape[-1]))
    for t in range(g.shape[1]):
        out[:, t] = x
        x = cascade_step(x, a, beta, gamma, seed_input=seeds[:, t],
                         kappa=kappa)
    return out


# --------------------------------------------------------------------------
# contact tracing: where to put the sentinels
# --------------------------------------------------------------------------


def sentinel_sets(adjacency: np.ndarray, n_sentinels: int, seed: int = 0
                  ) -> dict:
    """Three ways to choose which regions to monitor, at equal budget.

    The contact-tracing half of the borrow. Surveillance is the cheap
    intervention when the network cannot be changed: you cannot stop regions
    trading, but you can choose where to look. Returns index arrays for

    ``eigenvector``  the regions the leading eigenvector weights most, which is
                     the direction the linearized cascade grows along
    ``degree``       the most-connected regions, the obvious heuristic
    ``random``       the control, without which "central regions detect early"
                     is unfalsifiable -- with enough sentinels every rule works

    Ties are broken by index, which is deterministic and, on a regular lattice
    where every degree is equal, makes the degree rule a fixed arbitrary choice.
    That is the honest behaviour: on a lattice there *is* no informative degree
    signal, and the result should show the heuristic failing there rather than
    quietly randomising.
    """
    a = np.asarray(adjacency, dtype=float)
    n = a.shape[0]
    if not 1 <= n_sentinels <= n:
        raise ValueError(f"cannot place {n_sentinels} sentinels on {n} regions")
    centrality = eigenvector_centrality(a)
    degree = a.sum(axis=1)
    rng = np.random.default_rng(seed)
    return {
        "eigenvector": np.argsort(-centrality, kind="stable")[:n_sentinels],
        "degree": np.argsort(-degree, kind="stable")[:n_sentinels],
        "random": rng.choice(n, size=n_sentinels, replace=False),
    }


def detection_delay(scarcity: np.ndarray, sentinels: np.ndarray,
                    threshold: float = 0.05) -> dict:
    """How much later a sentinel set notices a shortfall than the network does.

    ``scarcity`` is (T, R) for one trajectory. Returns the first step at which
    any monitored region crosses ``threshold``, the first step at which any
    region does, their difference, and the share of the network already above
    threshold at the moment of detection -- which is the quantity that matters
    for an intervention, since a delay is only costly in proportion to what
    spread during it.

    ``np.inf`` delay when the sentinels never detect a shortfall that did occur;
    ``nan`` when there was no shortfall to detect, so that runs with nothing
    happening are dropped from the average rather than counted as instant
    detection.
    """
    x = np.asarray(scarcity, dtype=float)
    hit_any = np.nonzero((x > threshold).any(axis=1))[0]
    hit_sen = np.nonzero((x[:, sentinels] > threshold).any(axis=1))[0]
    if hit_any.size == 0:
        return {"first_any": np.nan, "first_sentinel": np.nan,
                "delay": np.nan, "spread_at_detection": np.nan}
    first_any = int(hit_any[0])
    if hit_sen.size == 0:
        return {"first_any": first_any, "first_sentinel": np.inf,
                "delay": np.inf,
                "spread_at_detection": float((x[-1] > threshold).mean())}
    first_sen = int(hit_sen[0])
    return {"first_any": first_any, "first_sentinel": first_sen,
            "delay": float(first_sen - first_any),
            "spread_at_detection": float((x[first_sen] > threshold).mean())}
