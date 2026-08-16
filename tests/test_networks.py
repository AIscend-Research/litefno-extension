"""Validation for litefno.networks.

Three claims are pinned here, and the extension's headline rests on each of
them being true rather than merely plausible.

The lattice identity -- that the Fourier modes are the periodic lattice
Laplacian's eigenvectors -- is what makes "a spectral convolution is already a
graph convolution" a fact rather than an analogy, and everything ext24 declines
to claim follows from it. It is checked against the closed-form spectrum, not
against numpy's eigendecomposition, because an eigensolver would agree with any
consistent pair of conventions including a wrong one.

The epidemic threshold is checked by simulating the dynamics and finding where
they die out, against ``1 / lambda_1``. A cascade implementation with the
adjacency transposed, or with the seed applied on the wrong side of the update,
still produces a threshold; it is just not this one.

The node ordering is checked against ``region_gains``. Nothing else in the
module can detect a permuted graph -- every metric is permutation-covariant --
so if this test is missing, an ext24 that silently ran on scrambled regions
would look exactly like an ext24 that found no network effect.

No network access, no data files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litefno.allocation import region_gains  # noqa: E402
from litefno.networks import (  # noqa: E402
    cascade_step, cascade_threshold, degree_preserving_rewire,
    demand_pressure, detection_delay, eigenvector_centrality,
    epidemic_threshold, fourier_eigenbasis_residual, grid_graph,
    normalized_adjacency, preferential_attachment, propagate_scarcity,
    sentinel_sets, shortcut_fraction, small_world, spectral_radius,
    torus_laplacian_eigenvalues)


# --------------------------------------------------------------------------
# the lattice identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 6, 8])
def test_fourier_modes_are_lattice_eigenvectors(n):
    assert fourier_eigenbasis_residual(n) < 1e-10


def test_lattice_spectrum_matches_closed_form():
    """The dense Laplacian's eigenvalues are the closed form's, as a multiset."""
    n = 6
    lap = 4.0 * np.eye(n * n)
    grid = grid_graph(n)          # the region-graph builder, same construction
    lap -= grid
    dense = np.sort(np.linalg.eigvalsh(lap))
    closed = np.sort(torus_laplacian_eigenvalues(n).ravel())
    assert np.allclose(dense, closed, atol=1e-9)


def test_lattice_is_regular_and_circulant():
    a = grid_graph(4)
    assert np.allclose(a.sum(axis=1), 4.0)
    assert np.allclose(a, a.T)
    # circulant on the torus: shifting both indices by one row leaves it fixed
    perm = np.roll(np.arange(16).reshape(4, 4), 1, axis=0).ravel()
    assert np.allclose(a[np.ix_(perm, perm)], a)


def test_lattice_spectral_radius_is_the_degree():
    assert spectral_radius(grid_graph(4)) == pytest.approx(4.0)
    assert epidemic_threshold(grid_graph(4)) == pytest.approx(0.25)


# --------------------------------------------------------------------------
# node ordering: the failure nothing else can see
# --------------------------------------------------------------------------


def test_graph_nodes_match_region_gains_ordering():
    """Region ``r`` of the graph is region ``r`` of the gain vector.

    Built by putting a bump in one block of a field, reading the gains, and
    checking that the loud region's graph neighbours are the blocks physically
    adjacent to it on the torus.
    """
    blocks, size = 4, 8
    field = np.zeros((1, size, size, 1))
    field[0, 2:4, 4:6, 0] = 1.0            # block row 1, block col 2 -> node 6
    gains = region_gains(field, blocks=blocks, offset=1.0)
    assert int(np.argmax(gains[0])) == 6

    a = grid_graph(blocks)
    neighbours = set(np.nonzero(a[6])[0].tolist())
    assert neighbours == {2, 10, 5, 7}     # up, down, left, right on the torus


# --------------------------------------------------------------------------
# graph families and the controls
# --------------------------------------------------------------------------


def test_small_world_preserves_edge_count():
    base = grid_graph(4)
    for p in (0.0, 0.3, 1.0):
        a = small_world(4, p, seed=1)
        assert a.sum() == base.sum()
        assert np.allclose(a, a.T)
        assert np.all(np.diag(a) == 0)


def test_shortcut_fraction_is_zero_on_the_lattice_and_rises_with_p():
    assert shortcut_fraction(grid_graph(4)) == 0.0
    assert shortcut_fraction(small_world(4, 0.0, seed=0)) == 0.0
    low = shortcut_fraction(small_world(4, 0.25, seed=0))
    high = shortcut_fraction(small_world(4, 1.0, seed=0))
    assert 0.0 < low < high <= 1.0


def test_degree_preserving_rewire_preserves_every_degree():
    for graph in (grid_graph(4), preferential_attachment(16, 2, seed=0)):
        rewired = degree_preserving_rewire(graph, seed=2)
        assert np.array_equal(np.sort(graph.sum(axis=1)),
                              np.sort(rewired.sum(axis=1)))
        assert np.allclose(rewired, rewired.T)
        assert np.all(np.diag(rewired) == 0)
        assert rewired.sum() == graph.sum()


def test_degree_preserving_rewire_actually_moves_edges():
    """The control has to differ from what it controls for."""
    graph = small_world(5, 0.3, seed=0)
    rewired = degree_preserving_rewire(graph, seed=0)
    assert not np.allclose(graph, rewired)


def test_normalized_adjacency_spectrum_is_bounded():
    for graph in (grid_graph(4), small_world(4, 1.0, seed=3),
                  preferential_attachment(16, 2, seed=1)):
        prop = normalized_adjacency(graph)
        assert np.allclose(prop, prop.T)
        assert spectral_radius(prop) <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# the epidemic threshold
# --------------------------------------------------------------------------


@pytest.mark.parametrize("graph,name", [
    (grid_graph(4), "lattice"),
    (small_world(4, 1.0, seed=1), "trade network"),
    (preferential_attachment(16, 2, seed=0), "scale free"),
])
def test_cascade_threshold_matches_one_over_lambda_1(graph, name):
    found = cascade_threshold(graph, gamma=0.3, steps=300)
    assert found["rel_error"] < 0.02, name


def test_subcritical_cascade_dies_and_supercritical_one_does_not():
    graph = small_world(4, 0.5, seed=0)
    gamma, tau_c = 0.3, epidemic_threshold(graph)
    for tau, should_grow in ((0.5 * tau_c, False), (1.5 * tau_c, True)):
        x = np.full(16, 1e-3)
        for _ in range(300):
            x = cascade_step(x, graph, beta=tau * gamma, gamma=gamma)
        assert bool(x.mean() > 1e-3) is should_grow


def test_topology_moves_the_threshold_at_fixed_mean_degree():
    """The point of the whole extension, in one assertion.

    Rewiring keeps the edge count -- and here the degree sequence too, up to the
    lattice being regular -- and still moves the threshold, because it moves
    ``lambda_1``. Whatever a graph layer is for, it is not for counting edges.
    """
    lattice = grid_graph(4)
    trade = small_world(4, 1.0, seed=1)
    assert trade.sum() == lattice.sum()
    assert spectral_radius(trade) > spectral_radius(lattice)
    assert epidemic_threshold(trade) < epidemic_threshold(lattice)


def test_cascade_stays_in_the_unit_interval():
    graph = grid_graph(4)
    rng = np.random.default_rng(0)
    x = rng.random(16)
    for _ in range(50):
        x = cascade_step(x, graph, beta=0.5, gamma=0.1,
                         seed_input=rng.random(16), kappa=0.5)
        assert np.all((x >= 0.0) & (x <= 1.0))


# --------------------------------------------------------------------------
# coupling to the field
# --------------------------------------------------------------------------


def test_demand_pressure_is_zero_when_everyone_is_equal():
    gains = np.full((3, 16), 2.0)
    assert np.allclose(demand_pressure(gains), 0.0)


def test_demand_pressure_is_scale_free_and_rectified():
    rng = np.random.default_rng(0)
    gains = rng.random((4, 16)) + 0.5
    base = demand_pressure(gains)
    assert np.allclose(demand_pressure(gains * 7.5), base)
    assert np.all(base >= 0.0)
    assert np.any(base > 0.0)


def test_propagate_scarcity_starts_at_zero_and_is_causal():
    """``x[:, t]`` must not know about the field at ``t``.

    A model predicting ``x[t+h]`` from the frame at ``t`` would be reading the
    answer off its own input if the alignment slipped by one step, and the
    graph arms would look exactly as good as the no-graph arm.
    """
    rng = np.random.default_rng(0)
    gains = rng.random((2, 12, 16)) + 1.0
    graph = grid_graph(4)
    x = propagate_scarcity(gains, graph, beta=0.05, gamma=0.3, kappa=0.2)
    assert np.allclose(x[:, 0], 0.0)

    bumped = gains.copy()
    bumped[:, 5, 3] *= 3.0                  # one region's shock at t = 5
    y = propagate_scarcity(bumped, graph, beta=0.05, gamma=0.3, kappa=0.2)
    assert np.allclose(x[:, :6], y[:, :6])  # nothing before t = 6 may move
    assert not np.allclose(x[:, 6], y[:, 6])


def test_scarcity_travels_further_on_a_network_with_shortcuts():
    """A single seeded region reaches more of the network when it has shortcuts.

    The mechanism ext24 is about, isolated from any model: same seed, same beta
    and gamma, same edge count, different topology.
    """
    gains = np.ones((1, 20, 16))
    gains[0, :, 0] = 4.0                    # region 0 is permanently in deficit
    reach = {}
    for name, graph in (("lattice", grid_graph(4)),
                        ("trade", small_world(4, 1.0, seed=1))):
        x = propagate_scarcity(gains, graph, beta=0.05, gamma=0.3, kappa=0.2)
        reach[name] = float((x[0, -1] > 1e-4).sum())
    assert reach["trade"] >= reach["lattice"]


def test_propagate_scarcity_rejects_a_mismatched_graph():
    gains = np.ones((1, 4, 16))
    with pytest.raises(ValueError):
        propagate_scarcity(gains, grid_graph(3))


# --------------------------------------------------------------------------
# contact tracing
# --------------------------------------------------------------------------


def test_eigenvector_centrality_is_uniform_on_a_regular_graph():
    c = eigenvector_centrality(grid_graph(4))
    assert np.allclose(c, 1.0 / 16, atol=1e-9)
    assert c.sum() == pytest.approx(1.0)


def test_eigenvector_centrality_finds_the_hub():
    star = np.zeros((6, 6))
    star[0, 1:] = star[1:, 0] = 1.0
    c = eigenvector_centrality(star)
    assert int(np.argmax(c)) == 0


def test_sentinel_sets_are_distinct_indices_of_the_right_size():
    graph = preferential_attachment(16, 2, seed=0)
    sets = sentinel_sets(graph, 3, seed=0)
    assert set(sets) == {"eigenvector", "degree", "random"}
    for idx in sets.values():
        assert len(set(idx.tolist())) == 3
        assert np.all((idx >= 0) & (idx < 16))


def test_detection_delay_reports_nan_when_nothing_happens():
    quiet = np.zeros((10, 16))
    found = detection_delay(quiet, np.array([0, 1]), threshold=0.05)
    assert np.isnan(found["delay"])


def test_detection_delay_is_zero_when_the_sentinel_is_the_source():
    x = np.zeros((10, 16))
    x[3:, 4] = 0.5
    assert detection_delay(x, np.array([4]))["delay"] == 0.0
    assert np.isinf(detection_delay(x, np.array([7]))["delay"])
