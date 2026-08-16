"""Validation for litefno.models.graphfno.

The experiment's whole design is "four models that differ in one matrix", and
the things that can quietly break it are all here.

If the arms do not have equal parameter counts, a topology result is a capacity
result. If the identity arm still passes information between regions, the
control is not a control. If the graph arm's propagator is not the graph it was
named after, every number is about some other network. And if the layer cannot
express a propagation that a per-region model provably cannot, the experiment
has no power to detect the effect it is looking for -- so there is a test that
fits both on a synthetic diffusion and demands the graph win.

No network access, no data files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litefno.models.graphfno import (  # noqa: E402
    GraphConv, GraphLiteFNO, bound_violation, fit_graph_model, paint_regions,
    predict, region_vrmse)
from litefno.models.litefno import LiteFNO  # noqa: E402
from litefno.networks import (  # noqa: E402
    grid_graph, normalized_adjacency, small_world)


def _model(propagator=None, blocks=4, hidden=8, trunk_channels=4, seed=0):
    torch.manual_seed(seed)
    trunk = LiteFNO(3, trunk_channels, width=8, rank=4, layers=1)
    return GraphLiteFNO(trunk, trunk_channels=trunk_channels, blocks=blocks,
                        hidden=hidden, propagator=propagator, order=2,
                        layers=2)


# --------------------------------------------------------------------------
# the layer
# --------------------------------------------------------------------------


def test_graph_conv_with_identity_propagator_does_not_mix_nodes():
    """The no-graph arm has to actually be a no-graph arm."""
    layer = GraphConv(3, 5, n_nodes=6, propagator=None, order=2)
    h = torch.randn(2, 6, 3)
    out = layer(h)
    perturbed = h.clone()
    perturbed[:, 4] += 10.0
    moved = layer(perturbed)
    untouched = [n for n in range(6) if n != 4]
    assert torch.allclose(out[:, untouched], moved[:, untouched], atol=1e-6)


def test_graph_conv_with_a_graph_does_mix_nodes():
    prop = normalized_adjacency(grid_graph(4))
    layer = GraphConv(3, 5, n_nodes=16, propagator=prop, order=2)
    h = torch.randn(2, 16, 3)
    perturbed = h.clone()
    perturbed[:, 0] += 10.0
    assert not torch.allclose(layer(h)[:, 5], layer(perturbed)[:, 5],
                              atol=1e-6)


def test_graph_conv_keeps_the_propagator_it_was_given():
    prop = normalized_adjacency(small_world(4, 1.0, seed=0))
    layer = GraphConv(2, 2, n_nodes=16, propagator=prop, order=2)
    assert np.allclose(layer.powers[1].numpy(), prop, atol=1e-6)
    assert np.allclose(layer.powers[0].numpy(), np.eye(16), atol=1e-6)
    assert np.allclose(layer.powers[2].numpy(), prop @ prop, atol=1e-5)


def test_graph_conv_rejects_a_mismatched_propagator():
    with pytest.raises(ValueError):
        GraphConv(2, 2, n_nodes=16, propagator=np.eye(9))


def test_order_zero_is_a_plain_per_node_linear_map():
    layer = GraphConv(3, 4, n_nodes=5,
                      propagator=normalized_adjacency(grid_graph(3))[:5, :5],
                      order=0)
    h = torch.randn(2, 5, 3)
    expected = h @ layer.weight[0] + layer.bias
    assert torch.allclose(layer(h), expected, atol=1e-6)


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------


def test_every_arm_has_the_same_parameter_count():
    """Without this the comparison is between model sizes, not topologies."""
    counts = {
        "identity": _model(None).n_parameters(),
        "lattice": _model(normalized_adjacency(grid_graph(4))).n_parameters(),
        "true": _model(
            normalized_adjacency(small_world(4, 1.0, seed=0))).n_parameters(),
    }
    assert len(set(counts.values())) == 1, counts


def test_arms_share_an_initialisation_given_a_seed():
    """The paired comparison assumes the arms start from the same weights."""
    a = _model(None, seed=3)
    b = _model(normalized_adjacency(grid_graph(4)), seed=3)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa, pb), name


def test_forward_shape_and_resolution_independence():
    model = _model(normalized_adjacency(grid_graph(4)))
    for size in (16, 32):
        out = model(torch.randn(3, 3, size, size))
        assert out.shape == (3, 16)


# --------------------------------------------------------------------------
# the input encoding
# --------------------------------------------------------------------------


def test_paint_regions_round_trips_through_block_means():
    values = np.arange(16, dtype=float).reshape(1, 16)
    painted = paint_regions(values, size=32, blocks=4)
    assert painted.shape == (1, 1, 32, 32)
    pooled = painted.reshape(1, 4, 8, 4, 8).mean(axis=(2, 4)).reshape(1, 16)
    assert np.allclose(pooled, values)


def test_paint_regions_matches_the_graph_node_ordering():
    """Node r of the painted channel is block r in row-major order."""
    values = np.zeros((1, 16))
    values[0, 6] = 1.0                       # block row 1, block col 2
    painted = paint_regions(values, size=8, blocks=4)[0, 0]
    assert painted[2:4, 4:6].min() == 1.0
    assert painted.sum() == pytest.approx(4.0)


def test_paint_regions_rejects_an_indivisible_grid():
    with pytest.raises(ValueError):
        paint_regions(np.zeros((1, 16)), size=30, blocks=4)


# --------------------------------------------------------------------------
# power: can the layer learn something the ablation cannot?
# --------------------------------------------------------------------------


def test_graph_arm_beats_the_identity_arm_on_pure_network_diffusion():
    """A target that is a graph average of the input, and nothing local.

    The experiment's negative results are only informative if a positive one is
    reachable, so this is the calibration: a synthetic task whose answer at node
    r is a weighted average over r's *neighbours*, with the node's own value
    carrying no information about it. A per-node model cannot do better than the
    mean; a graph model given the right propagator should nearly solve it.
    """
    rng = np.random.default_rng(0)
    graph = small_world(4, 1.0, seed=2)
    prop = normalized_adjacency(graph)
    # neighbour average with the self-loop removed, so the answer is genuinely
    # not a function of the node's own feature
    mix = prop - np.diag(np.diag(prop))
    node_values = rng.normal(size=(256, 16)).astype(np.float32)
    targets = (node_values @ mix.T).astype(np.float32)
    states = paint_regions(node_values, size=8, blocks=4)
    states = np.concatenate([np.zeros_like(states), np.zeros_like(states),
                             states], axis=1)

    scores = {}
    for name, matrix in (("identity", None), ("true", prop)):
        torch.manual_seed(0)
        trunk = LiteFNO(3, 4, width=8, rank=4, layers=1)
        model = GraphLiteFNO(trunk, trunk_channels=4, blocks=4, hidden=16,
                             propagator=matrix, order=2, layers=2)
        fit_graph_model(model, states[:192], targets[:192], epochs=60, lr=5e-3,
                        seed=0)
        scores[name] = region_vrmse(predict(model, states[192:]),
                                    targets[192:])
    assert scores["true"] < 0.6 * scores["identity"], scores


def test_fit_restores_the_best_validation_epoch():
    rng = np.random.default_rng(1)
    states = rng.normal(size=(64, 3, 8, 8)).astype(np.float32)
    targets = rng.normal(size=(64, 16)).astype(np.float32)
    model = _model(None)
    fit = fit_graph_model(model, states[:48], targets[:48], epochs=6, lr=1e-2,
                          valid=(states[48:], targets[48:]), seed=0)
    best = min(h["valid_mse"] for h in fit["history"])
    assert fit["history"][fit["best_epoch"] - 1]["valid_mse"] == best


def test_bound_violation_reports_the_worst_excursion():
    assert bound_violation(np.array([0.2, 0.9])) == 0.0
    assert bound_violation(np.array([-0.3, 0.5])) == pytest.approx(0.3)
    assert bound_violation(np.array([0.5, 1.25])) == pytest.approx(0.25)


def test_region_vrmse_is_one_for_a_mean_predictor():
    rng = np.random.default_rng(2)
    target = rng.normal(size=(100, 16))
    pred = np.full_like(target, target.mean())
    assert region_vrmse(pred, target) == pytest.approx(1.0, rel=1e-3)
