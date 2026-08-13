"""Validation for litefno.models.allocator, the auxiliary allocation network.

Two things need pinning here and they fail in different ways.

The constraint is architectural, so it is checked as a guarantee rather than as
a trend: if a softmax head ever stopped producing a point on the simplex, every
welfare number in the experiment would be computed on an allocation that
overspends, and nothing downstream would complain.

The objective is the subtle one. ``alpha_fair_loss`` is the training signal for
every learned arm, and a version of it that is merely *correlated* with the
welfare would still train, still converge, and still produce a network that
optimises the wrong thing -- so it is checked against the numpy welfare in
:mod:`litefno.allocation` value by value, not by watching a loss go down.

No network access, no data files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litefno.allocation import (  # noqa: E402
    alpha_fair_allocation, outcomes, region_gains, relative_welfare_loss,
    welfare_ce)
from litefno.models.allocator import (  # noqa: E402
    RegionAllocator, alpha_fair_loss, allocate, evaluate_loss, fit_allocator)

BLOCKS, SIZE = 4, 16
REGIONS = BLOCKS * BLOCKS


def _blocky_fields(n: int, seed: int = 0, noise: float = 0.05):
    """Fields that are constant on each region, plus noise.

    The allocation problem without the PDE: the gains are known exactly by
    construction, so a network that fails here has failed at pooling, not at
    reading an oscillatory medium.
    """
    rng = np.random.default_rng(seed)
    levels = rng.uniform(-0.6, 0.6, (n, BLOCKS, BLOCKS))
    field = np.repeat(np.repeat(levels, SIZE // BLOCKS, axis=1),
                      SIZE // BLOCKS, axis=2)
    field = field + rng.normal(0, noise, field.shape)
    stacked = np.stack([field, 0.3 * field], axis=-1)      # (n, H, W, 2)
    gains = region_gains(stacked, blocks=BLOCKS)
    states = np.ascontiguousarray(stacked.transpose(0, 3, 1, 2))
    return states.astype(np.float32), gains


# --------------------------------------------------------------------------
# the constraint
# --------------------------------------------------------------------------


def test_output_is_on_the_simplex_at_initialisation():
    torch.manual_seed(0)
    model = RegionAllocator(in_channels=2, blocks=BLOCKS, budget=2.5)
    states, _ = _blocky_fields(8)
    alloc = model(torch.as_tensor(states))
    assert alloc.shape == (8, REGIONS)
    assert torch.all(alloc >= 0)
    assert torch.allclose(alloc.sum(-1), torch.full((8,), 2.5), atol=1e-5)


def test_output_stays_on_the_simplex_after_training():
    torch.manual_seed(0)
    model = RegionAllocator(in_channels=2, blocks=BLOCKS)
    states, gains = _blocky_fields(48)
    fit_allocator(model, states, gains, alpha=4.0, epochs=8, seed=0)
    alloc = allocate(model, states)
    assert np.all(alloc >= 0)
    assert np.allclose(alloc.sum(-1), 1.0, atol=1e-5)


def test_softmax_cannot_reach_a_vertex():
    """The stated cost of enforcing the constraint architecturally.

    At alpha = 0 the optimum is the whole budget to one region. The network
    cannot get there, which is a structural limit worth failing loudly on if it
    ever silently changes, because the alpha = 0 row of the results is read
    against it.
    """
    torch.manual_seed(0)
    model = RegionAllocator(in_channels=2, blocks=BLOCKS)
    states, gains = _blocky_fields(48)
    fit_allocator(model, states, gains, alpha=0.0, epochs=15, seed=0)
    alloc = allocate(model, states)
    assert alloc.max() < 1.0


def test_resolution_independence():
    """Adaptive pooling, so the allocator outlives the grid it was trained on."""
    torch.manual_seed(0)
    model = RegionAllocator(in_channels=2, blocks=BLOCKS)
    for size in (16, 32, 64):
        x = torch.zeros(2, 2, size, size)
        assert model(x).shape == (2, REGIONS)


def test_stays_small():
    """A decision layer that outweighs the surrogate is not the deployment
    story this repo is about."""
    model = RegionAllocator(in_channels=2, blocks=BLOCKS, width=16)
    assert model.n_parameters() < 10_000


# --------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
def test_loss_is_minus_log_of_the_numpy_welfare(alpha):
    rng = np.random.default_rng(0)
    gains = np.exp(rng.normal(0, 0.4, (7, REGIONS)))
    alloc = rng.dirichlet(np.ones(REGIONS), size=7)
    got = float(alpha_fair_loss(torch.as_tensor(alloc, dtype=torch.float64),
                                torch.as_tensor(gains, dtype=torch.float64),
                                alpha))
    expected = float(np.mean(-np.log(welfare_ce(outcomes(gains, alloc), alpha))))
    assert got == pytest.approx(expected, rel=1e-6)


def test_loss_is_minimised_by_the_closed_form_allocation():
    """The training signal points at the rule, on data where the rule is known."""
    rng = np.random.default_rng(1)
    gains = np.exp(rng.normal(0, 0.4, (32, REGIONS)))
    g = torch.as_tensor(gains, dtype=torch.float64)
    for alpha in (0.5, 1.0, 2.0, 8.0):
        best = alpha_fair_allocation(gains, alpha=alpha)
        worse = rng.dirichlet(np.ones(REGIONS), size=32)
        assert float(alpha_fair_loss(torch.as_tensor(best), g, alpha)) <= \
            float(alpha_fair_loss(torch.as_tensor(worse), g, alpha))


def test_loss_refuses_the_max_min_limit_rather_than_approximating_it():
    alloc = torch.full((2, REGIONS), 1.0 / REGIONS)
    gains = torch.ones(2, REGIONS)
    with pytest.raises(ValueError, match="non-differentiable"):
        alpha_fair_loss(alloc, gains, float("inf"))


def test_fit_rejects_a_mismatched_partition():
    model = RegionAllocator(in_channels=2, blocks=BLOCKS)
    states, _ = _blocky_fields(4)
    with pytest.raises(ValueError, match="partitions do not match"):
        fit_allocator(model, states, np.ones((4, 9)), epochs=1)


# --------------------------------------------------------------------------
# does it learn the rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.5, 2.0, 8.0])
def test_training_moves_the_allocation_toward_the_closed_form_rule(alpha):
    """The network recovers the rule it was never shown.

    It is trained only on realised welfare -- no allocation labels anywhere --
    so agreement with the closed form is evidence the decision-focused loss
    works, and it is the baseline the experiment's "does the network add
    anything" question is asked against.
    """
    torch.manual_seed(0)
    states, gains = _blocky_fields(256, seed=2)
    model = RegionAllocator(in_channels=2, blocks=BLOCKS)
    before = relative_welfare_loss(gains, allocate(model, states), alpha).mean()
    fit_allocator(model, states, gains, alpha=alpha, epochs=60, lr=5e-3, seed=0)
    after = relative_welfare_loss(gains, allocate(model, states), alpha).mean()

    assert after < before
    assert after < 0.02, f"alpha={alpha}: regret {after:.4f} against the rule"

    # and it moved in the right direction: the fitted log-allocation should
    # track (1-alpha)/alpha times the log gain, which is the rule itself
    learned = np.log(allocate(model, states))
    target = np.log(alpha_fair_allocation(gains, alpha=alpha))
    corr = np.corrcoef(learned.ravel() - learned.mean(1).repeat(REGIONS),
                       target.ravel() - target.mean(1).repeat(REGIONS))[0, 1]
    assert corr > 0.9, f"alpha={alpha}: correlation with the rule {corr:.3f}"


def test_valid_loss_is_tracked_when_asked():
    torch.manual_seed(0)
    states, gains = _blocky_fields(64, seed=3)
    model = RegionAllocator(in_channels=2, blocks=BLOCKS)
    out = fit_allocator(model, states[:48], gains[:48], alpha=2.0, epochs=4,
                        seed=0, valid=(states[48:], gains[48:]))
    assert all("valid_loss" in record for record in out["history"])
    assert out["history"][-1]["valid_loss"] == pytest.approx(
        evaluate_loss(model, states[48:], gains[48:], 2.0), rel=1e-5)
