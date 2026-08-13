"""Validation for litefno.specscope.

The transplant machinery is the part of SpecScope where a silent failure is
most costly: an arm that is nominally frozen but actually trainable turns H2's
claim from "a transplanted subspace helps" into "a warm start helps", which is
a different and much weaker paper. So the freezing is tested by training and
checking the weights did not move, not by inspecting a flag.

Two bugs found while building ext21 are pinned here:

* ``rank_mode_energy`` normalised within the subset it was handed, so every
  subset summed to one, every component looked equally at home anywhere, and
  the partition split on a comparison that was 0.5 by construction.
* the component selection returned every component for both sets, collapsing
  the two transplant arms into the same arm.

No network access, no data files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from litefno.models.harmonic import HarmonicLiteFNO   # noqa: E402
from litefno.specscope import (  # noqa: E402
    fit, one_step_vrmse, partition_rank_components, rank_mode_energy,
    rollout_mode_error, to_pairs, transplant)
from litefno.systems import rotating_diffusion, split_trajectories  # noqa: E402


def build(seed: int = 0, rank: int = 6, modes: int = 8, layers: int = 2):
    torch.manual_seed(seed)
    return HarmonicLiteFNO(2, 2, width=8, modes=modes, layers=layers, rank=rank)


@pytest.fixture(scope="module")
def data():
    traj = rotating_diffusion(n_traj=6, n_steps=12, size=32, seed=0)
    return split_trajectories(traj, seed=0)


# --------------------------------------------------------------------------
# training plumbing
# --------------------------------------------------------------------------


def test_pairs_are_consecutive_states_channels_first():
    traj = np.arange(2 * 4 * 3 * 3 * 2).reshape(2, 4, 3, 3, 2).astype(np.float32)
    x, y = to_pairs(traj)
    assert x.shape == (2 * 3, 2, 3, 3)
    # y[i] must be the state one step after x[i]
    assert np.allclose(y[0].numpy(), traj[0, 1].transpose(2, 0, 1))
    assert np.allclose(x[0].numpy(), traj[0, 0].transpose(2, 0, 1))


def test_fit_reduces_training_loss(data):
    model = build()
    out = fit(model, data["train"], epochs=6, device="cpu", seed=0)
    assert out["history"][-1]["train_mse"] < out["history"][0]["train_mse"]


def test_fit_refuses_a_fully_frozen_model(data):
    model = build()
    with pytest.raises(ValueError, match="nothing to train"):
        fit(model, data["train"], epochs=1, frozen=list(model.parameters()))


def test_frozen_parameters_do_not_move(data):
    model = build()
    watched = model.spectral_layers[0].factor_m1
    before = watched.detach().clone()
    fit(model, data["train"], epochs=4, device="cpu", seed=0, frozen=[watched])
    assert torch.equal(watched.detach(), before)


# --------------------------------------------------------------------------
# per-mode rollout error
# --------------------------------------------------------------------------


def test_rollout_mode_error_is_zero_for_a_perfect_model(data):
    """A model that reproduces the data exactly must show no error anywhere.

    Built by handing back the true next state, so this checks the bookkeeping
    -- the step alignment, the spectra, the normaliser -- rather than a model.
    """
    traj = data["test"]

    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.step = 0
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            self.step += 1
            nxt = traj[:len(x), self.step].transpose(0, 3, 1, 2)
            return torch.from_numpy(np.ascontiguousarray(nxt))

    got = rollout_mode_error(Oracle(), traj, horizon=4, max_mode=4,
                             batch=len(traj))
    assert np.all(got["error"] < 1e-5)


def test_rollout_mode_error_shapes_and_growth(data):
    model = build()
    got = rollout_mode_error(model, data["test"], horizon=5, max_mode=4)
    assert got["error"].shape == (5, len(got["radius"]))
    assert got["growth"].shape == (len(got["radius"]),)
    assert np.all(np.isfinite(got["growth"]))


def test_rollout_horizon_is_clipped_to_the_data(data):
    got = rollout_mode_error(build(), data["test"], horizon=9999, max_mode=3)
    assert got["horizon"] == data["test"].shape[1] - 1


# --------------------------------------------------------------------------
# component bookkeeping
# --------------------------------------------------------------------------


def test_rank_mode_energy_is_a_share_of_the_whole_grid():
    """The regression for the normalise-within-the-subset bug.

    A component's share over every mode must be 1, and its share over half of
    them must not be -- if a subset always sums to one, the partition below has
    nothing to compare.
    """
    model = build(rank=4, modes=8)
    layer = model.spectral_layers[0]
    ky_all = np.concatenate([np.arange(5), np.arange(-3, 0)])
    all_ky = np.repeat(ky_all, 8)
    all_kx = np.tile(np.arange(8), 8)

    full = rank_mode_energy(layer, all_ky, all_kx).sum(axis=1)
    assert full == pytest.approx(np.ones(4), rel=1e-6)

    half = rank_mode_energy(layer, all_ky[:32], all_kx[:32]).sum(axis=1)
    assert np.all(half < 1.0)
    assert np.any(np.abs(half - 0.5) > 0.02)      # not uniform by construction


def test_partition_is_disjoint_and_covers_every_component():
    model = build(rank=6, modes=8)
    ky_all = np.concatenate([np.arange(5), np.arange(-3, 0)])
    low = (np.repeat(ky_all[:2], 4), np.tile(np.arange(4), 2))
    high = (np.repeat(ky_all[3:5], 4), np.tile(np.arange(4, 8), 2))
    split = partition_rank_components(model, low, high)
    assert set(split["a"]).isdisjoint(split["b"])
    assert sorted(list(split["a"]) + list(split["b"])) == list(range(6))


def test_partition_margin_is_not_degenerate():
    """The bug this pins reported a margin of exactly zero for every model."""
    model = build(rank=6, modes=8)
    ky_all = np.concatenate([np.arange(5), np.arange(-3, 0)])
    low = (np.repeat(ky_all[:2], 4), np.tile(np.arange(4), 2))
    high = (np.repeat(ky_all[3:5], 4), np.tile(np.arange(4, 8), 2))
    assert partition_rank_components(model, low, high)["margin"] > 0.01


def test_partition_puts_a_component_where_its_energy_is():
    """A component built to live on one mode must be assigned to that mode's set."""
    model = build(rank=2, modes=8)
    with torch.no_grad():
        for layer in model.spectral_layers:
            layer.factor_m1.zero_()
            layer.factor_m2.zero_()
            # component 0 lives at ky index 1, kx 1; component 1 at ky 4, kx 5
            layer.factor_m1[1, 0] = 1.0
            layer.factor_m2[1, 0] = 1.0
            layer.factor_m1[4, 1] = 1.0
            layer.factor_m2[5, 1] = 1.0
    split = partition_rank_components(model, (np.array([1]), np.array([1])),
                                      (np.array([4]), np.array([5])))
    assert list(split["a"]) == [0]
    assert list(split["b"]) == [1]


# --------------------------------------------------------------------------
# the transplant itself
# --------------------------------------------------------------------------


def test_transplant_copies_the_selected_components(data):
    source, target = build(seed=1), build(seed=2)
    components = [0, 2, 4]
    info = transplant(target, source, components)
    assert info["n_components"] == 3
    for tgt, src in zip(target.spectral_layers, source.spectral_layers):
        assert torch.allclose(tgt.factor_m1[:, components],
                              src.factor_m1[:, components])
        assert torch.allclose(tgt.factor_m2[:, components],
                              src.factor_m2[:, components])


def test_transplant_leaves_the_other_components_alone(data):
    source, target = build(seed=1), build(seed=2)
    untouched = target.spectral_layers[0].factor_m1[:, [1, 3, 5]].detach().clone()
    transplant(target, source, [0, 2, 4])
    assert torch.equal(target.spectral_layers[0].factor_m1[:, [1, 3, 5]].detach(),
                       untouched)


def test_transplanted_components_survive_training(data):
    """The claim is a *frozen* transplant, so this is the load-bearing test.

    If the gradient masking is dropped, the copied columns drift and the arm
    silently becomes a warm start.
    """
    source, target = build(seed=1), build(seed=2)
    components = [0, 2]
    transplant(target, source, components)
    frozen_before = target.spectral_layers[0].factor_m1[:, components].detach().clone()
    free_before = target.spectral_layers[0].factor_m1[:, [1, 3]].detach().clone()

    fit(target, data["train"], epochs=5, device="cpu", seed=0)

    after = target.spectral_layers[0].factor_m1.detach()
    assert torch.equal(after[:, components], frozen_before)
    assert not torch.allclose(after[:, [1, 3]], free_before)


def test_removing_the_handles_unfreezes(data):
    source, target = build(seed=1), build(seed=2)
    info = transplant(target, source, [0, 2])
    for handle in info["handles"]:
        handle.remove()
    before = target.spectral_layers[0].factor_m1[:, [0, 2]].detach().clone()
    fit(target, data["train"], epochs=5, device="cpu", seed=0)
    assert not torch.allclose(
        target.spectral_layers[0].factor_m1[:, [0, 2]].detach(), before)


def test_empty_transplant_is_a_no_op(data):
    source, target = build(seed=1), build(seed=2)
    before = target.spectral_layers[0].factor_m1.detach().clone()
    info = transplant(target, source, [])
    assert info["n_components"] == 0
    assert info["handles"] == []
    assert torch.equal(target.spectral_layers[0].factor_m1.detach(), before)


def test_duplicate_components_are_deduplicated(data):
    source, target = build(seed=1), build(seed=2)
    assert transplant(target, source, [1, 1, 3])["n_components"] == 2


def test_one_step_vrmse_is_zero_for_an_exact_model(data):
    class Echo(torch.nn.Module):
        """Returns the target by construction: VRMSE must be 0, not merely small."""

        def __init__(self, traj):
            super().__init__()
            self.lookup = {}
            for i in range(traj.shape[0]):
                for t in range(traj.shape[1] - 1):
                    key = traj[i, t].tobytes()
                    self.lookup[key] = traj[i, t + 1].transpose(2, 0, 1)

        def forward(self, x):
            out = [self.lookup[np.ascontiguousarray(
                sample.numpy().transpose(1, 2, 0)).tobytes()] for sample in x]
            return torch.from_numpy(np.stack(out))

    assert one_step_vrmse(Echo(data["test"]), data["test"]) == pytest.approx(
        0.0, abs=1e-6)
