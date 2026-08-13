"""Validation for litefno.operator.

The module claims to read a trained network's per-mode propagator out of it. The
way to test that is to build operators whose propagator is known by
construction and check the extractor returns it, rather than to check two
approximations against each other.

Three ground truths are used, in descending order of strength:

1. A hand-built diagonal spectral operator with per-mode gains chosen by the
   test. The correct answer is the number written into the module.
2. An exactly linear PDE (advection-diffusion), whose propagator is
   ``exp((-nu|k|^2 - i c.k) dt)`` in closed form.
3. The two independent extraction routes against each other on a real
   ``HarmonicLiteFNO``, which pins the one place they are known to disagree.

No network access, no data files, no training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from litefno.operator import (  # noqa: E402
    analytic_mode_operators, classify_operator_modes, compare_operators,
    empirical_mode_operators, gray_scott_linear_spectrum, linear_pde_propagator,
    mode_basis, mode_grid, operator_poles, principal_angles, probe_convergence,
    stability_margin, subspace_overlap)
from litefno.models.harmonic import HarmonicLiteFNO  # noqa: E402

H = W = 32


class DiagonalSpectral(torch.nn.Module):
    """A model that multiplies each Fourier mode by a gain the test chooses.

    This is the simplest object with a known per-mode propagator: gains are
    supplied as a (H, W//2+1) complex array, applied in Fourier space, and the
    same scalar gain acts on every channel. Conjugate symmetry is imposed so
    the output is real without ``irfft2`` having to discard anything -- the one
    effect that makes the analytic route inexact on the real model.
    """

    def __init__(self, gains: np.ndarray):
        super().__init__()
        self.register_buffer("gains", torch.as_tensor(gains, dtype=torch.cfloat))

    def forward(self, x):
        return torch.fft.irfft2(torch.fft.rfft2(x) * self.gains,
                                s=x.shape[-2:])


def hermitian_gains(fn) -> np.ndarray:
    """Per-mode gains from ``fn(ky, kx)``, made conjugate-symmetric."""
    ky = np.fft.fftfreq(H, d=1.0 / H).astype(int)
    kx = np.arange(W // 2 + 1)
    g = fn(ky[:, None], kx[None, :]).astype(np.complex128)
    # rows +a and -a in the kx=0 column must be conjugates for a real output
    for a in range(1, H // 2):
        g[H - a, 0] = np.conj(g[a, 0])
    g[0, 0] = g[0, 0].real
    return g


# --------------------------------------------------------------------------
# 1. a propagator the test wrote itself
# --------------------------------------------------------------------------


def test_probe_recovers_known_per_mode_gains():
    rng = np.random.default_rng(0)
    phases = rng.uniform(-np.pi, np.pi, (H, W // 2 + 1))
    gains = hermitian_gains(lambda a, b: 0.8 * np.exp(1j * phases))
    model = DiagonalSpectral(gains)

    got = empirical_mode_operators(model, np.zeros((2, H, W)), max_mode=4,
                                   eps=1e-2)
    for i, (a, b) in enumerate(zip(got["ky"], got["kx"])):
        expected = gains[a % H, b]
        # scalar gain on every channel => the matrix is g * I
        assert got["operators"][i] == pytest.approx(
            expected * np.eye(2), abs=1e-4), f"mode ({a}, {b})"


def test_probe_is_channel_resolved():
    """A gain that mixes channels must show up off-diagonal, not just in norm."""
    class Mixer(torch.nn.Module):
        def forward(self, x):
            # channel 1 of the output is channel 0 of the input, halved
            return torch.stack([0.5 * x[:, 1], 0.5 * x[:, 0]], dim=1)

    got = empirical_mode_operators(Mixer(), np.zeros((2, H, W)), max_mode=2,
                                   eps=1e-2)
    for op in got["operators"]:
        assert op == pytest.approx(np.array([[0, 0.5], [0.5, 0]]), abs=1e-5)


# --------------------------------------------------------------------------
# 2. an exactly linear PDE
# --------------------------------------------------------------------------


def test_advection_diffusion_poles_are_recovered_exactly():
    """A model that *is* the propagator must yield the analytic poles.

    No linearization enters anywhere: the system is linear, the model is the
    exact step map, so any gap is extractor error.
    """
    nu, dt, vel = 0.02, 0.5, (0.0, 1.0)
    ky_all = np.fft.fftfreq(H, d=1.0 / H).astype(int)
    kx_all = np.arange(W // 2 + 1)
    truth_grid = linear_pde_propagator(ky_all, kx_all, dt=dt, nu=nu,
                                       velocity=vel, height=H, width=W)
    model = DiagonalSpectral(hermitian_gains(
        lambda a, b: linear_pde_propagator(np.ravel(a), np.ravel(b), dt=dt,
                                           nu=nu, velocity=vel, height=H,
                                           width=W)))

    got = empirical_mode_operators(model, np.zeros((1, H, W)), max_mode=5,
                                   eps=1e-2)
    poles = operator_poles(got["operators"])
    for i, (a, b) in enumerate(zip(got["ky"], got["kx"])):
        assert poles["magnitude"][i].max() == pytest.approx(
            abs(truth_grid[a % H, b]), abs=1e-4), f"mode ({a}, {b})"


def test_pure_diffusion_has_no_oscillation_and_pure_advection_is_neutral():
    ky, kx = np.arange(0, 5), np.arange(0, 5)
    diffusion = linear_pde_propagator(ky, kx, dt=1.0, nu=0.05, height=H, width=W)
    assert np.all(np.abs(np.angle(diffusion)) < 1e-12)     # real, no rotation
    assert np.all(np.abs(diffusion) <= 1.0 + 1e-12)        # decays

    advection = linear_pde_propagator(ky, kx, dt=1.0, nu=0.0, velocity=(1.0, 0.0),
                                      height=H, width=W)
    assert np.abs(advection) == pytest.approx(1.0)         # neutrally stable
    assert np.abs(np.angle(advection[1, 0])) > 1e-6        # and it rotates


# --------------------------------------------------------------------------
# 3. the two routes against each other, and where they part company
# --------------------------------------------------------------------------


def _linear_model(seed: int = 0):
    """A HarmonicLiteFNO with the activation removed, so composition is exact."""
    torch.manual_seed(seed)
    model = HarmonicLiteFNO(2, 2, width=8, modes=8, layers=2, rank=4)
    model.act = torch.nn.Identity()
    torch.nn.init.zeros_(model.input_proj.bias)
    torch.nn.init.zeros_(model.output_proj.bias)
    for skip in model.skips:
        torch.nn.init.zeros_(skip.bias)
    return model


def test_routes_agree_away_from_the_kx_zero_column():
    model = _linear_model()
    rows = compare_operators(analytic_mode_operators(model, gelu_gain=1.0),
                             empirical_mode_operators(model, np.zeros((2, H, W)),
                                                      max_mode=4, eps=1e-2))
    off_axis = [r["rel_norm_diff"] for r in rows if r["kx"] != 0]
    assert off_axis and max(off_axis) < 1e-5


def test_kx_zero_column_disagrees_and_the_size_of_it_is_pinned():
    """The one known inexactness, kept as a number rather than a caveat.

    At kx = 0 the retained block holds +ky and -ky with independent weights, so
    it is not conjugate-symmetric, and ``irfft2`` quietly symmetrizes it on the
    way out. The composed operator does not model that step, so it is wrong
    there -- by under half a percent on this model, which is why the poles are
    still usable, but it is not zero and should not silently become larger.
    """
    model = _linear_model()
    rows = compare_operators(analytic_mode_operators(model, gelu_gain=1.0),
                             empirical_mode_operators(model, np.zeros((2, H, W)),
                                                      max_mode=4, eps=1e-2))
    on_axis = [r["rel_norm_diff"] for r in rows if r["kx"] == 0]
    assert max(on_axis) > 1e-5          # a real effect, not noise
    assert max(on_axis) < 1e-2          # and a small one


def test_probe_convergence_reports_a_plateau():
    model = _linear_model()
    steps = probe_convergence(model, np.zeros((2, H, W)), mode=(1, 1),
                              eps_values=(1e-1, 1e-2, 1e-3))
    changes = [s["rel_change"] for s in steps[1:]]
    # the model is linear here, so every step size must give the same answer
    assert max(changes) < 1e-4


def test_gelu_gain_moves_magnitudes_but_not_frequencies():
    model = _linear_model()
    a = analytic_mode_operators(model, gelu_gain=1.0)
    b = analytic_mode_operators(model, gelu_gain=0.5)
    pa, pb = operator_poles(a["operators"]), operator_poles(b["operators"])
    assert np.allclose(np.sort(pa["freq"], axis=-1), np.sort(pb["freq"], axis=-1),
                       atol=1e-8)
    # two layers, so every magnitude drops by the same factor of 0.5^2
    ratio = pb["magnitude"] / np.maximum(pa["magnitude"], 1e-30)
    assert ratio == pytest.approx(0.25, rel=1e-6)


# --------------------------------------------------------------------------
# pole bookkeeping
# --------------------------------------------------------------------------


def test_pole_conventions_on_a_known_rotation():
    """A rotation by 2pi/8 scaled by r has period 8 and sigma = log r."""
    theta, r = 2 * np.pi / 8, 0.9
    rot = r * np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    got = operator_poles(rot[None])
    assert got["magnitude"][0] == pytest.approx(r)
    assert got["freq"][0] == pytest.approx(1 / 8)
    assert got["sigma"][0] == pytest.approx(np.log(r))


def test_operator_poles_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        operator_poles(np.zeros((3, 2, 4)))


def test_stability_margin_takes_the_least_damped_pole():
    sigma = np.array([[[-1.0, -0.1], [-3.0, -2.0]]])
    assert stability_margin(sigma) == pytest.approx(np.array([[-0.1, -2.0]]))


def test_classification_separates_the_three_specscope_classes():
    def poles_of(z_list):
        z = np.array(z_list)[None]
        return {"sigma": np.log(np.abs(z)), "freq": np.abs(np.angle(z)) / (2 * np.pi)}

    # neutral + rotating -> resonant; neutral + real -> primary; decaying -> damped
    labels = classify_operator_modes(poles_of([
        [np.exp(1j * 0.4), 0.1],           # resonant
        [1.0, 0.5],                        # primary
        [0.3, 0.2],                        # damped
        [1.5, 0.1],                        # unstable
    ]))
    assert list(labels[0]) == ["resonant", "primary", "damped", "unstable"]


def test_unstable_is_not_reported_as_resonant():
    """A growing oscillation is the failure H1 predicts, not physics to move."""
    z = np.array([[1.4 * np.exp(1j * 0.4), 0.0]])[None]
    labels = classify_operator_modes(
        {"sigma": np.log(np.abs(z) + 1e-300), "freq": np.abs(np.angle(z)) / (2 * np.pi)})
    assert labels[0, 0] == "unstable"


# --------------------------------------------------------------------------
# mode grid and ground-truth spectra
# --------------------------------------------------------------------------


def test_mode_grid_matches_the_layer_storage_order():
    ky, kx = mode_grid(8, 5)
    assert list(ky) == [0, 1, 2, 3, 4, -3, -2, -1]
    assert list(kx) == [0, 1, 2, 3, 4]


def test_gray_scott_trivial_state_is_always_decaying():
    """About (1, 0) both branches are damped: no Turing instability there."""
    z = gray_scott_linear_spectrum(np.arange(0, 8), feed=0.03, kill=0.062)
    assert np.all(np.abs(z) < 1.0)
    assert np.all(np.abs(np.angle(z)) < 1e-12)      # real, so not oscillatory


def test_gray_scott_refuses_states_it_cannot_linearize_about():
    with pytest.raises(ValueError, match="closed-form"):
        gray_scott_linear_spectrum(np.arange(3), 0.03, 0.062, state="patterned")


# --------------------------------------------------------------------------
# shared bases (H2)
# --------------------------------------------------------------------------


def test_principal_angles_of_identical_subspaces_are_zero():
    rng = np.random.default_rng(1)
    basis = rng.normal(size=(12, 3)) + 1j * rng.normal(size=(12, 3))
    # 1e-6 rather than machine precision: arccos near 1 resolves an angle only
    # to sqrt(eps), which is why the scalar overlap does not go through it
    assert principal_angles(basis, basis) == pytest.approx(np.zeros(3), abs=1e-6)
    assert subspace_overlap(basis, basis) == pytest.approx(1.0)


def test_orthogonal_subspaces_have_zero_overlap():
    a = np.eye(6, dtype=complex)[:, :2]
    b = np.eye(6, dtype=complex)[:, 2:4]
    assert principal_angles(a, b) == pytest.approx(np.full(2, np.pi / 2))
    assert subspace_overlap(a, b) == pytest.approx(0.0, abs=1e-12)


def test_overlap_is_between_zero_and_one_on_random_bases():
    rng = np.random.default_rng(2)
    for _ in range(5):
        a = rng.normal(size=(20, 4)) + 1j * rng.normal(size=(20, 4))
        b = rng.normal(size=(20, 4)) + 1j * rng.normal(size=(20, 4))
        assert 0.0 <= subspace_overlap(a, b) <= 1.0


def test_mode_basis_has_one_column_per_rank_component():
    model = HarmonicLiteFNO(2, 2, width=8, modes=6, layers=2, rank=3)
    basis = mode_basis(model, layer=0)
    assert basis.shape == (6 * 6, 3)


def test_mode_basis_rejects_models_without_cp_factors():
    class Plain(torch.nn.Module):
        spectral_layers = torch.nn.ModuleList([torch.nn.Identity()])

    with pytest.raises(TypeError, match="CP-factorized"):
        mode_basis(Plain())
