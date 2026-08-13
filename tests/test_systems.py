"""Validation for litefno.systems.

These systems exist to be ground truth for everything else, so they are the one
place in the project where being wrong is silent: a mis-integrated testbed does
not fail, it just moves the "correct" answer and every downstream comparison
agrees with it. So each system is checked against the property it is carried
for, not merely for running.

The integrator bug these pin: ETD1's first-order error put the lambda-omega
limit cycle at 0.1706 rad/step against an exact 0.15, a 13.6% frequency error
in the one quantity that system is used for.

No network access, no data files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litefno.poles import analyse_series  # noqa: E402
from litefno.systems import (  # noqa: E402
    advection_diffusion, lambda_omega, lambda_omega_frequency,
    rotating_diffusion, rotating_diffusion_pole, split_trajectories)

D, OMEGA, DT, LENGTH, SIZE = 0.4, 0.6, 0.25, 32.0, 32


def _mode_ratio(field_complex: np.ndarray, ky: int, kx: int) -> complex:
    """Measured step multiplier of one Fourier mode, from the data itself."""
    spec = np.fft.fft2(field_complex, axes=(1, 2))[:, ky, kx]
    ratio = spec[1:] / np.where(np.abs(spec[:-1]) > 0, spec[:-1], 1)
    return complex(np.median(ratio[np.isfinite(ratio)]))


# --------------------------------------------------------------------------
# rotating diffusion: the exact oscillatory case
# --------------------------------------------------------------------------


def test_rotating_diffusion_matches_its_closed_form_per_mode():
    """Every mode's measured step multiplier is the analytic pole.

    This is the check the whole ground-truth claim rests on. If it holds, a gap
    between a *model's* extracted poles and this formula is the model's, which
    is the only reason the comparison is worth making.
    """
    traj = rotating_diffusion(n_traj=1, n_steps=100, size=SIZE, diffusion=D,
                              omega=OMEGA, dt=DT, length=LENGTH, seed=0)
    field = traj[0, :, :, :, 0] + 1j * traj[0, :, :, :, 1]
    for ky, kx in [(0, 0), (1, 0), (0, 2), (3, 1), (4, 4)]:
        exact = rotating_diffusion_pole(np.hypot(ky, kx), D, OMEGA, DT, LENGTH)
        assert _mode_ratio(field, ky, kx) == pytest.approx(exact, abs=1e-6), \
            f"mode ({ky}, {kx})"


def test_rotating_diffusion_is_neutral_at_dc_and_damped_at_high_k():
    """The spread H1 needs: some modes ring, some die, and they are ordered."""
    poles = rotating_diffusion_pole(np.arange(0, 9), D, OMEGA, DT, LENGTH)
    assert abs(poles[0]) == pytest.approx(1.0)              # DC is neutral
    assert abs(poles[-1]) < 0.9                             # high k is damped
    assert np.all(np.diff(np.abs(poles)) < 0)               # monotone in k
    assert np.all(np.abs(np.angle(poles)) > 1e-6)           # all oscillate


def test_rotating_diffusion_rotates_at_the_requested_frequency():
    poles = rotating_diffusion_pole(np.arange(0, 5), D, OMEGA, DT, LENGTH)
    assert np.angle(poles) == pytest.approx(np.full(5, OMEGA * DT))


# --------------------------------------------------------------------------
# advection-diffusion: exact, and deliberately without oscillation
# --------------------------------------------------------------------------


def test_advection_diffusion_energy_decays_monotonically():
    traj = advection_diffusion(n_traj=2, n_steps=16, size=SIZE, nu=0.02, seed=0)
    energy = (traj ** 2).mean(axis=(2, 3, 4))
    assert np.all(np.diff(energy, axis=1) < 0)


def test_pure_advection_conserves_energy():
    """With nu = 0 nothing damps, so a drift in energy would be integrator error."""
    traj = advection_diffusion(n_traj=1, n_steps=32, size=SIZE, nu=0.0,
                               velocity=(0.0, 1.0), seed=0)
    energy = (traj ** 2).mean(axis=(2, 3, 4))[0]
    assert energy.max() / energy.min() == pytest.approx(1.0, rel=1e-5)


def test_advection_diffusion_initial_conditions_are_band_limited():
    """Energy above max_mode would be content the operator cannot represent."""
    traj = advection_diffusion(n_traj=1, n_steps=1, size=SIZE, max_mode=4, seed=0)
    spec = np.abs(np.fft.fft2(traj[0, 0, :, :, 0])) ** 2
    freq = np.fft.fftfreq(SIZE, d=1.0 / SIZE)
    radius = np.hypot(freq[:, None], freq[None, :])
    assert spec[radius > 4].sum() / spec.sum() < 1e-12


# --------------------------------------------------------------------------
# lambda-omega: nonlinear, and only its frequency is claimed
# --------------------------------------------------------------------------


def test_lambda_omega_settles_on_the_limit_cycle():
    """|A| -> 1. Without the +1 growth term the field collapses to zero instead."""
    traj = lambda_omega(n_traj=1, n_steps=40, size=SIZE, diffusion=D,
                        omega=OMEGA, dt=DT, length=LENGTH, seed=0)
    amplitude = np.hypot(traj[..., 0], traj[..., 1]).mean()
    assert amplitude == pytest.approx(1.0, rel=0.02)


def test_lambda_omega_recovers_its_documented_frequency():
    traj = lambda_omega(n_traj=1, n_steps=300, size=SIZE, diffusion=D,
                        omega=OMEGA, dt=DT, length=LENGTH, seed=0)
    u = traj[0, :, :, :, 0].mean(axis=(1, 2))
    v = traj[0, :, :, :, 1].mean(axis=(1, 2))
    got = analyse_series(u + 1j * v, order=8)
    assert 1.0 / got["dominant_period"] == pytest.approx(
        lambda_omega_frequency(OMEGA, 0.0, DT), rel=0.02)
    assert got["sigma_reliable"]        # sustained, not a decaying transient


def test_substepping_converges_on_the_exact_frequency():
    """The regression for the 13.6% ETD1 error, kept as a convergence check.

    One substep is the old behaviour and must still be visibly wrong; refining
    must reduce the error, roughly linearly, because ETD1 is first order.
    """
    want = lambda_omega_frequency(OMEGA, 0.0, DT)
    errors = []
    for substeps in (1, 4, 16):
        traj = lambda_omega(n_traj=1, n_steps=200, size=SIZE, diffusion=D,
                            omega=OMEGA, dt=DT, length=LENGTH,
                            substeps=substeps, seed=0)
        u = traj[0, :, :, :, 0].mean(axis=(1, 2))
        v = traj[0, :, :, :, 1].mean(axis=(1, 2))
        got = analyse_series(u + 1j * v, order=8)
        errors.append(abs(1.0 / got["dominant_period"] - want) / want)
    assert errors[0] > 0.10                       # unrefined is badly wrong
    assert errors[-1] < 0.02                      # the default is not
    assert errors[0] > errors[1] > errors[2]      # and refining helps


def test_lambda_omega_stays_spatially_structured():
    """A uniform field would make the nonlinear testbed a scalar ODE."""
    traj = lambda_omega(n_traj=1, n_steps=20, size=SIZE, seed=0)
    assert traj[0].std(axis=0).mean() > 0.1


def test_substeps_must_be_positive():
    with pytest.raises(ValueError, match="substeps"):
        lambda_omega(n_traj=1, n_steps=2, size=8, substeps=0)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def test_split_is_by_trajectory_and_covers_everything():
    traj = np.arange(10 * 3 * 4 * 4).reshape(10, 3, 4, 4, 1).astype(np.float32)
    parts = split_trajectories(traj, seed=0)
    total = sum(len(p) for p in parts.values())
    assert total == 10
    seen = np.concatenate([p[:, 0, 0, 0, 0] for p in parts.values()])
    assert len(set(seen.tolist())) == 10        # no trajectory in two splits


def test_split_refuses_to_hand_back_an_empty_test_set():
    with pytest.raises(ValueError, match="cannot fill"):
        split_trajectories(np.zeros((3, 2, 4, 4, 1)), fractions=(0.9, 0.09, 0.01))
