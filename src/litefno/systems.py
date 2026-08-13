r"""Oscillatory PDE testbeds with known spectra.

SpecScope step 3: "Because your PDEs have analytically derivable linearized
spectra, you can score the extracted poles against the true ones. This is the
honesty check no interpretability paper for neural operators currently has: a
domain where the 'correct' answer for what the network should have learned is
known in closed form."

That check needs systems whose answer is known, which means systems this repo
generates rather than downloads. Two are provided, and the difference between
them is the whole point:

``advection_diffusion``  linear, non-oscillatory. The step propagator is
                         ``exp((-nu|k|^2 - i c.k) dt)`` exactly, for every mode,
                         with no linearization anywhere. A model trained on this
                         has poles with a *correct value*, so the extractor can
                         be scored rather than argued about.
``rotating_diffusion``   linear and genuinely oscillatory: every mode rings at
                         ``omega`` while diffusion damps it by ``-D q^2``, so
                         the poles ``exp((-D q^2 + i omega) dt)`` are complex,
                         near-neutral at low wavenumber and strongly damped at
                         high, and still exactly known. This is the system H1 is
                         scored on, because "which modes are resonant" has a
                         correct answer here.
``lambda_omega``         nonlinear, with a stable limit cycle. Oscillates
                         forever at a known frequency, but is *not* pole-exact:
                         linearizing about a limit cycle is a Floquet problem
                         (see :func:`lambda_omega_frequency`), so what is known
                         here is the frequency, not the full pole set. Used
                         where a nonlinear oscillatory medium is needed and the
                         claim is correspondingly weaker.

A boundary this repo has already drawn
--------------------------------------
The reproduction repo removed a self-simulated Gray-Scott dataset because a
proxy for The Well's data has no place in a reproduction. Nothing here is a
proxy for anything: these are not stand-ins for a benchmark and no accuracy
number from them is comparable to a published one. They exist because
"the network should have learned X" is only checkable when X is known, and for
The Well's settled Turing patterns it is not known -- ``poles.py``'s own
reliability flag says as much. Results on The Well's data are reported
separately and are never averaged with these.

Both integrators are pseudo-spectral with exact treatment of the linear part
(ETD1), so the diffusion is not itself an approximation contributing error to a
comparison that is about the network.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _wavenumbers(n: int, length: float) -> np.ndarray:
    return 2 * np.pi * np.fft.fftfreq(n, d=length / n)


# --------------------------------------------------------------------------
# linear: the exact case
# --------------------------------------------------------------------------


def advection_diffusion(n_traj: int = 8, n_steps: int = 64, size: int = 32,
                        nu: float = 0.02, velocity=(0.0, 1.0), dt: float = 0.5,
                        length: float = 32.0, max_mode: int = 8,
                        seed: int = 0) -> np.ndarray:
    """Trajectories of ``u_t + c.grad u = nu lap u`` on a periodic square.

    Stepped in Fourier space by multiplying each mode by its exact propagator,
    so the data contains no time-discretisation error at all and the only
    approximation in a comparison against :func:`~litefno.operator.
    linear_pde_propagator` is whatever the network did.

    Initial conditions are band-limited to ``max_mode``: a white-noise start
    would put most of its energy in modes the model truncates away, and the
    resulting trajectory would be dominated by content the operator cannot
    represent, which is a data problem masquerading as a model failure.

    Returns (n_traj, n_steps, size, size, 1), the layout the rest of the repo's
    loaders expect.
    """
    rng = np.random.default_rng(seed)
    ky = _wavenumbers(size, length)[:, None]
    kx = _wavenumbers(size, length)[None, :]
    k2 = ky ** 2 + kx ** 2
    cy, cx = velocity
    step = np.exp((-nu * k2 - 1j * (cy * ky + cx * kx)) * dt)

    radius = np.sqrt(np.fft.fftfreq(size, d=1.0 / size)[:, None] ** 2
                     + np.fft.fftfreq(size, d=1.0 / size)[None, :] ** 2)
    band = radius <= max_mode

    out = np.empty((n_traj, n_steps, size, size, 1), dtype=np.float32)
    for t in range(n_traj):
        spec = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
        spec *= band
        field = np.real(np.fft.ifft2(spec))
        field /= np.std(field) + 1e-12
        cur = np.fft.fft2(field)
        for s in range(n_steps):
            out[t, s, :, :, 0] = np.real(np.fft.ifft2(cur))
            cur = cur * step
    return out


# --------------------------------------------------------------------------
# linear and oscillatory: the case H1 is scored on
# --------------------------------------------------------------------------


def rotating_diffusion(n_traj: int = 8, n_steps: int = 64, size: int = 32,
                       diffusion: float = 0.4, omega: float = 0.6,
                       dt: float = 0.25, length: float = 32.0,
                       max_mode: int = 8, seed: int = 0) -> np.ndarray:
    """Trajectories of ``A_t = D lap A + i omega A`` for complex ``A = u + iv``.

    Every mode rotates at ``omega`` and decays at ``D q^2``, independently, so
    the field is an oscillatory medium whose per-mode poles are complex and
    exactly known -- ``exp((-D q^2 + i omega) dt)`` -- with no linearization
    anywhere, because the system is already linear.

    That combination is what H1 needs and what neither of the other two systems
    provides. ``advection_diffusion`` is exact but has no oscillation to detect,
    so "which modes are resonant" has the degenerate answer *none*.
    ``lambda_omega`` oscillates but only its frequency is known in closed form.
    Here the spread of pole margins across modes is wide, ordered, and correct
    by construction, so a claim that pole margin predicts rollout error can be
    checked against the truth rather than against another estimate.

    Returns (n_traj, n_steps, size, size, 2) holding (u, v).
    """
    rng = np.random.default_rng(seed)
    ky = _wavenumbers(size, length)[:, None]
    kx = _wavenumbers(size, length)[None, :]
    step = np.exp((-diffusion * (ky ** 2 + kx ** 2) + 1j * omega) * dt)

    radius = np.sqrt(np.fft.fftfreq(size, d=1.0 / size)[:, None] ** 2
                     + np.fft.fftfreq(size, d=1.0 / size)[None, :] ** 2)
    band = radius <= max_mode

    out = np.empty((n_traj, n_steps, size, size, 2), dtype=np.float32)
    for t in range(n_traj):
        spec = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
        a = np.fft.ifft2(spec * band)
        a /= np.abs(a).std() + 1e-12
        a_hat = np.fft.fft2(a)
        for s in range(n_steps):
            a = np.fft.ifft2(a_hat)
            out[t, s, :, :, 0] = np.real(a)
            out[t, s, :, :, 1] = np.imag(a)
            a_hat = a_hat * step
    return out


def rotating_diffusion_pole(radius, diffusion: float = 0.4, omega: float = 0.6,
                            dt: float = 0.25, length: float = 32.0
                            ) -> np.ndarray:
    """Exact per-mode pole of :func:`rotating_diffusion`.

    ``exp((-D q^2 + i omega) dt)`` for physical wavenumber ``q = 2 pi k / L``.
    One pole per mode, not two: the (u, v) pair is one complex field, and its
    conjugate partner lives at ``-k``.
    """
    q2 = (np.asarray(radius, dtype=float) * 2 * np.pi / length) ** 2
    return np.exp((-diffusion * q2 + 1j * omega) * dt)


# --------------------------------------------------------------------------
# nonlinear: a limit cycle, where only the frequency is known
# --------------------------------------------------------------------------


def lambda_omega(n_traj: int = 8, n_steps: int = 64, size: int = 32,
                 diffusion: float = 0.4, omega: float = 0.6,
                 omega1: float = 0.0, dt: float = 0.25, length: float = 32.0,
                 max_mode: int = 6, amplitude: float = 1.0,
                 perturbation: float = 0.3, seed: int = 0,
                 spinup: int = 50, substeps: int = 16) -> np.ndarray:
    """Trajectories of a lambda-omega reaction-diffusion system.

    In complex form ``A = u + iv``,

        A_t = D lap A + (1 - |A|^2) A + i (omega - omega1 |A|^2) A

    which has a stable limit cycle at ``|A| = 1``, on which the uniform state
    rotates at angular frequency ``omega - omega1`` forever. This is the classic
    normal form for an oscillatory medium (predator-prey cycles near a Hopf
    bifurcation reduce to it), and it produces exactly the traveling waves and
    spiral cores the roadmap describes as the ecosystem case.

    Why it is the right nonlinear testbed
    -------------------------------------
    Its oscillation is *sustained*, not a decaying transient, which is the
    distinction ``poles.py`` exists to make and the one Gray-Scott's settled
    patterns could not supply -- every regime tested there came back flagged
    unreliable because the patterns drift in phase rather than ring. Here the
    frequency is a parameter, so what a correct pole finder must return is
    known before the model is trained rather than inferred from its output.

    ``spinup`` steps are integrated and discarded so the returned trajectories
    sit on the limit cycle rather than on the way to it; the formation ramp has
    a 1/f spectrum that would swamp the oscillation, which is the same trap
    ext10 documented.

    Returns (n_traj, n_steps, size, size, 2) holding (u, v).
    """
    rng = np.random.default_rng(seed)
    ky = _wavenumbers(size, length)[:, None]
    kx = _wavenumbers(size, length)[None, :]
    k2 = ky ** 2 + kx ** 2
    # growth (+1) and rotation (i omega) alongside diffusion. The +1 is what
    # makes this a limit cycle rather than a decay: without it the reaction term
    # -|A|^2 A is the only amplitude dynamics and the field collapses to zero.
    linear = 1.0 - diffusion * k2 + 1j * omega
    # ETD1 is exact on the linear part and first order on the reaction term, and
    # that first order is not good enough here. At dt = 0.25 the uniform mode
    # comes out rotating at 0.1706 rad/step against the exact 0.15 -- 13.6% fast
    # -- so the "known frequency" this system is carried for would be wrong by
    # more than any effect being measured. The error is O(dt), so the stored
    # step is subdivided and only every ``substeps``-th state is kept: the data
    # has the requested dt while the integrator runs at dt/substeps.
    if substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    inner_dt = dt / substeps
    expl = np.exp(linear * inner_dt)
    phi = np.where(np.abs(linear) > 1e-12, (expl - 1) / np.where(
        np.abs(linear) > 1e-12, linear, 1.0), inner_dt)

    radius = np.sqrt(np.fft.fftfreq(size, d=1.0 / size)[:, None] ** 2
                     + np.fft.fftfreq(size, d=1.0 / size)[None, :] ** 2)
    band = radius <= max_mode

    def nonlinear(a):
        r2 = np.abs(a) ** 2
        return (-r2 + 1j * (-omega1 * r2)) * a

    out = np.empty((n_traj, n_steps, size, size, 2), dtype=np.float32)
    for t in range(n_traj):
        spec = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
        noise = np.fft.ifft2(spec * band)
        noise = noise / (np.abs(noise).mean() + 1e-12)
        # start on the limit cycle and perturb it, rather than starting from
        # noise and waiting. From noise the medium settles into spiral
        # turbulence whose spatial mean drifts in frequency, and the one thing
        # known in closed form here -- that the uniform solution rotates at
        # exactly omega -- stops being measurable. Perturbing a uniform state
        # keeps the medium nonlinear and spatially structured while leaving the
        # documented frequency recoverable, which is the property this system
        # is carried for.
        a = amplitude * (1.0 + perturbation * noise)
        a_hat = np.fft.fft2(a)

        def advance(state):
            for _ in range(substeps):
                state = expl * state + phi * np.fft.fft2(
                    nonlinear(np.fft.ifft2(state)))
            return state

        for _ in range(spinup):
            a_hat = advance(a_hat)
        for s in range(n_steps):
            a = np.fft.ifft2(a_hat)
            out[t, s, :, :, 0] = np.real(a)
            out[t, s, :, :, 1] = np.imag(a)
            a_hat = advance(a_hat)
    return out


def lambda_omega_frequency(omega: float = 0.6, omega1: float = 0.0,
                           dt: float = 0.25) -> float:
    """Cycles per step of the uniform limit cycle. The known part.

    On the limit cycle ``|A| = 1``, so the uniform solution is
    ``A = exp(i(omega - omega1) t)`` and the field completes
    ``(omega - omega1) dt / 2pi`` of a cycle per stored step. This is the same
    kind of ground truth ext12 used on planetswe -- a documented period the
    analysis must recover -- and it is checkable by eye in the trajectory.

    Why there is no matching closed form for the poles
    --------------------------------------------------
    Writing ``A = (1 + a) exp(i omega t)`` and linearizing gives
    ``a_t = D lap a - (a + conj(a))``, which is autonomous in the *co-rotating*
    variable but not in the lab frame: transforming back attaches ``exp(2 i
    omega t)`` to the conjugate term, so the lab-frame linearization is
    time-periodic and its natural invariants are Floquet multipliers over a
    period, not eigenvalues of a fixed matrix. A network extracts a fixed
    matrix, so the two objects are not the same thing and pretending otherwise
    would manufacture agreement or disagreement out of a frame choice.

    :func:`rotating_diffusion_pole` is the exact comparison; this system is for
    the questions that need a nonlinear medium, with the weaker claim attached.
    """
    return float((omega - omega1) * dt / (2 * np.pi))


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def split_trajectories(traj: np.ndarray, fractions=(0.6, 0.2, 0.2),
                       seed: Optional[int] = None) -> dict:
    """Trajectory-level train/valid/test split.

    Split on whole trajectories, never on time steps: consecutive states of one
    trajectory are near-duplicates, so a step-level split leaks the test set
    into training and every error reported afterwards is optimistic.
    """
    n = len(traj)
    idx = np.arange(n)
    if seed is not None:
        np.random.default_rng(seed).shuffle(idx)
    n_train = max(1, int(round(fractions[0] * n)))
    n_valid = max(1, int(round(fractions[1] * n)))
    if n_train + n_valid >= n:
        raise ValueError(
            f"{n} trajectories cannot fill splits {fractions}; ask for more")
    return {"train": traj[idx[:n_train]],
            "valid": traj[idx[n_train:n_train + n_valid]],
            "test": traj[idx[n_train + n_valid:]]}
