r"""Pole-residue analysis: which spectral modes actually oscillate.

Board task: "borrow from control theory / pole-residue analysis -- systems with
stable oscillations have poles near the imaginary axis (neutral stability); use
these transfer-function insights to identify which spectral modes are 'actually'
oscillatory vs. transient; this gives you a principled way to decide which modes
to preserve."

The idea is that a power spectrum cannot tell those apart. A decaying transient
and a sustained oscillation at the same frequency deposit power in the same bin,
and ext10 already ran into the consequence: over a full Gray-Scott trajectory
every scenario looks "low-frequency dominated" because the one-time
pattern-formation ramp has a 1/f spectrum, whether or not anything oscillates.
ext10's fix was an AR(1) null, which is a one-pole model -- a single real pole,
so it can express decay but never oscillation. This generalises that to p poles,
which can be complex, and complex poles are what oscillation looks like.

The model
---------
Each spatial Fourier mode's time series is treated as the impulse response of a
linear system, fitted as

    x[n] = sum_i c_i x[n-i]        (AR(p), least squares)

whose characteristic roots are the discrete-time poles z_i. Amplitudes follow by
projecting the series onto ``z_i^n`` -- the residues. Then, per step,

    sigma_i = log|z_i|          growth rate; 0 is neutral, negative is decaying
    f_i     = arg(z_i) / 2pi    frequency in cycles per step

In continuous time a pole on the imaginary axis is neutrally stable and rings
forever; in discrete time that is the unit circle, so ``|log|z||`` near zero is
the neutral-stability condition and ``arg(z)`` away from 0 and pi is what makes
it oscillatory rather than a slow drift.

    oscillatory   near-neutral and complex -- rings for longer than the record
    transient     decaying, whatever its frequency
    stationary    near-neutral and real, positive -- a constant, not a cycle
    unstable      growing; on settled data this usually means the fit is
                  extrapolating and the mode should be treated with suspicion

Residues are not decoration
---------------------------
A pole with a negligible residue is a fitting artifact, and an order-16 fit
always produces 16 of them. So every classification here is weighted by the
energy each pole actually contributes to the reconstruction,
``sum_n |r_i z_i^n|^2``, and the headline number per mode is the *share* of its
energy sitting in near-neutral complex poles. Counting poles instead of energy
would let sixteen numerical ghosts outvote the one pole carrying the signal.

Why this can be checked rather than believed
--------------------------------------------
planetswe has documented forcing at exactly 24 and 1008 steps (ext12), so a pole
finder that works must place near-neutral complex poles at those two
frequencies and nowhere else in particular. That is a ground truth on real data,
not a synthetic self-test, and ``scripts/pole_analysis.py`` runs it.

Where the model does not apply, and how it says so
--------------------------------------------------
An AR fit assumes the series is a sum of fixed-frequency exponentials. A mode
whose phase wanders is not, and the fit cannot say so directly -- it absorbs the
misfit into damping and reports a decaying mode. On Gray-Scott's settled spirals
the fit gives sigma = -0.0067 for the dominant mode while its amplitude falls by
4% across 501 steps, a true sigma of -0.00012: a factor of 56, and enough to
flip the mode from oscillatory to transient.

:func:`envelope_sigma` is the guard. It measures decay from the amplitude
envelope alone, assuming nothing about frequency, so it disagrees with the poles
exactly when the constant-frequency model is wrong. ``sigma_reliable`` asks
whether the two land on the same side of the neutral threshold -- the same
*label*, not the same number, since two tiny rates of opposite sign mean the
same thing.

That check separates the two datasets cleanly. planetswe's forced diurnal mode
is reliable, and reads as 100% oscillatory. All four Gray-Scott regimes tested
are flagged unreliable: their patterns are self-organised and drift in phase, so
their "0% oscillatory" reading is a statement about the model, not about the
physics, and is reported as such.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# A pole is "neutral" when it decays by less than this per step. The default
# corresponds to an e-folding time of 200 steps: over a 3024-step planetswe
# record that is a mode still ringing at the end, and over a 60-step training
# window it is indistinguishable from constant.
NEUTRAL_TOL = 5e-3


def fit_ar_poles(series: np.ndarray, order: int = 12,
                 rcond: float = 1e-10) -> np.ndarray:
    """Discrete-time poles of an AR(``order``) fit to one series.

    Least squares on the Hankel system, then the roots of the characteristic
    polynomial. ``rcond`` truncates the least-squares solve: these systems are
    routinely rank-deficient (a mode with two real oscillations in it does not
    need twelve poles) and without truncation the surplus roots land wherever
    rounding sends them, sometimes outside the unit circle, which would read as
    a spurious instability.
    """
    x = np.asarray(series)
    n = x.shape[0]
    if n <= order + 1:
        raise ValueError(f"series of length {n} too short for order {order}")

    rows = n - order
    hankel = np.stack([x[order - 1 - i: order - 1 - i + rows]
                       for i in range(order)], axis=1)
    target = x[order:order + rows]
    coeffs, *_ = np.linalg.lstsq(hankel, target, rcond=rcond)
    # characteristic polynomial z^p - c_1 z^(p-1) - ... - c_p
    poly = np.concatenate([[1.0], -np.asarray(coeffs)])
    return np.roots(poly)


def pole_residues(series: np.ndarray, poles: np.ndarray,
                  rcond: float = 1e-10):
    """Amplitudes of each pole, and the energy each contributes.

    Solves ``x[n] = sum_i r_i z_i^n``. Energy is the actual contribution
    ``sum_n |r_i z_i^n|^2`` rather than ``|r_i|^2``, because a large residue on
    a fast-decaying pole contributes almost nothing after the first few steps.
    """
    x = np.asarray(series)
    n = x.shape[0]
    powers = np.arange(n)[:, None]
    # guard against overflow from a badly placed pole before it becomes a nan
    with np.errstate(over="ignore", invalid="ignore"):
        vander = np.power(poles[None, :].astype(np.complex128), powers)
    vander = np.nan_to_num(vander, nan=0.0, posinf=0.0, neginf=0.0)
    residues, *_ = np.linalg.lstsq(vander, x.astype(np.complex128), rcond=rcond)
    energy = (np.abs(vander * residues[None, :]) ** 2).sum(axis=0)
    return residues, np.real(energy)


def envelope_sigma(series: np.ndarray, smooth: Optional[int] = None) -> float:
    """Decay rate estimated from the amplitude envelope, independent of any fit.

    A regression of log|s(t)| against t. This exists as a cross-check on the
    pole magnitudes, and it is needed: on Gray-Scott's spirals the pole fit
    reported sigma = -0.0067 while the envelope only fell by 4% across 501 steps
    (sigma = -0.00012), a factor of 56. A fixed-frequency exponential cannot
    represent a mode whose phase wanders, so it buys the misfit with damping,
    and the damping is what the neutral-vs-transient call depends on.

    The envelope makes no assumption about frequency, so the two disagree
    exactly when the constant-frequency model is wrong.
    """
    x = np.asarray(series)
    x = x - x.mean()
    if np.isrealobj(x):
        # |cos| touches zero twice per period, and a log-regression through
        # those zeros measures the zero-crossings rather than the envelope.
        # The analytic signal has the envelope as its modulus. (Fourier-mode
        # series are already complex and need no transform.)
        spec = np.fft.fft(x)
        h = np.zeros(len(x))
        h[0] = 1
        if len(x) % 2 == 0:
            h[len(x) // 2] = 1
            h[1:len(x) // 2] = 2
        else:
            h[1:(len(x) + 1) // 2] = 2
        x = np.fft.ifft(spec * h)
    amp = np.abs(x)
    if smooth is None:
        smooth = max(3, len(x) // 50)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        amp = np.convolve(amp, kernel, mode="valid")
    floor = amp[amp > 0].min() * 1e-3 if np.any(amp > 0) else 1e-30
    t = np.arange(len(amp), dtype=float)
    slope, _ = np.polyfit(t, np.log(np.maximum(amp, floor)), 1)
    return float(slope)


def classify_poles(poles: np.ndarray, energy: np.ndarray,
                   neutral_tol: float = NEUTRAL_TOL,
                   n_time: Optional[int] = None,
                   min_cycles_in_record: float = 2.0,
                   min_cycles_per_step: Optional[float] = None) -> dict:
    """Label each pole and return the share of energy in each class.

    A pole on the positive real axis is a constant, not a cycle, so there has to
    be a floor on frequency. Setting that floor as a fixed constant is a trap:
    an earlier default of 1e-3 cycles per step silently excluded every period
    above 1000 steps, and planetswe's annual forcing is 1008 -- it was being
    labelled "stationary" by a hair, on the one dataset where the right answer
    is documented.

    The floor is now tied to the record: a cycle counts as resolved only if the
    series contains ``min_cycles_in_record`` of it, which is the same condition
    under which it would be visible in a spectrum. Pass ``n_time`` to use it;
    ``min_cycles_per_step`` still overrides explicitly.
    """
    if min_cycles_per_step is None:
        min_cycles_per_step = (min_cycles_in_record / n_time) if n_time \
            else 1e-3
    poles = np.asarray(poles)
    energy = np.asarray(energy, dtype=float)
    magnitude = np.abs(poles)
    with np.errstate(divide="ignore"):
        sigma = np.log(np.where(magnitude > 0, magnitude, 1e-300))
    freq = np.abs(np.angle(poles)) / (2 * np.pi)

    oscillatory = freq > min_cycles_per_step
    near_neutral = np.abs(sigma) <= neutral_tol
    growing = sigma > neutral_tol

    labels = np.full(poles.shape, "transient", dtype=object)
    labels[near_neutral & oscillatory] = "oscillatory"
    labels[near_neutral & ~oscillatory] = "stationary"
    labels[growing] = "unstable"

    total = energy.sum()
    shares = {}
    for name in ("oscillatory", "stationary", "transient", "unstable"):
        sel = labels == name
        shares[f"{name}_share"] = float(energy[sel].sum() / total) if total > 0 \
            else 0.0

    # the dominant oscillatory pole, which is the one worth naming
    osc = np.flatnonzero(labels == "oscillatory")
    if osc.size:
        best = osc[int(np.argmax(energy[osc]))]
        shares.update(dominant_freq=float(freq[best]),
                      dominant_period=float(1.0 / freq[best]) if freq[best] > 0
                      else float("inf"),
                      dominant_sigma=float(sigma[best]),
                      dominant_energy_share=float(energy[best] / total)
                      if total > 0 else 0.0)
    else:
        shares.update(dominant_freq=float("nan"),
                      dominant_period=float("nan"),
                      dominant_sigma=float("nan"),
                      dominant_energy_share=0.0)
    return {"labels": labels, "sigma": sigma, "freq": freq, **shares}


def analyse_series(series: np.ndarray, order: int = 12,
                   neutral_tol: float = NEUTRAL_TOL) -> dict:
    """Fit, weight by residue energy, and classify one time series.

    The temporal mean is removed first. Without it a constant offset takes a
    pole of its own and, on planetswe's zonal mean, 99% of the energy -- the
    classification then describes the offset rather than the dynamics.
    """
    series = np.asarray(series)
    series = series - series.mean()
    poles = fit_ar_poles(series, order=order)
    residues, energy = pole_residues(series, poles)
    out = classify_poles(poles, energy, neutral_tol=neutral_tol,
                         n_time=len(series))

    # Independent damping estimate, as a check on the fit. The comparison is
    # whether the two agree on the *label*, not on the number: two rates that
    # differ by a factor of three but are both far inside the neutral band
    # produce the same answer, while a relative difference between two
    # near-zero values is large and means nothing. Compare against the
    # energy-weighted mean pole decay rather than the dominant oscillatory
    # pole, which is undefined when the fit found no oscillation at all -- and
    # that is exactly the case worth flagging.
    env = envelope_sigma(series)
    total = energy.sum()
    fit_sigma = float((out["sigma"] * energy).sum() / total) if total > 0 \
        else float("nan")
    same_call = (np.isfinite(fit_sigma)
                 and (abs(fit_sigma) <= neutral_tol) == (abs(env) <= neutral_tol))
    out.update(poles=poles, residues=residues, energy=energy,
               envelope_sigma=env, fit_sigma=fit_sigma,
               sigma_gap=abs(fit_sigma - env) if np.isfinite(fit_sigma)
               else float("nan"),
               sigma_reliable=bool(same_call))
    return out


def spatial_mode_series(field: np.ndarray, max_mode: int) -> tuple:
    """Time series of each spatial Fourier mode, low modes first.

    ``field`` is (T, H, W) real. Returns the complex coefficient series with
    shape (T, n_modes) and the integer radial wavenumber of each mode, so a
    caller can group results by shell.
    """
    spec = np.fft.rfft2(np.asarray(field, dtype=np.float64), axes=(1, 2))
    n_time, height, half = spec.shape
    ky = np.fft.fftfreq(height, d=1.0 / height).astype(int)
    kx = np.arange(half)
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).round().astype(int)
    keep = radius <= max_mode
    rows, cols = np.nonzero(keep)
    order = np.argsort(radius[rows, cols], kind="stable")
    rows, cols = rows[order], cols[order]
    return spec[:, rows, cols], radius[rows, cols]


def analyse_field(field: np.ndarray, max_mode: int = 8, order: int = 12,
                  neutral_tol: float = NEUTRAL_TOL,
                  min_energy_frac: float = 1e-6) -> list[dict]:
    """Pole analysis of every retained spatial mode of a field.

    Modes carrying a negligible share of the field's total variance are skipped:
    fitting twelve poles to numerical noise produces twelve confident-looking
    poles, and they would dominate any per-mode average.
    """
    series, radii = spatial_mode_series(field, max_mode)
    weights = (np.abs(series) ** 2).sum(axis=0)
    total = weights.sum()

    out = []
    for i in range(series.shape[1]):
        if total > 0 and weights[i] / total < min_energy_frac:
            continue
        got = analyse_series(series[:, i], order=order, neutral_tol=neutral_tol)
        out.append({"mode_index": int(i), "k": int(radii[i]),
                    "energy_weight": float(weights[i] / total) if total else 0.0,
                    **{key: got[key] for key in got
                       if key not in ("labels", "sigma", "freq", "poles",
                                      "residues", "energy")}})
    return out
