r"""The trained operator as an empirical transfer function.

SpecScope step 1-2: "For a trained LiteFNO, the low-rank factors define, per
Fourier mode k, an effective linear map on that mode. Composing layers gives a
per-mode frequency response. This is checkpoint analysis -- no training."

``litefno.poles`` already fits poles, but to *data*: it takes a measured field,
extracts each spatial mode's time series, and asks what the system does. This
module asks the other question -- what the trained network thinks the system
does -- by reading the poles out of the weights. The two are directly
comparable, because both end up as discrete-time poles per spatial mode with the
same convention (``sigma = log|z|`` per step, ``freq = arg(z)/2pi``), so
:func:`litefno.poles.classify_poles` labels either one.

Why a spectral operator has poles at all
----------------------------------------
An FNO layer is diagonal in Fourier space by construction: the spectral
convolution multiplies mode ``k`` by a channel-mixing matrix ``W_k`` and touches
no other mode, and the pointwise skip is the same matrix ``S`` at every mode. So
one layer acts on the channel vector of mode k as

    z  ->  phi(W_k^T z + S z)

and, with the activation linearized at gain ``g``, a whole L-layer network with
lift ``Q`` and projection ``P`` collapses to one matrix per mode,

    M_k = P (g L_k^{(L)}) ... (g L_k^{(1)}) Q,     L_k^{(l)} = W_k^{(l)T} + S^{(l)}

of shape (out_channels, in_channels) -- for Gray-Scott, 2x2. The model predicts
the next state directly, so ``M_k`` *is* the one-step propagator restricted to
mode k, and its eigenvalues are discrete-time poles in the ordinary sense. An
eigenvalue on the unit circle is a mode the operator will ring on forever; one
inside decays; one outside means the surrogate will blow up in rollout, which is
the connection to H1.

The linearization is the one real assumption
--------------------------------------------
GELU is pointwise in space, so it is *not* diagonal in Fourier space: it couples
modes, and the collapse above is exact only for the linearized network. That is
a genuine approximation and it is why this module ships two independent routes
to the same object:

``analytic_mode_operators``   composes the weights as above. Exact for the
                              linearized network, zero cost, needs to know the
                              architecture.
``empirical_mode_operators``  probes the real model with finite differences
                              around a real state and reads the response off in
                              Fourier space. Makes no architectural assumption,
                              includes whatever the activation actually does at
                              that operating point, and works on FNO-S or any
                              other model with no code changes.

Agreement between them says the linearization is harmless at that state;
disagreement is the honest signal that it is not, and
:func:`compare_operators` reports it as a number rather than leaving it to
faith. Neither route is trusted by default -- ``scripts/operator_poles.py``
runs both.

Ground truth exists here, which is the unusual part
---------------------------------------------------
For a *linear* PDE the true per-mode propagator is known in closed form:
diffusion-advection over a step ``dt`` multiplies mode k by
``exp((-nu|k|^2 - i c.k) dt)`` exactly. A model trained on that system has a
correct answer its extracted poles can be scored against, with no linearization
error anywhere in the comparison, because the system itself is linear.
:func:`linear_pde_propagator` provides it. For Gray-Scott,
:func:`gray_scott_linear_spectrum` gives the Turing analysis about the
homogeneous steady state -- the same object, but only valid near that state, and
the committed trajectories are settled patterns far from it. Both are exposed;
only the first is a clean check, and the scripts say which is which.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# GELU'(0) = 0.5. The default linearization gain, used when composing weights
# without reference to an operating point. It scales every layer identically, so
# it rescales |z| by g^L and leaves arg(z) -- the frequency -- untouched.
GELU_GAIN_AT_ZERO = 0.5


# --------------------------------------------------------------------------
# route 1: compose the weights
# --------------------------------------------------------------------------


def _layer_matrices(model):
    """(spectral weight, skip matrix) per layer, as numpy, plus lift/project.

    Supports the repo's two spectral architectures. ``HarmonicLiteFNO`` stores
    its weight in CP form and reconstructs it on demand; ``FNOS`` stores it
    dense. Both index the retained block the same way, which is what lets one
    extractor serve both.
    """
    import torch

    def np_(t):
        return t.detach().cpu().numpy()

    layers = getattr(model, "spectral_layers", None)
    if layers is None:
        raise TypeError(
            f"{type(model).__name__} has no .spectral_layers; use "
            "empirical_mode_operators, which needs no architecture knowledge")

    weights, skips = [], []
    for i, layer in enumerate(layers):
        if hasattr(layer, "weight") and callable(layer.weight):
            w = np_(layer.weight())                    # CP -> dense (i,o,m1,m2)
        elif hasattr(layer, "weights"):
            w = np_(layer.weights)
        else:
            raise TypeError(f"spectral layer {i} exposes no weight tensor")
        weights.append(w)
        skip = model.skips[i]
        skips.append(np_(skip.weight)[:, :, 0, 0])     # (out, in)

    with torch.no_grad():
        lift = np_(model.input_proj.weight)[:, :, 0, 0]
        project = np_(model.output_proj.weight)[:, :, 0, 0]
    return weights, skips, lift, project


def mode_grid(modes1: int, modes2: int) -> tuple:
    """Signed (ky, kx) of every retained mode, in the layer's storage order.

    ``modes1`` holds positive vertical wavenumbers first and the negative ones
    folded onto the end, which is how ``CPSpectralConv2d`` slices the spectrum;
    getting this wrong silently mislabels half the modes as high-frequency.
    """
    pos = modes1 // 2 + 1
    ky = np.concatenate([np.arange(pos), np.arange(-(modes1 - pos), 0)])
    kx = np.arange(modes2)
    return ky, kx


def analytic_mode_operators(model, gelu_gain: float = GELU_GAIN_AT_ZERO
                            ) -> dict:
    """Per-mode one-step propagator ``M_k``, composed from the weights.

    Returns ``ky``, ``kx`` (signed wavenumbers), ``radius`` and ``operators``
    with shape (modes1, modes2, out_channels, in_channels), complex.

    The activation enters only as the scalar ``gelu_gain`` per layer. It is a
    uniform rescaling of every pole magnitude by ``gain**n_layers`` and does not
    move any pole in angle, so a wrong gain shifts every growth rate by the same
    additive constant in log space and cannot change which mode is the most
    resonant -- the ranking H1 depends on survives it, the absolute stability
    call does not. ``scripts/operator_poles.py`` reports the gain it used.
    """
    weights, skips, lift, project = _layer_matrices(model)
    m1, m2 = weights[0].shape[2], weights[0].shape[3]
    ky, kx = mode_grid(m1, m2)

    ops = np.zeros((m1, m2, project.shape[0], lift.shape[1]), dtype=np.complex128)
    for a in range(m1):
        for b in range(m2):
            acc = lift.astype(np.complex128)
            for w, s in zip(weights, skips):
                layer = w[:, :, a, b].T + s            # (out, in), see docstring
                acc = gelu_gain * (layer @ acc)
            ops[a, b] = project @ acc
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    return {"ky": ky, "kx": kx, "radius": radius, "operators": ops,
            "gelu_gain": gelu_gain, "route": "analytic"}


# --------------------------------------------------------------------------
# route 2: probe the model
# --------------------------------------------------------------------------


def empirical_mode_operators(model, base_state, max_mode: int = 8,
                             eps: float = 1e-3, device: str = "cpu") -> dict:
    """Per-mode propagator measured by finite differences around ``base_state``.

    ``base_state`` is (C, H, W) real. For each retained mode k and each input
    channel j the model is probed with ``cos`` and ``sin`` perturbations of that
    single mode; the response is read at mode k in Fourier space, giving one
    complex column of ``M_k``. Assembling over j gives the matrix.

    This is the empirical transfer function of the actual network, activation
    and all, at that operating point. It costs ``2 * n_modes * C`` forward
    passes -- a few thousand on a 32x32 grid, seconds on CPU.

    ``eps`` matters in both directions: too large and the difference quotient
    picks up the curvature that the linear response is supposed to exclude, too
    small and it is float32 noise. The default is calibrated for fields
    normalised to order 1; :func:`probe_convergence` checks it rather than
    assuming it.
    """
    import torch

    model.eval()
    base = torch.as_tensor(np.asarray(base_state), dtype=torch.float32)
    if base.ndim != 3:
        raise ValueError(f"base_state must be (C, H, W), got {tuple(base.shape)}")
    channels, height, width = base.shape
    base = base.to(device)

    with torch.no_grad():
        f0 = model(base[None])[0]
    out_channels = f0.shape[0]

    yy, xx = np.mgrid[0:height, 0:width]
    modes = [(a, b) for a in range(-max_mode, max_mode + 1)
             for b in range(0, max_mode + 1)
             if a * a + b * b <= max_mode * max_mode
             and abs(a) != height // 2 and b != width // 2]

    ops = np.zeros((len(modes), out_channels, channels), dtype=np.complex128)
    for mi, (a, b) in enumerate(modes):
        phase = 2 * np.pi * (a * yy / height + b * xx / width)
        # A real field cannot carry an independent +k and -k, so probing with
        # cos(k.r) excites both and the response at bin +k is (HW/2) T_k, while
        # sin(k.r) gives (HW/2)(-i T_k). Recovering T_k needs both -- one probe
        # alone leaves the phase of the response undetermined. The DC mode is
        # its own case: +k and -k coincide there, the factor of two disappears
        # and the sin probe is identically zero.
        self_conjugate = (a == 0 and b == 0)
        cos = torch.as_tensor(np.cos(phase), dtype=torch.float32, device=device)
        sin = torch.as_tensor(np.sin(phase), dtype=torch.float32, device=device)
        scale = (1.0 if self_conjugate else 2.0) / (height * width)
        for j in range(channels):
            resp = []
            for pattern in ((cos,) if self_conjugate else (cos, sin)):
                pert = torch.zeros_like(base)
                pert[j] = eps * pattern
                with torch.no_grad():
                    delta = (model((base + pert)[None])[0] - f0) / eps
                spec = np.fft.rfft2(delta.cpu().numpy().astype(np.float64))
                # rfft2 stores negative ky folded to the end; kx >= 0 only
                resp.append(spec[:, a % height, b] * scale)
            ops[mi, :, j] = resp[0] if self_conjugate \
                else 0.5 * (resp[0] + 1j * resp[1])

    modes = np.asarray(modes)
    radius = np.sqrt((modes ** 2).sum(axis=1))
    return {"ky": modes[:, 0], "kx": modes[:, 1], "radius": radius,
            "operators": ops, "eps": eps, "route": "empirical"}


def probe_convergence(model, base_state, mode=(1, 1), device: str = "cpu",
                      eps_values: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4)
                      ) -> list[dict]:
    """Sensitivity of one probed mode to the step size.

    A finite-difference Jacobian is only a Jacobian on the plateau between
    curvature error (large eps) and cancellation noise (small eps). This walks
    eps down and reports the relative change per halving of scale, so the caller
    can see the plateau instead of trusting a default.
    """
    a, b = mode
    # the probe keeps modes inside a disc, so the radius has to cover (a, b)
    # itself -- max(|a|, b) drops the diagonal ones and leaves nothing to read
    reach = int(np.ceil(np.hypot(a, b)))
    prev, out = None, []
    for eps in eps_values:
        got = empirical_mode_operators(model, base_state, max_mode=reach,
                                       eps=eps, device=device)
        sel = np.flatnonzero((got["ky"] == a) & (got["kx"] == b))
        m = got["operators"][sel[0]]
        rel = float(np.linalg.norm(m - prev) / max(np.linalg.norm(prev), 1e-30)) \
            if prev is not None else float("nan")
        out.append({"eps": eps, "norm": float(np.linalg.norm(m)),
                    "rel_change": rel})
        prev = m
    return out


# --------------------------------------------------------------------------
# poles of a per-mode operator
# --------------------------------------------------------------------------


def operator_poles(operators: np.ndarray) -> dict:
    """Eigenvalues of each per-mode propagator, as discrete-time poles.

    ``operators`` is (..., C, C). Returns ``z`` (the eigenvalues), ``sigma =
    log|z|`` (growth rate per step; 0 is neutral) and ``freq = |arg z| / 2pi``
    (cycles per step), matching the convention in :mod:`litefno.poles` so the
    same classifier applies to a model's poles and to the data's.

    Non-square or empty operators are a caller error and raise; a silent reshape
    here would produce plausible eigenvalues of the wrong matrix.
    """
    ops = np.asarray(operators)
    if ops.ndim < 2 or ops.shape[-1] != ops.shape[-2]:
        raise ValueError(f"expected square per-mode operators, got {ops.shape}")
    z = np.linalg.eigvals(ops)
    magnitude = np.abs(z)
    with np.errstate(divide="ignore"):
        sigma = np.log(np.where(magnitude > 0, magnitude, 1e-300))
    return {"z": z, "sigma": sigma, "freq": np.abs(np.angle(z)) / (2 * np.pi),
            "magnitude": magnitude}


def stability_margin(sigma: np.ndarray) -> np.ndarray:
    """Distance of the least-damped pole from neutrality, per mode.

    Negative means every pole decays (the margin is how fast the slowest one
    does); positive means at least one pole grows, and the surrogate amplifies
    that mode every step it is rolled out. This is the scalar H1 predicts
    rollout error growth from.
    """
    return np.asarray(sigma).max(axis=-1)


def classify_operator_modes(poles: dict, neutral_tol: float = 5e-3,
                            min_cycles_per_step: float = 1e-3) -> np.ndarray:
    """Per-mode label from that mode's least-damped pole.

    The three classes are SpecScope's: ``resonant`` (near-neutral and genuinely
    oscillating -- the modes H2 transplants), ``primary`` (near-neutral but real
    and positive: a slowly-varying or held component, also transplanted), and
    ``damped`` (decaying, the control arm's set). ``unstable`` is split out from
    resonant rather than folded into it, because a mode the operator amplifies
    is the failure H1 is trying to predict, not a physics mode worth moving.

    The tolerance is :data:`litefno.poles.NEUTRAL_TOL` by default, so a mode
    called resonant here and an oscillation called neutral there mean the same
    thing.
    """
    sigma, freq = np.asarray(poles["sigma"]), np.asarray(poles["freq"])
    idx = np.argmax(sigma, axis=-1)
    lead_sigma = np.take_along_axis(sigma, idx[..., None], axis=-1)[..., 0]
    lead_freq = np.take_along_axis(freq, idx[..., None], axis=-1)[..., 0]

    labels = np.full(lead_sigma.shape, "damped", dtype=object)
    near = np.abs(lead_sigma) <= neutral_tol
    labels[near & (lead_freq > min_cycles_per_step)] = "resonant"
    labels[near & (lead_freq <= min_cycles_per_step)] = "primary"
    labels[lead_sigma > neutral_tol] = "unstable"
    return labels


def compare_operators(analytic: dict, empirical: dict) -> list[dict]:
    """Per-mode agreement between the composed and the probed operator.

    Matched on signed (ky, kx), so it is immune to the two routes storing their
    modes in different orders -- which they do. Reports the relative operator
    norm difference and, more usefully, the gap in leading pole magnitude: the
    norms can differ by an overall scale (the activation gain) while the
    stability call, which is what everything downstream uses, still agrees.
    """
    out = []
    lookup = {(int(a), int(b)): i for i, (a, b)
              in enumerate(zip(np.ravel(empirical["ky"]), np.ravel(empirical["kx"])))}
    ky, kx = analytic["ky"], analytic["kx"]
    for a in range(len(ky)):
        for b in range(len(kx)):
            key = (int(ky[a]), int(kx[b]))
            if key not in lookup:
                continue
            m_a = analytic["operators"][a, b]
            m_e = empirical["operators"][lookup[key]]
            denom = max(np.linalg.norm(m_e), 1e-30)
            za = np.abs(np.linalg.eigvals(m_a)).max()
            ze = np.abs(np.linalg.eigvals(m_e)).max()
            out.append({
                "ky": key[0], "kx": key[1],
                "radius": float(np.hypot(*key)),
                "rel_norm_diff": float(np.linalg.norm(m_a - m_e) / denom),
                "analytic_lead_magnitude": float(za),
                "empirical_lead_magnitude": float(ze),
                "log_magnitude_gap": float(np.log(max(za, 1e-300))
                                           - np.log(max(ze, 1e-300))),
            })
    return out


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------


def linear_pde_propagator(ky: np.ndarray, kx: np.ndarray, dt: float,
                          nu: float = 0.0, velocity=(0.0, 0.0),
                          height: int = 32, width: int = 32) -> np.ndarray:
    """Exact per-mode propagator of advection-diffusion over one step.

    ``u_t + c.grad u = nu lap u`` is diagonal in Fourier space, so the step map
    multiplies mode k by ``exp((-nu|k|^2 - i c.k) dt)`` and nothing is
    approximated anywhere. That is the point of using it: a model trained on
    this system has poles with a known correct value, so any gap between the
    extracted poles and these is extractor error rather than a modelling
    argument. Wavenumbers are converted to physical ones by the grid size, so
    ``ky``/``kx`` are integer mode indices as the rest of this module uses them.
    """
    ky = np.asarray(ky, dtype=float)[:, None] * (2 * np.pi / height)
    kx = np.asarray(kx, dtype=float)[None, :] * (2 * np.pi / width)
    k2 = ky ** 2 + kx ** 2
    cy, cx = velocity
    return np.exp((-nu * k2 - 1j * (cy * ky + cx * kx)) * dt)


def gray_scott_linear_spectrum(radius: np.ndarray, feed: float, kill: float,
                               d_u: float = 2e-5, d_v: float = 1e-5,
                               dt: float = 10.0, domain: float = 1.0,
                               state: str = "trivial") -> np.ndarray:
    """Linearized Gray-Scott eigenvalues per radial wavenumber, as step poles.

    Gray-Scott,

        u_t = d_u lap u - u v^2 + F (1 - u)
        v_t = d_v lap v + u v^2 - (F + k) v

    has the homogeneous state (u, v) = (1, 0), about which the Jacobian is
    diagonal, giving continuous-time eigenvalues ``-F - d_u q^2`` and
    ``-(F + k) - d_v q^2`` for physical wavenumber q. Exponentiating over ``dt``
    puts them in the same discrete-time convention as everything else here.

    Two caveats, stated because the comparison is worthless without them. This
    state is the *unpatterned* one: the committed trajectories are settled
    Turing patterns nowhere near it, so this is a reference point, not the
    ground truth that :func:`linear_pde_propagator` is. And ``d_u``, ``d_v`` and
    ``domain`` are properties of the generating simulation rather than of the
    stored file; the defaults are the standard Pearson values, and the scripts
    pass them explicitly so a wrong assumption is visible in the output rather
    than buried here.
    """
    if state != "trivial":
        raise ValueError(
            f"unknown state {state!r}; only the homogeneous (1, 0) state has a "
            "closed-form Jacobian, and the patterned states do not have one at "
            "all -- use linear_pde_propagator for an exact check")
    q2 = (np.asarray(radius, dtype=float) * 2 * np.pi / domain) ** 2
    lam_u = -feed - d_u * q2
    lam_v = -(feed + kill) - d_v * q2
    return np.exp(np.stack([lam_u, lam_v], axis=-1) * dt)


# --------------------------------------------------------------------------
# shared spectral bases (H2's overlap matrix)
# --------------------------------------------------------------------------


def mode_basis(model, layer: int = 0) -> np.ndarray:
    """The spectral factor a model learned over the mode axes, as a basis.

    For a CP-factorized layer the mode dependence lives entirely in
    ``factor_m1`` and ``factor_m2``: their outer product per rank component is
    the operator's footprint over the mode grid. Flattening that to one vector
    per rank component gives an ``(n_modes, rank)`` basis whose span is what the
    layer can express spatially, and which is the object H2 asks whether two
    disaster regimes share.
    """
    layers = getattr(model, "spectral_layers", None)
    if layers is None:
        raise TypeError(f"{type(model).__name__} has no spectral_layers")
    lay = layers[layer]
    if not hasattr(lay, "factor_m1"):
        raise TypeError("mode_basis needs a CP-factorized layer "
                        "(factor_m1 / factor_m2)")
    f1 = lay.factor_m1.detach().cpu().numpy()          # (modes1, rank)
    f2 = lay.factor_m2.detach().cpu().numpy()          # (modes2, rank)
    outer = f1[:, None, :] * f2[None, :, :]            # (modes1, modes2, rank)
    return outer.reshape(-1, outer.shape[-1])


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two subspaces.

    The standard SVD construction: orthonormalize both, take singular values of
    ``Qa^H Qb``, arccos them. Small angles mean the two models learned
    overlapping spectral structure; angles near pi/2 mean they did not, which is
    the negative result H2 has to be able to return.

    Singular values are clipped into [0, 1] before the arccos -- they leave the
    SVD a few ulp outside it and ``arccos(1 + 1e-16)`` is nan, which would
    propagate silently into the overlap matrix as a blank cell.

    Small angles are the inaccurate end of this formulation: ``arccos(1 - e)``
    grows like ``sqrt(2e)``, so a singular value correct to machine precision
    still gives an angle only to about ``1e-8`` radians. That is far below any
    threshold used here, but it is why :func:`subspace_overlap` reports
    ``cos^2`` -- computed before the arccos -- rather than an angle.
    """
    qa, _ = np.linalg.qr(np.asarray(basis_a, dtype=np.complex128))
    qb, _ = np.linalg.qr(np.asarray(basis_b, dtype=np.complex128))
    s = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
    return np.arccos(np.clip(s, 0.0, 1.0))


def subspace_overlap(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """One number for how much two learned spectral bases share, in [0, 1].

    ``mean(cos^2(theta))`` over the principal angles: 1 when the subspaces
    coincide, 0 when every direction of one is orthogonal to all of the other.
    Squared cosines rather than cosines because that makes it the fraction of
    energy of a random vector in one subspace that survives projection into the
    other, which is the quantity the transplant actually cares about.
    """
    theta = principal_angles(basis_a, basis_b)
    return float(np.mean(np.cos(theta) ** 2)) if theta.size else 0.0
