r"""Training, per-mode rollout error, and factor transplants.

The machinery the three SpecScope experiments share, kept in the package rather
than in one script so the other two do not import it through a file path and so
the tests can reach it.

Three things live here:

``fit``                  a compact one-step trainer. Deliberately not
                         ``litefno.train.run_training``: that one is driven by a
                         config file and writes checkpoints and JSONL, which is
                         right for a reproduction run and wrong for a study that
                         trains dozens of small models in a loop and only wants
                         the weights back.
``rollout_mode_error``   H1's dependent variable. Autoregressive rollout error
                         resolved *per Fourier mode* rather than summed over the
                         field, because the hypothesis is about which modes go
                         wrong, and a scalar VRMSE cannot answer that.
``transplant``           H2's intervention. Copies a chosen subset of one
                         model's spectral factors into another and freezes them.

Why the per-mode error is defined the way it is
-----------------------------------------------
The obvious definition -- error in mode k at step t -- grows because the
*signal* in mode k grows or shrinks, not only because the model is wrong, so
comparing modes on it would mostly rank the modes by how much energy they carry.
:func:`rollout_mode_error` normalises each mode by that mode's own truth energy
at the same step, so what is compared across modes is relative error. That is
the per-mode analogue of the repo's VRMSE, which normalises by the variance of
the target for exactly the same reason.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def to_pairs(traj: np.ndarray):
    """(n_traj, T, H, W, C) -> one-step tensors (x, y), channels first."""
    import torch
    x, y = traj[:, :-1], traj[:, 1:]
    n, t, h, w, c = x.shape
    x = x.reshape(n * t, h, w, c).transpose(0, 3, 1, 2)
    y = y.reshape(n * t, h, w, c).transpose(0, 3, 1, 2)
    return (torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)))


def fit(model, train_traj: np.ndarray, valid_traj: Optional[np.ndarray] = None,
        epochs: int = 40, batch: int = 64, lr: float = 1e-3,
        device: str = "cpu", seed: int = 0, log_every: int = 0,
        frozen: Sequence = ()) -> dict:
    """Train ``model`` on one-step prediction. Returns the fit history.

    ``frozen`` is a list of parameters excluded from the optimizer *and* from
    gradient tracking. Excluding them from the optimizer alone is not enough:
    the transplant arms exist to show that a frozen subspace still helps, and a
    parameter left with ``requires_grad`` set is only frozen until someone adds
    a scheduler or a second optimizer, at which point the arm quietly becomes a
    fine-tune and the result inverts.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    frozen_ids = {id(p) for p in frozen}
    for p in frozen:
        p.requires_grad_(False)
    trainable = [p for p in model.parameters() if id(p) not in frozen_ids]
    if not trainable:
        raise ValueError("every parameter is frozen; there is nothing to train")

    model = model.to(device)
    x_tr, y_tr = to_pairs(train_traj)
    opt = torch.optim.Adam(trainable, lr=lr)
    loss_fn = nn.MSELoss()

    history = []
    n = len(x_tr)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = x_tr[idx].to(device), y_tr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        record = {"epoch": epoch + 1, "train_mse": total / n}
        if valid_traj is not None and (log_every and
                                       (epoch + 1) % log_every == 0
                                       or epoch == epochs - 1):
            record["valid_vrmse"] = one_step_vrmse(model, valid_traj, device)
        history.append(record)
    return {"model": model, "history": history,
            "n_train_pairs": int(n),
            "n_trainable": int(sum(p.numel() for p in trainable))}


def one_step_vrmse(model, traj: np.ndarray, device: str = "cpu",
                   batch: int = 128) -> float:
    """VRMSE over a whole split, computed on the concatenated predictions.

    Not a mean of per-batch values: VRMSE normalises by the variance of the
    target, and averaging ratios is not the ratio of the averages.
    """
    import torch
    from litefno.metrics import vrmse

    model.eval()
    x, y = to_pairs(traj)
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            preds.append(model(x[i:i + batch].to(device)).cpu())
    return float(vrmse(torch.cat(preds), y))


# --------------------------------------------------------------------------
# H1's dependent variable
# --------------------------------------------------------------------------


def rollout_mode_error(model, traj: np.ndarray, horizon: int = 16,
                       max_mode: int = 8, device: str = "cpu",
                       batch: int = 16) -> dict:
    """Relative rollout error per spatial Fourier mode, per step.

    Rolls the model autoregressively from step 0 and, at each step, compares the
    predicted and true spectra mode by mode. Returns ``ky``, ``kx``, ``radius``
    and ``error`` with shape (horizon, n_modes), where each entry is

        |pred_k - true_k| / rms(|true_k|)

    with the normaliser taken over trajectories at that step, so a mode holding
    little energy is not automatically the most accurate one.

    ``growth`` is the fitted slope of ``log(error)`` against step, which is the
    quantity H1 says the pole margin predicts: not how wrong a mode is, but how
    fast being wrong compounds.
    """
    import torch

    model.eval()
    n_traj, n_time, height, width, _ = traj.shape
    horizon = min(horizon, n_time - 1)

    ky_full = np.fft.fftfreq(height, d=1.0 / height).astype(int)
    kx_full = np.arange(width // 2 + 1)
    radius_grid = np.hypot(ky_full[:, None], kx_full[None, :])
    keep = radius_grid <= max_mode
    rows, cols = np.nonzero(keep)

    state = torch.from_numpy(
        np.ascontiguousarray(traj[:, 0].transpose(0, 3, 1, 2)))
    preds = []
    with torch.no_grad():
        for i in range(0, n_traj, batch):
            cur = state[i:i + batch].to(device)
            steps = []
            for _ in range(horizon):
                cur = model(cur)
                steps.append(cur.cpu())
            preds.append(torch.stack(steps, dim=1))
    pred = torch.cat(preds).numpy()                    # (n, horizon, C, H, W)
    truth = traj[:, 1:horizon + 1].transpose(0, 1, 4, 2, 3)

    # sum the two channels' spectra in quadrature: the error of a mode of the
    # state, not of one component of it
    pred_spec = np.fft.rfft2(pred, axes=(-2, -1))[..., rows, cols]
    true_spec = np.fft.rfft2(truth, axes=(-2, -1))[..., rows, cols]
    diff = np.sqrt((np.abs(pred_spec - true_spec) ** 2).sum(axis=2))
    scale = np.sqrt((np.abs(true_spec) ** 2).sum(axis=2))
    denom = np.sqrt((scale ** 2).mean(axis=0))         # rms over trajectories
    error = diff.mean(axis=0) / np.maximum(denom, 1e-12)

    steps = np.arange(1, horizon + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_err = np.log(np.maximum(error, 1e-30))
    growth = np.polyfit(steps, log_err, 1)[0]

    return {"ky": ky_full[rows], "kx": kx_full[cols],
            "radius": radius_grid[rows, cols], "error": error,
            "growth": growth, "horizon": horizon}


# --------------------------------------------------------------------------
# H2's intervention
# --------------------------------------------------------------------------


def rank_mode_energy(layer, ky: np.ndarray, kx: np.ndarray) -> np.ndarray:
    """What share of each CP rank component's footprint sits on given modes.

    A CP component ``r`` contributes ``factor_m1[:, r] outer factor_m2[:, r]``
    over the mode grid, so ``|f1[a, r] f2[b, r]|^2`` is its footprint on mode
    (a, b). Returned as (rank, n_modes).

    The normaliser is the component's energy over the *whole* mode grid, not
    over the modes passed in. Normalising within the subset is the obvious
    mistake and it is silent: every subset then sums to one, so a component
    looks equally at home in any set of modes, and
    :func:`partition_rank_components` splits on a comparison that is exactly
    0.5 by construction. That produced a reported split margin of 0.0 and a
    partition carrying no information.
    """
    f1 = layer.factor_m1.detach().cpu().numpy()
    f2 = layer.factor_m2.detach().cpu().numpy()
    grid_ky, grid_kx = _layer_mode_axes(layer)

    # Modes outside the layer's retained block are dropped rather than looked
    # up. The probe walks a disc of radius max_mode while the layer keeps a
    # rectangle of signed wavenumbers, so the two sets are never equal: with
    # modes=10 the grid stops at ky=-4 and a probe reaching ky=-5 has no weight
    # to attribute. Past the truncation the operator does not act on the mode at
    # all, so its correct share is nothing, not an index error.
    ky = np.asarray(ky)
    kx = np.asarray(kx)
    lookup = {int(k): i for i, k in enumerate(grid_ky)}
    keep = np.array([int(a) in lookup and 0 <= int(b) < len(grid_kx)
                     for a, b in zip(ky, kx)], dtype=bool)
    rows = np.array([lookup[int(a)] for a in ky[keep]], dtype=int)
    cols = kx[keep].astype(int)

    full = np.abs(f1[:, None, :] * f2[None, :, :]) ** 2      # (m1, m2, rank)
    total = full.sum(axis=(0, 1))                            # (rank,)
    if rows.size == 0:
        return np.zeros((f1.shape[1], 0))
    energy = (np.abs(f1[rows, :] * f2[cols, :]) ** 2).T      # (rank, n_kept)
    return energy / np.maximum(total[:, None], 1e-30)


def _layer_mode_axes(layer):
    from litefno.operator import mode_grid
    return mode_grid(layer.modes1, layer.modes2)


def select_rank_components(model, keep_ky: np.ndarray, keep_kx: np.ndarray,
                           layer: int = 0, threshold: float = 0.5
                           ) -> np.ndarray:
    """Which rank components belong to a given set of modes.

    A CP factorization does not store one weight per mode, so "transplant the
    resonant modes" cannot be done by copying rows: the rank components are
    shared across the whole mode grid. What is transplantable is the components
    whose footprint sits mostly on the selected modes, and ``threshold`` is how
    much of a component's mode energy has to be there for it to count.

    This is a real limitation of transplanting a factorized operator and is
    stated here rather than hidden: with a low rank the components are not
    cleanly separable by mode, so the selection is approximate, and if it
    selects everything or nothing the transplant arms collapse into the
    fine-tune and from-scratch arms. ``scripts/mode_transplant.py`` reports the
    count it selected for exactly that reason.
    """
    lay = model.spectral_layers[layer]
    share = rank_mode_energy(lay, keep_ky, keep_kx).sum(axis=1)
    return np.flatnonzero(share >= threshold)


def partition_rank_components(model, modes_a, modes_b, layer: int = 0) -> dict:
    """Assign each rank component to whichever of two mode sets it favours.

    :func:`select_rank_components` thresholds each set independently, and on a
    real model that returns everything twice: a CP component's footprint is an
    outer product spread over the whole mode grid, so essentially every
    component has some energy on any half of it. At threshold 0.25 all eight
    components qualified as both resonant and damped and the two transplant arms
    became the same arm -- the guard caught it, but a guard that always fires is
    not a usable selection.

    Comparing the two shares instead always produces a partition: a component
    goes wherever more of its mode energy sits. That is weaker than "this
    component *is* the resonant physics" and is the honest form of the claim a
    factorized operator can support, since the factorization simply does not
    store per-mode weights to move.

    Returns the two index arrays and, as the thing to judge the split by, the
    margin between the shares -- a component at 0.51/0.49 has been assigned, not
    identified.
    """
    lay = model.spectral_layers[layer]
    share_a = rank_mode_energy(lay, modes_a[0], modes_a[1]).sum(axis=1)
    share_b = rank_mode_energy(lay, modes_b[0], modes_b[1]).sum(axis=1)
    total = np.maximum(share_a + share_b, 1e-30)
    frac_a = share_a / total
    return {"a": np.flatnonzero(frac_a >= 0.5),
            "b": np.flatnonzero(frac_a < 0.5),
            "frac_a": frac_a,
            "margin": float(np.mean(np.abs(frac_a - 0.5)) * 2)}


def transplant(target, source, components: Sequence[int]) -> dict:
    """Copy chosen CP rank components from ``source`` into ``target``, frozen.

    Copies the mode-axis factors (``factor_m1``, ``factor_m2``) and the rank
    weights for the selected components in every spectral layer, leaving the
    channel-axis factors to be retrained: the mode structure is the claimed
    shared physics, while how it maps onto channels is regime-specific
    bookkeeping.

    The freezing is per *component*, not per tensor, and that distinction is the
    experiment. H2 claims a transplanted subspace helps while frozen; if the
    copied columns were merely an initialisation that training then moved, the
    arm would be measuring a warm start instead, and a warm start helping is a
    much weaker and much less interesting statement. Since ``requires_grad``
    only exists per tensor, the columns are held by masking their gradient:
    every backward pass zeroes the gradient of the transplanted columns, so
    they cannot move no matter what optimizer or schedule is wrapped around
    them.

    Returns the handles (so a caller can undo the masking) and the count of
    components actually held.
    """
    import torch

    components = np.asarray(sorted(set(int(c) for c in components)), dtype=int)
    handles = []
    if components.size == 0:
        return {"handles": handles, "n_components": 0, "components": components}

    with torch.no_grad():
        for tgt, src in zip(target.spectral_layers, source.spectral_layers):
            tgt.factor_m1[:, components] = src.factor_m1[:, components]
            tgt.factor_m2[:, components] = src.factor_m2[:, components]
            tgt.rank_weights[components] = src.rank_weights[components]

    for tgt in target.spectral_layers:
        for param, axis in ((tgt.factor_m1, 1), (tgt.factor_m2, 1),
                            (tgt.rank_weights, 0)):
            mask = torch.ones_like(param, dtype=torch.bool)
            if axis == 1:
                mask[:, components] = False
            else:
                mask[components] = False
            handles.append(param.register_hook(
                lambda grad, m=mask: grad * m))
    return {"handles": handles, "n_components": int(components.size),
            "components": components}
