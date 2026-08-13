r"""The auxiliary network: a reconstructed state in, a fair allocation out.

Board task: "train an auxiliary network that, given the reconstructed ecosystem
state, predicts a fair resource allocation across populations/regions."

The fairness mathematics is :mod:`litefno.allocation`; this is the network and
its training loop. The two are separate because the interesting comparison in
``scripts/fair_allocation.py`` is between them -- the closed-form rule applied
to pooled gains versus a network asked to produce the same thing from the raw
field -- and a shared implementation would make that comparison vacuous.

Trained on the decision, not on a label
---------------------------------------
The obvious way to train this is supervised regression onto the closed-form
allocation. That is a worse experiment and a worse method. It caps the network
at imitating the rule, cannot express any hedge against its input being wrong,
and its loss (allocation MSE) is not the quantity anyone cares about, which is
realised welfare.

So the loss *is* the welfare objective:

    L(state) = -log CE_alpha( g_true  x  allocate(state) )

with the gains from the *true* state while the network sees a reconstructed
one. This is decision-focused learning in the sense of Elmachtoub & Grigas
(2022) and Wilder, Dilkina & Tambe (2019): the network is scored on the
decision's downstream value, not on the accuracy of an intermediate estimate.
It also makes the robustness arm expressible -- a network trained on the
surrogate's own predictions can learn to distrust them, which a regression onto
the rule's output structurally cannot.

Why the simplex constraint is architectural
-------------------------------------------
The allocation must be non-negative and sum to the budget. Enforcing that with a
penalty term would make "did it satisfy the constraint" a result rather than a
guarantee, and every arm's welfare would then be reported on allocations that
overspend by an unknown margin. A softmax over regions satisfies it exactly, at
every point in training, including at initialisation.

The cost of that choice is stated rather than hidden: a softmax cannot reach a
vertex of the simplex, so at alpha = 0 -- where the optimum is to give the whole
budget to one region -- the network is structurally unable to match the rule.
That is a real limit and it shows up in the alpha = 0 row of the results.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


def alpha_fair_loss(allocation: torch.Tensor, gains: torch.Tensor,
                    alpha: float = 1.0) -> torch.Tensor:
    """``-log CE_alpha(g * a)``, averaged over the batch.

    The log of :func:`litefno.allocation.welfare_ce`, which is the same
    maximiser and a far better conditioned objective: CE at alpha = 8 varies
    over orders of magnitude across a batch, and its gradient is dominated by
    whichever sample happens to have the smallest outcome.

    ``alpha = inf`` (exact max-min) is refused rather than approximated
    silently. Its objective is ``min_r``, whose gradient reaches one region per
    step; the family is continuous in alpha, so a large finite alpha is the
    honest way to sit near that limit, and the caller should choose it and say
    so.
    """
    if not np.isfinite(alpha):
        raise ValueError(
            "alpha = inf has a non-differentiable min objective; train at a "
            "large finite alpha and report it as an approximation to max-min")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    logx = torch.log(torch.clamp(gains * allocation, min=1e-12))
    if alpha == 1.0:
        return -logx.mean(dim=-1).mean()
    scaled = (1.0 - alpha) * logx
    log_ce = (torch.logsumexp(scaled, dim=-1)
              - np.log(logx.shape[-1])) / (1.0 - alpha)
    return -log_ce.mean()


class RegionAllocator(nn.Module):
    """Convolutional encoder over the field, softmax allocation over regions.

    Input (batch, channels, H, W); output (batch, blocks*blocks) non-negative
    and summing to ``budget``, in the same row-major region order as
    :func:`litefno.allocation.region_gains`.

    The encoder is deliberately small -- two convolutions and a pooled head,
    a few thousand parameters -- for the same reason the rest of this repo is:
    a decision layer that costs more than the surrogate it sits on is not a
    lightweight deployment story. It is also enough. The information the rule
    needs is a per-region average, which one pooling layer can express, so a
    larger network would only buy a better fit to the part of the problem that
    is not the question.

    Adaptive average pooling to the region grid, rather than a fixed stride, so
    the model is independent of the input resolution -- the same allocator runs
    on a 32x32 field and on a 64x64 one, which matters because the surrogate is
    a neural operator and resolution independence is the property it is sold on.
    """

    def __init__(self, in_channels: int = 2, blocks: int = 4, width: int = 16,
                 budget: float = 1.0, temperature: float = 1.0):
        super().__init__()
        self.blocks = blocks
        self.budget = budget
        self.temperature = temperature
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(blocks)
        self.head = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, 1, kernel_size=1),
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(self.pool(self.encoder(x)))
        return h.flatten(1) / self.temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.budget * torch.softmax(self.logits(x), dim=-1)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def fit_allocator(model: RegionAllocator, states: np.ndarray,
                  gains: np.ndarray, alpha: float = 1.0, epochs: int = 60,
                  batch: int = 64, lr: float = 3e-3, device: str = "cpu",
                  seed: int = 0, valid: Optional[tuple] = None) -> dict:
    """Train the allocator on realised welfare. Returns the fit history.

    ``states`` is (N, C, H, W) -- what the network sees, which may be a
    surrogate's reconstruction -- and ``gains`` is (N, R) read from the *true*
    state, which is what the welfare is realised against. Keeping those two
    arguments separate is the whole design: passing the same field to both is
    the clean-input arm, and passing a reconstruction as ``states`` with true
    gains is the arm that measures error propagation.

    ``valid`` is an optional ``(states, gains)`` pair, scored each epoch with
    the same loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    x = torch.as_tensor(np.ascontiguousarray(states), dtype=torch.float32)
    g = torch.as_tensor(np.ascontiguousarray(gains), dtype=torch.float32)
    if len(x) != len(g):
        raise ValueError(f"{len(x)} states against {len(g)} gain vectors")
    if g.shape[-1] != model.blocks ** 2:
        raise ValueError(
            f"gains have {g.shape[-1]} regions, allocator has "
            f"{model.blocks ** 2}; the partitions do not match")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    n = len(x)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, gb = x[idx].to(device), g[idx].to(device)
            opt.zero_grad()
            loss = alpha_fair_loss(model(xb), gb, alpha)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        record = {"epoch": epoch + 1, "train_loss": total / n}
        if valid is not None:
            record["valid_loss"] = evaluate_loss(model, valid[0], valid[1],
                                                 alpha, device)
        history.append(record)
    return {"model": model, "history": history,
            "n_parameters": model.n_parameters()}


@torch.no_grad()
def allocate(model: RegionAllocator, states: np.ndarray, device: str = "cpu",
             batch: int = 256) -> np.ndarray:
    """Run the allocator over a stack of states. Returns (N, R) as numpy."""
    model.eval()
    x = torch.as_tensor(np.ascontiguousarray(states), dtype=torch.float32)
    out = [model(x[i:i + batch].to(device)).cpu().numpy()
           for i in range(0, len(x), batch)]
    return np.concatenate(out) if out else np.zeros((0, model.blocks ** 2))


@torch.no_grad()
def evaluate_loss(model: RegionAllocator, states: np.ndarray,
                  gains: np.ndarray, alpha: float = 1.0, device: str = "cpu",
                  batch: int = 256) -> float:
    """The training objective on held-out data, without the optimizer."""
    model.eval()
    x = torch.as_tensor(np.ascontiguousarray(states), dtype=torch.float32)
    g = torch.as_tensor(np.ascontiguousarray(gains), dtype=torch.float32)
    total = 0.0
    for i in range(0, len(x), batch):
        xb, gb = x[i:i + batch].to(device), g[i:i + batch].to(device)
        total += float(alpha_fair_loss(model(xb), gb, alpha)) * len(xb)
    return total / max(len(x), 1)
