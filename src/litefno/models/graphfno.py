r"""A graph-convolutional head over LiteFNO's field features.

Board task: "add a graph-convolutional layer on top of LiteFNO to capture
network effects in resource flow, not just local diffusion."

The network mathematics is :mod:`litefno.networks`; this is the layer and its
training loop.

The architecture, and why the arms differ in exactly one matrix
---------------------------------------------------------------
::

    field (B, C, H, W)
      -> LiteFNO trunk           local physics, unchanged from the repo
      -> adaptive average pool   to the blocks x blocks region grid
      -> K graph convolutions    propagation on the trade network
      -> per-region scalar       predicted scarcity at t + horizon

Every arm of the experiment instantiates *this* module. They differ only in the
propagator handed to :class:`GraphConv`: the true trade network, a
degree-preserving rewiring of it, the spatial lattice, or the identity. Same
trunk, same widths, same parameter count, same optimizer, same seed. That is
deliberate and it is the only way the comparison means anything -- a "no graph"
arm built by deleting the graph layers would have fewer parameters, and a loss
against the graph arm would be a capacity result wearing a topology result's
clothes.

The identity arm deserves a note, because it looks like a trick and is not. Set
the propagator to ``I`` and the graph convolution becomes a per-region MLP:
every polynomial power collapses to ``I``, so the ``K+1`` weight matrices sum
into one effective matrix. The parameters are all still there and still trained;
what is gone is any exchange of information between regions. It is the cleanest
possible ablation of *network effects* while holding capacity fixed, and its
weights are not wasted -- they fit the local map as well as the same-shaped
graph arm's do.

Why a polynomial filter rather than one Kipf-Welling propagation
-----------------------------------------------------------------
``sum_k Ahat^k H W_k`` (Defferrard et al. 2016, in the monomial rather than
Chebyshev basis) keeps the ``k = 0`` term, so a graph layer can always fall back
to *not* propagating if propagation does not help. A single ``Ahat H W`` cannot:
it forces every region's output through its neighbours' features and would make
a wrong graph arbitrarily damaging, which would flatter the true-graph arm for
the wrong reason. With the ``k = 0`` term present, the wrong-graph control has
the option of ignoring the graph and paying only what the extra terms cost in
optimisation -- so if it still loses, it loses on topology.

The monomial basis is numerically worse than Chebyshev at large ``K``; at the
``K = 2`` this experiment uses on a 16-node graph with a spectrally normalised
propagator, the difference is nil, and the monomials are what makes the identity
collapse above exact rather than approximate.

What this module does not claim
-------------------------------
It does not claim a graph layer improves PDE surrogacy. The trunk's output is
pooled to 16 numbers before the graph ever sees it; nothing here changes the
field prediction, and the repo's headline reproduction result stands untouched.
The claim under test is narrower: that a *region-level* quantity which
propagates on a non-lattice network is predicted better when the network is
supplied than when it is not, and that the gain comes from the non-lattice
edges. See ``docs/network_scarcity.md``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


class GraphConv(nn.Module):
    """``sum_{k=0..K} Ahat^k H W_k``: a polynomial spectral filter on a graph.

    Input and output are (batch, n_nodes, channels). The propagator powers are
    precomputed as one stacked buffer, which is both faster (a single einsum)
    and inspectable -- ``self.powers[1]`` is the normalised adjacency the arm
    was actually given, so a test can check an arm is running on the graph it
    was named after.

    ``propagator=None`` means the identity: no message passing, same parameters.
    """

    def __init__(self, in_channels: int, out_channels: int, n_nodes: int,
                 propagator: Optional[np.ndarray] = None, order: int = 2,
                 bias: bool = True):
        super().__init__()
        if order < 0:
            raise ValueError(f"filter order must be >= 0, got {order}")
        if propagator is None:
            prop = np.eye(n_nodes)
        else:
            prop = np.asarray(propagator, dtype=float)
            if prop.shape != (n_nodes, n_nodes):
                raise ValueError(
                    f"propagator is {prop.shape}, expected "
                    f"({n_nodes}, {n_nodes})")
        powers = [np.eye(n_nodes)]
        for _ in range(order):
            powers.append(powers[-1] @ prop)
        self.register_buffer("powers",
                             torch.tensor(np.stack(powers), dtype=torch.float32))
        self.order = order
        self.weight = nn.Parameter(
            torch.empty(order + 1, in_channels, out_channels))
        nn.init.normal_(self.weight, std=(in_channels * (order + 1)) ** -0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # (K+1, N, N) x (B, N, C) -> (B, K+1, N, C), then contract k and C
        spread = torch.einsum("knm,bmc->bknc", self.powers, h)
        out = torch.einsum("bknc,kcd->bnd", spread, self.weight)
        return out + self.bias if self.bias is not None else out


class GraphLiteFNO(nn.Module):
    """LiteFNO trunk, region pooling, graph convolutions, one scalar per region.

    ``trunk`` is any field-to-field module; the experiment passes the repo's own
    :class:`~litefno.models.litefno.LiteFNO` or
    :class:`~litefno.models.harmonic.HarmonicLiteFNO` so that what sits under
    the graph layer is the model this repository is about, not a stand-in.

    Pooling is ``AdaptiveAvgPool2d(blocks)``, matching how
    :func:`litefno.allocation.region_gains` forms regions -- a block mean over
    the same square partition -- so the node features are the trunk's reading of
    exactly the regions the target is defined on. Adaptive rather than
    fixed-stride for the same reason as in
    :class:`~litefno.models.allocator.RegionAllocator`: the surrogate is a
    neural operator and resolution independence is the property it is sold on.

    The head is linear and unconstrained even though the target is a share in
    [0, 1]. Squashing it through a sigmoid is the tempting alternative and is a
    trap here: the scarcity levels this experiment runs at sit near 0.06 with a
    spread of 0.04, so the correct logits are far out on the sigmoid's flat
    tail, and every arm trains into the saturation and predicts a constant --
    worse than the mean, identically for all four arms, which would read as "the
    graph does not help" when what happened is that nothing learned anything.
    The bound is left to be learned, and :func:`bound_violation` reports how far
    outside it the fitted models actually go so the omission is checkable rather
    than assumed harmless.
    """

    def __init__(self, trunk: nn.Module, trunk_channels: int, blocks: int = 4,
                 hidden: int = 32, propagator: Optional[np.ndarray] = None,
                 order: int = 2, layers: int = 2):
        super().__init__()
        if layers < 1:
            raise ValueError(f"need at least one graph layer, got {layers}")
        n_nodes = blocks * blocks
        self.trunk = trunk
        self.blocks = blocks
        self.pool = nn.AdaptiveAvgPool2d(blocks)
        dims = [trunk_channels] + [hidden] * layers
        self.graph_layers = nn.ModuleList([
            GraphConv(dims[i], dims[i + 1], n_nodes, propagator=propagator,
                      order=order)
            for i in range(layers)])
        self.head = nn.Linear(hidden, 1)
        self.act = nn.GELU()

    def node_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.trunk(x))                       # (B, C, b, b)
        return h.flatten(2).transpose(1, 2)                # (B, N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.node_features(x)
        for layer in self.graph_layers:
            h = self.act(layer(h))
        return self.head(h).squeeze(-1)                    # (B, N)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def paint_regions(values: np.ndarray, size: int, blocks: int) -> np.ndarray:
    """Broadcast a per-region vector back onto the pixel grid as a channel.

    (N, R) -> (N, 1, size, size), constant on each block. The current scarcity
    state has to reach the model somehow, and painting it on the field is the
    representation that gives *every* arm the same information: the field-only
    arm can read it with a convolution, and the graph arms get it through the
    same pooling as the rest of the trunk's features. Handing the graph arms a
    separate node-level input and the field arm a painted one would confound the
    comparison with an encoding difference.
    """
    v = np.asarray(values, dtype=np.float32)
    if v.shape[-1] != blocks * blocks:
        raise ValueError(
            f"{v.shape[-1]} values do not fill a {blocks}x{blocks} partition")
    if size % blocks:
        raise ValueError(f"{size} does not divide into {blocks} blocks")
    grid = v.reshape(-1, blocks, blocks)
    grid = np.repeat(np.repeat(grid, size // blocks, axis=1),
                     size // blocks, axis=2)
    return grid[:, None].astype(np.float32)


def fit_graph_model(model: GraphLiteFNO, states: np.ndarray,
                    targets: np.ndarray, epochs: int = 40, batch: int = 64,
                    lr: float = 3e-3, device: str = "cpu", seed: int = 0,
                    valid: Optional[tuple] = None) -> dict:
    """Train on region-level MSE. Returns the fit history and the best state.

    ``valid`` is an optional ``(states, targets)`` pair, scored every epoch. The
    parameters at the best validation epoch are restored at the end, because the
    arms are compared on held-out error and letting each arm run to its own last
    epoch would compare them at different points on their own curves -- the
    kind of difference that reads as a topology effect and is a training-length
    effect.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    x = torch.as_tensor(np.ascontiguousarray(states), dtype=torch.float32)
    y = torch.as_tensor(np.ascontiguousarray(targets), dtype=torch.float32)
    if len(x) != len(y):
        raise ValueError(f"{len(x)} states against {len(y)} targets")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    best = {"valid_mse": np.inf, "state": None, "epoch": -1}
    n = len(x)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx].to(device), y[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        record = {"epoch": epoch + 1, "train_mse": total / n}
        if valid is not None:
            record["valid_mse"] = predict_mse(model, valid[0], valid[1], device)
            if record["valid_mse"] < best["valid_mse"]:
                best = {"valid_mse": record["valid_mse"],
                        "state": {k: v.detach().clone()
                                  for k, v in model.state_dict().items()},
                        "epoch": epoch + 1}
        history.append(record)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"model": model, "history": history, "best_epoch": best["epoch"],
            "n_parameters": model.n_parameters()}


@torch.no_grad()
def predict(model: GraphLiteFNO, states: np.ndarray, device: str = "cpu",
            batch: int = 256) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(np.ascontiguousarray(states), dtype=torch.float32)
    out = [model(x[i:i + batch].to(device)).cpu().numpy()
           for i in range(0, len(x), batch)]
    return np.concatenate(out) if out else np.zeros((0, model.blocks ** 2))


def predict_mse(model: GraphLiteFNO, states: np.ndarray, targets: np.ndarray,
                device: str = "cpu") -> float:
    pred = predict(model, states, device)
    return float(np.mean((pred - np.asarray(targets, dtype=float)) ** 2))


def bound_violation(pred: np.ndarray) -> float:
    """Largest excursion of a prediction outside the target's [0, 1] range.

    The head is unconstrained, so this is what that costs. Reported next to the
    errors: a model whose predictions leave the feasible set by a hair has an
    encoding inefficiency, and one that leaves it by a wide margin is not
    modelling a share at all.
    """
    p = np.asarray(pred, dtype=float)
    return float(max(0.0, np.max(np.maximum(-p, p - 1.0))))


def region_vrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """RMSE normalised by the target's variance, the repo's headline metric.

    Reported instead of raw MSE for the reason the reproduction reports it: the
    scarcity target's scale depends on beta, gamma and the ecosystem, so an MSE
    is uninterpretable across settings while a VRMSE of 1 always means "no
    better than predicting the mean".
    """
    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    return float(np.sqrt(np.mean((p - t) ** 2) / (np.var(t) + 1e-12)))
