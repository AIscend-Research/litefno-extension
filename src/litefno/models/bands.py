r"""Classify Fourier modes, then spend the low-rank budget where it matters.

Board task: "during spectral decomposition, identify which Fourier modes are
'primary' (low-freq, stable) vs. 'resonant' (peaks in power spectrum, typically
linked to feedback) vs. 'damped' (high-freq decay); weight the low-rank
factorization to preserve primary + resonant modes, compress damped modes."

Two halves. :func:`classify_modes` reads a radial power spectrum and partitions
the wavenumber shells into the three classes. :class:`BandedCPSpectralConv2d`
takes that partition and gives each class its own CP rank, so capacity follows
the classification instead of being spread evenly.

Classifying the modes
---------------------
The classes are read off the measured spectrum, not from fixed wavenumber
cutoffs -- a cutoff would just be the low-frequency prior again, the one
ext9/PR #15 refuted for this data. Gray-Scott's spectrum does not decay
monotonically from k=1; three of its six regimes climb by orders of magnitude to
an interior peak at the Turing wavelength.

    resonant   the contiguous band around the dominant peak whose power stays
               within a factor of it. In Gray-Scott this is the
               activator-inhibitor wavelength -- feedback in the literal sense
    primary    the shells below that band: the large, slow scales
    damped     everything above it: the decaying tail

Whether a resonance exists at all is decided by ``rise``, the peak's power over
the largest scale's. On the committed spectra that is maze 2501x, spots 1222x,
worms 12.4x against bubbles 6.7x, gliders 4.6x, spirals 2.0x -- and the default
threshold of 10 splits those two groups exactly where ext10 split the same six
regimes by an unrelated statistic, the share of variance below mode 8. Two
independent measures agreeing on the partition is the reason to trust it.

The result is asserted to be a partition: every shell gets exactly one label.
Overlapping classes would make the rank budget ambiguous.

Weighting the factorization
---------------------------
A CP factorization of the spectral weight shares one set of rank-R components
across every retained mode. Banding it means one CP term per class, each
multiplied by its class's mask:

    W = sum_c  CP_c(rank r_c)  *  mask_c

Setting r_damped below r_primary and r_resonant is the "compress damped modes"
half. Parameter count is ``sum_c r_c * (in + out + m1 + m2)``, so a banded model
with ranks summing to R costs the same as a uniform model of rank R -- which is
what makes the comparison a test of *where* capacity is spent rather than of how
much there is. The experiment script uses that matched-parameter pairing, and a
test pins the equality.

The masks are radial and therefore not separable into an m1 mask times an m2
mask, so they are applied to the reconstructed per-class weight rather than to
the factor matrices. That costs a dense reconstruction per class, and it is the
reason the class exposes ``n_bands`` -- the cost scales with the number of
classes, not with their ranks.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

CLASSES = ("primary", "resonant", "damped")


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def classify_modes(k: Sequence[int], power: Sequence[float],
                   peak_drop: float = 0.5, min_rise: float = 10.0) -> dict:
    """Partition wavenumber shells into primary / resonant / damped.

    The resonant class is the contiguous band around the dominant peak whose
    power stays above ``peak_drop`` times the peak's -- a Turing band has finite
    width, and clipping it to the single argmax shell would leave most of its
    energy classed as something else. Below that band is primary; above it is
    damped.

    Whether there is a resonance at all is decided by ``rise``, the ratio of the
    peak's power to the largest scale's. A spectrum that decays from the first
    shell has rise <= 1 and no feedback peak to preserve; one that climbs by
    orders of magnitude to an interior peak has a scale the dynamics select.

    Two earlier gates were tried and discarded on the real spectra:

    * Excess over a fitted power law, per shell. That classed 26-35 shells as
      resonant, nearly all at k=25-72 where the power is at the noise floor,
      while missing the dominant peak entirely -- a single power law is a poor
      background for a spectrum that rises before it decays, and per-shell
      local-maximum tests are dominated by tail noise. Spending rank there is
      exactly the mistake this function exists to avoid.
    * The peak band's share of total variance. That does not discriminate: the
      band rule adapts to whatever shape the spectrum has, so every Gray-Scott
      regime scores 50-74% and the measure separates nothing.

    ``rise`` does separate them, and at the same place ext10 did, which is the
    reason to trust it. Measured on the committed radial spectra: maze 2501x,
    spots 1222x, worms 12.4x against bubbles 6.7x, gliders 4.6x, spirals 2.0x.
    ext10 independently ranked the same six by the share of variance below mode
    8 -- spots 0.6%, maze 1.3%, worms 31% against bubbles 58%, gliders 69%,
    spirals 77% -- and the default threshold of 10 splits the two groups
    identically. Two unrelated statistics agreeing on the partition is better
    evidence than either alone.
    """
    k = np.asarray(list(k), dtype=int)
    power = np.asarray(list(power), dtype=np.float64)
    if k.size == 0:
        return {c: [] for c in CLASSES} | {
            "peak_k": None, "rise": float("nan"), "has_resonance": False}

    peak_i = int(np.argmax(power))
    first = power[0] if power[0] > 0 else np.nan
    rise = float(power[peak_i] / first) if first and np.isfinite(first) \
        else float("inf")

    # A peak in the first shell is the top of a monotone decay, not a resonance.
    has_resonance = peak_i > 0 and rise >= min_rise
    if has_resonance:
        floor = power[peak_i] * peak_drop
        lo = peak_i
        while lo - 1 >= 0 and power[lo - 1] >= floor:
            lo -= 1
        hi = peak_i
        while hi + 1 < k.size and power[hi + 1] >= floor:
            hi += 1
        resonant, primary, damped = k[lo:hi + 1], k[:lo], k[hi + 1:]
    else:
        resonant, primary, damped = k[:0], k[:peak_i + 1], k[peak_i + 1:]

    assert len(resonant) + len(primary) + len(damped) == k.size, "not a partition"
    return {"primary": primary.tolist(), "resonant": resonant.tolist(),
            "damped": damped.tolist(), "peak_k": int(k[peak_i]),
            "rise": rise, "has_resonance": bool(has_resonance)}


def shell_mask(modes1: int, modes2: int, shells: Sequence[int]) -> torch.Tensor:
    """Mask over the retained rfft2 block selecting integer radial shells.

    Radius is rounded to the nearest integer so the shells tile the plane
    without gaps -- the classification is defined on integer shells, and a mode
    that fell between two of them would belong to no class and silently lose all
    its capacity.
    """
    ky = torch.cat([torch.arange(modes1 // 2 + 1),
                    torch.arange(-(modes1 - modes1 // 2 - 1), 0)]).float()
    kx = torch.arange(modes2).float()
    radius = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).round().long()
    want = torch.tensor(sorted(set(int(s) for s in shells)), dtype=torch.long)
    if want.numel() == 0:
        return torch.zeros_like(radius, dtype=torch.bool)
    return torch.isin(radius, want)


def partition_masks(modes1: int, modes2: int, classes: Mapping[str, Sequence[int]]
                    ) -> dict:
    """Masks for each class, with anything unclaimed folded into 'damped'.

    A mode at a radius no class listed -- beyond the measured spectrum, say --
    would otherwise get zero capacity from every band and be silently zeroed by
    the layer. Folding the remainder into the compressed class is the
    conservative choice: it still gets represented, just cheaply.
    """
    masks = {c: shell_mask(modes1, modes2, classes.get(c, [])) for c in CLASSES}
    claimed = masks["primary"] | masks["resonant"] | masks["damped"]
    masks["damped"] = masks["damped"] | ~claimed
    return masks


# --------------------------------------------------------------------------
# banded factorization
# --------------------------------------------------------------------------


class BandedCPSpectralConv2d(nn.Module):
    """Spectral convolution with a separate CP rank per mode class."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int,
                 modes2: int, ranks: Mapping[str, int],
                 classes: Optional[Mapping[str, Sequence[int]]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.ranks = {c: int(ranks.get(c, 0)) for c in CLASSES}

        classes = classes or {"primary": [], "resonant": [], "damped": []}
        masks = partition_masks(modes1, modes2, classes)
        for c in CLASSES:
            self.register_buffer(f"mask_{c}", masks[c])

        scale = 1.0 / (in_channels * out_channels) ** 0.5
        for c in CLASSES:
            rank = self.ranks[c]
            if rank <= 0:
                continue
            self.register_parameter(
                f"{c}_in", nn.Parameter(
                    torch.randn(in_channels, rank, dtype=torch.cfloat) * scale))
            self.register_parameter(
                f"{c}_out", nn.Parameter(
                    torch.randn(out_channels, rank, dtype=torch.cfloat) * scale))
            self.register_parameter(
                f"{c}_m1", nn.Parameter(
                    torch.randn(modes1, rank, dtype=torch.cfloat) * scale))
            self.register_parameter(
                f"{c}_m2", nn.Parameter(
                    torch.randn(modes2, rank, dtype=torch.cfloat) * scale))
            self.register_parameter(
                f"{c}_w", nn.Parameter(torch.ones(rank, dtype=torch.cfloat)))

    @property
    def n_bands(self) -> int:
        return sum(1 for c in CLASSES if self.ranks[c] > 0)

    def modes_per_class(self) -> dict:
        return {c: int(getattr(self, f"mask_{c}").sum()) for c in CLASSES}

    def weight(self) -> torch.Tensor:
        """Reconstruct (in, out, m1, m2), each class confined to its own mask."""
        total = None
        for c in CLASSES:
            if self.ranks[c] <= 0:
                continue
            band = torch.einsum(
                "r,ir,or,ar,br->ioab", getattr(self, f"{c}_w"),
                getattr(self, f"{c}_in"), getattr(self, f"{c}_out"),
                getattr(self, f"{c}_m1"), getattr(self, f"{c}_m2"))
            band = band * getattr(self, f"mask_{c}")
            total = band if total is None else total + band
        if total is None:
            return torch.zeros(self.in_channels, self.out_channels,
                               self.modes1, self.modes2, dtype=torch.cfloat)
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)

        m1, m2 = self.modes1, self.modes2
        pos = m1 // 2 + 1
        neg = m1 - pos
        idx = torch.cat([torch.arange(pos, device=x.device),
                         torch.arange(height - neg, height, device=x.device)])
        block = x_ft[:, :, idx][:, :, :, :m2]

        out_block = torch.einsum("bimn,iomn->bomn", block, self.weight())
        out_ft = torch.zeros(batch, self.out_channels, height, width // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, idx[:, None], torch.arange(m2, device=x.device)[None, :]] = \
            out_block
        return torch.fft.irfft2(out_ft, s=(height, width))


class BandedLiteFNO(nn.Module):
    """LiteFNO whose CP rank is allocated by mode class.

    ``ranks`` maps class name to rank. A uniform baseline at matched parameter
    count is ``{"primary": R, "resonant": 0, "damped": 0}`` with every shell
    labelled primary, which is what ``uniform_ranks`` builds.
    """

    def __init__(self, in_channels: int, out_channels: int, width: int = 32,
                 modes: int = 16, layers: int = 4,
                 ranks: Optional[Mapping[str, int]] = None,
                 classes: Optional[Mapping[str, Sequence[int]]] = None):
        super().__init__()
        ranks = ranks or {"primary": 6, "resonant": 6, "damped": 2}
        self.ranks = dict(ranks)
        self.classes = {c: list(classes.get(c, [])) for c in CLASSES} \
            if classes else {c: [] for c in CLASSES}
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            BandedCPSpectralConv2d(width, width, modes, modes, ranks, classes)
            for _ in range(layers)])
        self.skips = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(layers)])
        self.output_proj = nn.Conv2d(width, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def modes_per_class(self) -> dict:
        return self.spectral_layers[0].modes_per_class()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for spectral, skip in zip(self.spectral_layers, self.skips):
            x = self.act(spectral(x) + skip(x))
        return self.output_proj(x)


def uniform_ranks(total_rank: int, max_shell: int) -> tuple[dict, dict]:
    """Ranks and classes for the matched-parameter uniform control.

    Everything is labelled primary and given the whole budget, so the control
    has the same parameter count and the same code path as the banded model and
    differs only in how capacity is distributed.
    """
    return ({"primary": total_rank, "resonant": 0, "damped": 0},
            {"primary": list(range(0, max_shell + 1)), "resonant": [],
             "damped": []})
