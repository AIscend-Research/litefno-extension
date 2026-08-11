r"""CP-factorized spectral convolution with a harmonic-mode bias.

Board task: "Add harmonic conditioning: modify LITEFNO's spectral factorization
to add a harmonic-mode bias."

Two things happen here, and they are separable on purpose.

**The spectral factorization.** ``CPSpectralConv2d`` is a genuine spectral
convolution -- ``rfft2``, complex weights, mode truncation -- whose weight
tensor is stored in CP form: a rank vector plus one factor matrix per tensor
mode, contracted on the fly. This is the architecture
``docs/reproducibility_findings.md`` records as missing from
``models/litefno.py``, which is a low-rank CNN with no FFT in it. Until now the
only CP-factorized spectral model in this project came from ``neuralop``, inside
a notebook.

**The harmonic-mode bias.** On top of that, designated harmonic modes get their
own learnable complex bias, added to the spectral output:

    out[k] = (W_k . x[k]) + b_k        for k in H
    out[k] =  W_k . x[k]               otherwise

where H is a set of radial shells -- a fundamental wavenumber and its integer
multiples. The bias is input-independent, so in space it is a fixed learnable
pattern confined to those shells: cheap capacity for a persistent harmonic
structure the operator would otherwise have to synthesise from the input every
step.

Why those shells, and what the repo's own measurements say
----------------------------------------------------------
This is not a free parameter. ext10 measured where each Gray-Scott regime keeps
its spatial variance and found maze and spots hold ~1% of theirs below mode 8,
against spirals' 77% -- their energy sits in a narrow band at the Turing
wavelength near mode 13-16 on the native 128x128 grid, which is mode 3-4 after
the 4x downsampling the training pipeline applies. So the shells worth
conditioning on are not a guess; they are the ones the data concentrates in.

The measurements also predict this will be a small effect, and the honest thing
is to say so before running it rather than after. ext9/PR #15 showed Gray-Scott's
variance is not low-wavenumber dominated, which killed the spatial harmonic
prior in its original form. ext12 found that on planetswe -- a system with
*documented, exactly periodic* forcing, the best case available -- the forcing
accounts for 5.4% of temporal variance globally and 11.5% in the most favourable
band. A prior keyed to harmonic structure can only reach variance that sits at
those harmonics.

What this design does differently is not assume the harmonics are low-frequency.
The Turing band is mid-spectrum, and it is where two of the six regimes keep
essentially all of their energy. That is a narrower and better-targeted claim
than the one PR #15 refuted, and it is why the flag exists rather than the
behaviour being baked in: ``harmonic_bias=False`` reproduces the plain
CP-factorized model exactly, so the two can be run as a controlled A/B with
nothing else changed. A test pins that exactness.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
from torch import nn


def harmonic_shells(fundamental: float, n_harmonics: int, max_mode: int
                    ) -> list[float]:
    """Radial wavenumbers of a fundamental and its integer multiples.

    Multiples beyond ``max_mode`` are dropped rather than wrapped: past the
    truncation there is no mode to bias, and aliasing them back onto low
    wavenumbers would put the bias somewhere it was never asked to go.
    """
    if fundamental <= 0:
        raise ValueError(f"fundamental must be positive, got {fundamental}")
    shells = [fundamental * n for n in range(1, n_harmonics + 1)]
    return [s for s in shells if s <= max_mode]


def harmonic_mask(modes1: int, modes2: int, shells: Sequence[float],
                  width: float = 0.5) -> torch.Tensor:
    """Boolean mask over the retained rfft2 grid selecting the harmonic shells.

    ``modes1`` indexes signed vertical wavenumbers stored as [0..m-1] for the
    positive half and the negative half folded to the end, matching how the
    layer slices the spectrum; ``modes2`` indexes non-negative horizontal
    wavenumbers, as rfft2 produces.

    A shell is a radial annulus of half-width ``width`` around |k|, so a shell
    at 3.0 catches the modes whose radius rounds to 3. Selecting on radius
    rather than on a single (kx, ky) pair matters because a Turing pattern picks
    a wavelength, not an orientation -- the energy is spread around the ring.
    """
    ky = torch.cat([torch.arange(modes1 // 2 + 1),
                    torch.arange(-(modes1 - modes1 // 2 - 1), 0)]).float()
    kx = torch.arange(modes2).float()
    radius = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    mask = torch.zeros_like(radius, dtype=torch.bool)
    for shell in shells:
        mask |= (radius - shell).abs() <= width
    return mask


class CPSpectralConv2d(nn.Module):
    """Spectral convolution with CP-factorized complex weights.

    The dense weight is (in_channels, out_channels, modes1, modes2) complex.
    CP stores it as ``rank`` weights plus one factor matrix per tensor mode and
    contracts them, so the parameter count grows with the sum of the dimensions
    rather than their product.

    ``harmonic_bias`` adds a learnable complex bias on the selected shells only.

    It is one complex value per selected mode, broadcast over output channels,
    and stored packed rather than as a full grid.

    Both of those choices are forced by the same constraint: the bias must not
    be a capacity increase in disguise, or an improvement from harmonic
    conditioning becomes indistinguishable from an improvement from being a
    bigger model. Measured on a width-32, modes-12, 4-layer model, a full-grid
    per-channel bias costs 255% of the base parameter count and a packed
    per-channel one still costs 62% -- because CP weights are themselves small,
    so anything scaling with out_channels x modes dwarfs them. Per-mode and
    packed costs about 4%. A test holds the ratio under 10%.

    The mask is kept as a buffer so the selection stays inspectable.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int,
                 modes2: int, rank: int = 8, harmonic_bias: bool = False,
                 shells: Optional[Iterable[float]] = None,
                 shell_width: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.rank = rank
        self.use_harmonic_bias = harmonic_bias

        scale = 1.0 / (in_channels * out_channels) ** 0.5
        # CP factors, one per tensor mode; complex so the factorization is of
        # the spectral weight itself rather than of some real surrogate
        self.factor_in = nn.Parameter(
            torch.randn(in_channels, rank, dtype=torch.cfloat) * scale)
        self.factor_out = nn.Parameter(
            torch.randn(out_channels, rank, dtype=torch.cfloat) * scale)
        self.factor_m1 = nn.Parameter(
            torch.randn(modes1, rank, dtype=torch.cfloat) * scale)
        self.factor_m2 = nn.Parameter(
            torch.randn(modes2, rank, dtype=torch.cfloat) * scale)
        self.rank_weights = nn.Parameter(torch.ones(rank, dtype=torch.cfloat))

        if harmonic_bias:
            shells = list(shells) if shells is not None else []
            mask = harmonic_mask(modes1, modes2, shells, shell_width)
            self.register_buffer("bias_mask", mask)
            self.register_buffer("bias_index", mask.nonzero(as_tuple=False))
            self.harmonic_bias = nn.Parameter(
                torch.zeros(int(mask.sum()), dtype=torch.cfloat))
            self.shells = shells
        else:
            self.register_buffer("bias_mask", None)
            self.register_buffer("bias_index", None)
            self.shells = []

    def weight(self) -> torch.Tensor:
        """Reconstruct the dense (in, out, m1, m2) weight from the CP factors."""
        return torch.einsum(
            "r,ir,or,ar,br->ioab", self.rank_weights, self.factor_in,
            self.factor_out, self.factor_m1, self.factor_m2)

    def n_harmonic_modes(self) -> int:
        if not self.use_harmonic_bias or self.bias_mask is None:
            return 0
        return int(self.bias_mask.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)

        m1, m2 = self.modes1, self.modes2
        pos = m1 // 2 + 1
        neg = m1 - pos
        # gather the retained block: low positive vertical modes and, folded
        # after them, the matching negative ones
        idx = torch.cat([torch.arange(pos, device=x.device),
                         torch.arange(height - neg, height, device=x.device)])
        block = x_ft[:, :, idx][:, :, :, :m2]

        out_block = torch.einsum("bimn,iomn->bomn", block, self.weight())
        if self.use_harmonic_bias and self.harmonic_bias.numel():
            # per-mode, broadcast over channels: shape (1, 1, m1, m2)
            bias_grid = out_block.new_zeros(1, 1, m1, m2)
            bias_grid[0, 0, self.bias_index[:, 0], self.bias_index[:, 1]] = \
                self.harmonic_bias
            out_block = out_block + bias_grid

        out_ft = torch.zeros(batch, self.out_channels, height, width // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, idx[:, None], torch.arange(m2, device=x.device)[None, :]] = \
            out_block
        return torch.fft.irfft2(out_ft, s=(height, width))


class HarmonicLiteFNO(nn.Module):
    """LiteFNO with CP-factorized spectral layers and optional harmonic bias.

    Architecture follows the repo's FNO-S: lift, then per layer a spectral
    convolution plus a pointwise skip, then project. ``harmonic_bias=False``
    gives the plain CP-factorized spectral model, which is the control arm.
    """

    def __init__(self, in_channels: int, out_channels: int, width: int = 64,
                 modes: int = 16, layers: int = 8, rank: int = 8,
                 harmonic_bias: bool = False,
                 fundamental: float = 4.0, n_harmonics: int = 3,
                 shell_width: float = 0.5):
        super().__init__()
        shells = harmonic_shells(fundamental, n_harmonics, modes) \
            if harmonic_bias else []
        self.shells = shells
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            CPSpectralConv2d(width, width, modes, modes, rank=rank,
                             harmonic_bias=harmonic_bias, shells=shells,
                             shell_width=shell_width)
            for _ in range(layers)])
        self.skips = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(layers)])
        self.output_proj = nn.Conv2d(width, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def n_harmonic_modes(self) -> int:
        return sum(l.n_harmonic_modes() for l in self.spectral_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for spectral, skip in zip(self.spectral_layers, self.skips):
            x = self.act(spectral(x) + skip(x))
        return self.output_proj(x)
