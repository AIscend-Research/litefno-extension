"""Validation for litefno.models.harmonic.

Two claims have to hold for the A/B this module exists to enable.

The layer must be a genuine spectral convolution with a CP-factorized weight --
the repo already shipped a model named LiteFNO that was a CNN, and
docs/reproducibility_findings.md records the confusion that caused.

And the harmonic bias must touch only the shells it was given. If it leaked into
other modes it would be a general-purpose extra parameter, and any improvement
would say nothing about harmonic conditioning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from litefno.models.harmonic import (  # noqa: E402
    CPSpectralConv2d, HarmonicLiteFNO, harmonic_mask, harmonic_shells)


# --------------------------------------------------------------------------
# shells and mask
# --------------------------------------------------------------------------


def test_shells_are_integer_multiples_of_the_fundamental():
    assert harmonic_shells(4.0, 3, 32) == [4.0, 8.0, 12.0]
    assert harmonic_shells(3.0, 5, 32) == [3.0, 6.0, 9.0, 12.0, 15.0]


def test_shells_beyond_truncation_are_dropped_not_wrapped():
    """Aliasing a harmonic back onto a low mode would bias the wrong place."""
    assert harmonic_shells(6.0, 4, 16) == [6.0, 12.0]
    assert harmonic_shells(20.0, 3, 16) == []


def test_shells_reject_a_nonpositive_fundamental():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            harmonic_shells(bad, 2, 16)


def test_mask_selects_a_radial_annulus_not_a_single_mode():
    """A Turing pattern picks a wavelength, not an orientation."""
    mask = harmonic_mask(16, 16, [4.0], width=0.5)
    assert mask[4, 0] and mask[0, 4]              # both axes
    assert mask[3, 3]                             # radius 4.24 -> within 0.5
    assert not mask[0, 6]
    assert mask.sum() > 4                         # a ring, not four points


def test_mask_is_empty_for_no_shells():
    assert harmonic_mask(8, 8, []).sum() == 0


def test_mask_covers_negative_vertical_wavenumbers():
    """The retained block folds negative ky to the end; the ring must too."""
    mask = harmonic_mask(16, 16, [3.0], width=0.5)
    assert mask[3, 0]                             # +3
    assert mask[-3, 0]                            # -3, folded


# --------------------------------------------------------------------------
# the layer is spectral, and CP
# --------------------------------------------------------------------------


def test_layer_uses_an_fft():
    calls = []
    real = torch.fft.rfft2

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    torch.fft.rfft2 = spy
    try:
        CPSpectralConv2d(2, 2, 4, 4, rank=2)(torch.randn(1, 2, 16, 16))
    finally:
        torch.fft.rfft2 = real
    assert calls, "no FFT in a spectral convolution"


def test_weights_are_complex():
    layer = CPSpectralConv2d(2, 3, 4, 4, rank=2)
    complex_params = [n for n, p in layer.named_parameters() if p.is_complex()]
    assert {"factor_in", "factor_out", "factor_m1", "factor_m2",
            "rank_weights"} <= set(complex_params)


def test_cp_reconstructs_a_known_rank_one_tensor():
    """The contraction must be the CP decomposition, not something adjacent."""
    layer = CPSpectralConv2d(2, 3, 4, 5, rank=1)
    with torch.no_grad():
        layer.rank_weights.fill_(1.0)
        a = (torch.arange(2).reshape(2, 1) + 1).to(torch.cfloat)
        b = (torch.arange(3).reshape(3, 1) + 1).to(torch.cfloat)
        c = (torch.arange(4).reshape(4, 1) + 1).to(torch.cfloat)
        d = (torch.arange(5).reshape(5, 1) + 1).to(torch.cfloat)
        layer.factor_in.copy_(a)
        layer.factor_out.copy_(b)
        layer.factor_m1.copy_(c)
        layer.factor_m2.copy_(d)
    expected = (a.squeeze(1)[:, None, None, None] * b.squeeze(1)[None, :, None, None]
                * c.squeeze(1)[None, None, :, None] * d.squeeze(1)[None, None, None, :])
    assert torch.allclose(layer.weight(), expected)


def test_output_is_real():
    out = CPSpectralConv2d(2, 2, 4, 4, rank=3)(torch.randn(2, 2, 16, 16))
    assert out.dtype == torch.float32 and torch.isfinite(out).all()


def test_cp_is_cheaper_than_a_dense_spectral_weight():
    ch, m, rank = 32, 12, 8
    layer = CPSpectralConv2d(ch, ch, m, m, rank=rank)
    cp = sum(p.numel() for n, p in layer.named_parameters() if "factor" in n
             or n == "rank_weights")
    assert cp < ch * ch * m * m


def test_layer_transfers_across_resolutions():
    """The FNO property: the same weights apply at a different grid size."""
    layer = CPSpectralConv2d(2, 2, 4, 4, rank=2)
    for size in (16, 32, 64):
        assert layer(torch.randn(1, 2, size, size)).shape == (1, 2, size, size)


# --------------------------------------------------------------------------
# the bias touches only its shells
# --------------------------------------------------------------------------


def _spectrum_of_bias_only(layer, size=32):
    """Response to a zero input isolates the bias: W.0 = 0, so out = irfft(b)."""
    out = layer(torch.zeros(1, layer.out_channels, size, size)
                if layer.in_channels == layer.out_channels
                else torch.zeros(1, layer.in_channels, size, size))
    return torch.fft.rfft2(out)


def test_bias_perturbs_only_the_designated_shells():
    torch.manual_seed(0)
    layer = CPSpectralConv2d(2, 2, 8, 8, rank=2, harmonic_bias=True,
                             shells=[3.0])
    with torch.no_grad():
        layer.harmonic_bias.normal_()               # make the bias non-trivial
    spec = _spectrum_of_bias_only(layer)[0, 0]

    m1, m2 = layer.modes1, layer.modes2
    pos = m1 // 2 + 1
    idx = list(range(pos)) + list(range(32 - (m1 - pos), 32))
    mask = layer.bias_mask
    for a, row in enumerate(idx):
        for col in range(m2):
            got = spec[row, col].abs().item()
            if mask[a, col]:
                continue                            # allowed to be non-zero
            assert got < 1e-5, (row, col, got)


def test_bias_is_nonzero_somewhere_on_its_shells():
    """Guard the mirror image of the test above: a bias that does nothing."""
    torch.manual_seed(1)
    layer = CPSpectralConv2d(2, 2, 8, 8, rank=2, harmonic_bias=True,
                             shells=[3.0])
    with torch.no_grad():
        layer.harmonic_bias.normal_()
    assert _spectrum_of_bias_only(layer).abs().max() > 1e-3


def test_disabled_bias_reproduces_the_plain_model_exactly():
    """The control arm must differ in nothing but the bias.

    This is what makes the A/B a controlled comparison rather than two
    different models.
    """
    x = torch.randn(2, 2, 32, 32)
    torch.manual_seed(7)
    plain = CPSpectralConv2d(2, 2, 8, 8, rank=3, harmonic_bias=False)
    torch.manual_seed(7)
    biased = CPSpectralConv2d(2, 2, 8, 8, rank=3, harmonic_bias=True,
                              shells=[4.0])
    assert torch.equal(plain(x), biased(x))         # bias initialises to zero


def test_bias_changes_the_output_once_trained_away_from_zero():
    x = torch.randn(2, 2, 32, 32)
    torch.manual_seed(7)
    layer = CPSpectralConv2d(2, 2, 8, 8, rank=3, harmonic_bias=True,
                             shells=[4.0])
    before = layer(x)
    with torch.no_grad():
        layer.harmonic_bias.normal_()
    assert not torch.allclose(before, layer(x))


def test_empty_shell_set_yields_zero_bias_not_an_error():
    layer = CPSpectralConv2d(2, 2, 8, 8, rank=2, harmonic_bias=True, shells=[])
    assert layer.n_harmonic_modes() == 0
    assert torch.isfinite(layer(torch.randn(1, 2, 16, 16))).all()


def test_gradients_reach_the_bias():
    layer = CPSpectralConv2d(2, 2, 8, 8, rank=2, harmonic_bias=True,
                             shells=[3.0])
    layer(torch.randn(2, 2, 16, 16)).pow(2).mean().backward()
    grad = layer.harmonic_bias.grad
    assert grad is not None and grad.abs().sum() > 0
    # packed storage means there is no off-shell entry to receive a gradient
    assert grad.shape == (layer.n_harmonic_modes(),)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def test_model_shapes_and_reality():
    model = HarmonicLiteFNO(2, 2, width=8, modes=8, layers=2, rank=2)
    out = model(torch.randn(3, 2, 32, 32))
    assert out.shape == (3, 2, 32, 32) and out.dtype == torch.float32


def test_model_ab_pair_is_identical_at_initialisation():
    x = torch.randn(2, 2, 32, 32)
    torch.manual_seed(3)
    off = HarmonicLiteFNO(2, 2, width=8, modes=8, layers=2, rank=2,
                          harmonic_bias=False)
    torch.manual_seed(3)
    on = HarmonicLiteFNO(2, 2, width=8, modes=8, layers=2, rank=2,
                         harmonic_bias=True, fundamental=3.0, n_harmonics=2)
    assert torch.equal(off(x), on(x))
    assert on.n_harmonic_modes() > 0
    assert off.n_harmonic_modes() == 0


def test_bias_adds_few_parameters_relative_to_the_model():
    """Harmonic conditioning must not be a capacity increase in disguise.

    If the bias were a large fraction of the parameters, an improvement could
    be explained by size alone.
    """
    torch.manual_seed(4)
    off = HarmonicLiteFNO(2, 2, width=32, modes=12, layers=4, rank=8,
                          harmonic_bias=False)
    torch.manual_seed(4)
    on = HarmonicLiteFNO(2, 2, width=32, modes=12, layers=4, rank=8,
                         harmonic_bias=True, fundamental=4.0, n_harmonics=3)
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on > n_off
    assert (n_on - n_off) / n_off < 0.10, (n_off, n_on)


def test_model_uses_the_turing_band_by_default_configuration():
    """The shells are chosen from ext10's measurement, not arbitrarily.

    Gray-Scott's energy sits near mode 13-16 at 128x128, which is mode 3-4
    after the pipeline's 4x downsample.
    """
    model = HarmonicLiteFNO(2, 2, width=8, modes=16, layers=1, rank=2,
                            harmonic_bias=True, fundamental=4.0, n_harmonics=3)
    assert model.shells == [4.0, 8.0, 12.0]
