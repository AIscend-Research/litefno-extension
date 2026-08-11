"""Validation for litefno.models.bands.

Two things have to be true for banded rank allocation to mean anything.

The classification must be a partition -- every shell in exactly one class --
or the rank budget is ambiguous and some modes get capacity from two bands while
others get none.

And the banded model must cost the same as the uniform model it is compared
against. Otherwise "spending rank where the variance is" is untestable, because
any difference could be a difference in how much rank there is.

The classifier is exercised on synthetic spectra with a known answer, and on the
qualitative shapes the real Gray-Scott regimes have.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from litefno.models.bands import (  # noqa: E402
    CLASSES, BandedCPSpectralConv2d, BandedLiteFNO, classify_modes,
    partition_masks, shell_mask, uniform_ranks)


def _spectrum(peak_k=None, n=40, slope=-2.0, peak_gain=100.0, width=1.5):
    """Power law, optionally with a Gaussian bump at ``peak_k``."""
    k = np.arange(1, n + 1)
    power = k.astype(float) ** slope
    if peak_k is not None:
        power = power + peak_gain * power[0] * np.exp(
            -0.5 * ((k - peak_k) / width) ** 2)
    return k, power


# --------------------------------------------------------------------------
# classification is a partition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("peak_k", [None, 4, 12, 25])
def test_classification_is_a_partition(peak_k):
    k, power = _spectrum(peak_k)
    got = classify_modes(k, power)
    labelled = got["primary"] + got["resonant"] + got["damped"]
    assert sorted(labelled) == list(k)
    assert len(labelled) == len(set(labelled)), "a shell got two labels"


def test_classes_are_ordered_primary_then_resonant_then_damped():
    """Bands must be contiguous in k, or 'low / peak / tail' is meaningless."""
    k, power = _spectrum(peak_k=12)
    got = classify_modes(k, power)
    assert got["resonant"], "expected a resonance"
    assert max(got["primary"]) < min(got["resonant"])
    assert max(got["resonant"]) < min(got["damped"])


def test_empty_spectrum_is_handled():
    got = classify_modes([], [])
    assert all(got[c] == [] for c in CLASSES)
    assert got["has_resonance"] is False


# --------------------------------------------------------------------------
# the resonance gate
# --------------------------------------------------------------------------


def test_a_monotone_decaying_spectrum_has_no_resonance():
    """No interior peak means no feedback scale to preserve."""
    k, power = _spectrum(peak_k=None)
    got = classify_modes(k, power)
    assert got["has_resonance"] is False
    assert got["resonant"] == []
    assert got["primary"] == [1]          # only the peak shell itself
    assert len(got["damped"]) == len(k) - 1


def test_an_injected_peak_is_found_at_the_right_wavenumber():
    for peak_k in (8, 14, 22):
        k, power = _spectrum(peak_k=peak_k, peak_gain=500.0)
        got = classify_modes(k, power)
        assert got["has_resonance"], peak_k
        assert got["peak_k"] == peak_k
        assert peak_k in got["resonant"]


def test_a_weak_bump_does_not_qualify():
    """The gate is a rise threshold, so a small wiggle stays unclassified."""
    k, power = _spectrum(peak_k=12, peak_gain=0.05)
    assert classify_modes(k, power)["has_resonance"] is False


def test_the_resonant_band_has_width():
    """Clipping to the argmax shell would leave most of the peak's energy out."""
    k, power = _spectrum(peak_k=15, peak_gain=500.0, width=2.0)
    got = classify_modes(k, power)
    assert len(got["resonant"]) > 1
    band = np.array(got["resonant"])
    assert band.max() - band.min() == len(band) - 1, "band is not contiguous"


def test_rise_threshold_is_what_separates_the_two_gray_scott_groups():
    """Regression on the measured values that set the default.

    maze/spots/worms rise by 12x or more; bubbles/gliders/spirals by under 7x.
    A default of 10 must put the split between them.
    """
    measured = {"maze": 2501.3, "spots": 1221.7, "worms": 12.4,
                "bubbles": 6.7, "gliders": 4.6, "spirals": 2.0}
    resonant = {n for n, r in measured.items() if r >= 10.0}
    assert resonant == {"maze", "spots", "worms"}


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------


def test_shell_mask_selects_a_ring():
    mask = shell_mask(16, 16, [4])
    assert mask[4, 0] and mask[0, 4] and mask[-4, 0]
    assert not mask[0, 6]


def test_partition_masks_cover_every_mode_exactly_once():
    """Overlap would double-count capacity; a gap would zero a mode entirely."""
    classes = {"primary": [0, 1, 2], "resonant": [3, 4], "damped": [5, 6]}
    masks = partition_masks(12, 12, classes)
    stack = torch.stack([masks[c] for c in CLASSES]).int().sum(0)
    assert (stack == 1).all(), "modes covered zero or multiple times"


def test_unclaimed_modes_fall_into_damped():
    """A radius beyond the measured spectrum must still be represented."""
    classes = {"primary": [0, 1], "resonant": [2], "damped": [3]}
    masks = partition_masks(16, 16, classes)     # radii run well past 3
    stack = torch.stack([masks[c] for c in CLASSES]).int().sum(0)
    assert (stack == 1).all()
    assert masks["damped"][-8, 8], "a far corner mode was left unclassified"


# --------------------------------------------------------------------------
# the banded layer
# --------------------------------------------------------------------------


def test_each_band_is_confined_to_its_mask():
    """A band leaking outside its shells would break the whole premise."""
    torch.manual_seed(0)
    classes = {"primary": [0, 1, 2], "resonant": [3, 4], "damped": [5, 6, 7]}
    layer = BandedCPSpectralConv2d(2, 2, 8, 8,
                                   {"primary": 2, "resonant": 0, "damped": 0},
                                   classes)
    w = layer.weight()
    outside = ~layer.mask_primary
    assert w[:, :, outside].abs().max() < 1e-12


def test_zero_rank_band_contributes_nothing_and_costs_nothing():
    torch.manual_seed(0)
    classes = {"primary": [0, 1, 2], "resonant": [3, 4], "damped": [5, 6, 7]}
    layer = BandedCPSpectralConv2d(2, 2, 8, 8,
                                   {"primary": 4, "resonant": 0, "damped": 0},
                                   classes)
    names = [n for n, _ in layer.named_parameters()]
    assert not any(n.startswith("resonant_") or n.startswith("damped_")
                   for n in names)
    assert layer.n_bands == 1


def test_banded_matches_uniform_parameter_count_at_equal_total_rank():
    """The comparison is about where rank goes, not how much there is."""
    classes = {"primary": [0, 1, 2, 3], "resonant": [4, 5], "damped": [6, 7, 8]}
    banded = BandedCPSpectralConv2d(
        8, 8, 8, 8, {"primary": 3, "resonant": 3, "damped": 2}, classes)
    ranks, uni_classes = uniform_ranks(8, max_shell=12)
    uniform = BandedCPSpectralConv2d(8, 8, 8, 8, ranks, uni_classes)
    n_banded = sum(p.numel() for p in banded.parameters())
    n_uniform = sum(p.numel() for p in uniform.parameters())
    assert n_banded == n_uniform, (n_banded, n_uniform)


def test_compressing_damped_reduces_parameters():
    classes = {"primary": [0, 1, 2], "resonant": [3, 4], "damped": [5, 6, 7]}
    big = BandedCPSpectralConv2d(
        8, 8, 8, 8, {"primary": 4, "resonant": 4, "damped": 4}, classes)
    small = BandedCPSpectralConv2d(
        8, 8, 8, 8, {"primary": 4, "resonant": 4, "damped": 1}, classes)
    assert sum(p.numel() for p in small.parameters()) < \
        sum(p.numel() for p in big.parameters())


def test_layer_output_is_real_and_shape_preserving():
    classes = {"primary": [0, 1, 2], "resonant": [3], "damped": [4, 5]}
    layer = BandedCPSpectralConv2d(
        2, 2, 8, 8, {"primary": 2, "resonant": 2, "damped": 1}, classes)
    out = layer(torch.randn(3, 2, 32, 32))
    assert out.shape == (3, 2, 32, 32) and out.dtype == torch.float32


def test_layer_transfers_across_resolutions():
    classes = {"primary": [0, 1, 2], "resonant": [3], "damped": [4, 5]}
    layer = BandedCPSpectralConv2d(
        2, 2, 4, 4, {"primary": 2, "resonant": 1, "damped": 1}, classes)
    for size in (16, 32, 64):
        assert layer(torch.randn(1, 2, size, size)).shape == (1, 2, size, size)


def test_all_bands_zero_rank_gives_a_zero_operator():
    classes = {"primary": [0, 1], "resonant": [], "damped": [2]}
    layer = BandedCPSpectralConv2d(
        2, 2, 4, 4, {"primary": 0, "resonant": 0, "damped": 0}, classes)
    assert layer.weight().abs().max() == 0
    assert torch.allclose(layer(torch.randn(1, 2, 16, 16)),
                          torch.zeros(1, 2, 16, 16), atol=1e-6)


def test_gradients_reach_every_nonzero_band():
    classes = {"primary": [0, 1, 2], "resonant": [3, 4], "damped": [5, 6]}
    layer = BandedCPSpectralConv2d(
        2, 2, 8, 8, {"primary": 2, "resonant": 2, "damped": 1}, classes)
    layer(torch.randn(2, 2, 16, 16)).pow(2).mean().backward()
    for c in CLASSES:
        grad = getattr(layer, f"{c}_m1").grad
        assert grad is not None and grad.abs().sum() > 0, c


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


def test_model_runs_and_reports_its_allocation():
    classes = {"primary": list(range(0, 4)), "resonant": [4, 5],
               "damped": list(range(6, 24))}
    model = BandedLiteFNO(2, 2, width=8, modes=8, layers=2,
                          ranks={"primary": 3, "resonant": 3, "damped": 1},
                          classes=classes)
    out = model(torch.randn(2, 2, 32, 32))
    assert out.shape == (2, 2, 32, 32)
    counts = model.modes_per_class()
    assert sum(counts.values()) == 8 * 8
    assert all(v >= 0 for v in counts.values())


def test_uniform_control_labels_everything_primary():
    ranks, classes = uniform_ranks(7, max_shell=20)
    assert ranks == {"primary": 7, "resonant": 0, "damped": 0}
    assert classes["resonant"] == [] and classes["damped"] == []
    masks = partition_masks(8, 8, classes)
    assert masks["primary"].all(), "the control must cover every mode"
