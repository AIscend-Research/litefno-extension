"""Validation for litefno.bench.

ext25's headline is a comparison between a closed-form FLOP model and a wall
clock, so the closed form has to be right for any of it to mean anything. It is
checked here against ``torch``'s own tracer -- exactly, on the two architectures
whose operations the tracer fully covers, and to within the documented residual
on the CP model.

The second thing pinned here is the fusion transform. It is only a legitimate
optimisation if it changes nothing observable: same outputs, same parameters,
same checkpoint. If any of those drifted, ext25's "free 1.4-1.9x" would be a
measurement of a different model.

Deliberately absent: assertions on wall-clock durations. Timing is measured by
the experiment and reported with its spread; a test that asserts one model is
faster than another would fail on a loaded CI box and prove nothing when it
passed. What is tested is that the timing *harness* cannot mislead -- that it
restores thread state, and that its derived quantities are consistent.

No network access, no data files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litefno.bench import (  # noqa: E402
    analytic_flops, conv_flops, count_parameters, counted_flops,
    cp_reconstruction_flops, fft_flops, flop_audit, fuse_spectral_weights,
    latency, peak_rss_bytes, resident_bytes, spearman, state_dict_bytes,
    unfuse_spectral_weights)
from litefno.models.fno_s import FNOS  # noqa: E402
from litefno.models.harmonic import HarmonicLiteFNO  # noqa: E402
from litefno.models.litefno import LiteFNO  # noqa: E402


def _cnn():
    return LiteFNO(2, 2, width=16, rank=8, layers=2)


def _fno():
    return FNOS(2, 2, width=16, modes=8, layers=2)


def _cp(rank=4):
    return HarmonicLiteFNO(2, 2, width=16, modes=8, layers=2, rank=rank)


# --------------------------------------------------------------------------
# the closed-form FLOP model
# --------------------------------------------------------------------------


def test_conv_flops_matches_a_hand_count():
    # 3x3, 4 in, 8 out, on 5x5: 4*8*9*25 MACs at 2 flops each, plus 8*25 bias
    assert conv_flops(4, 8, 5, 5, kernel=3, bias=True) == 2 * 4 * 8 * 9 * 25 + 200
    assert conv_flops(4, 8, 5, 5, kernel=3, bias=False) == 2 * 4 * 8 * 9 * 25


def test_fft_flops_follows_the_stated_convention():
    from math import log2
    assert fft_flops(8, 8) == pytest.approx(2.5 * 64 * log2(64))
    assert fft_flops(8, 8, n_fields=3) == pytest.approx(3 * fft_flops(8, 8))
    assert fft_flops(1, 1) == 0.0


@pytest.mark.parametrize("builder", [_cnn, _fno])
def test_analytic_flops_matches_the_tracer_exactly(builder):
    """No tolerance. These architectures are convolutions and matmuls only."""
    model = builder()
    audit = flop_audit(model, size=16, batch=1)
    assert audit["counted"] is not None
    assert audit["counted"] == audit["analytic_counter_convention"]


def test_analytic_flops_matches_the_tracer_on_cp_within_the_residual():
    """The residual is the two small factor contractions torch does not emit."""
    audit = flop_audit(_cp(), size=16, batch=1)
    assert audit["rel_error"] < 0.01


@pytest.mark.parametrize("batch", [1, 4, 16])
def test_the_audit_holds_across_batch_sizes(batch):
    audit = flop_audit(_fno(), size=16, batch=batch)
    assert audit["counted"] == audit["analytic_counter_convention"]


def test_the_real_convention_costs_more_than_the_tracer_convention():
    """A complex MAC is 8 real flops and an FFT is not free; the tracer says
    otherwise, and the gap is the reason the experiment does not use it."""
    model = _fno()
    real = analytic_flops(model, 16, batch=1)
    counter = analytic_flops(model, 16, batch=1, convention="counter")
    assert real["total"] > counter["total"]
    assert real["fft"] > 0.0 and counter["fft"] == 0.0


def test_unknown_convention_is_rejected():
    with pytest.raises(ValueError):
        analytic_flops(_cnn(), 16, convention="whatever")


def test_the_families_are_recognised():
    assert analytic_flops(_cnn(), 16)["family"] == "cnn"
    assert analytic_flops(_fno(), 16)["family"] == "dense_spectral"
    assert analytic_flops(_cp(), 16)["family"] == "cp"


# --------------------------------------------------------------------------
# the cost that does not scale with batch -- ext25's whole finding
# --------------------------------------------------------------------------


def test_cp_reconstruction_is_independent_of_batch_size():
    """The claim the extension rests on, isolated from any timer."""
    model = _cp()
    one = analytic_flops(model, 16, batch=1)
    many = analytic_flops(model, 16, batch=64)
    assert one["fixed"] == many["fixed"] > 0.0
    assert many["batched"] == pytest.approx(64 * one["batched"])
    # ... so its share collapses as the batch grows
    assert many["fixed_share"] < one["fixed_share"]


def test_only_cp_models_have_a_fixed_cost():
    assert analytic_flops(_cnn(), 16)["fixed"] == 0.0
    assert analytic_flops(_fno(), 16)["fixed"] == 0.0
    assert analytic_flops(_cp(), 16)["fixed"] > 0.0


def test_cp_reconstruction_grows_with_rank_but_parameters_barely_do():
    """Low rank buys parameters and costs compute -- in the same direction."""
    cheap, dear = _cp(rank=2), _cp(rank=16)
    assert count_parameters(dear) > count_parameters(cheap)
    ratio_params = count_parameters(dear) / count_parameters(cheap)
    ratio_flops = (analytic_flops(dear, 16)["fixed"]
                   / analytic_flops(cheap, 16)["fixed"])
    assert ratio_flops == pytest.approx(8.0)          # exactly rank-linear
    assert ratio_params < ratio_flops


def test_cp_reconstruction_closed_form_is_rank_linear():
    single = cp_reconstruction_flops(8, 8, 4, 4, rank=1)
    assert cp_reconstruction_flops(8, 8, 4, 4, rank=5) == pytest.approx(5 * single)


def test_a_tiny_cp_model_can_cost_more_flops_than_a_huge_dense_one():
    """The headline, as an assertion with no clock in it.

    This is the whole of H6's refutation in deterministic form: parameter count
    and forward cost move in opposite directions here, so any argument that
    reads the first as the second is unsound before a single timing is taken.
    """
    small = HarmonicLiteFNO(2, 2, width=32, modes=16, layers=4, rank=32)
    large = FNOS(2, 2, width=32, modes=16, layers=4)
    assert count_parameters(small) < count_parameters(large) / 100
    assert analytic_flops(small, 32, batch=1)["total"] > \
        analytic_flops(large, 32, batch=1)["total"]


# --------------------------------------------------------------------------
# fusing
# --------------------------------------------------------------------------


def test_fusing_leaves_the_output_bitwise_identical():
    torch.manual_seed(0)
    model = _cp().eval()
    x = torch.randn(2, 2, 16, 16)
    with torch.inference_mode():
        before = model(x).clone()
    assert fuse_spectral_weights(model) == 2
    with torch.inference_mode():
        after = model(x)
    assert torch.equal(before, after)


def test_fusing_changes_neither_parameters_nor_the_checkpoint():
    model = _cp()
    params, disk = count_parameters(model), state_dict_bytes(model)
    resident = resident_bytes(model)
    fuse_spectral_weights(model)
    assert count_parameters(model) == params
    assert state_dict_bytes(model) == disk          # non-persistent buffer
    assert resident_bytes(model) > resident         # ... but it does cost RAM


def test_fusing_removes_the_fixed_flop_cost():
    model = _cp()
    assert analytic_flops(model, 16)["fixed"] > 0.0
    fuse_spectral_weights(model)
    assert analytic_flops(model, 16)["fixed"] == 0.0
    assert analytic_flops(model, 16)["spectral"] > 0.0   # the real work remains


def test_fusing_is_idempotent_and_reversible():
    torch.manual_seed(0)
    model = _cp().eval()
    x = torch.randn(1, 2, 16, 16)
    with torch.inference_mode():
        reference = model(x).clone()
    assert fuse_spectral_weights(model) == 2
    assert fuse_spectral_weights(model) == 0           # nothing left to fuse
    assert unfuse_spectral_weights(model) == 2
    assert unfuse_spectral_weights(model) == 0
    with torch.inference_mode():
        assert torch.equal(model(x), reference)
    assert analytic_flops(model, 16)["fixed"] > 0.0


@pytest.mark.parametrize("builder", [_cnn, _fno])
def test_fusing_is_a_no_op_on_models_without_cp_layers(builder):
    model = builder()
    assert fuse_spectral_weights(model) == 0
    assert unfuse_spectral_weights(model) == 0


def test_a_fused_model_still_tracks_gradients_after_unfusing():
    """Fusing is inference-only; unfusing has to give training back."""
    model = _cp()
    fuse_spectral_weights(model)
    unfuse_spectral_weights(model)
    out = model(torch.randn(1, 2, 16, 16)).sum()
    out.backward()
    assert model.spectral_layers[0].factor_in.grad is not None


# --------------------------------------------------------------------------
# size and memory
# --------------------------------------------------------------------------


def test_state_dict_bytes_is_four_bytes_per_float32_parameter():
    model = _cnn()                       # no buffers, all fp32
    assert state_dict_bytes(model) == 4 * count_parameters(model)


def test_peak_rss_is_reported_in_plausible_bytes():
    """A factor-1024 unit error here would misreport a model's memory."""
    rss = peak_rss_bytes()
    assert 8 * 2 ** 20 < rss < 2 ** 41   # between 8 MB and 2 TB


# --------------------------------------------------------------------------
# the rank correlation
# --------------------------------------------------------------------------


def test_spearman_is_one_for_a_monotone_pair_and_minus_one_when_reversed():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_ignores_scale_and_sees_only_order():
    assert spearman([1, 2, 3], [1, 100, 10000]) == pytest.approx(1.0)


def test_spearman_averages_ties():
    # ranks of the second series are 1.5, 1.5, 3 -- a partial correlation
    rho = spearman([1, 2, 3], [5, 5, 9])
    assert 0.0 < rho < 1.0


def test_spearman_is_nan_when_one_series_is_constant():
    assert spearman([1, 2, 3], [7, 7, 7]) != spearman([1, 2, 3], [7, 7, 7])


def test_spearman_rejects_bad_input():
    with pytest.raises(ValueError):
        spearman([1, 2], [1])
    with pytest.raises(ValueError):
        spearman([1], [1])


# --------------------------------------------------------------------------
# the timing harness
# --------------------------------------------------------------------------


def test_latency_reports_consistent_derived_quantities():
    model = _cnn()
    result = latency(model, torch.randn(4, 2, 16, 16), repeats=3, warmup=1)
    assert result["ms"] > 0.0
    assert result["ms_per_sample"] == pytest.approx(result["ms"] / 4)
    assert result["samples_per_s"] == pytest.approx(4 / (result["ms"] / 1e3))
    assert result["ms_min"] <= result["ms"]
    assert result["ms_iqr"] >= 0.0


def test_latency_restores_the_thread_count_it_was_given():
    """A benchmark that leaks thread state changes every later measurement."""
    before = torch.get_num_threads()
    latency(_cnn(), torch.randn(1, 2, 16, 16), repeats=2, warmup=0, threads=1)
    assert torch.get_num_threads() == before


def test_latency_rejects_zero_repeats():
    with pytest.raises(ValueError):
        latency(_cnn(), torch.randn(1, 2, 16, 16), repeats=0)


def test_counted_flops_runs_under_inference_mode_without_touching_grads():
    model = _cnn()
    counted_flops(model, torch.zeros(1, 2, 16, 16))
    assert all(p.grad is None for p in model.parameters())
