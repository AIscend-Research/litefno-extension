"""Validation for scripts/degradation_sweep.py.

The claim is "accuracy at increasing degradation severity", so the things that
would quietly invalidate it are properties of the severity knob itself: that
s = 0 is genuinely the clean field rather than an almost-clean one, that
severity is monotone, and that both arms meet the same corruption draw. Those
are the first tests here. The rest cover the gap arithmetic.

Training is not exercised. No network access.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch                                                    # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "degradation_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location("degradation_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["degradation_sweep"] = module
    spec.loader.exec_module(module)
    return module


ds = _load()


def _field(n=16, c=2, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, c, h, w, generator=g) * 3.0 + 1.0


# --------------------------------------------------------------------------
# the severity knob
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(ds.CORRUPTIONS))
def test_zero_severity_is_exactly_the_clean_field(kind):
    # "0% artifacts" has to mean the untouched field. If any component were
    # merely near-identity at s = 0 the whole sweep would be measured against a
    # slightly corrupted reference, and every reported excess would be wrong.
    x = _field()
    g = torch.Generator().manual_seed(0)
    assert torch.equal(ds.CORRUPTIONS[kind](x, 0.0, g), x)


def test_blur_is_exactly_identity_at_zero_sigma():
    x = _field()
    assert torch.equal(ds._periodic_blur(x, 0.0), x)


def test_blur_preserves_the_mean_and_returns_real_values():
    # a Gaussian blur is a low-pass with unit DC gain: the field's mean must
    # survive it, and the inverse FFT must not leak an imaginary part
    x = _field()
    b = ds._periodic_blur(x, 1.2)
    assert b.dtype == x.dtype
    assert float(b.mean()) == pytest.approx(float(x.mean()), abs=1e-5)
    # it really did smooth: high-frequency energy fell
    assert float(b.std()) < float(x.std())


def test_blur_wraps_around_rather_than_padding_with_zeros():
    # the grid is periodic, so a bright edge column must bleed onto the far
    # edge. A zero-padded blur would darken it instead.
    x = torch.zeros(1, 1, 16, 16)
    x[0, 0, :, 0] = 1.0
    b = ds._periodic_blur(x, 1.5)
    assert float(b[0, 0, 0, -1]) > 0.01


@pytest.mark.parametrize("kind", list(ds.CORRUPTIONS))
def test_severity_is_monotone(kind):
    x = _field()
    prev = 0.0
    for s in (0.1, 0.25, 0.5, 0.75, 1.0):
        g = torch.Generator().manual_seed(0)
        dev = float((ds.CORRUPTIONS[kind](x, s, g) - x).norm())
        assert dev > prev
        prev = dev


def test_degradation_scale_follows_the_field_not_an_absolute_number():
    # the two Gray-Scott regimes differ by orders of magnitude in amplitude, so
    # a fixed absolute noise floor would mean something different on each. The
    # relative deviation must be roughly invariant to a rescaling of the field.
    x = _field()
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    rel_small = float((ds.degrade(x, 0.5, g1) - x).norm() / x.norm())
    big = x * 1000.0
    rel_big = float((ds.degrade(big, 0.5, g2) - big).norm() / big.norm())
    assert rel_small == pytest.approx(rel_big, rel=0.05)


def test_dropout_zeroes_pixels_rather_than_perturbing_them():
    # the held-out corruption must be structurally unlike the training chain:
    # it removes values instead of nudging them
    x = _field() + 10.0                       # keep values away from zero
    g = torch.Generator().manual_seed(0)
    d = ds.drop_pixels(x, 1.0, g)
    changed = d != x
    assert bool(changed.any())
    assert float(d[changed].abs().max()) == 0.0
    frac = float(changed.float().mean())
    assert frac == pytest.approx(ds.DROP_MAX, abs=0.05)


def test_both_arms_meet_the_same_corruption_draw():
    # evaluate_at reseeds per batch from the run seed, so an arm cannot look
    # better by drawing a luckier noise sample
    x, y = _field(n=8), _field(n=8, seed=1)
    ident = torch.nn.Identity()
    a = ds.evaluate_at(ident, x, y, 0.5, "smartphone", "cpu", seed=3)
    b = ds.evaluate_at(ident, x, y, 0.5, "smartphone", "cpu", seed=3)
    assert a == pytest.approx(b, rel=1e-12)
    # and a different seed really does give a different draw
    c = ds.evaluate_at(ident, x, y, 0.5, "smartphone", "cpu", seed=99)
    assert c != pytest.approx(a, rel=1e-12)


# --------------------------------------------------------------------------
# the gap arithmetic
# --------------------------------------------------------------------------


def _rows(pairs, corr="smartphone"):
    return [{"arm": arm, "corruption": corr, "severity": s, "vrmse": v}
            for s, (arm, v) in pairs]


def test_excess_is_measured_against_each_arms_own_clean_error():
    # robust starts worse on clean input (0.03 vs 0.02) but degrades less.
    # Measured against its own clean error it closes half the rise; measured
    # against baseline's it would look worse than it is.
    rows = [
        {"arm": "baseline", "corruption": "smartphone", "severity": 0.0, "vrmse": 0.02},
        {"arm": "robust", "corruption": "smartphone", "severity": 0.0, "vrmse": 0.03},
        {"arm": "baseline", "corruption": "smartphone", "severity": 1.0, "vrmse": 0.22},
        {"arm": "robust", "corruption": "smartphone", "severity": 1.0, "vrmse": 0.13},
    ]
    out = {c["severity"]: c for c in ds.gap_closed(rows)}
    assert out[1.0]["baseline_excess"] == pytest.approx(0.20)
    assert out[1.0]["robust_excess"] == pytest.approx(0.10)
    assert out[1.0]["frac_gap_closed"] == pytest.approx(0.5)


def test_clean_severity_has_no_gap_to_close():
    rows = [
        {"arm": "baseline", "corruption": "smartphone", "severity": 0.0, "vrmse": 0.02},
        {"arm": "robust", "corruption": "smartphone", "severity": 0.0, "vrmse": 0.03},
    ]
    out = ds.gap_closed(rows)
    # both excesses are zero by construction; a percentage here would be invented
    assert np.isnan(out[0]["frac_gap_closed"])


def test_absolute_comparison_is_reported_separately_from_the_fraction():
    # robust closes most of the *rise* yet is still worse in absolute terms,
    # because it started with a large clean-input tax. Both facts are needed.
    rows = [
        {"arm": "baseline", "corruption": "smartphone", "severity": 0.0, "vrmse": 0.02},
        {"arm": "robust", "corruption": "smartphone", "severity": 0.0, "vrmse": 0.09},
        {"arm": "baseline", "corruption": "smartphone", "severity": 1.0, "vrmse": 0.22},
        {"arm": "robust", "corruption": "smartphone", "severity": 1.0, "vrmse": 0.11},
    ]
    c = [x for x in ds.gap_closed(rows) if x["severity"] == 1.0][0]
    assert c["frac_gap_closed"] == pytest.approx(0.9)
    assert c["robust_better_absolute"] is True
    # at a milder severity the tax can still dominate
    rows2 = rows[:2] + [
        {"arm": "baseline", "corruption": "smartphone", "severity": 0.1, "vrmse": 0.03},
        {"arm": "robust", "corruption": "smartphone", "severity": 0.1, "vrmse": 0.095},
    ]
    c2 = [x for x in ds.gap_closed(rows2) if x["severity"] == 0.1][0]
    assert c2["robust_better_absolute"] is False


def test_negative_fraction_when_the_robust_arm_degrades_faster():
    rows = [
        {"arm": "baseline", "corruption": "dropout", "severity": 0.0, "vrmse": 0.02},
        {"arm": "robust", "corruption": "dropout", "severity": 0.0, "vrmse": 0.02},
        {"arm": "baseline", "corruption": "dropout", "severity": 1.0, "vrmse": 0.12},
        {"arm": "robust", "corruption": "dropout", "severity": 1.0, "vrmse": 0.17},
    ]
    c = [x for x in ds.gap_closed(rows) if x["severity"] == 1.0][0]
    assert c["frac_gap_closed"] == pytest.approx(-0.5)


def test_corruption_without_a_clean_reference_is_skipped():
    # rebasing on the mildest available severity would understate every excess
    rows = [
        {"arm": "baseline", "corruption": "smartphone", "severity": 0.5, "vrmse": 0.10},
        {"arm": "robust", "corruption": "smartphone", "severity": 0.5, "vrmse": 0.08},
    ]
    assert ds.gap_closed(rows) == []


def test_the_two_corruptions_are_scored_independently():
    rows = [
        {"arm": "baseline", "corruption": "smartphone", "severity": 0.0, "vrmse": 0.02},
        {"arm": "robust", "corruption": "smartphone", "severity": 0.0, "vrmse": 0.02},
        {"arm": "baseline", "corruption": "smartphone", "severity": 1.0, "vrmse": 0.12},
        {"arm": "robust", "corruption": "smartphone", "severity": 1.0, "vrmse": 0.07},
        {"arm": "baseline", "corruption": "dropout", "severity": 0.0, "vrmse": 0.02},
        {"arm": "robust", "corruption": "dropout", "severity": 0.0, "vrmse": 0.02},
        {"arm": "baseline", "corruption": "dropout", "severity": 1.0, "vrmse": 0.12},
        {"arm": "robust", "corruption": "dropout", "severity": 1.0, "vrmse": 0.12},
    ]
    out = {(c["corruption"], c["severity"]): c for c in ds.gap_closed(rows)}
    assert out[("smartphone", 1.0)]["frac_gap_closed"] == pytest.approx(0.5)
    # in-family gain must not leak into the held-out corruption's number
    assert out[("dropout", 1.0)]["frac_gap_closed"] == pytest.approx(0.0)
