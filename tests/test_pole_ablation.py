"""Validation for scripts/pole_ablation.py.

The ablation only means something if the controls really are controls: a
shuffle that leaves tensors untouched, or a resample that quietly changes the
scale, would manufacture the gap this study reports. Those properties are pinned
first. The partial-correlation degeneracy is pinned second, because the study's
central methodological claim is that partialling is *undefined* here rather than
merely inconvenient.

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pole_ablation.py"


def _load():
    spec = importlib.util.spec_from_file_location("pole_ablation", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["pole_ablation"] = m
    spec.loader.exec_module(m)
    return m


ab = _load()


class Toy(torch.nn.Module):
    def __init__(self, complex_too=True):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.real = torch.nn.Parameter(torch.randn(6, 7, generator=g))
        self.scalar = torch.nn.Parameter(torch.tensor(3.0))
        if complex_too:
            self.cplx = torch.nn.Parameter(
                torch.complex(torch.randn(4, 5, generator=g),
                              torch.randn(4, 5, generator=g)))


# --------------------------------------------------------------------------
# shuffled: same values, different arrangement
# --------------------------------------------------------------------------


def test_shuffle_preserves_the_exact_multiset_of_values():
    # the whole point of this control: identical distribution, scale, extremes
    # and sparsity; only the arrangement differs
    m = Toy()
    before = m.real.detach().clone()
    ab.shuffle_weights(m, seed=1)
    after = m.real.detach()
    assert torch.allclose(before.flatten().sort().values,
                          after.flatten().sort().values)
    assert float(before.mean()) == pytest.approx(float(after.mean()), abs=1e-6)
    assert float(before.std()) == pytest.approx(float(after.std()), abs=1e-6)


def test_shuffle_actually_rearranges():
    m = Toy()
    before = m.real.detach().clone()
    ab.shuffle_weights(m, seed=1)
    assert not torch.equal(before, m.real.detach())


def test_shuffle_handles_complex_tensors():
    m = Toy()
    before = m.cplx.detach().clone()
    ab.shuffle_weights(m, seed=1)
    after = m.cplx.detach()
    assert after.is_complex()
    assert not torch.equal(before, after)
    # complex entries move as units, so the multiset of moduli is preserved
    assert torch.allclose(before.abs().flatten().sort().values,
                          after.abs().flatten().sort().values, atol=1e-6)


def test_shuffle_leaves_scalars_alone():
    # nothing to permute in a one-element tensor; "shuffling" it would overstate
    # how much of the model was scrambled
    m = Toy()
    ab.shuffle_weights(m, seed=1)
    assert float(m.scalar) == pytest.approx(3.0)


def test_shuffle_is_deterministic_per_seed():
    a, b = Toy(), Toy()
    ab.shuffle_weights(a, seed=7)
    ab.shuffle_weights(b, seed=7)
    assert torch.equal(a.real.detach(), b.real.detach())


# --------------------------------------------------------------------------
# resampled: same moments, new values
# --------------------------------------------------------------------------


def test_resample_matches_mean_and_std_without_reusing_values():
    m = Toy()
    before = m.real.detach().clone()
    ab.resample_weights(m, seed=2)
    after = m.real.detach()
    assert float(after.mean()) == pytest.approx(float(before.mean()), abs=0.35)
    assert float(after.std()) == pytest.approx(float(before.std()), abs=0.35)
    assert not torch.allclose(before.flatten().sort().values,
                              after.flatten().sort().values)


def test_resample_handles_complex_without_overflow():
    # float() of a complex mean raises; the CP spectral weights are complex, so
    # this is the crash the real run hit
    m = Toy()
    ab.resample_weights(m, seed=2)
    assert m.cplx.detach().is_complex()
    assert torch.isfinite(m.cplx.detach().real).all()
    assert torch.isfinite(m.cplx.detach().imag).all()


def test_resample_matches_real_and_imaginary_moments_separately():
    m = Toy()
    b = m.cplx.detach().clone()
    ab.resample_weights(m, seed=3)
    a = m.cplx.detach()
    assert float(a.real.std()) == pytest.approx(float(b.real.std()), abs=0.4)
    assert float(a.imag.std()) == pytest.approx(float(b.imag.std()), abs=0.4)


# --------------------------------------------------------------------------
# the partial-correlation degeneracy
# --------------------------------------------------------------------------


def test_partial_is_undefined_when_the_target_is_a_function_of_the_covariate():
    # this is the study's methodological claim: on these systems the exact pole
    # is a function of |k| alone, so partialling removes the target entirely
    z = np.arange(20, dtype=float)
    y = -2.0 * z + 5.0                 # perfectly determined by z
    x = np.sin(z)
    assert np.isnan(ab.partial_spearman(x, y, z))


def test_partial_recovers_a_relationship_that_survives_the_covariate():
    rng = np.random.default_rng(0)
    z = rng.uniform(0, 10, 400)
    extra = rng.standard_normal(400)
    x = z + extra
    y = z + extra                      # shared component beyond z
    assert ab.partial_spearman(x, y, z) > 0.5


def test_partial_reports_zero_when_only_the_covariate_is_shared():
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 10, 400)
    x = z + 0.01 * rng.standard_normal(400)
    y = z + 0.01 * rng.standard_normal(400)
    assert abs(ab.partial_spearman(x, y, z)) < 0.3


def test_arms_are_named_and_distinct():
    assert set(ab.ARMS) == {"trained", "untrained", "shuffled", "resampled"}
