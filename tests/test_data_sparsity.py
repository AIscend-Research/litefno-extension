"""Validation for scripts/data_sparsity.py.

The reconstruction claim rests on three things being true: the Fourier basis
spans exactly the mode box an FNO truncation keeps, the full-data fit is
genuinely the best that model class can do (so the oracle floor is a floor), and
the masks realise the coverage they claim. Each is checked here on cases with a
known answer.

No network access -- the tests that need field data synthesise it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_sparsity.py"


def _load():
    spec = importlib.util.spec_from_file_location("data_sparsity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_sparsity"] = module
    spec.loader.exec_module(module)
    return module


ds = _load()

GRID = (32, 32)


def recon(basis, idx, fields):
    """fit_and_reconstruct also returns the constrained-mode count; drop it."""
    out, _ = ds.fit_and_reconstruct(basis, idx, fields)
    return out


def fft_truncate(u: np.ndarray, modes: int) -> np.ndarray:
    """Reference: zero every mode outside the box, then invert."""
    h, w = u.shape
    spec = np.fft.fft2(u)
    ky = np.fft.fftfreq(h, d=1 / h)
    kx = np.fft.fftfreq(w, d=1 / w)
    box = np.maximum(np.abs(ky[:, None]), np.abs(kx[None, :]))
    spec[box > modes] = 0
    return np.real(np.fft.ifft2(spec))


# --------------------------------------------------------------------------
# the basis and the oracle floor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("modes", [1, 2, 4, 8])
def test_basis_has_the_right_dimension(modes):
    basis = ds.fourier_basis(GRID, modes)
    assert basis.shape == (GRID[0] * GRID[1], (2 * modes + 1) ** 2)


@pytest.mark.parametrize("modes", [2, 4, 8])
def test_full_data_fit_equals_fft_truncation(modes):
    """The oracle floor is provably the best any M-mode model can do.

    A least-squares fit over the whole grid is an orthogonal projection onto the
    basis span; if that span is exactly the mode box, the fit must coincide with
    simply zeroing the modes outside it. If this drifts, the 'floor' stops being
    a floor and every sparsity penalty becomes meaningless.
    """
    rng = np.random.default_rng(0)
    u = rng.normal(size=GRID)
    basis = ds.fourier_basis(GRID, modes)
    fit = recon(
        basis, np.arange(u.size), u.reshape(-1, 1)).reshape(GRID)
    assert np.abs(fit - fft_truncate(u, modes)).max() < 1e-10


def test_band_limited_field_is_recovered_exactly():
    """With enough incoherent samples, a field inside the box comes back exact."""
    modes = 4
    rng = np.random.default_rng(1)
    basis = ds.fourier_basis(GRID, modes)
    field = basis @ rng.normal(size=basis.shape[1])
    idx = rng.choice(field.size, size=3 * basis.shape[1], replace=False)
    rec = recon(basis, idx, field.reshape(-1, 1)).ravel()
    assert ds.vrmse(rec, field) < 1e-9


def test_oracle_is_a_lower_bound_on_sparse_reconstruction():
    """No mask may beat the full-data fit of the same mode budget."""
    modes = 4
    rng = np.random.default_rng(2)
    u = rng.normal(size=GRID) + fft_truncate(rng.normal(size=GRID), 2) * 5
    basis = ds.fourier_basis(GRID, modes)
    flat = u.reshape(-1, 1)
    oracle = ds.vrmse(recon(basis, np.arange(u.size), flat), flat)
    for regime, fn in ds.REGIMES.items():
        for cov in (0.5, 0.25, 0.1):
            m = fn(GRID, cov, np.random.default_rng(ds.mask_seed(regime, cov, 0)))
            idx = np.flatnonzero(m.ravel())
            v = ds.vrmse(recon(basis, idx, flat), flat)
            assert v >= oracle - 1e-9, (regime, cov, v, oracle)


# --------------------------------------------------------------------------
# sampling regimes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("regime", sorted(ds.REGIMES))
@pytest.mark.parametrize("coverage", [0.25, 0.1, 0.02])
def test_masks_realise_roughly_their_coverage(regime, coverage):
    """Realised coverage is reported, not assumed -- but it must be close."""
    m = ds.REGIMES[regime](
        (128, 128), coverage, np.random.default_rng(ds.mask_seed(regime, coverage, 0)))
    got = m.mean()
    # stations overlap and blocks/grid quantise, so allow a relative slack
    assert got == pytest.approx(coverage, rel=0.35), f"{regime}: {got:.4f}"


def test_masks_are_boolean_and_nonempty():
    for regime, fn in ds.REGIMES.items():
        m = fn((128, 128), 0.01, np.random.default_rng(0))
        assert m.dtype == bool and m.shape == (128, 128)
        assert m.any(), regime


def test_grid_mask_is_a_lattice():
    m = ds.mask_grid((128, 128), 0.25, np.random.default_rng(0))
    rows = np.flatnonzero(m.any(axis=1))
    assert len(set(np.diff(rows))) == 1          # evenly spaced
    assert np.diff(rows)[0] == 2                 # stride 2 for 1/4 coverage


def test_blocks_mask_is_contiguous():
    """Observed pixels come in solid BLOCK x BLOCK patches, not scattered."""
    m = ds.mask_blocks((128, 128), 0.25, np.random.default_rng(0))
    b = ds.BLOCK
    tiles = m.reshape(128 // b, b, 128 // b, b).transpose(0, 2, 1, 3)
    sums = tiles.reshape(-1, b * b).sum(axis=1)
    assert set(np.unique(sums)) <= {0, b * b}


def test_mask_seed_is_stable_across_processes():
    """Regression guard: hash() is salted per process and must not be used.

    A hash-derived seed silently redraws every mask on each run, which would
    make the committed results irreproducible without failing anything loudly.
    """
    import subprocess
    code = (f"import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('ds',r'{SCRIPT}');"
            f"m=importlib.util.module_from_spec(s);sys.modules['ds']=m;"
            f"s.loader.exec_module(m);print(m.mask_seed('random',10.0,0))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout.strip()
            for _ in range(2)}
    assert len(outs) == 1
    assert outs.pop() == str(ds.mask_seed("random", 10.0, 0))


# --------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------


def test_vrmse_matches_litefno():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(SCRIPT.parents[1] / "src"))
    from litefno.metrics import vrmse as reference

    rng = np.random.default_rng(3)
    for shape in [(4, 16, 16), (32, 32), (7, 5)]:
        a = rng.normal(size=shape)
        b = rng.normal(size=shape) * 2 + 1
        assert ds.vrmse(a, b) == pytest.approx(float(reference(a, b)), rel=1e-5)


def test_vrmse_is_one_for_a_mean_prediction():
    """VRMSE = 1 is the 'predict the mean' baseline the figure marks in red."""
    rng = np.random.default_rng(4)
    target = rng.normal(size=(8, 32, 32))
    assert ds.vrmse(np.full_like(target, target.mean()), target) == pytest.approx(
        1.0, rel=1e-4)


def test_vrmse_zero_for_exact_prediction():
    rng = np.random.default_rng(5)
    target = rng.normal(size=(4, 8, 8))
    assert ds.vrmse(target, target) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# end to end on synthetic fields
# --------------------------------------------------------------------------


GRID64 = (64, 64)


def _plane_wave(k: int) -> np.ndarray:
    yy, xx = np.indices(GRID64)
    return np.cos(2 * np.pi * k * yy / 64) * np.cos(2 * np.pi * k * xx / 64)


def test_where_a_scenario_keeps_its_variance_decides_whether_it_survives():
    """The mechanism behind the Gray-Scott prediction, in isolation.

    Two fields of equal variance, one with all its energy at mode 2 and one at
    mode 12, sampled by the same stride-4 lattice and fitted with a mode budget
    the lattice can support. The low-mode field comes back; the high-mode one is
    beyond the lattice Nyquist (64 / (2 * 4) = 8) and does not.
    """
    idx = np.flatnonzero(
        ds.mask_grid(GRID64, 0.0625, np.random.default_rng(0)).ravel())
    basis = ds.fourier_basis(GRID64, 4)

    for k, bound, worse_than in [(2, 0.05, None), (12, None, 0.5)]:
        field = _plane_wave(k).reshape(-1, 1)
        v = ds.vrmse(recon(basis, idx, field), field)
        if bound is not None:
            assert v < bound, (k, v)
        else:
            assert v > worse_than, (k, v)


def test_observations_per_dof_is_what_governs_recovery():
    """Overshooting the mode budget destroys content the sampling resolves.

    A stride-4 lattice gives 256 observations. Asked for 4 modes (81 dof, 3.2
    observations each) it returns the mode-2 field exactly. Asked for 16 modes
    (1089 dof, 0.24 each) the system is four times underdetermined and the
    min-norm solution spreads the energy, so the same well-sampled field is
    lost. Min-norm least squares is not sparsity-promoting -- it has no reason
    to concentrate energy in the one mode that generated the data.

    This is why results are reported per mode budget with obs/dof alongside,
    rather than at a single fixed M.
    """
    idx = np.flatnonzero(
        ds.mask_grid(GRID64, 0.0625, np.random.default_rng(0)).ravel())
    field = _plane_wave(2).reshape(-1, 1)

    within = ds.vrmse(
        recon(ds.fourier_basis(GRID64, 4), idx, field), field)
    beyond = ds.vrmse(
        recon(ds.fourier_basis(GRID64, 16), idx, field), field)
    assert within < 0.01
    assert beyond > 0.5


def test_the_underdetermined_limit_is_pattern_agnostic():
    """Far below one observation per dof, lattice and random fail alike.

    Worth pinning: it means a regime comparison at very thin coverage is
    measuring the mode budget, not the sampling geometry.
    """
    n_obs = int(ds.mask_grid(GRID64, 0.0625, np.random.default_rng(0)).sum())
    grid_idx = np.flatnonzero(
        ds.mask_grid(GRID64, 0.0625, np.random.default_rng(0)).ravel())
    rand_idx = np.random.default_rng(7).choice(64 * 64, size=n_obs, replace=False)
    basis = ds.fourier_basis(GRID64, 16)          # 1089 dof vs 256 obs

    field = _plane_wave(2).reshape(-1, 1)
    v_grid = ds.vrmse(recon(basis, grid_idx, field), field)
    v_rand = ds.vrmse(recon(basis, rand_idx, field), field)
    assert v_grid == pytest.approx(v_rand, abs=0.05)


def test_a_lattice_aliases_out_of_band_content_to_a_specific_wavenumber():
    """The failure mode that makes `grid` worse than useless.

    A stride-4 lattice samples at frequency 16, so a mode-12 component folds to
    |12 - 16| = 4 -- which sits *inside* a 4-mode budget. The fit therefore
    reports a confident, well-conditioned answer at the wrong wavenumber, and
    scores VRMSE sqrt(2): worse than predicting the mean, which would score 1.

    An unresolved mode is recoverable-in-principle with more sensors. An
    aliased one is not: the information is destroyed at acquisition.
    """
    idx = np.flatnonzero(
        ds.mask_grid(GRID64, 0.0625, np.random.default_rng(0)).ravel())
    field = _plane_wave(12).reshape(-1, 1)
    basis = ds.fourier_basis(GRID64, 4)

    fitted = recon(basis, idx, field)
    assert ds.vrmse(fitted, field) > 1.0         # worse than the mean baseline

    spec = np.abs(np.fft.fft2(fitted.reshape(GRID64)))
    k = np.fft.fftfreq(64, d=1 / 64).astype(int)
    peak = np.unravel_index(np.argmax(spec), GRID64)
    assert abs(k[peak[0]]) == 4 and abs(k[peak[1]]) == 4


def test_random_sampling_is_ill_conditioned_near_one_obs_per_dof():
    """Guard on a real instability rather than a claim of robustness.

    With 256 observations against 289 dof the design matrix is nearly square
    and badly conditioned, and min-norm least squares can return a fit far
    worse than the mean. The sweep reports obs/dof so these rows are
    identifiable instead of being read as physics.
    """
    n_obs = int(ds.mask_grid(GRID64, 0.0625, np.random.default_rng(0)).sum())
    idx = np.random.default_rng(7).choice(64 * 64, size=n_obs, replace=False)
    basis = ds.fourier_basis(GRID64, 8)                # 289 dof vs 256 obs
    field = _plane_wave(12).reshape(-1, 1)
    assert ds.vrmse(recon(basis, idx, field), field) > 2.0
