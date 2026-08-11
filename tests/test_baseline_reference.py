"""Validation for scripts/baseline_reference.py.

The reference number is only worth anything if the arm labelled ``litefno`` is
really the CP-factorized spectral architecture. The repo has already been bitten
by this once: ``results/extensions/logs_reproduction_table.csv`` carries LiteFNO
in its filename but was produced by the low-rank CNN placeholder. The first
tests here make that mislabelling impossible to repeat silently.

Training is not exercised (it needs the dataset and a GPU); everything around it
is -- pair construction, metric agreement, rollout window indexing, and the
summary statistics the reported number is read from.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "baseline_reference.py"


def _load():
    spec = importlib.util.spec_from_file_location("baseline_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["baseline_reference"] = module
    spec.loader.exec_module(module)
    return module


br = _load()


# --------------------------------------------------------------------------
# the arms are what they say they are
# --------------------------------------------------------------------------


def _has_complex_parameter(model) -> bool:
    """A genuine spectral layer carries complex weights."""
    return any(p.is_complex() for p in model.parameters()) or any(
        getattr(b, "is_complex", lambda: False)() for b in model.buffers())


def _uses_fft(model, in_ch=2) -> bool:
    """Does a forward pass go through torch.fft at all?"""
    calls = []
    real_rfft2, real_fft2 = torch.fft.rfft2, torch.fft.fft2

    def spy(fn):
        def inner(*a, **k):
            calls.append(fn.__name__)
            return fn(*a, **k)
        return inner

    torch.fft.rfft2, torch.fft.fft2 = spy(real_rfft2), spy(real_fft2)
    try:
        model(torch.randn(1, in_ch, 32, 32))
    finally:
        torch.fft.rfft2, torch.fft.fft2 = real_rfft2, real_fft2
    return bool(calls)


def test_litefno_arm_is_spectral_and_cp_factorized():
    """The exact confusion that mislabelled the committed reproduction table.

    Checked on the parameters rather than by intercepting torch.fft, because
    neuralop does not reach the FFT through the attribute this module could
    patch -- a spy there reports "no FFT" for a model that plainly has complex
    spectral weights, which would be a false alarm rather than a guard.

    Two things must hold. The spectral weights are complex, which a CNN's never
    are. And under CP factorization the weight is stored as a rank vector plus
    one factor matrix per tensor mode, so those parameter names must be present
    -- if neuralop silently fell back to dense, they would not be, and the
    returned label changes with it so a dense model can never be reported as CP.
    """
    pytest.importorskip("neuralop")
    model, factorization = br.build_model("litefno", 2, 2)
    assert factorization in ("cp", "tucker", "dense")

    complex_params = [n for n, p in model.named_parameters() if p.is_complex()]
    assert complex_params, "litefno arm has no complex spectral weights"

    if factorization == "cp":
        names = [n for n, _ in model.named_parameters()]
        assert any(n.endswith("weight.weights") for n in names), names[:6]
        factors = [n for n in names if "weight.factors.factor_" in n]
        assert factors, "no CP factor matrices"
        # every CP factor is itself complex: the factorization is of the
        # spectral weight, not of some real-valued side tensor
        by_name = dict(model.named_parameters())
        assert all(by_name[n].is_complex() for n in factors)


def test_litefno_cp_rank_follows_the_configured_fraction():
    """rank=0.02 is a fraction; a larger model must get a larger CP rank.

    Pins that the rank argument is actually reaching the factorization, rather
    than being accepted and ignored.
    """
    pytest.importorskip("neuralop")
    ranks = []
    for width in (32, 128):
        model, fac = br.build_litefno(2, 2, width=width)
        if fac != "cp":
            pytest.skip("neuralop did not build a CP factorization")
        w = dict(model.named_parameters())
        key = next(n for n in w if n.endswith("weight.weights"))
        ranks.append(w[key].numel())
    assert ranks[1] > ranks[0], ranks


def test_cnn_arm_is_not_spectral_and_is_labelled_so():
    """src/litefno/models/litefno.py is a CNN; the arm name and label say so."""
    model, factorization = br.build_model("cnn", 2, 2)
    assert not _uses_fft(model)
    assert not _has_complex_parameter(model)
    assert factorization == "n/a"


def test_fno_s_arm_is_spectral():
    model, _ = br.build_model("fno_s", 2, 2)
    assert _uses_fft(model)


def test_arms_agree_on_shape():
    x = torch.randn(3, 2, 32, 32)
    for arm in ("fno_s", "cnn"):
        model, _ = br.build_model(arm, 2, 2)
        assert model(x).shape == x.shape, arm


def test_protocol_constants_match_the_notebook():
    """Guard against the reference protocol drifting away from the source."""
    assert (br.MODES, br.WIDTH, br.LAYERS, br.RANK) == (16, 64, 8, 0.02)
    assert (br.EPOCHS, br.BATCH, br.LR) == (200, 64, 1e-3)
    assert (br.LR_STEP, br.LR_GAMMA) == (100, 0.5)
    assert br.ROLL_WINDOWS == ((6, 12), (13, 30))


# --------------------------------------------------------------------------
# data handling
# --------------------------------------------------------------------------


def test_to_pairs_pairs_consecutive_frames():
    rng = np.random.default_rng(0)
    traj = rng.normal(size=(3, 5, 8, 8, 2)).astype(np.float32)
    x, y = br.to_pairs(traj)
    assert x.shape == (3 * 4, 2, 8, 8) == y.shape
    # first pair of the first trajectory is (frame 0, frame 1), channels first
    assert np.allclose(x[0].numpy(), traj[0, 0].transpose(2, 0, 1))
    assert np.allclose(y[0].numpy(), traj[0, 1].transpose(2, 0, 1))
    # pairs do not straddle a trajectory boundary
    assert np.allclose(x[4].numpy(), traj[1, 0].transpose(2, 0, 1))


def test_to_pairs_uses_every_transition():
    traj = np.zeros((2, 7, 4, 4, 1), dtype=np.float32)
    x, _ = br.to_pairs(traj)
    assert len(x) == 2 * 6


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


class _Scale(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x * self.k


def test_evaluate_one_step_matches_the_repo_metric():
    sys.path.insert(0, str(SCRIPT.parents[1] / "src"))
    from litefno.metrics import vrmse

    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.normal(size=(40, 2, 8, 8)).astype(np.float32))
    y = torch.from_numpy(rng.normal(size=(40, 2, 8, 8)).astype(np.float32))
    got = br.evaluate_one_step(_Identity(), x, y, "cpu", batch=7)
    assert got == pytest.approx(float(vrmse(x, y)), rel=1e-6)


def test_evaluate_one_step_is_batch_size_invariant():
    """Batching must not change the number -- VRMSE is not a mean of ratios."""
    rng = np.random.default_rng(2)
    x = torch.from_numpy(rng.normal(size=(37, 2, 8, 8)).astype(np.float32))
    y = torch.from_numpy(rng.normal(size=(37, 2, 8, 8)).astype(np.float32))
    vals = [br.evaluate_one_step(_Scale(0.5), x, y, "cpu", batch=b)
            for b in (1, 8, 37, 128)]
    assert max(vals) - min(vals) < 1e-6


def test_perfect_prediction_scores_zero():
    rng = np.random.default_rng(3)
    x = torch.from_numpy(rng.normal(size=(16, 2, 8, 8)).astype(np.float32))
    assert br.evaluate_one_step(_Identity(), x, x, "cpu") == pytest.approx(0, abs=1e-6)


# --------------------------------------------------------------------------
# rollout window indexing
# --------------------------------------------------------------------------


def test_rollout_of_a_constant_trajectory_is_exact():
    """An identity model on a time-constant field must roll out perfectly."""
    frame = np.random.default_rng(4).normal(size=(2, 8, 8, 2)).astype(np.float32)
    traj = np.repeat(frame[:, None], 40, axis=1)          # (2, 40, 8, 8, 2)
    out = br.evaluate_rollout(_Identity(), traj, "cpu")
    assert set(out) == {"roll_6_12", "roll_13_30"}
    for v in out.values():
        assert v == pytest.approx(0.0, abs=1e-5)


def test_rollout_windows_score_different_steps():
    """A model that drifts must score worse on the later window.

    This is what catches an off-by-one in the window slice: if both windows
    read the same steps, the two numbers would be equal.
    """
    rng = np.random.default_rng(5)
    frame = rng.normal(size=(2, 8, 8, 2)).astype(np.float32)
    traj = np.repeat(frame[:, None], 40, axis=1)
    out = br.evaluate_rollout(_Scale(1.05), traj, "cpu")
    assert out["roll_13_30"] > out["roll_6_12"] > 0


def test_rollout_horizon_is_capped_by_the_trajectory():
    frame = np.random.default_rng(6).normal(size=(1, 4, 4, 1)).astype(np.float32)
    traj = np.repeat(frame[:, None], 10, axis=1)
    out = br.evaluate_rollout(_Identity(), traj, "cpu")
    assert "roll_6_12" in out          # truncated to what the data supports
    for v in out.values():
        assert np.isfinite(v)


# --------------------------------------------------------------------------
# summary statistics
# --------------------------------------------------------------------------


def test_summarise_reports_mean_and_sample_sd():
    recs = [{"arm": "litefno", "seed": s, "factorization": "cp", "params": 10,
             "epochs": 200, "train_s": 1.0, "onestep_test_vrmse": v,
             "onestep_valid_vrmse": v, "roll_6_12": v}
            for s, v in zip((0, 1, 2), (0.01, 0.02, 0.03))]
    got = br.summarise(recs)[0]
    assert got["onestep_test_vrmse_mean"] == pytest.approx(0.02)
    assert got["onestep_test_vrmse_std"] == pytest.approx(np.std([0.01, 0.02, 0.03],
                                                                ddof=1))
    assert got["n_seeds"] == 3


def test_summarise_single_seed_reports_zero_spread_not_nan():
    recs = [{"arm": "cnn", "seed": 0, "factorization": "n/a", "params": 1,
             "epochs": 1, "train_s": 1.0, "onestep_test_vrmse": 0.5}]
    got = br.summarise(recs)[0]
    assert got["onestep_test_vrmse_std"] == 0.0
    assert np.isfinite(got["onestep_test_vrmse_mean"])
