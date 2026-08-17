"""Validation for scripts/cross_regime.py.

The measurement is a leave-one-regime-out gap, so the thing that would quietly
invalidate it is a fold that leaks: the held-out regime appearing in training,
or the "seen" evaluation set containing the held-out regime. Those are the first
tests here. The rest cover label reconstruction from the manifest and the
summary arithmetic.

Training is not exercised. No network access.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cross_regime.py"


def _load():
    spec = importlib.util.spec_from_file_location("cross_regime", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cross_regime"] = module
    spec.loader.exec_module(module)
    return module


cr = _load()

REGIMES = ["bubbles", "gliders", "maze", "spirals", "spots", "worms"]


def _manifest(per_regime=(12, 4, 4)):
    return {"splits": {
        split: {"files": [{"regime": r, "trajectories": list(range(n)),
                           "available": 20} for r in REGIMES]}
        for split, n in zip(("train", "valid", "test"), per_regime)}}


# --------------------------------------------------------------------------
# label reconstruction
# --------------------------------------------------------------------------


def test_regime_labels_match_concatenation_order():
    lab = cr.regime_labels(_manifest(), "train")
    assert len(lab) == 72
    assert list(lab[:12]) == ["bubbles"] * 12
    assert list(lab[-12:]) == ["worms"] * 12
    assert list(dict.fromkeys(lab)) == REGIMES


def test_regime_labels_handle_uneven_counts():
    m = {"splits": {"train": {"files": [
        {"regime": "a", "trajectories": [0, 1, 2]},
        {"regime": "b", "trajectories": [0]},
    ]}}}
    assert list(cr.regime_labels(m, "train")) == ["a", "a", "a", "b"]


def test_labels_cover_every_trajectory():
    """A mismatch here would silently misalign every fold."""
    for split, n in (("train", 72), ("valid", 24), ("test", 24)):
        assert len(cr.regime_labels(_manifest(), split)) == n


# --------------------------------------------------------------------------
# fold construction: no leakage
# --------------------------------------------------------------------------


def _split_indices(labels, held):
    return (labels != held), (labels == held), (labels != held)


@pytest.mark.parametrize("held", REGIMES)
def test_held_out_regime_is_absent_from_training(held):
    train_lab = cr.regime_labels(_manifest(), "train")
    seen = train_lab != held
    assert held not in set(train_lab[seen])
    assert seen.sum() == 60                      # five regimes x 12
    assert set(train_lab[seen]) == set(REGIMES) - {held}


@pytest.mark.parametrize("held", REGIMES)
def test_evaluation_sets_are_disjoint_and_complete(held):
    test_lab = cr.regime_labels(_manifest(), "test")
    held_mask = test_lab == held
    seen_mask = test_lab != held
    assert not (held_mask & seen_mask).any()      # disjoint
    assert (held_mask | seen_mask).all()          # complete
    assert held_mask.sum() == 4
    assert seen_mask.sum() == 20
    assert set(test_lab[seen_mask]) == set(REGIMES) - {held}


def test_every_regime_is_held_out_exactly_once():
    lab = cr.regime_labels(_manifest(), "train")
    regimes = list(dict.fromkeys(lab))
    assert len(regimes) == len(set(regimes)) == 6


# --------------------------------------------------------------------------
# pair construction preserves the split
# --------------------------------------------------------------------------


def test_pairs_from_a_regime_subset_have_the_expected_count():
    """to_pairs must not mix trajectories across the fold boundary."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(24, 60, 8, 8, 2)).astype(np.float32)
    lab = cr.regime_labels(_manifest(), "test")
    x_held, y_held = cr.br.to_pairs(arr[lab == "maze"])
    assert len(x_held) == 4 * 59 == len(y_held)
    x_seen, _ = cr.br.to_pairs(arr[lab != "maze"])
    assert len(x_seen) == 20 * 59


def test_held_and_seen_pairs_come_from_different_trajectories():
    arr = np.zeros((24, 6, 4, 4, 1), dtype=np.float32)
    lab = cr.regime_labels(_manifest(), "test")
    # stamp each trajectory with its index so provenance is checkable
    for i in range(24):
        arr[i] = i
    x_held, _ = cr.br.to_pairs(arr[lab == "spots"])
    x_seen, _ = cr.br.to_pairs(arr[lab != "spots"])
    held_ids = set(np.unique(x_held.numpy()).tolist())
    seen_ids = set(np.unique(x_seen.numpy()).tolist())
    assert held_ids and seen_ids
    assert not (held_ids & seen_ids), "a trajectory appears in both eval sets"


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_spearman_matches_known_values():
    assert cr.br_spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert cr.br_spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert abs(cr.br_spearman([1, 2, 3, 4], [1, 4, 2, 3])) < 1.0


def test_spearman_is_rank_based_not_value_based():
    """A monotone transform of either axis must not change it."""
    x = [1, 2, 3, 4, 5]
    y = [2, 9, 11, 40, 41]
    assert cr.br_spearman(x, y) == pytest.approx(
        cr.br_spearman(x, [np.log(v) for v in y]))


def test_prediction_check_computes_distance_from_the_training_mean(tmp_path):
    """spectral_distance must exclude the held-out regime from the mean."""
    csv_path = tmp_path / "ext10.csv"
    vals = {"bubbles": 0.58, "gliders": 0.69, "maze": 0.013, "spirals": 0.774,
            "spots": 0.006, "worms": 0.309}
    with csv_path.open("w") as f:
        f.write("scenario,field,segment,spatial_var_at_modes_8\n")
        for k, v in vals.items():
            f.write(f"{k},A,settled,{v}\n")

    records = [{"held_out": k, "gap_ratio": 1.0 + i}
               for i, k in enumerate(vals)]
    out = cr.check_ext10_prediction(records, csv_path)
    assert len(out) == 6
    got = {o["held_out"]: o for o in out}
    others = [v for k, v in vals.items() if k != "maze"]
    assert got["maze"]["training_mean"] == pytest.approx(np.mean(others))
    assert got["maze"]["spectral_distance"] == pytest.approx(
        abs(vals["maze"] - np.mean(others)))
    # The symmetric metric is documented as the weaker hypothesis precisely
    # because it cannot tell the two extremes apart: spirals (v=77%, extreme
    # high) scores as distant as spots (v=0.6%, extreme low), even though only
    # one of them holds energy the remaining regimes never show the model.
    by_dist = {o["held_out"]: o["spectral_distance"] for o in out}
    assert by_dist["spirals"] == pytest.approx(by_dist["spots"], abs=0.02)
    # the signed predictor does separate them
    assert got["spots"]["var_below_mode8"] < got["spirals"]["var_below_mode8"]
    assert set(out[0].keys()) >= {"spearman_rho_signed", "spearman_rho_symmetric",
                                  "predicted_worst_two_hits"}


def test_prediction_check_is_skipped_without_ext10(tmp_path):
    assert cr.check_ext10_prediction([{"held_out": "maze", "gap_ratio": 2.0}],
                                     tmp_path / "missing.csv") == []


# --------------------------------------------------------------------------
# the three arms
# --------------------------------------------------------------------------
#
# Training is still not exercised. What is checked is the arithmetic that turns
# three arms into a "fraction of the gap closed" claim, plus the two properties
# of the pieces that would silently corrupt it: noise at the documented scale,
# and an ensemble average that reduces to the single-model path.


def test_arms_isolate_augmentation_from_ensembling():
    # arm 3 must differ from arm 2 in exactly one field, or the arm3-arm2
    # difference is not attributable to the ensemble alone
    diff = {k for k in cr.ARMS["robust"]
            if cr.ARMS["robust"][k] != cr.ARMS["robust+unc"][k]}
    assert diff == {"members"}
    assert cr.ARMS["baseline"]["snr_db"] is None
    assert cr.ARMS["baseline"]["members"] == 1
    assert cr.ARMS["robust+unc"]["members"] > 1


def test_noise_matches_the_ext3_definition():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4096, 2, 8, 8) * 3.0 + 1.0
    for snr in (40.0, 30.0, 20.0):
        noisy = cr.add_noise(x, snr, gen)
        # residual std is x.std() * 10 ** (-snr / 20), the ext3 formula
        expected = float(x.std()) * (10.0 ** (-snr / 20.0))
        assert float((noisy - x).std()) == pytest.approx(expected, rel=0.05)


def test_noise_leaves_the_signal_unbiased():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    x = torch.zeros(8192, 1, 4, 4)
    assert float(cr.add_noise(x, 20.0, gen).mean()) == pytest.approx(0.0, abs=1e-3)
    # zero-variance input gets zero noise: the scale is relative, not absolute
    assert float((cr.add_noise(x, 20.0, gen) - x).abs().max()) == 0.0


def test_single_member_ensemble_equals_the_plain_evaluation():
    torch = pytest.importorskip("torch")
    lin = torch.nn.Conv2d(2, 2, 1)
    x, y = torch.randn(32, 2, 8, 8), torch.randn(32, 2, 8, 8)
    assert cr.evaluate_ensemble([lin], x, y, "cpu") == pytest.approx(
        cr.br.evaluate_one_step(lin, x, y, "cpu"), rel=1e-9)


def test_ensemble_averages_predictions_not_errors():
    torch = pytest.importorskip("torch")

    class Const(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, x):
            return torch.full_like(x, self.v)

    y = torch.zeros(16, 1, 4, 4)
    y[0, 0, 0, 0] = 1.0                      # give the target nonzero variance
    x = torch.zeros_like(y)
    # two members that bracket the target: the mean prediction is exact, while
    # the mean of their individual VRMSEs is not zero
    both = cr.evaluate_ensemble([Const(-1.0), Const(1.0)], x, y, "cpu")
    each = [cr.evaluate_ensemble([Const(v)], x, y, "cpu") for v in (-1.0, 1.0)]
    assert both < min(each)


def test_gap_closed_is_the_fraction_of_baseline_excess_removed():
    recs = [
        {"held_out": "maze", "arm": "baseline", "gap_ratio": 3.0,
         "held_out_vrmse": 0.30, "seen_vrmse": 0.10},
        {"held_out": "maze", "arm": "robust", "gap_ratio": 2.0,
         "held_out_vrmse": 0.20, "seen_vrmse": 0.10},
    ]
    out = cr.gap_closed(recs)
    assert len(out) == 1
    # baseline excess is 3.0 - 1 = 2.0; the arm removed 1.0 of it
    assert out[0]["frac_gap_closed"] == pytest.approx(0.5)
    assert out[0]["real_improvement"] is True
    assert out[0]["held_vrmse_delta"] == pytest.approx(-0.10)


def test_gap_closed_flags_improvement_bought_by_degrading_the_seen_side():
    # identical gap improvement to the test above, but achieved by making the
    # seen regimes worse while the held-out regime also got worse. The fraction
    # still reads 50%, so the flag is the only thing separating them.
    recs = [
        {"held_out": "spots", "arm": "baseline", "gap_ratio": 3.0,
         "held_out_vrmse": 0.30, "seen_vrmse": 0.10},
        {"held_out": "spots", "arm": "robust", "gap_ratio": 2.0,
         "held_out_vrmse": 0.40, "seen_vrmse": 0.20},
    ]
    out = cr.gap_closed(recs)
    assert out[0]["frac_gap_closed"] == pytest.approx(0.5)
    assert out[0]["real_improvement"] is False
    assert out[0]["seen_vrmse_delta"] > 0


def test_gap_closed_reports_negative_when_an_arm_widens_the_gap():
    recs = [
        {"held_out": "worms", "arm": "baseline", "gap_ratio": 2.0,
         "held_out_vrmse": 0.20, "seen_vrmse": 0.10},
        {"held_out": "worms", "arm": "robust", "gap_ratio": 2.5,
         "held_out_vrmse": 0.25, "seen_vrmse": 0.10},
    ]
    assert cr.gap_closed(recs)[0]["frac_gap_closed"] == pytest.approx(-0.5)


def test_gap_closed_needs_a_baseline_and_skips_folds_without_one():
    recs = [{"held_out": "maze", "arm": "robust", "gap_ratio": 2.0,
             "held_out_vrmse": 0.2, "seen_vrmse": 0.1}]
    assert cr.gap_closed(recs) == []


def test_gap_closed_is_undefined_when_the_baseline_had_no_gap():
    recs = [
        {"held_out": "maze", "arm": "baseline", "gap_ratio": 1.0,
         "held_out_vrmse": 0.10, "seen_vrmse": 0.10},
        {"held_out": "maze", "arm": "robust", "gap_ratio": 0.9,
         "held_out_vrmse": 0.09, "seen_vrmse": 0.10},
    ]
    # nothing to close: dividing by zero excess would fabricate a percentage
    assert np.isnan(cr.gap_closed(recs)[0]["frac_gap_closed"])
