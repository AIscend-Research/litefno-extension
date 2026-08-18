"""Validation for scripts/transplant_distance.py.

This experiment exists because ext21's null was read off a single dose, and a
null transplant is indistinguishable from a null *dose* unless the amount of
network actually copied is measured. So the first tests here pin the parameter
accounting, and the rest pin the two axes -- distance and dose -- that the
dose-response is read along.

The ladder is the other thing worth testing: a distance axis that does not
achieve the distance it claims would make every correlation in the report a
correlation with the wrong variable.

Training is not exercised. No network access.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts"
          / "transplant_distance.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load():
    spec = importlib.util.spec_from_file_location("transplant_distance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["transplant_distance"] = module
    spec.loader.exec_module(module)
    return module


td = _load()


def _model(rank=8, modes=10, layers=4, width=32):
    import argparse
    return td.mt.build(argparse.Namespace(width=width, modes=modes,
                                          layers=layers, rank=rank), 0)


# --------------------------------------------------------------------------
# the accounting that unblocked this experiment
# --------------------------------------------------------------------------


def test_dose_accounting_is_a_strictly_growing_fraction_of_the_model():
    m = _model()
    fracs = [td.params_moved(m, k)["frac_of_model"] for k in range(0, 9)]
    assert fracs[0] == 0.0
    assert all(b > a for a, b in zip(fracs, fracs[1:]))
    assert fracs[-1] < 0.15          # even all components is a small slice


def test_three_components_is_a_few_percent_of_the_model_not_most_of_it():
    # the fact ext21's null was missing: 3 components is 3.5% of the weights,
    # so "transplant barely moved the error" and "fine-tune moved it 5x" were
    # never comparing equal doses
    got = td.params_moved(_model(), 3)
    assert got["params_moved"] == 252
    assert got["frac_of_model"] == pytest.approx(0.0355, abs=5e-4)
    assert got["frac_of_spectral"] == pytest.approx(0.0926, abs=5e-4)


def test_dose_accounting_is_linear_in_components():
    m = _model()
    one = td.params_moved(m, 1)["params_moved"]
    for k in (2, 3, 5, 8):
        assert td.params_moved(m, k)["params_moved"] == one * k


def test_dose_accounting_tracks_the_architecture_rather_than_being_hardcoded():
    # one component costs (2 * modes + 1) per layer: both mode-axis factors
    # plus the single rank weight, which does not scale with modes
    for modes in (5, 10, 20):
        got = td.params_moved(_model(modes=modes), 1)["params_moved"]
        assert got == (2 * modes + 1) * 4
    deep = td.params_moved(_model(layers=8), 1)["params_moved"]
    assert deep == 2 * td.params_moved(_model(layers=4), 1)["params_moved"]


# --------------------------------------------------------------------------
# the distance axis achieves the distance it claims
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ray", ["diffuse", "sharp"])
@pytest.mark.parametrize("d", [0.0, 0.25, 0.5, 1.0, 1.75, 3.0])
def test_the_ladder_lands_exactly_on_the_requested_distance(ray, d):
    rung = td.ladder([d], ray)
    assert td.log_distance(rung[round(d, 4)]) == pytest.approx(d, abs=1e-12)


def test_zero_distance_is_the_source_regime_itself():
    for ray in ("diffuse", "sharp"):
        assert td.ladder([0.0], ray)[0.0] == pytest.approx(td.SOURCE)


def test_the_two_rays_move_in_opposite_directions():
    a = td.ladder([1.0], "diffuse")[1.0]
    b = td.ladder([1.0], "sharp")[1.0]
    assert a["diffusion"] > td.SOURCE["diffusion"] > b["diffusion"]
    assert a["omega"] < td.SOURCE["omega"] < b["omega"]


def test_distance_is_symmetric_between_the_rays():
    # if one ray were systematically farther the distance axis would be
    # confounded with which direction was taken
    for d in (0.25, 1.0, 2.0):
        near = td.log_distance(td.ladder([d], "diffuse")[d])
        far = td.log_distance(td.ladder([d], "sharp")[d])
        assert near == pytest.approx(far, rel=1e-12)


def test_the_ladder_is_monotone_in_the_requested_distance():
    rung = td.ladder([0.0, 0.25, 0.5, 1.0, 1.75], "diffuse")
    got = [td.log_distance(v) for _, v in sorted(rung.items())]
    assert all(b > a for a, b in zip(got, got[1:]))


# --------------------------------------------------------------------------
# arm construction
# --------------------------------------------------------------------------


def test_every_transplant_dose_gets_a_size_matched_control():
    arms = td.build_arms([1, 2, 3])
    res = sorted(d for k, d in arms if k == "transplant_resonant")
    dam = sorted(d for k, d in arms if k == "transplant_damped")
    assert res == dam == [1, 2, 3]


def test_the_controls_and_both_ceilings_are_present():
    kinds = {k for k, _ in td.build_arms([1])}
    assert {"scratch", "finetune", "transplant_all"} <= kinds


def test_arm_names_separate_doses_but_not_the_ceilings():
    assert td.arm_name("transplant_resonant", 2) == "transplant_resonant_2"
    assert td.arm_name("transplant_resonant", 3) != \
        td.arm_name("transplant_resonant", 2)
    for kind in ("scratch", "finetune", "transplant_all"):
        assert td.arm_name(kind, 7) == kind


# --------------------------------------------------------------------------
# the summary reads the gap in the direction the doc claims
# --------------------------------------------------------------------------


def _rows(res, dam, dose=3, ray="diffuse", d=0.5):
    out = []
    for seed, (a, b) in enumerate(zip(res, dam)):
        out += [
            {"ray": ray, "distance": d, "dose": dose, "seed": seed,
             "kind": "transplant_resonant", "test_vrmse": a,
             "frac_of_model": 0.0355},
            {"ray": ray, "distance": d, "dose": dose, "seed": seed,
             "kind": "transplant_damped", "test_vrmse": b,
             "frac_of_model": 0.0355},
            {"ray": ray, "distance": d, "dose": 0, "seed": seed,
             "kind": "scratch", "test_vrmse": 1.0, "frac_of_model": 0.0},
            {"ray": ray, "distance": d, "dose": -1, "seed": seed,
             "kind": "finetune", "test_vrmse": 0.5, "frac_of_model": 1.0},
            {"ray": ray, "distance": d, "dose": -2, "seed": seed,
             "kind": "transplant_all", "test_vrmse": 0.8,
             "frac_of_model": 0.0946},
        ]
    return out


def test_a_better_resonant_arm_gives_a_positive_gap():
    # positive must mean "resonant won", which is what every sentence in the
    # write-up depends on
    got = td.summarise(_rows([0.9, 0.9], [1.0, 1.0]))[0]
    assert got["gap_rel"] == pytest.approx(0.1)
    assert got["resonant_wins"] == 2


def test_a_worse_resonant_arm_gives_a_negative_gap():
    got = td.summarise(_rows([1.1, 1.1], [1.0, 1.0]))[0]
    assert got["gap_rel"] < 0
    assert got["resonant_wins"] == 0


def test_the_gap_is_paired_per_seed_not_computed_on_means():
    # means would hide a cell where resonant wins big once and loses slightly
    # twice; the win count is what makes that visible
    got = td.summarise(_rows([0.1, 1.05, 1.05], [1.0, 1.0, 1.0]))[0]
    assert got["resonant_wins"] == 1
    assert got["n_seeds"] == 3


def test_the_ceilings_are_carried_into_the_summary():
    got = td.summarise(_rows([0.9, 0.9], [1.0, 1.0]))[0]
    assert got["finetune"] == pytest.approx(0.5)
    assert got["transplant_all"] == pytest.approx(0.8)
    assert got["finetune_vs_scratch"] == pytest.approx(0.5)
    assert got["all_vs_scratch"] == pytest.approx(0.2)


def test_each_dose_is_summarised_separately():
    rows = _rows([0.9, 0.9], [1.0, 1.0], dose=1) + \
        _rows([0.7, 0.7], [1.0, 1.0], dose=3)
    got = {s["dose"]: s["gap_rel"] for s in td.summarise(rows)}
    assert set(got) == {1, 3}
    assert got[3] > got[1]


def test_a_cell_missing_its_control_is_dropped_rather_than_guessed():
    rows = [r for r in _rows([0.9], [1.0])
            if r["kind"] != "transplant_damped"]
    assert td.summarise(rows) == []


# --------------------------------------------------------------------------
# the rank correlation the verdict is read from
# --------------------------------------------------------------------------


def test_spearman_is_exact_on_a_monotone_relationship():
    x = np.array([0.0, 0.25, 0.5, 1.0, 1.75])
    assert td.mt_spearman(x, 2 * x + 1) == pytest.approx(1.0)
    assert td.mt_spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_handles_ties_without_blowing_up():
    x = np.array([1.0, 1.0, 2.0, 2.0])
    got = td.mt_spearman(x, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.isfinite(got) and -1.0 <= got <= 1.0
