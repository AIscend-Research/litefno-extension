r"""Dose-response on regime distance: is H2 a curve or a cliff? (ext34)

ext21 asked whether learned spectral factors are shared physics and answered no:
0 of 8 (target, budget) cells put the resonant transplant ahead of a size-matched
damped control. Its explanation was the overlap matrix -- two models trained on
the *same* regime share no more spectral basis (0.258) than models trained on
different ones (0.230-0.283), so the CP basis looks set by initialization rather
than physics.

But ext21 used **two** target regimes, both far from the source. A null on two far
points is consistent with two very different worlds:

    the cliff     factors never transplant, at any distance, because the basis
                  is arbitrary -- ext21's reading
    the curve     factors do transplant between nearby regimes and the two
                  targets were simply too far to show it

Those are distinguishable, and only by varying the distance. This turns H2 from a
yes/no into a dose-response.

Distance, defined before the runs
---------------------------------
The regimes are points in ``(diffusion, omega)``. Both are positive scale
parameters, so distance is measured in **log-ratio** space,

    d = sqrt( ln(D/D0)^2 + ln(w/w0)^2 )

which makes "twice the diffusion" the same distance as "half the diffusion", as
it should be for a scale parameter, and which a raw Euclidean distance would get
wrong.

Targets sit on a ladder along two rays from the source -- one toward more
diffusion and less rotation, one toward less diffusion and more rotation -- so a
distance effect can be separated from a direction effect. If only one ray shows
a trend, that is about that direction, not about distance.

The anchor that makes the curve interpretable
---------------------------------------------
``d = 0`` is included: the target *is* the source regime, drawn with a different
data seed and a different model init. This is the strongest possible case for
transplanting, and ext21's overlap result makes a sharp prediction about it --
if the basis is set by initialization, then even at zero regime distance the
resonant transplant should not beat the damped control, because the two models
never shared a basis to begin with.

A curve that is flat *including at d = 0* says the factors are arbitrary. A curve
that is positive at d = 0 and decays says they are physical and ext21 measured
the tail.

The reported quantity
---------------------
ext21's headline is the **gap** between the resonant transplant and the
size-matched damped control, not either arm's error, because any warm start
helps and the resonant set is also simply more of the source model. That gap is
what is plotted against distance here, with from-scratch and full fine-tune
carried alongside as the floor and ceiling.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_MT = Path(__file__).resolve().parent / "mode_transplant.py"


def _load_ext21():
    spec = importlib.util.spec_from_file_location("mode_transplant", _MT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["mode_transplant"] = m
    spec.loader.exec_module(m)
    return m


mt = _load_ext21()
import torch                                                    # noqa: E402

from litefno.specscope import fit, one_step_vrmse, transplant    # noqa: E402
from litefno.systems import rotating_diffusion                   # noqa: E402

ARMS = ("scratch", "finetune", "transplant_resonant", "transplant_damped")
SOURCE = dict(diffusion=0.4, omega=0.6)          # ext21's source, unchanged


def ladder(distances, ray: str) -> dict:
    """Regime parameters at each log-ratio distance along one ray.

    ``diffuse`` increases diffusion and decreases rotation; ``sharp`` does the
    reverse. Both move along a 45-degree line in log space, so the requested
    distance is achieved exactly.
    """
    sign = 1.0 if ray == "diffuse" else -1.0
    out = {}
    for d in distances:
        step = sign * d / math.sqrt(2.0)
        out[round(d, 4)] = dict(diffusion=SOURCE["diffusion"] * math.exp(step),
                                omega=SOURCE["omega"] * math.exp(-step))
    return out


def log_distance(params: dict) -> float:
    return float(math.hypot(math.log(params["diffusion"] / SOURCE["diffusion"]),
                            math.log(params["omega"] / SOURCE["omega"])))


def make_regime(params: dict, n_traj: int, n_steps: int, size: int, seed: int):
    return rotating_diffusion(n_traj=n_traj, n_steps=n_steps, size=size,
                              dt=mt.DT, length=mt.LENGTH, seed=seed, **params)


def run_cell(arm: str, source_model, params: dict, budget: int, args,
             device: str, seed: int, components: dict) -> dict:
    """One (arm, regime, seed). Identical to ext21's run_arm but regime-driven.

    The target data seed is offset from the source's so that ``d = 0`` is a
    genuinely fresh draw of the same regime rather than the source's own
    trajectories handed back.
    """
    train = make_regime(params, budget, args.n_steps, args.size,
                        seed=100 + seed)
    test = make_regime(params, args.n_test_traj, args.n_steps, args.size,
                       seed=999)
    model = mt.build(args, seed)
    info = {"n_components": 0}
    if arm == "finetune":
        model.load_state_dict(source_model.state_dict())
    elif arm == "transplant_resonant":
        info = transplant(model, source_model, components["resonant"])
    elif arm == "transplant_damped":
        info = transplant(model, source_model, components["damped"])
    t0 = time.time()
    fit(model, train, epochs=args.epochs, lr=args.lr, device=device, seed=seed)
    row = {"arm": arm, "n_components": info["n_components"],
           "test_vrmse": float(one_step_vrmse(model, test, device)),
           "train_s": round(time.time() - t0, 1)}
    for h in info.get("handles", []):
        h.remove()
    return row


def summarise(rows: list[dict]) -> list[dict]:
    """Per (ray, distance): mean error by arm and the resonant-minus-damped gap.

    The gap is signed so that **positive means the resonant transplant is
    better** (lower error), which is the direction H2 predicts. It is expressed
    relative to the damped control so distances with different absolute
    difficulty stay comparable.
    """
    keys = sorted({(r["ray"], r["distance"]) for r in rows})
    out = []
    for ray, d in keys:
        cell = [r for r in rows if r["ray"] == ray and r["distance"] == d]
        by = {a: np.array([r["test_vrmse"] for r in cell if r["arm"] == a],
                          dtype=float) for a in ARMS}
        if any(len(v) == 0 for v in by.values()):
            continue
        res, dam = by["transplant_resonant"], by["transplant_damped"]
        # paired per seed where possible: same seed means same init and data
        n = min(len(res), len(dam))
        rel = (dam[:n] - res[:n]) / np.where(dam[:n] > 0, dam[:n], np.nan)
        out.append({
            "ray": ray, "distance": d,
            "n_seeds": int(n),
            "scratch": float(by["scratch"].mean()),
            "finetune": float(by["finetune"].mean()),
            "resonant": float(res.mean()),
            "damped": float(dam.mean()),
            "gap_rel": float(np.nanmean(rel)),
            "gap_rel_sd": float(np.nanstd(rel)),
            "resonant_wins": int((res[:n] < dam[:n]).sum()),
            "finetune_vs_scratch": float(
                (by["scratch"].mean() - by["finetune"].mean())
                / by["scratch"].mean()) if by["scratch"].mean() > 0 else float("nan"),
        })
    return out


def print_report(summary: list[dict]) -> None:
    print("\n=== H2 as a dose-response on regime distance ===")
    print("    gap_rel > 0 means the resonant transplant beat the size-matched")
    print("    damped control -- the direction H2 predicts. d=0 is the same")
    print("    regime with a fresh draw and a fresh init: the easiest possible")
    print("    case for transplanting.\n")
    hdr = (f"    {'ray':>8s} {'dist':>6s} {'scratch':>9s} {'finetune':>9s} "
           f"{'resonant':>9s} {'damped':>9s} {'gap_rel':>9s} {'wins':>6s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for s in summary:
        print(f"    {s['ray']:>8s} {s['distance']:>6.2f} {s['scratch']:>9.5f} "
              f"{s['finetune']:>9.5f} {s['resonant']:>9.5f} {s['damped']:>9.5f} "
              f"{s['gap_rel']:>+8.1%} {s['resonant_wins']}/{s['n_seeds']:<4d}")

    g = np.array([s["gap_rel"] for s in summary], dtype=float)
    d = np.array([s["distance"] for s in summary], dtype=float)
    wins = sum(s["resonant_wins"] for s in summary)
    tot = sum(s["n_seeds"] for s in summary)
    print(f"\n    resonant beat damped in {wins}/{tot} paired runs "
          f"(a coin gives {tot/2:.0f})")
    print(f"    gap_rel: mean {g.mean():+.1%}  sd {g.std():.1%}  "
          f"range {g.min():+.1%} to {g.max():+.1%}")

    ok = np.isfinite(g) & np.isfinite(d)
    if ok.sum() >= 3 and np.ptp(d[ok]) > 0:
        rho = mt_spearman(d[ok], g[ok])
        print(f"    Spearman(distance, gap) = {rho:+.3f}   "
              "(H2-as-a-curve predicts negative: closer regimes transplant better)")

    zero = [s for s in summary if s["distance"] == 0.0]
    if zero:
        z = np.mean([s["gap_rel"] for s in zero])
        zw = sum(s["resonant_wins"] for s in zero)
        zn = sum(s["n_seeds"] for s in zero)
        print(f"\n    at d = 0 (same regime): gap {z:+.1%}, resonant wins "
              f"{zw}/{zn}")
        print("    a flat curve including d=0 says the basis is arbitrary, "
              "which is\n    ext21's reading; a positive d=0 decaying with "
              "distance would say\n    ext21 measured the tail of a real effect")

    ft = np.array([s["finetune_vs_scratch"] for s in summary], dtype=float)
    print(f"\n    full fine-tune vs scratch: mean {np.nanmean(ft):+.1%} "
          "(transfer itself, which ext21 found real and large)")


def mt_spearman(x, y) -> float:
    """Spearman via rank Pearson. Ties get average ranks."""
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ranks within ties so a plateau does not get an arbitrary order
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    a, b = rank(x), rank(y)
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--distances", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 1.0, 1.75])
    p.add_argument("--rays", nargs="+", default=["diffuse", "sharp"])
    p.add_argument("--budget", type=int, default=4,
                   help="target trajectories; ext21 saw transfer matter most "
                        "at the smallest budgets")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--source-traj", type=int, default=16)
    p.add_argument("--n-test-traj", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=10)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--source-epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--max-mode", type=int, default=6)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    args = p.parse_args()

    # one source model, reused for every distance: the experiment varies the
    # target, so re-training the source per cell would add noise it cannot
    # distinguish from a distance effect
    print("training source ...", flush=True)
    src_train = make_regime(SOURCE, args.source_traj, args.n_steps, args.size,
                            seed=0)
    source = mt.build(args, seed=0)
    fit(source, src_train, epochs=args.source_epochs, lr=args.lr,
        device=args.device, seed=0)
    base = src_train[0, 0].transpose(2, 0, 1)
    components = mt.classify_components(source, base, args, args.device)
    print(f"  resonant {len(components['resonant'])}, "
          f"damped {len(components['damped'])} components", flush=True)

    # d = 0 is the same regime on every ray, so it is measured once per ray and
    # those runs pool into a single anchor with more seeds behind it -- stated
    # here because otherwise the anchor silently carries twice the sample size
    rows = []
    for ray in args.rays:
        rung = ladder(args.distances, ray)
        for d, params in rung.items():
            got = log_distance(params)
            print(f"\n  [{ray} d={d:.2f} (check {got:.2f}) "
                  f"D={params['diffusion']:.3f} w={params['omega']:.3f}]",
                  flush=True)
            for arm in ARMS:
                for seed in args.seeds:
                    r = run_cell(arm, source, params, args.budget, args,
                                 args.device, seed, components)
                    r.update(ray=ray, distance=d, seed=seed,
                             diffusion=params["diffusion"],
                             omega=params["omega"], budget=args.budget)
                    rows.append(r)
                vals = [x['test_vrmse'] for x in rows
                        if x['ray'] == ray and x['distance'] == d
                        and x['arm'] == arm]
                print(f"      {arm:>20s} {np.mean(vals):.5f}", flush=True)

    summary = summarise(rows)
    print_report(summary)
    write_csv(args.out_dir / "ext34_distance_cells.csv", rows)
    write_csv(args.out_dir / "ext34_distance_summary.csv", summary)


if __name__ == "__main__":
    sys.exit(main())
