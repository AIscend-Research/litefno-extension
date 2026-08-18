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

SOURCE = dict(diffusion=0.4, omega=0.6)          # ext21's source, unchanged

# The dose axis, and the reason this experiment needed one. ext21 transplanted
# the matched resonant/damped sets -- 3 components -- and read a null. But 3 CP
# components carry the mode-axis factors and rank weight of 3 of 8 ranks across
# 4 spectral layers: 252 of the model's 7,106 parameters, or **3.55%**. The
# fine-tune ceiling it was implicitly compared against moves 100%. A 2e-5 move
# from 3.55% of the weights is not evidence that the subspace carries nothing;
# it is evidence that the dose was small. Varying dose is what separates those.
#
# The matched sets cap at 3 (the classifier finds 3 resonant and 5 damped, and
# ext21 trims to the smaller), so the matched ladder cannot go past k = 3. The
# ceiling is supplied instead by `transplant_all`, which copies **every** rank
# component's mode structure -- the entire claimed shared physics, 9.3% of the
# spectral parameters -- and has no size-matched control by construction. It is
# a dose ceiling, not an H2 test, and is labelled as one.
DOSES = (1, 2, 3)


def params_moved(model, n_components: int) -> dict:
    """How much of the network a k-component transplant actually writes.

    The number the blocked first run of this experiment was missing. Without it
    a null transplant and a null *dose* are indistinguishable.
    """
    total = sum(p.numel() for p in model.parameters())
    spectral = sum(p.numel() for L in model.spectral_layers
                   for p in L.parameters())
    per = sum(L.factor_m1.shape[0] + L.factor_m2.shape[0] + 1
              for L in model.spectral_layers)
    moved = per * n_components
    return {"params_moved": int(moved),
            "frac_of_model": moved / total,
            "frac_of_spectral": moved / spectral}


def arm_name(kind: str, dose: int) -> str:
    return kind if kind in ("scratch", "finetune", "transplant_all") \
        else f"{kind}_{dose}"


def build_arms(doses) -> list[tuple]:
    """(kind, dose) pairs. Order puts the controls first for readable logs."""
    arms = [("scratch", 0), ("finetune", -1), ("transplant_all", -2)]
    for k in doses:
        arms.append(("transplant_resonant", k))
        arms.append(("transplant_damped", k))
    return arms


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
             device: str, seed: int, components: dict, dose: int = 0) -> dict:
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
    elif arm == "transplant_all":
        info = transplant(model, source_model, range(args.rank))
    elif arm == "transplant_resonant":
        info = transplant(model, source_model, components["resonant"][:dose])
    elif arm == "transplant_damped":
        info = transplant(model, source_model, components["damped"][:dose])
    moved = params_moved(model, info["n_components"])
    if arm == "finetune":
        # a warm start writes every weight, which is the point of having it as
        # the ceiling: 100% dose, no freezing
        total = sum(q.numel() for q in model.parameters())
        moved = {"params_moved": total, "frac_of_model": 1.0,
                 "frac_of_spectral": 1.0}
    t0 = time.time()
    fit(model, train, epochs=args.epochs, lr=args.lr, device=device, seed=seed)
    row = {"arm": arm_name(arm, dose), "kind": arm, "dose": dose,
           "n_components": info["n_components"],
           "params_moved": moved["params_moved"],
           "frac_of_model": round(moved["frac_of_model"], 5),
           "test_vrmse": float(one_step_vrmse(model, test, device)),
           "train_s": round(time.time() - t0, 1)}
    for h in info.get("handles", []):
        h.remove()
    return row


def summarise(rows: list[dict]) -> list[dict]:
    """Per (ray, distance, dose): resonant vs its size-matched damped control.

    The gap is signed so **positive means the resonant transplant is better**
    (lower error), which is the direction H2 predicts, and is expressed relative
    to the damped control so cells of different absolute difficulty stay
    comparable. Pairing is per seed: same seed means same initialisation and
    same target draw, so the difference is the transplanted subspace and
    nothing else.
    """
    keys = sorted({(r["ray"], r["distance"], r["dose"]) for r in rows
                   if r["kind"] == "transplant_resonant"})
    out = []
    for ray, d, dose in keys:
        def pick(kind, k=None):
            return np.array([r["test_vrmse"] for r in rows
                             if r["ray"] == ray and r["distance"] == d
                             and r["kind"] == kind
                             and (k is None or r["dose"] == k)], dtype=float)
        res, dam = pick("transplant_resonant", dose), pick("transplant_damped", dose)
        scratch, ft = pick("scratch"), pick("finetune")
        allc = pick("transplant_all")
        if len(res) == 0 or len(dam) == 0 or len(scratch) == 0:
            continue
        n = min(len(res), len(dam))
        rel = (dam[:n] - res[:n]) / np.where(dam[:n] > 0, dam[:n], np.nan)
        moved = [r["frac_of_model"] for r in rows
                 if r["kind"] == "transplant_resonant" and r["dose"] == dose]
        out.append({
            "ray": ray, "distance": d, "dose": dose,
            "frac_of_model": float(np.mean(moved)) if moved else float("nan"),
            "n_seeds": int(n),
            "scratch": float(scratch.mean()),
            "finetune": float(ft.mean()) if len(ft) else float("nan"),
            "transplant_all": float(allc.mean()) if len(allc) else float("nan"),
            "resonant": float(res.mean()),
            "damped": float(dam.mean()),
            "gap_rel": float(np.nanmean(rel)),
            "gap_rel_sd": float(np.nanstd(rel)),
            "resonant_wins": int((res[:n] < dam[:n]).sum()),
            "finetune_vs_scratch": (
                float((scratch.mean() - ft.mean()) / scratch.mean())
                if len(ft) and scratch.mean() > 0 else float("nan")),
            "all_vs_scratch": (
                float((scratch.mean() - allc.mean()) / scratch.mean())
                if len(allc) and scratch.mean() > 0 else float("nan")),
        })
    return out


def print_report(summary: list[dict]) -> None:
    print("\n=== H2 as a dose-response on regime distance and on dose ===")
    print("    gap_rel > 0 means the resonant transplant beat its size-matched")
    print("    damped control -- the direction H2 predicts. d=0 is the same")
    print("    regime with a fresh draw and a fresh init: the easiest possible")
    print("    case for transplanting.\n")

    print("  A. the dose axis: does copying more of the operator do more?")
    print(f"    {'dose':>5s} {'%model':>7s} {'scratch':>9s} {'resonant':>9s} "
          f"{'damped':>9s} {'gap_rel':>9s} {'wins':>8s}")
    doses = sorted({s["dose"] for s in summary})
    for k in doses:
        sub = [s for s in summary if s["dose"] == k]
        w = sum(s["resonant_wins"] for s in sub)
        n = sum(s["n_seeds"] for s in sub)
        print(f"    {k:5d} {np.mean([s['frac_of_model'] for s in sub]):6.2%} "
              f"{np.mean([s['scratch'] for s in sub]):9.5f} "
              f"{np.mean([s['resonant'] for s in sub]):9.5f} "
              f"{np.mean([s['damped'] for s in sub]):9.5f} "
              f"{np.mean([s['gap_rel'] for s in sub]):>+8.1%} {w:4d}/{n:<4d}")
    one = [s for s in summary if s["dose"] == doses[-1]]
    print(f"\n    ceilings, averaged over every cell:")
    print(f"      scratch          {np.mean([s['scratch'] for s in one]):.5f}")
    print(f"      transplant_all   "
          f"{np.nanmean([s['transplant_all'] for s in one]):.5f}   "
          f"(all {len(doses) and ''}rank components frozen, no matched control)")
    print(f"      finetune         "
          f"{np.nanmean([s['finetune'] for s in one]):.5f}   (100% dose)")
    print(f"      all_vs_scratch   "
          f"{np.nanmean([s['all_vs_scratch'] for s in one]):+.1%}")
    print(f"      ft_vs_scratch    "
          f"{np.nanmean([s['finetune_vs_scratch'] for s in one]):+.1%}")

    print("\n  B. the distance axis, at the largest matched dose")
    hdr = (f"    {'ray':>8s} {'dist':>6s} {'scratch':>9s} {'finetune':>9s} "
           f"{'resonant':>9s} {'damped':>9s} {'gap_rel':>9s} {'wins':>6s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for s in sorted(one, key=lambda s: (s["ray"], s["distance"])):
        print(f"    {s['ray']:>8s} {s['distance']:>6.2f} {s['scratch']:>9.5f} "
              f"{s['finetune']:>9.5f} {s['resonant']:>9.5f} {s['damped']:>9.5f} "
              f"{s['gap_rel']:>+8.1%} {s['resonant_wins']}/{s['n_seeds']:<4d}")

    g = np.array([s["gap_rel"] for s in summary], dtype=float)
    d = np.array([s["distance"] for s in summary], dtype=float)
    k = np.array([s["dose"] for s in summary], dtype=float)
    wins = sum(s["resonant_wins"] for s in summary)
    tot = sum(s["n_seeds"] for s in summary)
    print(f"\n    resonant beat damped in {wins}/{tot} paired runs "
          f"(a coin gives {tot/2:.0f})")
    print(f"    gap_rel: mean {np.nanmean(g):+.1%}  sd {np.nanstd(g):.1%}  "
          f"range {np.nanmin(g):+.1%} to {np.nanmax(g):+.1%}")

    ok = np.isfinite(g) & np.isfinite(d)
    if ok.sum() >= 3 and np.ptp(d[ok]) > 0:
        print(f"    Spearman(distance, gap) = {mt_spearman(d[ok], g[ok]):+.3f}"
              "   (H2-as-a-curve predicts negative)")
    if ok.sum() >= 3 and np.ptp(k[ok]) > 0:
        print(f"    Spearman(dose, gap)     = {mt_spearman(k[ok], g[ok]):+.3f}"
              "   (a real subspace predicts positive)")

    zero = [s for s in summary if s["distance"] == 0.0]
    if zero:
        z = np.nanmean([s["gap_rel"] for s in zero])
        zw = sum(s["resonant_wins"] for s in zero)
        zn = sum(s["n_seeds"] for s in zero)
        print(f"\n    at d = 0 (same regime): gap {z:+.1%}, resonant wins "
              f"{zw}/{zn}")
        print("    a flat surface in BOTH dose and distance says the basis is")
        print("    arbitrary, which is ext21's reading. A positive d=0 decaying")
        print("    with distance, or a gap growing with dose, would say ext21")
        print("    measured the tail of a real effect at too small a dose.")


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
    p.add_argument("--doses", type=int, nargs="+", default=list(DOSES),
                   help="components transplanted per arm; capped at the "
                        "matched-set size, with transplant_all as the ceiling")
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
    doses = [k for k in args.doses if k <= components["n_matched"]]
    dropped = [k for k in args.doses if k > components["n_matched"]]
    arms = build_arms(doses)
    probe = mt.build(args, seed=0)
    print(f"  resonant {len(components['resonant'])}, "
          f"damped {len(components['damped'])} components; "
          f"matched at {components['n_matched']}", flush=True)
    for k in doses + [args.rank]:
        m = params_moved(probe, k)
        print(f"    dose {k}: {m['params_moved']:5d} params "
              f"= {m['frac_of_model']:.2%} of the model, "
              f"{m['frac_of_spectral']:.2%} of the spectral layers", flush=True)
    if dropped:
        print(f"    (doses {dropped} dropped: the size-matched sets cap at "
              f"{components['n_matched']}; transplant_all covers the ceiling)",
              flush=True)

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
            for kind, dose in arms:
                for seed in args.seeds:
                    r = run_cell(kind, source, params, args.budget, args,
                                 args.device, seed, components, dose=dose)
                    r.update(ray=ray, distance=d, seed=seed,
                             diffusion=params["diffusion"],
                             omega=params["omega"], budget=args.budget)
                    rows.append(r)
                name = arm_name(kind, dose)
                vals = [x['test_vrmse'] for x in rows
                        if x['ray'] == ray and x['distance'] == d
                        and x['arm'] == name]
                print(f"      {name:>22s} {np.mean(vals):.5f}", flush=True)

    summary = summarise(rows)
    print_report(summary)
    write_csv(args.out_dir / "ext34_distance_cells.csv", rows)
    write_csv(args.out_dir / "ext34_distance_summary.csv", summary)


if __name__ == "__main__":
    sys.exit(main())
