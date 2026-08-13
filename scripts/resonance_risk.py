r"""ext20: does the pole readout predict where the surrogate fails? (H1)

SpecScope step 4: "Correlate per-mode pole margin with per-mode rollout error
growth; build a scalar resonance risk score per input regime; show ROC-style
that it flags high-error scenarios before running the model long."

    python3 scripts/resonance_risk.py
    python3 scripts/resonance_risk.py --quick

The hypothesis has a sharp form and a weak form, reported separately because
they are not equally interesting:

  strong   the pole margin predicts *which modes* go wrong, mode by mode,
           inside one trained model
  weak     the risk score orders whole scenarios by how badly the model will do
           on them, without rolling the model out on any of them

H1 as written is the strong form. The weak form is the deployable one: a number
computed from the weights and one input frame, before any long rollout, that
says "do not trust this model here".

The control that makes the correlation mean something
-----------------------------------------------------
Pole margin correlates with wavenumber, and rollout error also correlates with
wavenumber, because high modes are both more damped and harder to predict. A
raw correlation between margin and error is therefore partly just both of them
tracking |k|, and reporting it alone would overstate the case. So the partial
correlation controlling for |k| is reported next to the raw one, along with a
wavenumber-only baseline. If margin adds nothing over "high k is worse", H1 has
not been demonstrated -- that is the "poles do not encode interpretable
dynamics" branch, and it is a result, not a failure of the script.

Scenarios as disaster regimes
-----------------------------
Each scenario is the same PDE family at different parameters, which is this
project's stand-in for a different disaster type in the same way
``cross_regime.py`` uses Gray-Scott's six named regimes. The model is trained on
one and scored on all, so every other scenario is genuinely out of distribution.

Outputs
-------
``results/extensions/ext20_mode_risk.csv``   per-mode margin vs error growth
``results/extensions/ext20_scenarios.csv``   per-scenario risk vs actual error
``results/extensions/ext20_summary.csv``     the two hypotheses, scored
``figures/extensions/ext20_resonance_risk.png``
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch                                                   # noqa: E402

from litefno.models.harmonic import HarmonicLiteFNO            # noqa: E402
from litefno.operator import (                                 # noqa: E402
    empirical_mode_operators, operator_poles, stability_margin)
from litefno.specscope import (                                # noqa: E402
    fit, one_step_vrmse, rollout_mode_error)
from litefno.systems import (                                  # noqa: E402
    rotating_diffusion, rotating_diffusion_pole, split_trajectories)

RESULTS = _ROOT / "results" / "extensions"
FIGURES = _ROOT / "figures" / "extensions"

DT, LENGTH = 0.25, 32.0

# The training regime first, then progressively less like it. Diffusion sets how
# strongly high modes are damped and omega sets the oscillation rate, so this
# sweep moves both the difficulty and the pole structure, which is what "a
# different disaster type" has to mean for the question to be non-trivial.
SCENARIOS = [("train_regime", dict(diffusion=0.4, omega=0.6))] + [
    (f"D{d:g}_w{w:g}", dict(diffusion=d, omega=w))
    for d in (0.05, 0.1, 0.4, 1.2, 1.6)
    for w in (0.15, 0.3, 0.6, 1.2)
    if not (d == 0.4 and w == 0.6)
]
# Twenty scenarios rather than a handful, because the weak form is scored with a
# rank correlation and an AUC and both are close to meaningless at n = 7: with
# four positives, one misordering moves the AUC by 0.08 and no amount of
# reporting precision fixes that. The grid also crosses damping and frequency
# independently, so the score cannot succeed by tracking one of them alone.


# --------------------------------------------------------------------------
# statistics, kept local so the script needs no scipy
# --------------------------------------------------------------------------


def rank(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(len(v), dtype=float)
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            sel = inverse == i
            ranks[sel] = ranks[sel].mean()
    return ranks


def spearman(x, y) -> float:
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def partial_spearman(x, y, control) -> float:
    """Rank correlation of x and y with ``control`` regressed out of both.

    Without this, a correlation between pole margin and rollout error is partly
    both of them following |k|. Removing the linear part of each on the control
    ranks leaves whatever the margin knows that wavenumber alone does not.
    """
    rx, ry, rc = rank(x), rank(y), rank(control)
    centred = rc - rc.mean()
    denom_c = max(float((centred ** 2).sum()), 1e-30)

    def residual(v):
        beta = float((v - v.mean()) @ centred) / denom_c
        return v - v.mean() - beta * centred

    ex, ey = residual(rx), residual(ry)
    denom = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / denom) if denom > 0 else float("nan")


def auc(scores, labels) -> float:
    """Area under the ROC curve, via the rank-sum identity.

    Returns nan rather than 0.5 when one class is empty: 0.5 reads as "no skill
    measured on a real test" and this case is "no test was possible".
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rank(scores)
    return float((r[labels].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------


def train_reference(args, device: str, seed: int = 0):
    traj = rotating_diffusion(n_traj=args.n_traj, n_steps=args.n_steps,
                              size=args.size, dt=DT, length=LENGTH, seed=0,
                              **SCENARIOS[0][1])
    splits = split_trajectories(traj, seed=0)
    # Seed before constructing: the initialization draws from the global RNG,
    # so seeding only inside fit() leaves the weights different every run. Two
    # identical invocations of this script disagreed on the partial correlation
    # by -0.257 (p = 0.046) versus -0.083 (p = 0.56) before this was pinned --
    # opposite conclusions from the same command, which is also why the study
    # now runs over several seeds instead of trusting one.
    torch.manual_seed(seed)
    model = HarmonicLiteFNO(2, 2, width=args.width, modes=args.modes,
                            layers=args.layers, rank=args.rank)
    t0 = time.time()
    fit(model, splits["train"], epochs=args.epochs, lr=args.lr, device=device,
        seed=seed)
    vrmse = one_step_vrmse(model, splits["test"], device)
    print(f"reference model trained in {time.time() - t0:.0f}s   "
          f"one-step VRMSE {vrmse:.5f}", flush=True)
    return model, splits, vrmse


def mode_level(model, splits, args, device: str) -> list[dict]:
    """H1 strong form: margin vs error growth, mode by mode."""
    base = splits["test"][0, 0].transpose(2, 0, 1)
    probed = empirical_mode_operators(model, base, max_mode=args.max_mode,
                                      eps=args.eps, device=device)
    margin = stability_margin(operator_poles(probed["operators"])["sigma"])
    err = rollout_mode_error(model, splits["test"], horizon=args.horizon,
                             max_mode=args.max_mode, device=device)

    # The two live on different mode lists -- the probe walks signed ky while
    # the rollout reads the rfft grid -- so they are matched on the signed pair
    # rather than by position. Zipping them directly would silently pair mode
    # (1, 0) with (-1, 0) and the correlation would be of noise.
    index = {(int(a), int(b)): i for i, (a, b)
             in enumerate(zip(err["ky"], err["kx"]))}
    rows = []
    for i, (a, b) in enumerate(zip(probed["ky"], probed["kx"])):
        j = index.get((int(a), int(b)))
        if j is None:
            continue
        exact = rotating_diffusion_pole(np.hypot(a, b),
                                        SCENARIOS[0][1]["diffusion"],
                                        SCENARIOS[0][1]["omega"], DT, LENGTH)
        rows.append({
            "ky": int(a), "kx": int(b), "radius": float(np.hypot(a, b)),
            "pole_margin": float(margin[i]),
            "exact_margin": float(np.log(abs(exact))),
            "error_growth": float(err["growth"][j]),
            "final_error": float(err["error"][-1, j]),
        })
    return rows


def scenario_level(model, args, device: str) -> list[dict]:
    """H1 weak form: one score per regime, computed without a long rollout."""
    rows = []
    for name, params in SCENARIOS:
        traj = rotating_diffusion(n_traj=max(4, args.n_traj // 2),
                                  n_steps=args.n_steps, size=args.size,
                                  dt=DT, length=LENGTH, seed=7, **params)
        # Probe at a state from the middle of the trajectory, not step 0. Every
        # scenario here is seeded identically and its initial condition is
        # generated before any physics is applied, so step 0 is the *same field*
        # in all of them -- probing there gave all seven scenarios an identical
        # risk score, which looked like a null result and was an artifact of
        # where the frame was taken.
        mid = traj.shape[1] // 2
        base = traj[0, mid].transpose(2, 0, 1)
        probed = empirical_mode_operators(model, base, max_mode=args.max_mode,
                                          eps=args.eps, device=device)
        margin = stability_margin(operator_poles(probed["operators"])["sigma"])

        # Weight each mode by how much of *this* scenario's energy it holds. A
        # mode the operator would amplify does not matter if the scenario never
        # excites it, which is why the score is per-regime rather than a
        # property of the weights alone. Taken over the whole trajectory for the
        # same reason as above: the scenarios differ in how they redistribute
        # energy over time, not in where they start.
        spec = np.fft.rfft2(traj.transpose(0, 1, 4, 2, 3), axes=(-2, -1))
        height = traj.shape[2]
        energy = np.array([
            float((np.abs(spec[:, :, :, int(a) % height, int(b)]) ** 2).sum())
            for a, b in zip(probed["ky"], probed["kx"])])
        energy = energy / max(energy.sum(), 1e-30)

        err = rollout_mode_error(model, traj, horizon=args.horizon,
                                 max_mode=args.max_mode, device=device)
        rows.append({
            "scenario": name, **params,
            "risk_max": float(margin.max()),
            "risk_weighted": float((margin * energy).sum()),
            "onestep_vrmse": one_step_vrmse(model, traj, device),
            "rollout_error": float(err["error"][-1].mean()),
            "rollout_growth": float(err["growth"].mean()),
        })
        print(f"  {name:<14} risk {rows[-1]['risk_weighted']:+.4f}   "
              f"rollout {rows[-1]['rollout_error']:.4f}", flush=True)
    return rows


def permutation_p(statistic, x, y, n_draws: int = 20000, seed: int = 0) -> float:
    """Two-sided p-value for a rank statistic, by shuffling the labels.

    The scenario-level test has a handful of points and a statistic with no
    convenient null distribution, so the null is built by shuffling: how often
    does a random pairing of risk scores to errors do at least as well? Without
    this the weak form is a suggestive number with no way to tell it from luck,
    and at these sample sizes luck is a live explanation.
    """
    rng = np.random.default_rng(seed)
    observed = statistic(x, y)
    if not np.isfinite(observed):
        return float("nan")
    y = np.asarray(y)
    hits = 0
    for _ in range(n_draws):
        value = statistic(x, rng.permutation(y))
        if np.isfinite(value) and abs(value) >= abs(observed):
            hits += 1
    # the +1s make the floor 1/(n_draws + 1) rather than zero: no finite number
    # of shuffles can show that a permutation is impossible, and rounding the
    # floor to "p = 0.0" would claim exactly that
    return float((hits + 1) / (n_draws + 1))


def format_p(value: float) -> str:
    """A p-value at the resolution the permutation count can actually support."""
    if not np.isfinite(value):
        return "n/a"
    return f"<{value:.1g}" if value <= 1e-3 else f"{value:.4f}"


def score(mode_rows: list[dict], scen_rows: list[dict]) -> dict:
    margin = [r["pole_margin"] for r in mode_rows]
    growth = [r["error_growth"] for r in mode_rows]
    radius = [r["radius"] for r in mode_rows]
    err = np.array([r["rollout_error"] for r in scen_rows])
    high = err > np.median(err)
    risk = np.array([r["risk_weighted"] for r in scen_rows])
    return {
        "n_modes": len(mode_rows),
        "mode_spearman_partial_p": permutation_p(
            lambda a, b: partial_spearman(a, b, radius), margin, growth,
            n_draws=2000),
        "scenario_spearman_p": permutation_p(spearman, risk, err,
                                             n_draws=20000),
        "scenario_auc_p": permutation_p(
            lambda s, lab: auc(s, lab > np.median(lab)), risk, err,
            n_draws=20000),
        "mode_spearman_raw": round(spearman(margin, growth), 4),
        "mode_spearman_partial_k": round(
            partial_spearman(margin, growth, radius), 4),
        "wavenumber_baseline_spearman": round(spearman(radius, growth), 4),
        "extracted_vs_exact_margin": round(
            spearman(margin, [r["exact_margin"] for r in mode_rows]), 4),
        "n_scenarios": len(scen_rows),
        "scenario_spearman_weighted": round(
            spearman([r["risk_weighted"] for r in scen_rows], err), 4),
        "scenario_spearman_max": round(
            spearman([r["risk_max"] for r in scen_rows], err), 4),
        "scenario_auc_weighted": round(
            auc([r["risk_weighted"] for r in scen_rows], high), 4),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(_ROOT)}")


def plot(mode_rows, scen_rows, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    sc = ax.scatter([r["pole_margin"] for r in mode_rows],
                    [r["error_growth"] for r in mode_rows],
                    c=[r["radius"] for r in mode_rows], cmap="viridis", s=22)
    fig.colorbar(sc, ax=ax, label="|k|")
    ax.set(xlabel="pole margin (log |z| of least-damped pole)",
           ylabel="rollout error growth (per step)",
           title="H1 strong form: mode by mode")

    ax = axes[1]
    ax.scatter([r["radius"] for r in mode_rows],
               [r["error_growth"] for r in mode_rows], s=22, color="tab:gray")
    ax.set(xlabel="|k|", ylabel="rollout error growth",
           title="the control: wavenumber alone")

    ax = axes[2]
    for row in scen_rows:
        ax.scatter(row["risk_weighted"], row["rollout_error"], s=45)
        ax.annotate(row["scenario"],
                    (row["risk_weighted"], row["rollout_error"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set(xlabel="resonance risk score (energy-weighted margin)",
           ylabel="rollout error", title="H1 weak form: per scenario")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png.relative_to(_ROOT)}")


def aggregate(summaries: list[dict]) -> dict:
    """Mean and spread of each statistic across seeds.

    Reported instead of a single run because a single run turned out not to be
    stable: the quantity H1 rests on moved by more than its own effect size
    between two seeds. A mean with a spread beside it is the honest form of a
    result like that, and if the spread covers zero then the effect is not
    there, however good the best seed looked.
    """
    out = {"n_seeds": len(summaries)}
    for key in summaries[0]:
        values = [s[key] for s in summaries]
        if not all(isinstance(v, (int, float)) and np.isfinite(v)
                   for v in values):
            continue
        out[f"{key}_mean"] = round(float(np.mean(values)), 4)
        out[f"{key}_std"] = round(float(np.std(values)), 4)
        out[f"{key}_min"] = round(float(np.min(values)), 4)
        out[f"{key}_max"] = round(float(np.max(values)), 4)
    return out


def print_aggregate(agg: dict) -> None:
    print("\n" + "=" * 74)
    print(f"ext20  H1 across {agg['n_seeds']} seeds  (mean +/- sd [min, max])")
    print("=" * 74)
    rows = [
        ("strong: margin vs growth, raw    ", "mode_spearman_raw"),
        ("strong: ... controlling for |k|  ", "mode_spearman_partial_k"),
        ("strong: |k|-only baseline        ", "wavenumber_baseline_spearman"),
        ("extracted vs exact margin        ", "extracted_vs_exact_margin"),
        ("weak:   risk vs error (weighted) ", "scenario_spearman_weighted"),
        ("weak:   AUC, worse half          ", "scenario_auc_weighted"),
    ]
    for label, key in rows:
        if f"{key}_mean" not in agg:
            continue
        print(f"  {label} {agg[f'{key}_mean']:+.4f} +/- {agg[f'{key}_std']:.4f}"
              f"   [{agg[f'{key}_min']:+.4f}, {agg[f'{key}_max']:+.4f}]")
    lo, hi = (agg.get("mode_spearman_partial_k_min"),
              agg.get("mode_spearman_partial_k_max"))
    if lo is not None and lo <= 0 <= hi:
        print("\n  the strong form's range spans zero: after controlling for "
              "wavenumber\n  the pole margin adds no consistent signal")


def print_report(summary: dict) -> None:
    print("\n" + "=" * 74)
    print(f"ext20  H1: does pole structure predict failure?  (seed "
          f"{summary.get('seed', 0)})")
    print("=" * 74)
    print(f"\nstrong form, {summary['n_modes']} modes")
    print(f"  margin vs error growth, raw      "
          f"{summary['mode_spearman_raw']:+.4f}")
    print(f"  ... controlling for wavenumber   "
          f"{summary['mode_spearman_partial_k']:+.4f}"
          f"   (p {format_p(summary['mode_spearman_partial_p'])})")
    print(f"  wavenumber-only baseline         "
          f"{summary['wavenumber_baseline_spearman']:+.4f}")
    print(f"  extracted vs exact margin        "
          f"{summary['extracted_vs_exact_margin']:+.4f}")
    print(f"\nweak form, {summary['n_scenarios']} scenarios")
    print(f"  risk vs rollout error (weighted) "
          f"{summary['scenario_spearman_weighted']:+.4f}"
          f"   (p {format_p(summary['scenario_spearman_p'])})")
    print(f"  risk vs rollout error (max)      "
          f"{summary['scenario_spearman_max']:+.4f}")
    print(f"  AUC flagging the worse half      "
          f"{summary['scenario_auc_weighted']:.4f}"
          f"   (p {format_p(summary['scenario_auc_p'])})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-traj", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=10)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--max-mode", type=int, default=6)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.quick:
        args.n_traj, args.n_steps, args.epochs = 8, 16, 8
        args.width, args.layers, args.max_mode, args.horizon = 16, 2, 4, 8
        args.seeds = args.seeds[:1]

    all_modes, all_scen, summaries = [], [], []
    for seed in args.seeds:
        print(f"\n----- seed {seed} -----")
        model, splits, _ = train_reference(args, args.device, seed=seed)
        mode_rows = mode_level(model, splits, args, args.device)
        print("\nscenarios:")
        scen_rows = scenario_level(model, args, args.device)
        summary = dict(seed=seed, **score(mode_rows, scen_rows))
        print_report(summary)
        all_modes.extend(dict(seed=seed, **r) for r in mode_rows)
        all_scen.extend(dict(seed=seed, **r) for r in scen_rows)
        summaries.append(summary)

    agg = aggregate(summaries)
    write_csv(RESULTS / "ext20_mode_risk.csv", all_modes)
    write_csv(RESULTS / "ext20_scenarios.csv", all_scen)
    write_csv(RESULTS / "ext20_summary.csv", summaries)
    write_csv(RESULTS / "ext20_across_seeds.csv", [agg])
    # the figure shows the first seed; the CSVs carry all of them
    first = args.seeds[0]
    plot([r for r in all_modes if r["seed"] == first],
         [r for r in all_scen if r["seed"] == first],
         FIGURES / "ext20_resonance_risk.png")
    print_aggregate(agg)


if __name__ == "__main__":
    main()
