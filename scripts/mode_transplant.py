r"""ext21: are the learned spectral factors shared physics? (H2)

SpecScope step 5: "Train on disaster type A. Freeze the stable+resonant factors,
re-initialize the rest, train on disaster type B with few trajectories. Compare
four arms: full fine-tune, transplant-stable+resonant, transplant-damped-only
(the control that must fail), from-scratch. Report accuracy vs training-set size
curves. Also report a mode-overlap matrix (principal angles between learned
bases across disaster types)."

    python3 scripts/mode_transplant.py
    python3 scripts/mode_transplant.py --quick

The asymmetry is the experiment
-------------------------------
"Transplanting helps" on its own is a weak claim: any warm start helps, and the
resonant factors are also simply *more* of the source model. What would make the
mode classification physically meaningful is the difference between the two
transplants -- resonant factors carrying across regimes while damped factors do
not, at matched component counts. So the control arm is matched in size, not
just present, and the headline number is the gap between the two rather than
either one's absolute error.

If both transplants help equally, the classification is not tracking physics and
the honest conclusion is that the win is a warm start. If neither helps over
from-scratch, the learned bases are regime-specific. Both are reported as
results.

What "freezing" means here, and what it costs
---------------------------------------------
CP factorization does not store one weight per mode: the rank components are
shared across the whole mode grid, so "transplant the resonant modes" cannot be
done by copying rows. What is transplantable is the components whose footprint
sits mostly on the selected modes, which is an approximation, and at low rank
the components may not separate by mode at all. The script reports how many
components each selection caught; if that count is 0 or the full rank, the arms
have collapsed into each other and the comparison is void, which is stated
rather than left for a reader to infer from suspiciously equal numbers.

Outputs
-------
``results/extensions/ext21_transplant.csv``    arm x budget x seed
``results/extensions/ext21_overlap.csv``       principal-angle overlap matrix
``results/extensions/ext21_summary.csv``       the asymmetry, per budget
``figures/extensions/ext21_mode_transplant.png``
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
    empirical_mode_operators, mode_basis, operator_poles, principal_angles,
    stability_margin, subspace_overlap)
from litefno.specscope import (                                # noqa: E402
    fit, one_step_vrmse, partition_rank_components, transplant)
from litefno.systems import (                                  # noqa: E402
    rotating_diffusion, split_trajectories)

RESULTS = _ROOT / "results" / "extensions"
FIGURES = _ROOT / "figures" / "extensions"

DT, LENGTH = 0.25, 32.0

# The regimes, in the same spirit as cross_regime.py's six Gray-Scott patterns:
# same PDE family, different parameters, different dynamics. The source is where
# the physics is learned; the targets are what it has to carry to.
REGIMES = {
    "source": dict(diffusion=0.4, omega=0.6),
    "slow_diffuse": dict(diffusion=1.2, omega=0.2),
    "fast_sharp": dict(diffusion=0.1, omega=1.4),
}
TARGETS = ["slow_diffuse", "fast_sharp"]


def make(regime: str, n_traj: int, n_steps: int, size: int, seed: int):
    return rotating_diffusion(n_traj=n_traj, n_steps=n_steps, size=size,
                              dt=DT, length=LENGTH, seed=seed,
                              **REGIMES[regime])


def build(args, seed: int) -> HarmonicLiteFNO:
    torch.manual_seed(seed)
    return HarmonicLiteFNO(2, 2, width=args.width, modes=args.modes,
                           layers=args.layers, rank=args.rank)


# --------------------------------------------------------------------------
# which components are which
# --------------------------------------------------------------------------


def classify_components(model, base, args, device: str) -> dict:
    """Split the source model's rank components into resonant and damped sets.

    The split runs through the pole readout, which is the point: H1's classifier
    chooses what H2 transplants, so if the transplant asymmetry appears it is
    evidence that the classification tracked something real, and if it does not,
    that is evidence against the classification rather than against transplants
    in general.

    The two sets are then trimmed to the same size. Without that, "resonant
    helps more" could just mean "resonant was the bigger set", and the whole
    comparison would be measuring how much of the source model got copied.
    """
    probed = empirical_mode_operators(model, base, max_mode=args.max_mode,
                                      eps=args.eps, device=device)
    margin = stability_margin(operator_poles(probed["operators"])["sigma"])
    order = np.argsort(margin)[::-1]          # least damped first
    half = max(1, len(order) // 2)
    resonant_modes, damped_modes = order[:half], order[-half:]

    split = partition_rank_components(
        model,
        (probed["ky"][resonant_modes], probed["kx"][resonant_modes]),
        (probed["ky"][damped_modes], probed["kx"][damped_modes]),
        layer=0)
    resonant, damped = split["a"], split["b"]

    # Matched sizes, so "resonant helps more" cannot just mean "resonant was the
    # bigger set". Each side keeps its most one-sided components, which are the
    # ones the split actually identified rather than merely assigned.
    confidence = np.abs(split["frac_a"] - 0.5)
    resonant = resonant[np.argsort(confidence[resonant])[::-1]]
    damped = damped[np.argsort(confidence[damped])[::-1]]
    size = min(len(resonant), len(damped))
    return {"resonant": np.sort(resonant[:size]),
            "damped": np.sort(damped[:size]),
            "n_matched": int(size), "rank": int(args.rank),
            "n_resonant_found": int(len(resonant)),
            "n_damped_found": int(len(damped)),
            "split_margin": round(split["margin"], 4)}


# --------------------------------------------------------------------------
# the four arms
# --------------------------------------------------------------------------


def run_arm(arm: str, source_model, target: str, budget: int, args,
            device: str, seed: int, components: dict) -> dict:
    train = make(target, budget, args.n_steps, args.size, seed=100 + seed)
    test = make(target, args.n_test_traj, args.n_steps, args.size, seed=999)

    model = build(args, seed)
    info = {"n_components": 0}
    if arm == "finetune":
        model.load_state_dict(source_model.state_dict())
    elif arm == "transplant_resonant":
        info = transplant(model, source_model, components["resonant"])
    elif arm == "transplant_damped":
        info = transplant(model, source_model, components["damped"])
    elif arm != "scratch":
        raise ValueError(arm)

    t0 = time.time()
    fit(model, train, epochs=args.epochs, lr=args.lr, device=device, seed=seed)
    row = {"arm": arm, "target": target, "budget": budget, "seed": seed,
           "n_components": info["n_components"],
           "test_vrmse": one_step_vrmse(model, test, device),
           "train_s": round(time.time() - t0, 1)}
    for handle in info.get("handles", []):
        handle.remove()
    return row


def overlap_matrix(models: dict, args) -> list[dict]:
    """Principal-angle overlap between every pair of independently trained models.

    The signature figure: if the spectral bases learned on different regimes
    overlap, there is shared structure to transplant, and if they are near
    orthogonal there is not. Same-regime pairs trained from different seeds are
    included as the reference -- they set how much overlap two models get from
    agreeing about the *task* rather than about the physics, and any
    cross-regime number has to be read against that, not against zero.
    """
    rows = []
    names = list(models)
    for i, a in enumerate(names):
        for b in names[i:]:
            for layer in range(min(args.layers, args.overlap_layers)):
                theta = principal_angles(mode_basis(models[a], layer),
                                         mode_basis(models[b], layer))
                rows.append({
                    "a": a, "b": b, "layer": layer,
                    "overlap": round(subspace_overlap(
                        mode_basis(models[a], layer),
                        mode_basis(models[b], layer)), 4),
                    "min_angle_deg": round(float(np.degrees(theta.min())), 2),
                    "median_angle_deg": round(
                        float(np.degrees(np.median(theta))), 2)})
    return rows


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    targets = dict.fromkeys(r["target"] for r in rows)
    budgets = sorted({r["budget"] for r in rows})
    for target in targets:
        for budget in budgets:
            sel = [r for r in rows if r["target"] == target
                   and r["budget"] == budget]
            by_arm = {}
            for arm in ("scratch", "finetune", "transplant_resonant",
                        "transplant_damped"):
                vals = [r["test_vrmse"] for r in sel if r["arm"] == arm]
                if vals:
                    by_arm[arm] = (float(np.mean(vals)), float(np.std(vals)))
            if len(by_arm) < 4:
                continue
            res, dam = by_arm["transplant_resonant"], by_arm["transplant_damped"]
            row = {"target": target, "budget": budget,
                   "n_seeds": len(sel) // 4}
            for arm, (mean, sd) in by_arm.items():
                row[f"{arm}_vrmse"] = round(mean, 5)
                row[f"{arm}_sd"] = round(sd, 5)
            # the headline: does the resonant set beat the size-matched damped
            # set, and does either beat starting over
            row["asymmetry"] = round(dam[0] - res[0], 5)
            # Effect size, but only where a spread exists to divide by. With a
            # single seed the standard deviation is exactly zero and any floor
            # turns a 1e-5 difference into a six-figure "effect", which is how
            # the first run of this script reported 4 of 4 cells as decisive on
            # differences in the fifth decimal place.
            spread = float(np.hypot(res[1], dam[1]))
            row["asymmetry_over_sd"] = round((dam[0] - res[0]) / spread, 3) \
                if spread > 1e-9 else float("nan")
            row["resonant_vs_scratch"] = round(
                by_arm["scratch"][0] - res[0], 5)
            out.append(row)
    return out


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


def plot(summary: list[dict], overlap: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = list(dict.fromkeys(r["target"] for r in summary))
    fig, axes = plt.subplots(1, len(targets) + 1,
                             figsize=(5.2 * (len(targets) + 1), 4.4))
    styles = {"scratch": ("tab:gray", "o", "from scratch"),
              "finetune": ("tab:blue", "s", "full fine-tune"),
              "transplant_resonant": ("tab:red", "^", "transplant resonant"),
              "transplant_damped": ("tab:orange", "v", "transplant damped")}

    for ax, target in zip(axes, targets):
        rows = sorted([r for r in summary if r["target"] == target],
                      key=lambda r: r["budget"])
        for arm, (colour, marker, label) in styles.items():
            ax.errorbar([r["budget"] for r in rows],
                        [r[f"{arm}_vrmse"] for r in rows],
                        yerr=[r[f"{arm}_sd"] for r in rows],
                        color=colour, marker=marker, capsize=3, label=label)
        ax.set(xscale="log", yscale="log", xlabel="training trajectories",
               ylabel="test one-step VRMSE", title=f"target: {target}")
        ax.legend(fontsize=8)

    ax = axes[-1]
    names = list(dict.fromkeys([r["a"] for r in overlap]
                               + [r["b"] for r in overlap]))
    grid = np.full((len(names), len(names)), np.nan)
    for row in overlap:
        if row["layer"] != 0:
            continue
        i, j = names.index(row["a"]), names.index(row["b"])
        grid[i, j] = grid[j, i] = row["overlap"]
    im = ax.imshow(grid, cmap="magma", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="subspace overlap")
    ax.set(xticks=range(len(names)), yticks=range(len(names)),
           title="shared spectral basis (layer 0)")
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png.relative_to(_ROOT)}")


def print_report(summary: list[dict], components: dict, overlap: list[dict]
                 ) -> None:
    print("\n" + "=" * 78)
    print("ext21  H2: do the resonant factors carry across regimes?")
    print("=" * 78)
    print(f"\ncomponents (rank {components['rank']}): "
          f"{components['n_resonant_found']} resonant, "
          f"{components['n_damped_found']} damped, "
          f"{components['n_matched']} used per arm after matching"
          f"   (split margin {components['split_margin']})")
    if components["n_matched"] == 0:
        print("  the two sets do not separate; the transplant arms are void")
    elif components["n_matched"] == components["rank"]:
        print("  every component was selected by both; the arms are identical")

    for target in dict.fromkeys(r["target"] for r in summary):
        print(f"\ntarget: {target}")
        print(f"  {'budget':>7}  {'scratch':>9}  {'finetune':>9}  "
              f"{'resonant':>9}  {'damped':>9}  {'asymmetry':>10}  {'/sd':>6}")
        for row in sorted([r for r in summary if r["target"] == target],
                          key=lambda r: r["budget"]):
            print(f"  {row['budget']:>7}  {row['scratch_vrmse']:>9.5f}  "
                  f"{row['finetune_vrmse']:>9.5f}  "
                  f"{row['transplant_resonant_vrmse']:>9.5f}  "
                  f"{row['transplant_damped_vrmse']:>9.5f}  "
                  f"{row['asymmetry']:>+10.5f}  "
                  + (f"{row['asymmetry_over_sd']:>+6.2f}"
                     if np.isfinite(row["asymmetry_over_sd"]) else "   n/a"))

    print("\nsubspace overlap, layer 0")
    for row in overlap:
        if row["layer"] == 0:
            print(f"  {row['a']:>14} vs {row['b']:<14} overlap "
                  f"{row['overlap']:.4f}   median angle "
                  f"{row['median_angle_deg']:5.1f} deg")

    positive = [r for r in summary
                if np.isfinite(r["asymmetry_over_sd"])
                and r["asymmetry_over_sd"] > 1.0]
    print(f"\n{len(positive)} of {len(summary)} (target, budget) cells put the "
          f"resonant transplant more than\none standard deviation ahead of the "
          f"size-matched damped control.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16])
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
    p.add_argument("--threshold", type=float, default=0.25,
                   help="share of a rank component's mode energy that must sit "
                        "on the selected modes for it to count")
    p.add_argument("--overlap-layers", type=int, default=1)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.quick:
        args.budgets, args.seeds = [2, 8], [0]
        args.n_steps, args.epochs, args.source_epochs = 16, 6, 8
        args.width, args.layers, args.max_mode = 16, 2, 4

    device = args.device
    print("training the source model", flush=True)
    source_traj = make("source", args.source_traj, args.n_steps, args.size,
                       seed=0)
    source_splits = split_trajectories(source_traj, seed=0)
    source = build(args, args.seeds[0])
    fit(source, source_splits["train"], epochs=args.source_epochs, lr=args.lr,
        device=device, seed=args.seeds[0])
    print(f"  source one-step VRMSE "
          f"{one_step_vrmse(source, source_splits['test'], device):.5f}",
          flush=True)

    base = source_splits["test"][0, args.n_steps // 2].transpose(2, 0, 1)
    components = classify_components(source, base, args, device)
    print(f"  resonant components "
          f"{[int(c) for c in components['resonant']]}, "
          f"damped {[int(c) for c in components['damped']]}", flush=True)

    rows = []
    for target in TARGETS:
        for budget in args.budgets:
            for seed in args.seeds:
                for arm in ("scratch", "finetune", "transplant_resonant",
                            "transplant_damped"):
                    rows.append(run_arm(arm, source, target, budget, args,
                                        device, seed, components))
            done = [r for r in rows if r["target"] == target
                    and r["budget"] == budget]
            print(f"  {target:<14} budget {budget:>3}  " + "  ".join(
                f"{a[:9]}={np.mean([r['test_vrmse'] for r in done if r['arm'] == a]):.4f}"
                for a in ("scratch", "finetune", "transplant_resonant",
                          "transplant_damped")), flush=True)

    # independently trained models per regime, for the overlap matrix
    print("\ntraining independent models for the overlap matrix", flush=True)
    models = {"source_s0": source}
    for regime in REGIMES:
        for seed in args.seeds[:2]:
            if regime == "source" and seed == args.seeds[0]:
                continue
            traj = make(regime, args.source_traj, args.n_steps, args.size,
                        seed=seed)
            model = build(args, seed + 50)
            fit(model, split_trajectories(traj, seed=0)["train"],
                epochs=args.source_epochs, lr=args.lr, device=device,
                seed=seed)
            models[f"{regime}_s{seed}"] = model

    summary = summarise(rows)
    overlap = overlap_matrix(models, args)

    write_csv(RESULTS / "ext21_transplant.csv", rows)
    write_csv(RESULTS / "ext21_overlap.csv", overlap)
    write_csv(RESULTS / "ext21_summary.csv", summary)
    plot(summary, overlap, FIGURES / "ext21_mode_transplant.png")
    print_report(summary, components, overlap)


if __name__ == "__main__":
    main()
