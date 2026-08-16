r"""ext24: does scarcity travel on a network the operator cannot see? (H5)

Board task: "borrow from epidemiology / contact tracing -- model how resource
scarcity propagates through social/ecological networks (contact graphs, trade
networks); add a graph-convolutional layer on top of LiteFNO to capture network
effects in resource flow, not just local diffusion."

    python3 scripts/network_scarcity.py
    python3 scripts/network_scarcity.py --quick

The framing that decides what can be claimed
--------------------------------------------
"Not just local diffusion" presumes LiteFNO does only local diffusion. It does
not. On a periodic grid the lattice Laplacian's eigenvectors *are* the Fourier
modes, so a spectral convolution with a free weight per mode is already a
spectral graph filter on the pixel lattice (Bruna et al. 2014; Defferrard et
al. 2016). Every filter on a translation-invariant graph is therefore in the
FNO's span before a graph layer is added, and the only capacity a graph layer
can contribute is whatever the *non-lattice* edges carry.

That turns a vague hypothesis into one with a null:

**H5. A graph-convolutional head improves region-level scarcity prediction in
proportion to the share of the trade network that is non-lattice, and by
nothing at all when the network is a pure spatial contact lattice.**

Four measurements
-----------------
1. **The lattice identity, and the epidemic threshold.** ``L e_k = lambda_k
   e_k`` to machine precision, and the die-out threshold of the scarcity
   cascade against the closed form ``1 / lambda_1(A)`` on four graph families.
   Both are ground truth known before the simulation runs, which is the same
   standard ``scripts/operator_poles.py`` holds itself to.
2. **The arms, at equal parameters.** Four models identical except for the
   matrix inside the graph layer -- the true trade network, a degree-preserving
   rewiring of it, the spatial lattice, and the identity -- predicting region
   scarcity ``h`` steps ahead from one field frame.
3. **The sweep that tests H5's proportionality.** The same comparison across
   Watts-Strogatz rewiring probabilities, plotted against the *realised*
   shortcut fraction.
4. **Contact tracing: where to watch.** Sentinel regions chosen by eigenvector
   centrality, by degree, and at random, scored on how late they notice a
   shortfall and how far it has spread by then.

Outputs
-------
``results/extensions/ext24_threshold.csv``   closed-form check on four families
``results/extensions/ext24_arms.csv``        per-arm, per-seed held-out error
``results/extensions/ext24_sweep.csv``       graph advantage vs shortcut share
``results/extensions/ext24_sentinels.csv``   detection delay by placement rule
``figures/extensions/ext24_network_scarcity.png``
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

import torch                                                    # noqa: E402

from litefno.allocation import region_gains                     # noqa: E402
from litefno.models.graphfno import (                           # noqa: E402
    GraphLiteFNO, bound_violation, fit_graph_model, paint_regions, predict,
    region_vrmse)
from litefno.models.harmonic import HarmonicLiteFNO             # noqa: E402
from litefno.models.litefno import LiteFNO                      # noqa: E402
from litefno.networks import (                                  # noqa: E402
    cascade_threshold, degree_preserving_rewire, detection_delay,
    eigenvector_centrality, fourier_eigenbasis_residual, grid_graph,
    normalized_adjacency, preferential_attachment, propagate_scarcity,
    sentinel_sets, shortcut_fraction, small_world, spectral_radius)
from litefno.systems import lambda_omega, split_trajectories     # noqa: E402

RESULTS = _ROOT / "results" / "extensions"
FIGURES = _ROOT / "figures" / "extensions"

# Must match scripts/fair_allocation.py and scripts/strategic_allocation.py.
# ext24 puts a third layer on the same regions of the same ecosystem, and a
# different medium would make the three sets of numbers incomparable.
ECOSYSTEM = dict(diffusion=0.4, omega=0.6, perturbation=0.8, max_mode=4,
                 spinup=20)

ARMS = ["identity", "lattice", "rewired", "true"]


# --------------------------------------------------------------------------
# the coupled system
# --------------------------------------------------------------------------


def build_dataset(traj: np.ndarray, adjacency: np.ndarray, args) -> dict:
    """Field frames plus the current scarcity, against scarcity ``h`` ahead.

    The input is the (u, v) field at ``t`` with ``x_t`` painted on as a third
    channel, and the target is ``x_{t+h}``. Both are exact functions of the
    input -- the ecosystem is deterministic and the cascade is driven by it --
    so there is no irreducible noise floor and every difference between arms is
    a difference in what the model can express, which is the point.
    """
    gains = region_gains(traj, blocks=args.blocks)
    lam = spectral_radius(adjacency)
    # subcritical by design: at tau > 1/lambda_1 a seeded cascade saturates the
    # whole network and the target becomes a constant 1, which no model can get
    # wrong and no comparison can see. Held at a fixed fraction of *each
    # graph's own* threshold so the sweep over topology does not silently sweep
    # over how supercritical the dynamics are.
    beta = args.tau_fraction * args.gamma / lam
    scarcity = propagate_scarcity(gains, adjacency, beta=beta,
                                  gamma=args.gamma, kappa=args.kappa)
    n_time = traj.shape[1]
    horizon = args.horizon
    fields, states, targets = [], [], []
    for t in range(n_time - horizon):
        fields.append(traj[:, t])
        states.append(scarcity[:, t])
        targets.append(scarcity[:, t + horizon])
    field = np.concatenate(fields)                       # (N, H, W, C)
    state = np.concatenate(states)                       # (N, R)
    target = np.concatenate(targets)
    painted = paint_regions(state, size=traj.shape[2], blocks=args.blocks)
    x = np.concatenate(
        [np.moveaxis(field, -1, -3).astype(np.float32), painted], axis=1)
    return {"x": np.ascontiguousarray(x), "y": target.astype(np.float32),
            "scarcity": scarcity, "beta": beta, "lambda_1": lam,
            "n_samples": len(x)}


def make_trunk(args, in_channels: int) -> torch.nn.Module:
    """The repo's own operator, spectral by default.

    ``--trunk cnn`` swaps in ``models/litefno.py``, the low-rank CNN the
    reproduction found matches the spectral model on Gray-Scott. It is offered
    because the Fourier-basis argument this experiment rests on is about the
    *spectral* layer, so it is worth being able to check whether the graph
    result survives when the trunk has no FFT in it.
    """
    if args.trunk == "cnn":
        return LiteFNO(in_channels, args.trunk_channels, width=args.width,
                       rank=args.rank, layers=args.layers)
    return HarmonicLiteFNO(in_channels, args.trunk_channels, width=args.width,
                           modes=args.modes, layers=args.layers, rank=args.rank)


def arm_propagator(arm: str, adjacency: np.ndarray, args, seed: int):
    """The one matrix that separates the arms."""
    if arm == "identity":
        return None
    if arm == "lattice":
        graph = grid_graph(args.blocks)
    elif arm == "rewired":
        graph = degree_preserving_rewire(adjacency, seed=1000 + seed)
    elif arm == "true":
        graph = adjacency
    else:
        raise ValueError(f"unknown arm {arm}")
    return normalized_adjacency(graph)


def run_arm(arm: str, data: dict, adjacency: np.ndarray, args, seed: int
            ) -> dict:
    torch.manual_seed(seed)
    trunk = make_trunk(args, in_channels=data["train"]["x"].shape[1])
    model = GraphLiteFNO(
        trunk, trunk_channels=args.trunk_channels, blocks=args.blocks,
        hidden=args.hidden, propagator=arm_propagator(arm, adjacency, args, seed),
        order=args.order, layers=args.graph_layers)
    fit = fit_graph_model(model, data["train"]["x"], data["train"]["y"],
                          epochs=args.epochs, lr=args.lr, batch=args.batch,
                          device=args.device, seed=seed,
                          valid=(data["valid"]["x"], data["valid"]["y"]))
    pred = predict(model, data["test"]["x"], args.device)
    return {"arm": arm, "seed": seed,
            "test_vrmse": region_vrmse(pred, data["test"]["y"]),
            "test_mse": float(np.mean((pred - data["test"]["y"]) ** 2)),
            "valid_mse": fit["history"][-1].get("valid_mse", np.nan),
            "bound_violation": bound_violation(pred),
            "best_epoch": fit["best_epoch"],
            "n_parameters": fit["n_parameters"]}


def split_dataset(splits: dict, adjacency: np.ndarray, args) -> dict:
    out = {}
    for name in ("train", "valid", "test"):
        out[name] = build_dataset(splits[name], adjacency, args)
    return out


# --------------------------------------------------------------------------
# 1. the two things known in closed form
# --------------------------------------------------------------------------


def threshold_table(args) -> list[dict]:
    """The cascade's die-out threshold against ``1 / lambda_1`` per family."""
    families = {
        "lattice (p=0)": grid_graph(args.blocks),
        "small world (p=0.3)": small_world(args.blocks, 0.3, seed=0),
        "trade network (p=1)": small_world(args.blocks, 1.0, seed=1),
        "scale free (BA m=2)": preferential_attachment(args.blocks ** 2, 2,
                                                       seed=0),
    }
    rows = []
    for name, graph in families.items():
        found = cascade_threshold(graph, gamma=args.gamma,
                                  steps=args.threshold_steps)
        rows.append({"family": name, "lambda_1": found["lambda_1"],
                     "predicted_tau_c": found["predicted"],
                     "measured_tau_c": found["measured"],
                     "rel_error": found["rel_error"],
                     "mean_degree": float(graph.sum(axis=1).mean()),
                     "shortcut_fraction": shortcut_fraction(graph, args.blocks)})
    return rows


# --------------------------------------------------------------------------
# 4. contact tracing
# --------------------------------------------------------------------------


def sentinel_table(traj: np.ndarray, args) -> list[dict]:
    """Detection delay by placement rule, on the lattice and on a trade network.

    Run at the supercritical setting rather than the subcritical one used for
    the prediction task: surveillance is a question about outbreaks that spread,
    and on a shortfall that dies out locally every placement rule detects it at
    the same time or not at all.
    """
    gains = region_gains(traj, blocks=args.blocks)
    rows = []
    graphs = {"lattice (p=0)": grid_graph(args.blocks),
              "trade network (p=1)": small_world(args.blocks, 1.0, seed=1)}
    for name, graph in graphs.items():
        beta = args.sentinel_tau * args.gamma / spectral_radius(graph)
        scarcity = propagate_scarcity(gains, graph, beta=beta,
                                      gamma=args.gamma, kappa=args.kappa)
        centrality = eigenvector_centrality(graph)
        degree = graph.sum(axis=1)
        # on a regular graph both heuristics are exact ties and their output is
        # whatever the tie-break says, which is an arbitrary set of regions and
        # must not be read as a placement rule beating another one. Flagged in
        # the results rather than left for the reader to notice.
        degenerate = {
            "eigenvector": float(centrality.max() - centrality.min()) < 1e-8,
            "degree": float(degree.max() - degree.min()) < 1e-8,
            "random": True,
        }
        # the random rule is averaged over draws; one draw is a sample of size
        # one from the very distribution the other rules are being compared to
        placements = {rule: [idx] for rule, idx in
                      sentinel_sets(graph, args.n_sentinels,
                                    seed=args.sentinel_seed).items()
                      if rule != "random"}
        placements["random"] = [
            sentinel_sets(graph, args.n_sentinels, seed=1000 + k)["random"]
            for k in range(args.sentinel_repeats)]
        for rule, draws in placements.items():
            delays, spreads, missed, considered = [], [], 0, 0
            for idx in draws:
                for run in scarcity:
                    found = detection_delay(run, idx, threshold=args.detect_at)
                    if np.isnan(found["delay"]):
                        continue
                    considered += 1
                    if np.isinf(found["delay"]):
                        missed += 1
                        continue
                    delays.append(found["delay"])
                    spreads.append(found["spread_at_detection"])
            rows.append({
                "graph": name, "rule": rule,
                "n_sentinels": args.n_sentinels,
                "mean_delay": float(np.mean(delays)) if delays else np.nan,
                "median_delay": float(np.median(delays)) if delays else np.nan,
                "spread_at_detection":
                    float(np.mean(spreads)) if spreads else np.nan,
                "missed": missed, "n_runs": considered,
                "score_is_a_tie": bool(degenerate[rule])})
    return rows


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict], key: str = "test_vrmse") -> dict:
    """Per-arm mean and spread, plus the paired advantage over the no-graph arm.

    Paired, because the arms share a seed, a trunk initialisation and a data
    split: the unpaired difference of means throws away the fact that a seed
    that was bad for one arm was bad for all of them, and with three seeds that
    is most of the information.
    """
    by_arm = {}
    for arm in ARMS:
        vals = [r[key] for r in rows if r["arm"] == arm]
        if vals:
            by_arm[arm] = np.asarray(vals, dtype=float)
    if "identity" not in by_arm:
        return {"by_arm": by_arm}
    base = by_arm["identity"]
    advantage = {arm: 1.0 - vals / base for arm, vals in by_arm.items()}
    return {"by_arm": by_arm, "advantage": advantage}


def paired_contrast(rows: list[dict], arm: str, against: str,
                    key: str = "test_vrmse") -> np.ndarray:
    """``1 - err(arm) / err(against)`` seed by seed, within one graph.

    Paired at the level of (rewiring probability, seed), because the two arms
    being compared share a graph, a data split and a trunk initialisation. The
    difference of the seed means would discard exactly the variance the pairing
    removes, and with three seeds that is the difference between a result and a
    coin flip.
    """
    out = []
    for seed in sorted({r["seed"] for r in rows}):
        a = next((r[key] for r in rows
                  if r["arm"] == arm and r["seed"] == seed), None)
        b = next((r[key] for r in rows
                  if r["arm"] == against and r["seed"] == seed), None)
        if a is not None and b is not None:
            out.append(1.0 - a / b)
    return np.asarray(out, dtype=float)


def plot(threshold: list[dict], arms: list[dict], sweep: list[dict],
         sentinels: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    pred = [r["predicted_tau_c"] for r in threshold]
    meas = [r["measured_tau_c"] for r in threshold]
    ax.scatter(pred, meas, s=60, color="#1f77b4", zorder=3)
    lo, hi = min(pred + meas) * 0.95, max(pred + meas) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1 / lambda_1")
    for row in threshold:
        ax.annotate(row["family"].split(" (")[0],
                    (row["predicted_tau_c"], row["measured_tau_c"]),
                    textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.set_xlabel("predicted threshold  1 / lambda_1")
    ax.set_ylabel("measured die-out threshold  beta / gamma")
    ax.set_title("1. the epidemic threshold is the graph's leading eigenvalue")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    labels, colours = [], {"identity": "#999999", "lattice": "#8c8cd9",
                           "rewired": "#d97b7b", "true": "#2ca02c"}
    for i, p in enumerate(sorted({r["p"] for r in arms})):
        for j, arm in enumerate(ARMS):
            vals = [r["test_vrmse"] for r in arms
                    if r["arm"] == arm and r["p"] == p]
            if not vals:
                continue
            pos = i * (len(ARMS) + 1) + j
            ax.bar(pos, np.mean(vals), yerr=np.std(vals), color=colours[arm],
                   capsize=3, label=arm if i == 0 else None)
        labels.append((i * (len(ARMS) + 1) + 1.5, f"p = {p:g}"))
    ax.set_xticks([x for x, _ in labels])
    ax.set_xticklabels([t for _, t in labels])
    ax.set_ylabel("held-out region VRMSE")
    ax.set_title("2. same parameters, different matrix")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    probs = sorted({r["p"] for r in sweep})
    xs = [next(r["shortcut_fraction"] for r in sweep if r["p"] == p)
          for p in probs]
    for against, label in (("identity", "true vs no graph"),
                           ("lattice", "true vs lattice graph"),
                           ("rewired", "true vs rewired graph")):
        ys, es = [], []
        for p in probs:
            vals = paired_contrast([r for r in sweep if r["p"] == p], "true",
                                   against)
            ys.append(vals.mean() if vals.size else np.nan)
            es.append(vals.std() if vals.size else np.nan)
        if not np.all(np.isnan(ys)):
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=label,
                        color=colours[against])
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.set_xlabel("realised shortcut fraction (share of non-lattice edges)")
    ax.set_ylabel("paired VRMSE reduction")
    ax.set_title("3. H5: what only the true topology could supply")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    rules = ["eigenvector", "degree", "random"]
    graphs = sorted({r["graph"] for r in sentinels})
    width = 0.25
    for j, rule in enumerate(rules):
        vals = [next((r["spread_at_detection"] for r in sentinels
                      if r["graph"] == g and r["rule"] == rule), np.nan)
                for g in graphs]
        ax.bar(np.arange(len(graphs)) + j * width, vals, width, label=rule)
    ax.set_xticks(np.arange(len(graphs)) + width)
    ax.set_xticklabels([g.split(" (")[0] for g in graphs])
    ax.set_ylabel("share of network already scarce at detection")
    ax.set_title("4. where to put the sentinels")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_report(residual: float, threshold: list[dict], arms: list[dict],
                 sweep: list[dict], sentinels: list[dict], meta: dict) -> None:
    print("\n" + "=" * 78)
    print("ext24: scarcity propagation on a trade network (H5)")
    print("=" * 78)

    print("\n1. the graph the operator is already convolving on")
    print(f"  max |L e_k - lambda_k e_k| over all modes of a "
          f"{meta['grid']}x{meta['grid']} periodic lattice: {residual:.2e}")
    print("  the Fourier modes ARE the lattice Laplacian's eigenvectors, so a "
          "spectral\n  convolution already spans every filter on that graph; "
          "only non-lattice edges\n  are new capacity.")

    print("\n   the epidemic threshold against its closed form")
    print(f"  {'family':>22} {'lambda_1':>10} {'1/lambda_1':>12} "
          f"{'measured':>10} {'rel err':>10} {'shortcuts':>10}")
    for row in threshold:
        print(f"  {row['family']:>22} {row['lambda_1']:>10.4f} "
              f"{row['predicted_tau_c']:>12.5f} {row['measured_tau_c']:>10.5f} "
              f"{row['rel_error']:>10.2e} {row['shortcut_fraction']:>10.3f}")

    print(f"\n2. the arms, at {meta['n_parameters']} parameters each "
          f"({meta['n_seeds']} seeds, horizon {meta['horizon']})")
    for p in sorted({r["p"] for r in arms}):
        rows = [r for r in arms if r["p"] == p]
        stats = summarise(rows)
        print(f"  p = {p:g}   (shortcut fraction "
              f"{rows[0]['shortcut_fraction']:.3f}, lambda_1 "
              f"{rows[0]['lambda_1']:.3f})")
        print(f"    {'arm':>10} {'VRMSE':>18} {'vs no-graph':>22}")
        for arm in ARMS:
            if arm not in stats["by_arm"]:
                continue
            vals = stats["by_arm"][arm]
            adv = stats["advantage"][arm]
            print(f"    {arm:>10} {vals.mean():>10.4f} +- {vals.std():<6.4f}"
                  f" {100 * adv.mean():>15.2f}% +- {100 * adv.std():<.2f}%")

    print("\n3. H5: gain against the share of non-lattice edges")
    print("   'vs X' is the paired VRMSE reduction of the true-graph arm "
          "against arm X,\n   seed by seed. The column that tests H5 is 'vs "
          "lattice': it is the part of\n   the gain that a convolution could "
          "not have supplied.")
    print(f"  {'p':>6} {'shortcuts':>11} {'true vs none':>18} "
          f"{'true vs lattice':>18} {'true vs rewired':>18}")
    for p in sorted({r["p"] for r in sweep}):
        rows = [r for r in sweep if r["p"] == p]
        line = f"  {p:>6.2f} {rows[0]['shortcut_fraction']:>11.3f}"
        for arm in ("identity", "lattice", "rewired"):
            vals = 100 * paired_contrast(rows, "true", arm)
            line += (f" {vals.mean():>11.2f}% +-{vals.std():<5.2f}"
                     if vals.size else f" {'-':>18}")
        print(line)

    print("\n4. sentinel placement")
    print(f"  {'graph':>22} {'rule':>12} {'delay':>10} {'spread at detect':>18}"
          f" {'missed':>8} {'tie':>5}")
    for row in sentinels:
        print(f"  {row['graph']:>22} {row['rule']:>12} "
              f"{row['mean_delay']:>10.2f} {row['spread_at_detection']:>18.4f}"
              f" {row['missed']:>8d} {'yes' if row['score_is_a_tie'] else 'no':>5}")
    print("  'tie' marks a rule whose score is constant over regions, so its "
          "choice is an\n  arbitrary tie-break and its row is not evidence "
          "about the rule.")


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # ecosystem
    p.add_argument("--n-traj", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4)
    # contagion
    p.add_argument("--gamma", type=float, default=0.3,
                   help="replenishment rate")
    p.add_argument("--tau-fraction", type=float, default=0.8,
                   help="beta/gamma as a fraction of each graph's threshold")
    p.add_argument("--kappa", type=float, default=0.15,
                   help="how hard local demand pressure seeds the cascade")
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--threshold-steps", type=int, default=400)
    # graphs
    p.add_argument("--p-grid", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--arm-p", type=float, nargs="+", default=None,
                   help="rewiring probabilities to run the full four-arm table "
                        "at; defaults to all of --p-grid, since the "
                        "true-vs-wrong-graph contrast is the point")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # model
    p.add_argument("--trunk", choices=["spectral", "cnn"], default="spectral")
    p.add_argument("--trunk-channels", type=int, default=8)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--graph-layers", type=int, default=2)
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    # sentinels
    p.add_argument("--n-sentinels", type=int, default=3)
    p.add_argument("--sentinel-tau", type=float, default=1.4)
    p.add_argument("--sentinel-seed", type=int, default=0)
    p.add_argument("--sentinel-repeats", type=int, default=20,
                   help="random placements to average the control over")
    p.add_argument("--detect-at", type=float, default=0.05)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.arm_p is None:
        args.arm_p = list(args.p_grid)

    if args.quick:
        args.n_traj, args.n_steps = 12, 20
        args.epochs = 4
        args.width, args.layers, args.hidden = 12, 1, 12
        args.p_grid = [0.0, 1.0]
        args.arm_p = [1.0]
        args.seeds = [0]
        args.threshold_steps = 60

    started = time.time()
    print("generating the ecosystem", flush=True)
    traj = lambda_omega(n_traj=args.n_traj, n_steps=args.n_steps,
                        size=args.size, seed=0, **ECOSYSTEM)
    splits = split_trajectories(traj, fractions=(0.5, 0.25, 0.25), seed=0)

    print("1. closed forms: the lattice eigenbasis and the threshold",
          flush=True)
    residual = fourier_eigenbasis_residual(8)
    threshold = threshold_table(args)
    print(f"   Fourier/Laplacian residual {residual:.2e}; threshold error "
          f"{max(r['rel_error'] for r in threshold):.2e} at worst "
          f"({time.time() - started:.0f}s)", flush=True)

    print("2-3. training the arms", flush=True)
    arm_rows, sweep_rows = [], []
    n_parameters = 0
    for prob in args.p_grid:
        full = prob in args.arm_p
        arms_here = ARMS if full else ["identity", "true"]
        for seed in args.seeds:
            graph = small_world(args.blocks, prob, seed=seed)
            data = split_dataset(splits, graph, args)
            shortcuts = shortcut_fraction(graph, args.blocks)
            results = {}
            for arm in arms_here:
                row = run_arm(arm, data, graph, args, seed)
                row.update({"p": prob, "shortcut_fraction": shortcuts,
                            "lambda_1": float(data["train"]["lambda_1"]),
                            "beta": float(data["train"]["beta"])})
                results[arm] = row
                n_parameters = row["n_parameters"]
                print(f"   p={prob:g} seed={seed} {arm:>9}: VRMSE "
                      f"{row['test_vrmse']:.4f}  ({time.time() - started:.0f}s)",
                      flush=True)
            base = results["identity"]["test_vrmse"]
            for arm, row in results.items():
                row["advantage"] = 1.0 - row["test_vrmse"] / base
                sweep_rows.append(row)
                if full:
                    arm_rows.append(row)

    print("4. sentinel placement", flush=True)
    # every trajectory, not just the test split: nothing here is trained, so
    # holding data out would buy nothing and cost three quarters of the runs
    sentinels = sentinel_table(traj, args)

    meta = {"grid": 8, "n_parameters": n_parameters,
            "n_seeds": len(args.seeds), "horizon": args.horizon}

    write_csv(RESULTS / "ext24_threshold.csv", threshold)
    write_csv(RESULTS / "ext24_arms.csv", arm_rows)
    write_csv(RESULTS / "ext24_sweep.csv", sweep_rows)
    write_csv(RESULTS / "ext24_sentinels.csv", sentinels)
    plot(threshold, arm_rows, sweep_rows, sentinels,
         FIGURES / "ext24_network_scarcity.png")
    print_report(residual, threshold, arm_rows, sweep_rows, sentinels, meta)
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
