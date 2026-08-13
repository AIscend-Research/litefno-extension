r"""ext22: fair allocation on a surrogate's forecast (H3)

Board task: "Add fairness-aware resource allocation layer: train an auxiliary
network that, given the reconstructed ecosystem state, predicts a fair resource
allocation across populations/regions (not just max-efficiency, but satisfying
fairness constraints like max-min or envy-free)."

    python3 scripts/fair_allocation.py
    python3 scripts/fair_allocation.py --quick

H3: how much a surrogate's error costs the decision taken on its output depends
on *which* fairness rule is used, and it does so in a way that is derivable
rather than empirical. :mod:`litefno.allocation` gives the prediction --
relative welfare loss scales as ``(1-alpha)^2/(2 alpha)`` times the variance of
the log gain error -- so fragility is U-shaped in the fairness parameter, zero
at the envy-free point, and rising toward *both* max-efficiency and max-min.

Four things are being asked, and they can fail separately
----------------------------------------------------------
1. **Is the fragility law right?** Tested on controlled noise, where the answer
   is known and the surrogate is not involved at all. If it fails here, nothing
   downstream means anything.
2. **Does the auxiliary network earn its place?** The honest control is not
   another network -- it is four lines of numpy: pool the reconstructed field
   into regions and apply the closed form. A network that cannot beat that is
   not a contribution, and this script reports which one wins rather than
   reporting only the network.
3. **Does the forecast earn its place?** The control that can kill the whole
   pipeline is ``persistence``: allocate from the last state actually observed
   and do not forecast at all. If a surrogate rollout is no better than that,
   the decision layer did not need the neural operator.
4. **Does any of it change when the surrogate is bad?** A network trained on the
   surrogate's own output could learn to distrust it, and a rule fed accurate
   gains has nothing to distrust -- so the whole question is vacuous at one
   error level. Every arm therefore runs against two surrogates, one trained on
   the full split and one deliberately starved, and the starved one is where
   hedging has something to do.

The arms
--------
======================  ====================================================
``oracle``              closed form on the true gains -- the 0 by definition
``plugin``              closed form on the surrogate's reconstructed field
``shrunk``              plugin with the exponent shrunk, fitted on validation
``learned``             the network, trained on true states
``learned_robust``      the network, trained on the surrogate's own outputs
``uniform``             equal division: the envy-free rule, state-free
``persistence``         closed form on the last observed state, no forecast
======================  ====================================================

Outputs
-------
``results/extensions/ext22_fragility.csv``   the law, on controlled noise
``results/extensions/ext22_arms.csv``        every arm x alpha x seed
``results/extensions/ext22_horizon.csv``     loss against rollout horizon
``results/extensions/ext22_summary.csv``     the headline table
``figures/extensions/ext22_fair_allocation.png``
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

from litefno.allocation import (                                # noqa: E402
    allocation_exponent, alpha_fair_allocation, fragility_coefficient,
    implied_gains, max_envy, max_min_ratio, outcome_envy, outcomes,
    predicted_welfare_loss, price_of_fairness, region_gains,
    relative_welfare_loss, tune_shrinkage, welfare_ce)
from litefno.models.allocator import (                          # noqa: E402
    RegionAllocator, allocate, fit_allocator)
from litefno.models.harmonic import HarmonicLiteFNO             # noqa: E402
from litefno.specscope import fit, one_step_vrmse               # noqa: E402
from litefno.systems import lambda_omega, split_trajectories    # noqa: E402

RESULTS = _ROOT / "results" / "extensions"
FIGURES = _ROOT / "figures" / "extensions"

# The ecosystem regime. lambda-omega is the repo's oscillatory medium and its
# own docstring names it the ecosystem case -- predator-prey cycles near a Hopf
# bifurcation reduce to this normal form. These parameters leave the medium
# *incompletely relaxed*, carrying amplitude defects and a spatially varying
# phase, which is what makes region populations differ at all: on the fully
# settled limit cycle every region holds the same amplitude and every allocation
# rule collapses onto equal division. Patchiness is the precondition for the
# allocation problem existing, so it is a parameter choice, stated here.
ECOSYSTEM = dict(diffusion=0.4, omega=0.6, perturbation=0.8, max_mode=4,
                 spinup=20)

# alpha = 0 is max-efficiency and alpha -> inf is max-min; 8 already sits at
# exponent -0.875, seven eighths of the way to the max-min limit of -1.
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
ARMS = ["plugin", "shrunk", "smoothed", "learned", "learned_robust", "uniform",
        "persistence"]


def make(n_traj: int, n_steps: int, size: int, seed: int) -> np.ndarray:
    return lambda_omega(n_traj=n_traj, n_steps=n_steps, size=size, seed=seed,
                        **ECOSYSTEM)


def as_states(field: np.ndarray) -> np.ndarray:
    """(..., H, W, C) -> (..., C, H, W) float32, the layout the models take."""
    return np.ascontiguousarray(
        np.moveaxis(field, -1, -3)).astype(np.float32)


# --------------------------------------------------------------------------
# 1. the law, with no surrogate involved
# --------------------------------------------------------------------------


def fragility_sweep(gains: np.ndarray, sigmas, seed: int = 0) -> list[dict]:
    """Inject known log-normal gain error and check the predicted welfare loss.

    The clean test. The gains are real -- read off the ecosystem field -- but
    the error is synthetic and its size is chosen, so the law is checked over a
    range of error magnitudes rather than at whatever single magnitude the
    surrogate happens to land on. That matters because the law is a
    second-order expansion: it has to hold at small error and is expected to
    drift at large error, and a test at one noise level cannot tell the two
    apart.

    The allocation sensitivity is reported against the *centred* error rather
    than against the injected sd. Both are honest, but only one is exact: the
    rule normalises over regions, so it can only respond to how much each
    region's error differs from the allocation-weighted mean error, and the
    common part cancels. Comparing to the raw sd builds in a deflation of
    ``sqrt(1 - 1/R)`` and makes an exact law look 3% wrong.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in sigmas:
        eta = rng.normal(0.0, sigma, gains.shape)
        noisy = gains * np.exp(eta)
        for alpha in ALPHAS:
            alloc = alpha_fair_allocation(noisy, alpha=alpha)
            best = alpha_fair_allocation(gains, alpha=alpha)
            measured = float(np.mean(
                relative_welfare_loss(gains, alloc, alpha)))
            predicted = (float("inf") if alpha == 0 else
                         float(np.mean(predicted_welfare_loss(
                             gains, noisy, alpha))))
            weights = best / best.sum(axis=-1, keepdims=True)
            centred = eta - np.sum(weights * eta, axis=-1, keepdims=True)
            with np.errstate(divide="ignore"):
                moved = (np.log(np.maximum(alloc, 1e-300))
                         - np.log(np.maximum(best, 1e-300)))
            finite = np.isfinite(moved) & np.isfinite(centred)
            spread = float(np.std(centred[finite]))
            rows.append({
                "alpha": alpha, "sigma": round(float(sigma), 4),
                "measured_loss": measured,
                "predicted_loss": predicted,
                "ratio": (measured / predicted
                          if np.isfinite(predicted) and predicted > 0
                          else float("nan")),
                "amplification": round(
                    float(np.std(moved[finite])) / max(spread, 1e-30), 4),
                "amplification_theory": (
                    round(abs(allocation_exponent(alpha)), 4) if alpha > 0
                    else float("inf")),
                "coefficient": fragility_coefficient(alpha)})
    return rows


# --------------------------------------------------------------------------
# 2. the surrogate, and the gains read off its output
# --------------------------------------------------------------------------


def train_surrogate(train_traj: np.ndarray, args, seed: int = 0):
    torch.manual_seed(seed)
    model = HarmonicLiteFNO(2, 2, width=args.width, modes=args.modes,
                            layers=args.layers, rank=args.rank)
    fit(model, train_traj, epochs=args.epochs, lr=args.lr, device=args.device,
        seed=seed)
    return model


@torch.no_grad()
def rollout_from_origins(model, split: np.ndarray, horizon: int, stride: int,
                         device: str = "cpu", batch: int = 64) -> dict:
    """Roll the surrogate forward from every ``stride``-th start time.

    One rollout per trajectory would give an evaluation sample the size of the
    split -- eight decisions, from which no arm difference is measurable. A
    decision is taken at a time as well as at a place, so every admissible start
    time is one, and the sample becomes ``n_traj x n_origins``.

    They are not independent (consecutive origins overlap in the state they roll
    from), so this buys resolution rather than degrees of freedom, and the
    seed-to-seed spread reported alongside is the honest error bar.

    Returns the surrogate's state at the target time, the true field there, and
    the field at the origin, which is the persistence arm's information.
    """
    model.eval()
    origins = list(range(0, split.shape[1] - horizon, stride))
    if not origins:
        raise ValueError(f"horizon {horizon} leaves no start times")
    starts = np.concatenate([as_states(split[:, o]) for o in origins])
    truth = np.concatenate([split[:, o + horizon] for o in origins])
    observed = np.concatenate([split[:, o] for o in origins])

    out = []
    for i in range(0, len(starts), batch):
        cur = torch.from_numpy(starts[i:i + batch]).to(device)
        for _ in range(horizon):
            cur = model(cur)
        out.append(cur.cpu().numpy())
    return {"pred_states": np.concatenate(out), "true_field": truth,
            "observed_field": observed, "n_origins": len(origins)}


def smooth(states: np.ndarray, kernel: int) -> np.ndarray:
    """Box-blur the field before pooling it. The 'it learned to blur' control.

    A network reading the whole field could beat block-pooling simply by being a
    softer pooling operator -- a block mean is a hard-edged filter and a
    convolutional encoder is not. If that were the whole story, blurring the
    field first would reproduce the win with no learning at all, and the
    interesting claim would evaporate. Circular padding, because the domain is
    periodic and zero padding would invent a boundary the PDE does not have.
    """
    if kernel <= 1:
        return states
    x = torch.from_numpy(np.ascontiguousarray(states))
    pad = kernel // 2
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="circular")
    return torch.nn.functional.avg_pool2d(x, kernel, stride=1).numpy()


def bundle(model, split: np.ndarray, horizon: int, stride: int, args) -> dict:
    """Everything an arm needs at one horizon: states, and the four gain sets."""
    got = rollout_from_origins(model, split, horizon, stride, args.device)
    blurred = smooth(got["pred_states"], args.smooth_kernel)
    return {
        "pred_states": got["pred_states"],
        "true_states": as_states(got["true_field"]),
        "true_gains": region_gains(got["true_field"], blocks=args.blocks),
        "pred_gains": region_gains(
            np.moveaxis(got["pred_states"], -3, -1), blocks=args.blocks),
        "smoothed_gains": region_gains(
            np.moveaxis(blurred, -3, -1), blocks=args.blocks),
        "observed_gains": region_gains(got["observed_field"],
                                       blocks=args.blocks),
        "n_samples": len(got["pred_states"]),
    }


def gain_error(pred_gains: np.ndarray, true_gains: np.ndarray) -> dict:
    """Relative error of the region gains -- the only channel through which the
    surrogate's error can reach the decision at all."""
    eta = np.log(pred_gains / true_gains)
    # The rule is scale-free in the gains, so an error common to every region
    # cancels in the normalisation and cannot reach the decision. Reporting only
    # the raw spread would overstate the error that matters, by a factor of 2.6
    # on the starved surrogate, whose error is mostly a level shift.
    return {"gain_rel_err": float(np.mean(np.abs(pred_gains / true_gains - 1))),
            "gain_log_sd": float(np.std(eta)),
            "gain_log_sd_centred": float(
                np.std(eta - eta.mean(axis=-1, keepdims=True)))}


# --------------------------------------------------------------------------
# 3. the arms
# --------------------------------------------------------------------------


def estimator_error(true_gains: np.ndarray, alloc: np.ndarray, alpha: float
                    ) -> float:
    """How well the gains implied by an allocation match the true ones.

    The question this answers is *why* an arm wins, which the welfare number
    alone cannot. An allocator can beat a plug-in rule two ways: by hedging
    against an input it does not trust, or by reading that input better than the
    rule does. Only the second shows up here, because hedging deliberately moves
    the allocation away from any gain estimate.

    Both are compared on relative gains, since neither the rule nor the network
    can see the overall level.
    """
    try:
        implied = implied_gains(alloc, alpha)
    except ValueError:
        return float("nan")            # alpha in {0, 1}: nothing to recover
    truth = true_gains / true_gains.mean(axis=-1, keepdims=True)
    return float(np.mean(np.abs(implied - truth)))


def diagnostics(true_gains: np.ndarray, alloc: np.ndarray, alpha: float
                ) -> dict:
    x = outcomes(true_gains, alloc)
    return {
        "rel_welfare_loss": float(np.mean(
            relative_welfare_loss(true_gains, alloc, alpha))),
        "gain_estimator_err": estimator_error(true_gains, alloc, alpha),
        "max_min_ratio": float(np.mean(max_min_ratio(x))),
        "bundle_envy": float(np.mean(max_envy(alloc))),
        "outcome_envy": float(np.mean(outcome_envy(x))),
        "mean_outcome": float(np.mean(welfare_ce(x, 0.0))),
    }


def train_allocators(alpha: float, args, clean_states, clean_gains,
                     dirty_states, dirty_gains, seed: int) -> dict:
    """One network per arm: trained on true states, and on the surrogate's.

    Both are scored on true gains in the loss -- welfare is realised against the
    real ecosystem either way. The difference is only what the network is *shown*
    at training time, which is the question: can a network that has seen the
    surrogate's error learn to distrust it?

    The clean arm trains on every state in the allocator split, the dirty arm on
    every rollout the surrogate produces from it. Those counts differ, and the
    script reports both, because a network handed a tenth of the data of its
    control would be losing a comparison about sample size dressed up as a
    comparison about robustness.
    """
    nets = {}
    for arm, states, gains in (("learned", clean_states, clean_gains),
                               ("learned_robust", dirty_states, dirty_gains)):
        torch.manual_seed(seed)
        model = RegionAllocator(in_channels=2, blocks=args.blocks,
                                width=args.alloc_width)
        fit_allocator(model, states, gains, alpha=alpha,
                      epochs=args.alloc_epochs, lr=args.alloc_lr,
                      device=args.device, seed=seed)
        nets[arm] = model
    return nets


def run_arms(alpha: float, args, data: dict, nets: dict, shrink: float,
             seed: int, horizon: int, surrogate: str) -> list[dict]:
    true_gains, pred_gains = data["true_gains"], data["pred_gains"]
    allocations = {
        "plugin": alpha_fair_allocation(pred_gains, alpha=alpha),
        "shrunk": alpha_fair_allocation(pred_gains, alpha=alpha,
                                        shrink=shrink),
        "smoothed": alpha_fair_allocation(data["smoothed_gains"], alpha=alpha),
        "uniform": np.full_like(true_gains, 1.0 / true_gains.shape[-1]),
        "persistence": alpha_fair_allocation(data["observed_gains"],
                                             alpha=alpha),
    }
    for arm, model in nets.items():
        allocations[arm] = allocate(model, data["pred_states"],
                                    device=args.device)

    # The bridge between the two halves of the experiment. The law was derived
    # for an arbitrary gain perturbation and checked on synthetic noise; the
    # surrogate's error is neither synthetic nor independent across regions, so
    # whether the law still predicts the plug-in arm's cost is a real question
    # and is answered here rather than assumed.
    law = (float("nan") if alpha == 0 else
           float(np.mean(predicted_welfare_loss(true_gains, pred_gains,
                                                alpha))))

    rows = []
    for arm, alloc in allocations.items():
        row = {"surrogate": surrogate, "alpha": alpha, "arm": arm,
               "horizon": horizon, "seed": seed,
               "n_samples": data["n_samples"]}
        row.update(diagnostics(true_gains, alloc, alpha))
        row["law_predicted_loss"] = law if arm == "plugin" else float("nan")
        rows.append(row)
    return rows


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


def summarise(arm_rows: list[dict], gains: dict, shrinks: dict, horizon: int
              ) -> list[dict]:
    out = []
    for surrogate in dict.fromkeys(r["surrogate"] for r in arm_rows):
        for alpha in ALPHAS:
            row = {"surrogate": surrogate, "alpha": alpha, "horizon": horizon,
                   "exponent": (round(allocation_exponent(alpha), 4)
                                if alpha > 0 else float("inf")),
                   "fragility_coefficient": fragility_coefficient(alpha),
                   "price_of_fairness": round(float(np.mean(
                       price_of_fairness(gains[surrogate], alpha))), 5),
                   "shrink": round(shrinks[surrogate].get(alpha, float("nan")),
                                   3)}
            for arm in ARMS:
                vals = [r["rel_welfare_loss"] for r in arm_rows
                        if r["alpha"] == alpha and r["arm"] == arm
                        and r["horizon"] == horizon
                        and r["surrogate"] == surrogate]
                if vals:
                    row[f"{arm}_loss"] = float(np.mean(vals))
                    row[f"{arm}_sd"] = float(np.std(vals))
                errs = [r["gain_estimator_err"] for r in arm_rows
                        if r["alpha"] == alpha and r["arm"] == arm
                        and r["horizon"] == horizon
                        and r["surrogate"] == surrogate]
                if errs:
                    row[f"{arm}_gain_err"] = float(np.mean(errs))
            law = [r["law_predicted_loss"] for r in arm_rows
                   if r["alpha"] == alpha and r["arm"] == "plugin"
                   and r["horizon"] == horizon
                   and r["surrogate"] == surrogate]
            if law and np.isfinite(law[0]):
                row["law_predicted_loss"] = float(np.mean(law))
                row["law_ratio"] = (row["plugin_loss"] / row["law_predicted_loss"]
                                    if row.get("plugin_loss") is not None
                                    and row["law_predicted_loss"] > 0
                                    else float("nan"))
            ratios = [r["max_min_ratio"] for r in arm_rows
                      if r["alpha"] == alpha and r["arm"] == "plugin"
                      and r["horizon"] == horizon
                      and r["surrogate"] == surrogate]
            row["plugin_max_min_ratio"] = (round(float(np.mean(ratios)), 4)
                                           if ratios else float("nan"))
            out.append(row)
    return out


def plot(fragility, summary, horizon_rows, headline: str, out_png: Path
         ) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    positive = [a for a in ALPHAS if a > 0]
    # the loss is exactly 0 at alpha = 1, which a log axis cannot draw; floor it
    # so the plunge to the envy-free point is visible as a plunge rather than
    # silently clipped, and label the line so nobody reads the floor as a value
    floor = 1e-9

    def alpha_axis(ax):
        ax.set_xscale("log")
        ax.set_xticks(positive)
        ax.set_xticklabels([f"{a:g}" for a in positive])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.axvline(1.0, color="k", ls=":", lw=1)

    # (a) the law
    ax = axes[0]
    sigmas = sorted({r["sigma"] for r in fragility})
    for sigma, colour in zip(sigmas, plt.cm.viridis(
            np.linspace(0.15, 0.85, len(sigmas)))):
        rows = sorted([r for r in fragility
                       if r["sigma"] == sigma and r["alpha"] > 0],
                      key=lambda r: r["alpha"])
        ax.plot([r["alpha"] for r in rows],
                [max(r["measured_loss"], floor) for r in rows],
                "o", color=colour, label=f"measured, sd {sigma}")
        ax.plot([r["alpha"] for r in rows],
                [max(r["predicted_loss"], floor) for r in rows],
                "-", color=colour, alpha=0.6)
    alpha_axis(ax)
    ax.set(yscale="log", ylim=(floor, None),
           xlabel="alpha (fairness aversion)",
           ylabel="relative welfare loss",
           title="fragility law: lines predicted, dots measured")
    ax.annotate("envy-free:\nexactly 0", xy=(1.0, floor * 3), fontsize=7,
                ha="center", va="bottom")
    ax.legend(fontsize=7)

    # (b) the arms, both surrogates
    ax = axes[1]
    styles = {"plugin": ("tab:blue", "o"), "shrunk": ("tab:cyan", "d"),
              "smoothed": ("tab:olive", "*"), "learned": ("tab:red", "^"),
              "learned_robust": ("tab:orange", "v"),
              "uniform": ("tab:gray", "s"), "persistence": ("tab:green", "x")}
    for surrogate, dashes in (("strong", "-"), ("weak", "--")):
        rows = sorted([r for r in summary if r["surrogate"] == surrogate
                       and r["alpha"] > 0], key=lambda r: r["alpha"])
        if not rows:
            continue
        for arm, (colour, marker) in styles.items():
            vals = [max(r.get(f"{arm}_loss", np.nan), floor) for r in rows]
            ax.plot([r["alpha"] for r in rows], vals, dashes, marker=marker,
                    color=colour, ms=4,
                    label=arm if surrogate == "strong" else None)
    alpha_axis(ax)
    ax.set(yscale="log", ylim=(floor, None),
           xlabel="alpha (fairness aversion)",
           ylabel="relative welfare loss",
           title="arms (solid: strong surrogate, dashed: weak)")
    ax.legend(fontsize=7, ncol=2)

    # (c) horizon
    ax = axes[2]
    shown = [a for a in (0.5, 2.0, 8.0) if a in ALPHAS]
    for alpha, colour in zip(shown, ("tab:blue", "tab:purple", "tab:red")):
        for arm, style in (("plugin", "-"), ("persistence", "--")):
            rows = sorted([r for r in horizon_rows if r["alpha"] == alpha
                           and r["arm"] == arm
                           and r["surrogate"] == headline],
                          key=lambda r: r["horizon"])
            if not rows:
                continue
            ax.plot([r["horizon"] for r in rows],
                    [max(r["rel_welfare_loss"], 1e-9) for r in rows],
                    style, color=colour, label=f"{arm}, alpha {alpha}")
    ax.set(yscale="log", xlabel="rollout horizon (steps)",
           ylabel="relative welfare loss", title="forecast vs no forecast")
    ax.legend(fontsize=7)

    # (d) the equity-efficiency trade-off
    ax = axes[3]
    rows = sorted([r for r in summary if r["surrogate"] == headline
                   and r["alpha"] > 0], key=lambda r: r["alpha"])
    ax.plot([r["alpha"] for r in rows], [r["price_of_fairness"] for r in rows],
            "o-", color="tab:red", label="price of fairness")
    ax.plot([r["alpha"] for r in rows],
            [r["plugin_max_min_ratio"] for r in rows], "s-", color="tab:blue",
            label="min/max outcome")
    alpha_axis(ax)
    ax.set(xlabel="alpha (fairness aversion)", ylabel="fraction",
           title="what fairness costs and buys")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png.relative_to(_ROOT)}")


def print_report(fragility, summary, meta) -> None:
    print("\n" + "=" * 78)
    print("ext22  H3: what a surrogate's error costs a fair allocation")
    print("=" * 78)

    print(f"\n{meta['n_regions']} regions, population spread (max/min) "
          f"{meta['gain_spread']:.3f}, {meta['n_samples']} evaluation "
          f"decisions at horizon {meta['horizon']}")
    for name, info in meta["surrogates"].items():
        print(f"  {name:<7} one-step VRMSE {info['vrmse']:.5f}   "
              f"gain error {info['gain_rel_err']:.5f} relative, log sd "
              f"{info['gain_log_sd']:.5f} ({info['gain_log_sd_centred']:.5f} "
              f"after the common mode the rule ignores)   "
              f"[{info['n_train']} training trajectories]")
    print(f"  {'persist':<7} {'':<21}   gain error "
          f"{meta['persist_rel_err']:.5f} relative -- no forecast at all")

    print("\nthe fragility law on controlled noise (measured / predicted)")
    sigmas = sorted({r["sigma"] for r in fragility})
    print("  " + "alpha".rjust(6) + "".join(f"    sd={s:<7}" for s in sigmas))
    for alpha in [a for a in ALPHAS if a > 0]:
        cells = []
        for sigma in sigmas:
            row = next(r for r in fragility
                       if r["alpha"] == alpha and r["sigma"] == sigma)
            cells.append("       n/a  " if not np.isfinite(row["ratio"])
                         else f"   {row['ratio']:8.3f}")
        print(f"  {alpha:>6}" + "".join(cells))

    smallest = min(r["sigma"] for r in fragility)
    print(f"\nallocation sensitivity at sd={smallest}, against |1-alpha|/alpha")
    print("  " + " " * 10 + "".join(f"{a:>9}" for a in ALPHAS if a > 0))
    for label, key in (("theory  ", "amplification_theory"),
                       ("measured", "amplification")):
        cells = [next(r[key] for r in fragility
                      if r["alpha"] == a and r["sigma"] == smallest)
                 for a in ALPHAS if a > 0]
        print(f"  {label}  " + "".join(f"{c:>9.3f}" for c in cells))

    print("\nthe same law against a real surrogate's error, not synthetic noise")
    print(f"  {'surrogate':>10} {'alpha':>6} {'plug-in':>11} {'predicted':>11}"
          f" {'ratio':>7}")
    for row in summary:
        if "law_ratio" not in row or not np.isfinite(row["law_ratio"]):
            continue
        print(f"  {row['surrogate']:>10} {row['alpha']:>6} "
              f"{row['plugin_loss']:>11.3e} {row['law_predicted_loss']:>11.3e} "
              f"{row['law_ratio']:>7.3f}")

    for surrogate in dict.fromkeys(r["surrogate"] for r in summary):
        rows = [r for r in summary if r["surrogate"] == surrogate]
        print(f"\n{surrogate} surrogate, horizon {rows[0]['horizon']} "
              "(relative welfare loss, lower is better)")
        head = f"  {'alpha':>6} {'exponent':>9} {'PoF':>7} {'min/max':>8}"
        head += "".join(f" {arm[:9]:>10}" for arm in ARMS)
        print(head)
        for row in rows:
            exponent = (f"{row['exponent']:>9.3f}"
                        if np.isfinite(row["exponent"]) else f"{'inf':>9}")
            line = (f"  {row['alpha']:>6} {exponent} "
                    f"{row['price_of_fairness']:>7.3f} "
                    f"{row['plugin_max_min_ratio']:>8.3f}")
            for arm in ARMS:
                val = row.get(f"{arm}_loss")
                line += f" {val:>10.2e}" if val is not None else f" {'-':>10}"
            print(line)

        print("  " + "-" * 76)
        for row in rows:
            scored = {arm: row[f"{arm}_loss"] for arm in ARMS
                      if f"{arm}_loss" in row}
            best = min(scored, key=scored.get)
            network = min(("learned", "learned_robust"),
                          key=lambda a: scored.get(a, float("inf")))
            # at alpha = 1 every rule is equal division and every arm scores an
            # exact zero, so the ratio is 0/0 and reporting it as a number
            # would invent a winner out of float noise
            gap = (f"{scored[network] / scored['plugin']:>7.1f}x"
                   if scored.get("plugin", 0.0) > 1e-9 else "    n/a")
            hedge = (f", shrink {row['shrink']:.2f}"
                     if np.isfinite(row["shrink"]) else "")
            print(f"  alpha {row['alpha']:>5}: best {best:<15} "
                  f"best network / plugin ={gap}{hedge}")

        # why, not just which: an allocator that wins by reading the field
        # better shows it here, and one that wins by hedging does not
        print(f"\n  implied region gains vs the truth ({surrogate} surrogate; "
              "mean absolute error, relative gains)")
        arms = [a for a in ARMS if a != "uniform"]
        print("   " + f"{'alpha':>6}" + "".join(f" {a[:9]:>10}" for a in arms))
        for row in rows:
            if row["alpha"] in (0.0, 1.0):
                continue           # the rule carries no gain information there
            line = f"   {row['alpha']:>6}"
            for arm in arms:
                val = row.get(f"{arm}_gain_err")
                line += (f" {val:>10.5f}" if val is not None
                         and np.isfinite(val) else f" {'-':>10}")
            print(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-traj", type=int, default=48)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4,
                   help="the domain is split into blocks x blocks regions")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--headline-horizon", type=int, default=8)
    p.add_argument("--origin-stride", type=int, default=2,
                   help="roll out from every n-th start time")
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.01, 0.05, 0.2])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    # surrogate
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=10)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weak-traj", type=int, default=2,
                   help="trajectories for the deliberately starved surrogate")
    p.add_argument("--smooth-kernel", type=int, default=3,
                   help="box blur applied before pooling, for the control arm")
    # allocator
    p.add_argument("--alloc-width", type=int, default=16)
    p.add_argument("--alloc-epochs", type=int, default=60)
    p.add_argument("--alloc-lr", type=float, default=5e-3)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.quick:
        args.n_traj, args.n_steps = 12, 20
        args.horizons, args.headline_horizon = [1, 4], 4
        args.seeds, args.sigmas = [0], [0.05]
        args.epochs, args.alloc_epochs = 6, 8
        args.width, args.layers = 16, 2
        args.origin_stride = 4

    if args.headline_horizon not in args.horizons:
        args.horizons = sorted(set(args.horizons) | {args.headline_horizon})
    if max(args.horizons) >= args.n_steps:
        raise SystemExit(
            f"horizon {max(args.horizons)} needs more than {args.n_steps} "
            "steps of trajectory to score against")

    started = time.time()
    print("generating the ecosystem", flush=True)
    traj = make(args.n_traj, args.n_steps, args.size, seed=0)
    # Three-way split with a job each: the surrogate never sees the allocator's
    # data and the allocator never sees the evaluation's. Without that the
    # surrogate's errors on the allocator's training set are in-sample and
    # unrepresentatively small, which is exactly the quantity learned_robust
    # exists to learn about.
    splits = split_trajectories(traj, fractions=(0.5, 0.25, 0.25), seed=0)
    print(f"  {len(splits['train'])} surrogate / {len(splits['valid'])} "
          f"allocator / {len(splits['test'])} evaluation trajectories",
          flush=True)

    print("training the surrogates", flush=True)
    surrogates, meta_surrogates = {}, {}
    for name, n_train in (("strong", len(splits["train"])),
                          ("weak", min(args.weak_traj, len(splits["train"])))):
        model = train_surrogate(splits["train"][:n_train], args,
                                seed=args.seeds[0])
        surrogates[name] = model
        meta_surrogates[name] = {
            "vrmse": one_step_vrmse(model, splits["test"], args.device),
            "n_train": n_train}
        print(f"  {name:<7} {n_train:>2} trajectories, one-step VRMSE "
              f"{meta_surrogates[name]['vrmse']:.5f}"
              f"   ({time.time() - started:.0f}s)", flush=True)

    head = args.headline_horizon
    valid, test = {}, {}
    for name, model in surrogates.items():
        valid[name] = bundle(model, splits["valid"], head, args.origin_stride,
                             args)
        test[name] = {h: bundle(model, splits["test"], h, args.origin_stride,
                                args) for h in args.horizons}
        meta_surrogates[name].update(
            gain_error(test[name][head]["pred_gains"],
                       test[name][head]["true_gains"]))

    reference = test["strong"][head]
    spread = float(np.mean(reference["true_gains"].max(-1)
                           / reference["true_gains"].min(-1)))
    persist = gain_error(reference["observed_gains"], reference["true_gains"])
    print(f"  {reference['n_samples']} evaluation decisions per arm; "
          f"population spread {spread:.3f}; persistence gain error "
          f"{persist['gain_rel_err']:.5f}", flush=True)

    print("\nchecking the fragility law on controlled noise", flush=True)
    fragility = fragility_sweep(reference["true_gains"], args.sigmas, seed=0)

    print("running the arms", flush=True)
    arm_rows, horizon_rows = [], []
    shrinks = {name: {} for name in surrogates}
    for name in surrogates:
        print(f"  {name} surrogate", flush=True)
        for alpha in ALPHAS:
            # the hedge is fitted on the allocator split, never on the
            # evaluation split, or "tuned on validation" would mean tuned on
            # the test set
            shrinks[name][alpha] = 1.0 if alpha == 0 else tune_shrinkage(
                valid[name]["pred_gains"], valid[name]["true_gains"],
                alpha)["shrink"]
            t0 = time.time()
            for seed in args.seeds:
                nets = train_allocators(
                    alpha, args, valid[name]["true_states"],
                    valid[name]["true_gains"], valid[name]["pred_states"],
                    valid[name]["true_gains"], seed)
                horizons = args.horizons if name == "strong" else [head]
                for horizon in horizons:
                    rows = run_arms(alpha, args, test[name][horizon], nets,
                                    shrinks[name][alpha], seed, horizon, name)
                    horizon_rows.extend(rows)
                    if horizon == head:
                        arm_rows.extend(rows)
            done = {r["arm"]: r["rel_welfare_loss"] for r in arm_rows
                    if r["alpha"] == alpha and r["surrogate"] == name}
            print(f"    alpha {alpha:>5}  shrink {shrinks[name][alpha]:.2f}  "
                  + "  ".join(f"{a[:9]}={done.get(a, float('nan')):.2e}"
                              for a in ARMS)
                  + f"   ({time.time() - t0:.0f}s)", flush=True)

    gains_by_surrogate = {name: test[name][head]["true_gains"]
                          for name in surrogates}
    summary = summarise(arm_rows, gains_by_surrogate, shrinks, head)
    meta = {"n_regions": args.blocks ** 2, "gain_spread": spread,
            "horizon": head, "n_samples": reference["n_samples"],
            "persist_rel_err": persist["gain_rel_err"],
            "surrogates": meta_surrogates}

    write_csv(RESULTS / "ext22_fragility.csv", fragility)
    write_csv(RESULTS / "ext22_arms.csv", arm_rows)
    write_csv(RESULTS / "ext22_horizon.csv", horizon_rows)
    write_csv(RESULTS / "ext22_summary.csv", summary)
    plot(fragility, summary, horizon_rows, "strong",
         FIGURES / "ext22_fair_allocation.png")
    print_report(fragility, summary, meta)
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
