r"""ext23: is the allocation robust to manipulation? (H4)

Board task: "borrow from algorithmic game theory / mechanism design -- use
game-theoretic fairness notions (strategy-proof, truthful, no-regret) to ensure
the allocation recommendation is not just statistically fair but robust to
manipulation; implement a lightweight version (e.g. lexicographic max-min
fairness)."

    python3 scripts/strategic_allocation.py
    python3 scripts/strategic_allocation.py --quick

ext22 asked what a rule loses when its input is wrong by accident. H4 asks what
it loses when the input is wrong on purpose, and the answer turns out to be the
same number: region r's allocation is ``a_r ∝ ghat_r^((1-alpha)/alpha)``, so the
elasticity of the allocation to a *reported* gain and to an *erroneous* gain are
one derivative. **The incentive to lie and the sensitivity to error cannot be
separated.** Both vanish at alpha = 1, where the rule ignores the state
entirely, and that is the whole of the strategy-proofness available here.

Four measurements
-----------------
1. **Who can lie, and by how much.** The incentive ratio of every member of the
   alpha-fair family on real ecosystem gains, against its closed form, plus what
   one region's lie costs everyone else.
2. **Leximin, and the cap as a mechanism.** The requested lightweight
   implementation, verified to be leximin, and then used for the useful part:
   a per-region capacity bounds what any lie can win, without payments,
   verification, or any truthfulness machinery. The cap is a dial and this
   measures its frontier.
3. **Is the learned allocator manipulable?** ext22 concluded that the auxiliary
   network beats the closed form when the surrogate is weak. That conclusion is
   incomplete if the network is easier to fool, so both are attacked under one
   threat model -- a bounded perturbation of the field inside the attacking
   region's own block -- and the closed form's exact corner response is compared
   against projected gradient descent on the network.
4. **What a no-regret guarantee is worth here.** An exponentiated-gradient
   learner needs no surrogate at all and provably has no regret against the best
   fixed allocation. Measured against a forecast rather than against its own
   bound, because in an oscillating ecosystem no fixed allocation is any good
   and being as good as the best one certifies very little.

Outputs
-------
``results/extensions/ext23_manipulation.csv``  incentive ratios by alpha
``results/extensions/ext23_leximin.csv``       the capacity dial's frontier
``results/extensions/ext23_attack.csv``        closed form vs network, attacked
``results/extensions/ext23_online.csv``        no-regret against a forecast
``figures/extensions/ext23_strategic.png``
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
    max_min_ratio, outcomes, region_gains, relative_welfare_loss, welfare_ce)
from litefno.mechanism import (                                 # noqa: E402
    incentive_ratio, incentive_ratio_formula, leximin_allocation,
    leximin_incentive_ratio, manipulation_damage, regret_curve)
from litefno.models.allocator import (                          # noqa: E402
    RegionAllocator, allocate, fit_allocator)
from litefno.models.harmonic import HarmonicLiteFNO             # noqa: E402
from litefno.specscope import fit, one_step_vrmse               # noqa: E402
from litefno.systems import lambda_omega, split_trajectories    # noqa: E402

RESULTS = _ROOT / "results" / "extensions"
FIGURES = _ROOT / "figures" / "extensions"

# Must match scripts/fair_allocation.py: ext23 reads its threat model against
# ext22's numbers, and a different ecosystem would make the two incomparable.
ECOSYSTEM = dict(diffusion=0.4, omega=0.6, perturbation=0.8, max_mode=4,
                 spinup=20)

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
ATTACK_ALPHAS = [0.5, 2.0, 8.0]


def make(n_traj: int, n_steps: int, size: int, seed: int) -> np.ndarray:
    return lambda_omega(n_traj=n_traj, n_steps=n_steps, size=size, seed=seed,
                        **ECOSYSTEM)


def as_states(field: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.moveaxis(field, -1, -3)).astype(np.float32)


def train_surrogate(train_traj: np.ndarray, args, seed: int = 0):
    torch.manual_seed(seed)
    model = HarmonicLiteFNO(2, 2, width=args.width, modes=args.modes,
                            layers=args.layers, rank=args.rank)
    fit(model, train_traj, epochs=args.epochs, lr=args.lr, device=args.device,
        seed=seed)
    return model


@torch.no_grad()
def rollout(model, split: np.ndarray, horizon: int, stride: int,
            device: str = "cpu", batch: int = 64) -> dict:
    model.eval()
    origins = list(range(0, split.shape[1] - horizon, stride))
    starts = np.concatenate([as_states(split[:, o]) for o in origins])
    truth = np.concatenate([split[:, o + horizon] for o in origins])
    out = []
    for i in range(0, len(starts), batch):
        cur = torch.from_numpy(starts[i:i + batch]).to(device)
        for _ in range(horizon):
            cur = model(cur)
        out.append(cur.cpu().numpy())
    return {"pred_states": np.concatenate(out), "true_field": truth}


# --------------------------------------------------------------------------
# 1. who can lie, and by how much
# --------------------------------------------------------------------------


def manipulation_table(gains: np.ndarray, kappa: float, sample: int = 64
                       ) -> list[dict]:
    """Incentive ratio per alpha, measured, against the closed form.

    Averaged over states rather than computed on one, because the ratio depends
    on the reporting region's share and that share moves with the ecosystem.
    """
    rows = []
    picked = gains[:sample]
    for alpha in ALPHAS:
        ratios, damages = [], []
        for g in picked:
            if alpha == 0:
                # the argmax rule is discontinuous: a lie either captures the
                # entire budget or changes nothing, so an elasticity is the
                # wrong description and the ratio is measured directly
                truthful = alpha_fair_allocation(g, alpha=0.0)
                best = 1.0
                for r in range(len(g)):
                    lied = g.copy()
                    lied[r] = g[r] * kappa
                    got = alpha_fair_allocation(lied, alpha=0.0)[r]
                    if truthful[r] > 1e-12:
                        best = max(best, got / truthful[r])
                    elif got > 1e-12:
                        best = np.inf          # from nothing to everything
                ratios.append(best)
            else:
                ratios.append(float(np.max(incentive_ratio(g, alpha,
                                                           kappa=kappa))))
            worst = int(np.argmax(g))
            damages.append(manipulation_damage(g, alpha, worst,
                                               kappa=kappa)["welfare_loss"]
                           if alpha > 0 else np.nan)

        clean = [d for d in damages if np.isfinite(d)]
        finite = [r for r in ratios if np.isfinite(r)]
        beta = allocation_exponent(alpha)
        row = {"alpha": alpha,
               "exponent": beta if np.isfinite(beta) else float("inf"),
               "fragility_coefficient": fragility_coefficient(alpha),
               "kappa": kappa,
               "incentive_ratio": float(np.mean(finite)) if finite else np.inf,
               "unbounded_fraction": float(np.mean(
                   [not np.isfinite(r) for r in ratios])),
               "welfare_loss_from_one_liar": (float(np.mean(clean)) if clean
                                              else float("nan")),
               "strategy_proof": bool(np.allclose(ratios, 1.0))}
        if alpha > 0:
            # the closed form, evaluated at the mean reporting share
            weights = picked ** beta
            shares = weights / weights.sum(axis=-1, keepdims=True)
            row["incentive_ratio_formula"] = float(np.mean(
                [incentive_ratio_formula(float(s.min()), alpha, kappa)
                 for s in shares]))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# 2. leximin, and the cap as a mechanism
# --------------------------------------------------------------------------


def leximin_frontier(gains: np.ndarray, multipliers, kappas,
                     sample: int = 32) -> list[dict]:
    """What a capacity cap buys and what it costs, across the dial.

    The cap is the entire mechanism: it is not a fairness criterion and it does
    not make anyone truthful, it just makes lying bounded. So both sides are
    reported -- the worst incentive ratio it permits, and the welfare it gives
    up relative to uncapped leximin, which is the allocation it is constraining.

    Swept over the misreport bound as well as over the cap, because the two
    interact and reporting one bound would misrepresent the mechanism. Against a
    small distortion, uncapped leximin is already boundedly manipulable and the
    cap looks worthless; against a large one, the uncapped rule's exposure grows
    without limit while the capped rule's cannot exceed ``c_r / a_r``. The cap
    is insurance against the tail, and a sweep is the only way to show that.
    """
    rows = []
    picked = gains[:sample]
    n = picked.shape[-1]
    for multiplier in multipliers:
        caps_of = (lambda: None) if not np.isfinite(multiplier) \
            else (lambda: np.full(n, multiplier / n))
        losses, floors, bounds = [], [], []
        for g in picked:
            alloc = leximin_allocation(g, budget=1.0, caps=caps_of())
            free = leximin_allocation(g, budget=1.0, caps=None)
            x = outcomes(g, alloc)
            # scored under the leximin objective itself: the worst-off region
            losses.append(1.0 - welfare_ce(x, np.inf)
                          / welfare_ce(outcomes(g, free), np.inf))
            floors.append(float(max_min_ratio(x)))
            if np.isfinite(multiplier):
                # the structural bound: nobody can be given more than the cap
                bounds.append(float(np.max((multiplier / n) / alloc)))
        for kappa in kappas:
            ratios = [float(np.max(leximin_incentive_ratio(
                g, 1.0, caps_of(), kappa=kappa))) for g in picked]
            rows.append({
                "cap_multiplier": multiplier, "kappa": kappa,
                "max_incentive_ratio": float(np.mean(ratios)),
                "structural_bound": (float(np.mean(bounds)) if bounds
                                     else float("inf")),
                "welfare_loss_vs_uncapped": float(np.mean(losses)),
                "max_min_ratio": float(np.mean(floors))})
    return rows


# --------------------------------------------------------------------------
# 3. attacking the learned allocator and the closed form alike
# --------------------------------------------------------------------------


def region_mask(blocks: int, region: int, height: int, width: int
                ) -> torch.Tensor:
    """1 inside the attacking region's own block, 0 everywhere else."""
    mask = torch.zeros(1, 1, height, width)
    row, col = divmod(region, blocks)
    bh, bw = height // blocks, width // blocks
    mask[:, :, row * bh:(row + 1) * bh, col * bw:(col + 1) * bw] = 1.0
    return mask


def attack_network(model, states: np.ndarray, region: int, blocks: int,
                   epsilon: float, steps: int = 40, device: str = "cpu"
                   ) -> np.ndarray:
    """Projected gradient ascent on one region's own share of the budget.

    L-infinity bounded and supported only on the attacker's block, which is the
    same licence the closed-form attacker gets. The difference is what the two
    can do with it: shifting a block mean is the *only* thing that moves the
    closed form, while a network reads the pattern as well, so the attacker has
    strictly more levers against it. Whether that translates into a bigger win
    is the measurement.

    Sign steps rather than raw gradient steps: the constraint is an L-infinity
    ball, and the steepest ascent direction inside one is the gradient's sign.
    """
    model.eval()
    x = torch.as_tensor(np.ascontiguousarray(states), dtype=torch.float32,
                        device=device)
    mask = region_mask(blocks, region, x.shape[-2], x.shape[-1]).to(device)
    delta = torch.zeros_like(x, requires_grad=True)
    step = 2.5 * epsilon / steps
    for _ in range(steps):
        share = model(x + delta * mask)[:, region].sum()
        grad, = torch.autograd.grad(share, delta)
        with torch.no_grad():
            delta += step * grad.sign()
            delta.clamp_(-epsilon, epsilon)
    with torch.no_grad():
        return model(x + delta * mask).cpu().numpy()


def attack_table(model, states: np.ndarray, true_gains: np.ndarray,
                 pred_gains: np.ndarray, alpha: float, args) -> list[dict]:
    """Both rules, one threat model, one metric.

    The metric is the incentive ratio -- what the attacking region multiplies
    its own allocation by -- so a rule that is merely *inaccurate* under attack
    does not score as manipulable. What is being measured is the payoff to the
    attacker, which is what decides whether anyone attacks.
    """
    blocks = args.blocks
    honest_net = allocate(model, states, device=args.device)
    honest_rule = alpha_fair_allocation(pred_gains, alpha=alpha)

    # leximin's best response has to be searched per state, not on an average
    # gain vector: the cap binds on whichever regions are currently small, and
    # averaging first hides exactly the states where it binds
    caps = np.full(blocks ** 2, args.cap / blocks ** 2)
    lex_ratios = np.array([
        leximin_incentive_ratio(g, 1.0, caps, epsilon=args.epsilon)
        for g in pred_gains[:args.leximin_sample]])

    rows = []
    for region in range(blocks ** 2):
        attacked = attack_network(model, states, region, blocks, args.epsilon,
                                  args.attack_steps, args.device)
        net_ratio = float(np.mean(attacked[:, region]
                                  / np.maximum(honest_net[:, region], 1e-12)))

        # the closed form sees only the block mean, so the best perturbation
        # inside the ball is a constant +eps or -eps -- no search needed
        beta = allocation_exponent(alpha)
        lied = pred_gains.copy()
        lied[:, region] = pred_gains[:, region] + (args.epsilon if beta > 0
                                                   else -args.epsilon)
        lied = np.maximum(lied, 1e-9)
        rule_ratio = float(np.mean(
            alpha_fair_allocation(lied, alpha=alpha)[:, region]
            / np.maximum(honest_rule[:, region], 1e-12)))

        lex_ratio = float(np.mean(lex_ratios[:, region]))

        rows.append({"alpha": alpha, "region": region,
                     "epsilon": args.epsilon,
                     "closed_form_ratio": rule_ratio,
                     "leximin_capped_ratio": lex_ratio,
                     "network_ratio": net_ratio,
                     "welfare_loss_network": float(np.mean(
                         relative_welfare_loss(true_gains, attacked, alpha))),
                     "welfare_loss_honest_network": float(np.mean(
                         relative_welfare_loss(true_gains, honest_net,
                                               alpha)))})
    return rows


# --------------------------------------------------------------------------
# 4. what a no-regret guarantee is worth
# --------------------------------------------------------------------------


def online_comparison(surrogate, split: np.ndarray, args) -> dict:
    """The model-free learner against a one-step forecast, on one lead time.

    Everything here decides at lead 1: the allocator must commit before the
    step it is scored on. That is the setting the online learner is built for --
    it sees a gain vector only after acting on it -- and putting the forecast on
    the same lead time is what makes the comparison about *information* rather
    than about how far ahead each method was asked to look.
    """
    true_gains = region_gains(split, blocks=args.blocks)      # (traj, T, R)
    rows, curves = [], []
    for index in range(len(split)):
        stream = true_gains[index]
        for alpha in args.online_alphas:
            run = regret_curve(stream[1:], alpha=alpha, eta=args.eta)
            curves.append({"trajectory": index, "alpha": alpha,
                           "steps": run["steps"],
                           "average_regret": run["average_regret"]})

            # one-step surrogate forecast from each state
            with torch.no_grad():
                states = torch.from_numpy(as_states(split[index, :-1]))
                predicted = surrogate(states.to(args.device)).cpu().numpy()
            forecast_gains = region_gains(np.moveaxis(predicted, -3, -1),
                                          blocks=args.blocks)

            target = stream[1:]
            arms = {
                "online_no_regret": run["allocations"],
                "best_fixed": np.repeat(run["best_fixed"][None], len(target),
                                        axis=0),
                "forecast": alpha_fair_allocation(forecast_gains, alpha=alpha),
                "persistence": alpha_fair_allocation(stream[:-1], alpha=alpha),
                "uniform": np.full_like(target, 1.0 / target.shape[-1]),
            }
            for arm, alloc in arms.items():
                rows.append({
                    "trajectory": index, "alpha": alpha, "arm": arm,
                    "rel_welfare_loss": float(np.mean(
                        relative_welfare_loss(target, alloc, alpha))),
                })
    return {"rows": rows, "curves": curves}


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


def _alpha_axis(ax, alphas) -> None:
    """Label a log alpha axis at the alphas actually run.

    Matplotlib's default log minor ticks collide into an unreadable smear at
    this range, and the axis is the independent variable of three of the four
    panels.
    """
    ax.set_xscale("log")
    ax.set_xticks(list(alphas))
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.minorticks_off()


def plot(manipulation, frontier, attack, online, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))

    # (a) the unification: one exponent governs lying and error alike
    ax = axes[0]
    rows = [r for r in manipulation if r["alpha"] > 0]
    alphas = [r["alpha"] for r in rows]
    ax.plot(alphas, [np.log(r["incentive_ratio"]) / np.log(r["kappa"])
                     for r in rows], "o-", color="tab:red",
            label="measured manipulation elasticity")
    ax.plot(alphas, [abs(r["exponent"]) for r in rows], "--", color="k",
            label=r"$|1-\alpha|/\alpha$ (also the error amplification)")
    ax.set(xlabel="alpha (fairness aversion)",
           ylabel="d log(allocation) / d log(report)",
           title="lying and erring are one derivative")
    _alpha_axis(ax, alphas)
    ax.axvline(1.0, color="k", ls=":", lw=1)
    ax.legend(fontsize=7)

    # (b) the capacity dial
    ax = axes[1]
    for kappa, colour in zip(sorted({r["kappa"] for r in frontier}),
                             ("tab:purple", "tab:red", "tab:brown")):
        finite = [r for r in frontier if np.isfinite(r["cap_multiplier"])
                  and r["kappa"] == kappa]
        finite.sort(key=lambda r: r["welfare_loss_vs_uncapped"])
        ax.plot([r["welfare_loss_vs_uncapped"] for r in finite],
                [r["max_incentive_ratio"] for r in finite], "o-",
                color=colour, label=f"misreport bound {kappa:g}x")
        for row in finite:
            ax.annotate(f"{row['cap_multiplier']:g}x",
                        (row["welfare_loss_vs_uncapped"],
                         row["max_incentive_ratio"]), fontsize=6,
                        textcoords="offset points", xytext=(4, 3))
    ax.set(yscale="log")
    ax.legend(fontsize=7)
    ax.set(xlabel="welfare given up vs uncapped leximin",
           ylabel="worst incentive ratio the cap permits",
           title="the capacity cap as a dial")

    # (c) attacked: closed form vs network
    ax = axes[2]
    width = 0.35
    shown = sorted({r["alpha"] for r in attack})
    for offset, (key, colour, label) in enumerate((
            ("closed_form_ratio", "tab:blue", "closed form"),
            ("leximin_capped_ratio", "tab:purple", "leximin (capped)"),
            ("network_ratio", "tab:red", "learned network"))):
        heights = [np.mean([r[key] for r in attack if r["alpha"] == a])
                   for a in shown]
        errs = [np.max([r[key] for r in attack if r["alpha"] == a])
                for a in shown]
        pos = np.arange(len(shown)) + (offset - 1) * width * 0.9
        ax.bar(pos, heights, width * 0.85, color=colour, label=label)
        ax.plot(pos, errs, "k_", ms=10)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ceiling = max(r[k] for r in attack for k in
                  ("closed_form_ratio", "leximin_capped_ratio",
                   "network_ratio"))
    ax.set(xticks=np.arange(len(shown)),
           xticklabels=[f"alpha {a:g}" for a in shown],
           ylim=(1.0, 1.0 + 1.25 * (ceiling - 1.0)),
           ylabel="incentive ratio (bars mean, ticks worst)",
           title=f"same attack, both rules (eps={attack[0]['epsilon']})")
    ax.legend(fontsize=7)

    # (d) no-regret against a forecast
    ax = axes[3]
    arms = ["online_no_regret", "best_fixed", "persistence", "uniform",
            "forecast"]
    colours = ["tab:red", "tab:pink", "tab:green", "tab:gray", "tab:blue"]
    shown = sorted({r["alpha"] for r in online["rows"]})
    for arm, colour in zip(arms, colours):
        vals = [np.mean([r["rel_welfare_loss"] for r in online["rows"]
                         if r["arm"] == arm and r["alpha"] == a])
                for a in shown]
        ax.plot(shown, np.maximum(vals, 1e-9), "o-", color=colour, label=arm)
    ax.set(yscale="log", xlabel="alpha (fairness aversion)",
           ylabel="relative welfare loss",
           title="model-free learning vs a one-step forecast")
    _alpha_axis(ax, shown)
    ax.legend(fontsize=7)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png.relative_to(_ROOT)}")


def print_report(manipulation, frontier, attack, online, meta) -> None:
    print("\n" + "=" * 78)
    print("ext23  H4: is the allocation robust to manipulation?")
    print("=" * 78)

    print(f"\nsurrogate one-step VRMSE {meta['vrmse']:.5f}; "
          f"{meta['n_regions']} regions; a region may distort the field inside "
          f"its own\nblock by at most {meta['epsilon']} per pixel, or its "
          f"reported gain by a factor {meta['kappa']}.")

    print("\n1. who can lie, and by how much")
    print(f"  {'alpha':>6} {'exponent':>9} {'incentive':>10} {'closed form':>12}"
          f" {'group loss':>11}  verdict")
    for row in manipulation:
        ratio = ("      inf" if not np.isfinite(row["incentive_ratio"])
                 or row["unbounded_fraction"] > 0
                 else f"{row['incentive_ratio']:>9.4f}")
        formula = (f"{row['incentive_ratio_formula']:>12.4f}"
                   if "incentive_ratio_formula" in row else f"{'-':>12}")
        loss = (f"{row['welfare_loss_from_one_liar']:>11.2e}"
                if np.isfinite(row["welfare_loss_from_one_liar"])
                else f"{'-':>11}")
        verdict = ("strategy-proof" if row["strategy_proof"]
                   else "manipulable")
        if row["unbounded_fraction"] > 0:
            verdict += (f" (unbounded on "
                        f"{row['unbounded_fraction']:.0%} of states)")
        exponent = (f"{row['exponent']:>9.3f}"
                    if np.isfinite(row["exponent"]) else f"{'inf':>9}")
        print(f"  {row['alpha']:>6} {exponent} {ratio} {formula} {loss}  "
              f"{verdict}")

    print("\n2. leximin: what a capacity cap buys and costs")
    kappas = sorted({r["kappa"] for r in frontier})
    print(f"  {'cap':>12} {'welfare given up':>17} {'min/max':>9}"
          + "".join(f"   worst lie, kappa={k:g}" for k in kappas))
    seen = []
    for row in frontier:
        if row["cap_multiplier"] in seen:
            continue
        seen.append(row["cap_multiplier"])
        cap = ("uncapped" if not np.isfinite(row["cap_multiplier"])
               else f"{row['cap_multiplier']:g}x equal")
        line = (f"  {cap:>12} {row['welfare_loss_vs_uncapped']:>17.4f} "
                f"{row['max_min_ratio']:>9.4f}")
        for kappa in kappas:
            match = next(r for r in frontier
                         if r["cap_multiplier"] == row["cap_multiplier"]
                         and r["kappa"] == kappa)
            line += f"{match['max_incentive_ratio']:>22.4f}"
        print(line)

    print(f"\n3. the same attack against both rules (eps = {meta['epsilon']})")
    print(f"  {'alpha':>6} {'closed form':>22} {'leximin capped':>22} "
          f"{'learned network':>22}")
    print(f"  {'':>6} {'mean':>10}{'worst':>12} {'mean':>10}{'worst':>12} "
          f"{'mean':>10}{'worst':>12} {'honest loss':>13}")
    for alpha in sorted({r["alpha"] for r in attack}):
        sel = [r for r in attack if r["alpha"] == alpha]
        line = f"  {alpha:>6}"
        for key in ("closed_form_ratio", "leximin_capped_ratio",
                    "network_ratio"):
            vals = [r[key] for r in sel]
            line += f" {np.mean(vals):>10.4f}{np.max(vals):>12.4f}"
        # an allocator that ignores its input cannot be manipulated and is
        # also worthless, so responsiveness is reported beside robustness
        line += f" {np.mean([r['welfare_loss_honest_network'] for r in sel]):>13.2e}"
        print(line)

    print("\n4. what the no-regret guarantee is worth")
    arms = ["forecast", "persistence", "online_no_regret", "best_fixed",
            "uniform"]
    print(f"  {'alpha':>6}" + "".join(f" {a[:16]:>17}" for a in arms))
    for alpha in sorted({r["alpha"] for r in online["rows"]}):
        line = f"  {alpha:>6}"
        for arm in arms:
            vals = [r["rel_welfare_loss"] for r in online["rows"]
                    if r["arm"] == arm and r["alpha"] == alpha]
            line += f" {np.mean(vals):>17.3e}"
        print(line)
    negative = [c for c in online["curves"] if c["average_regret"][-1] < 0]
    print(f"  the learner beats its own comparator on "
          f"{len(negative)}/{len(online['curves'])} runs -- regret against a "
          f"fixed\n  allocation is negative when the ecosystem oscillates, "
          f"which is what makes\n  the guarantee weak rather than reassuring.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-traj", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--origin-stride", type=int, default=2)
    p.add_argument("--kappa", type=float, default=1.2,
                   help="multiplicative bound on a misreported gain")
    p.add_argument("--kappa-grid", type=float, nargs="+", default=[1.2, 10.0],
                   help="misreport bounds to sweep the capacity dial against")
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="per-pixel bound on a field-level attack")
    p.add_argument("--cap", type=float, default=1.1,
                   help="leximin capacity, in multiples of an equal share")
    p.add_argument("--cap-grid", type=float, nargs="+",
                   default=[1.02, 1.05, 1.10, 1.15, 1.20, 1.30])
    p.add_argument("--leximin-sample", type=int, default=64,
                   help="states to search leximin best responses over")
    p.add_argument("--attack-steps", type=int, default=40)
    p.add_argument("--online-alphas", type=float, nargs="+",
                   default=[0.5, 2.0, 8.0])
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--weak-traj", type=int, default=2)
    # surrogate
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=10)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    # allocator
    p.add_argument("--alloc-width", type=int, default=16)
    p.add_argument("--alloc-epochs", type=int, default=60)
    p.add_argument("--alloc-lr", type=float, default=5e-3)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.quick:
        args.n_traj, args.n_steps = 12, 20
        args.horizon, args.origin_stride = 4, 4
        args.epochs, args.alloc_epochs, args.attack_steps = 6, 8, 8
        args.width, args.layers = 16, 2
        args.online_alphas = [2.0]
        args.cap_grid = [1.05, 1.2]
        args.leximin_sample = 8
        args.kappa_grid = [1.2, 10.0]

    started = time.time()
    print("generating the ecosystem", flush=True)
    traj = make(args.n_traj, args.n_steps, args.size, seed=0)
    splits = split_trajectories(traj, fractions=(0.5, 0.25, 0.25), seed=0)

    print("training the surrogate", flush=True)
    # the weak surrogate, because ext22 found the learned allocator is only
    # worth using there -- attacking the network in the regime where nobody
    # would deploy it would be attacking a straw man
    surrogate = train_surrogate(splits["train"][:args.weak_traj], args, seed=0)
    strong = train_surrogate(splits["train"], args, seed=0)
    vrmse = one_step_vrmse(surrogate, splits["test"], args.device)
    print(f"  weak surrogate one-step VRMSE {vrmse:.5f}; strong "
          f"{one_step_vrmse(strong, splits['test'], args.device):.5f}"
          f"   ({time.time() - started:.0f}s)", flush=True)

    valid = rollout(surrogate, splits["valid"], args.horizon,
                    args.origin_stride, args.device)
    test = rollout(surrogate, splits["test"], args.horizon, args.origin_stride,
                   args.device)
    valid_gains = region_gains(valid["true_field"], blocks=args.blocks)
    test_gains = region_gains(test["true_field"], blocks=args.blocks)
    test_pred_gains = region_gains(
        np.moveaxis(test["pred_states"], -3, -1), blocks=args.blocks)

    print("\n1. incentive ratios across the family", flush=True)
    manipulation = manipulation_table(test_gains, args.kappa)

    print("2. the leximin capacity dial", flush=True)
    frontier = leximin_frontier(test_gains, list(args.cap_grid) + [np.inf],
                                args.kappa_grid)

    print("3. attacking the closed form and the network", flush=True)
    attack = []
    for alpha in ATTACK_ALPHAS:
        torch.manual_seed(0)
        net = RegionAllocator(in_channels=2, blocks=args.blocks,
                              width=args.alloc_width)
        fit_allocator(net, valid["pred_states"], valid_gains, alpha=alpha,
                      epochs=args.alloc_epochs, lr=args.alloc_lr,
                      device=args.device, seed=0)
        rows = attack_table(net, test["pred_states"], test_gains,
                            test_pred_gains, alpha, args)
        attack.extend(rows)
        print(f"  alpha {alpha:>4}: closed form "
              f"{np.mean([r['closed_form_ratio'] for r in rows]):.4f}, "
              f"leximin {np.mean([r['leximin_capped_ratio'] for r in rows]):.4f}"
              f", network {np.mean([r['network_ratio'] for r in rows]):.4f}"
              f"   ({time.time() - started:.0f}s)", flush=True)

    print("4. the no-regret learner against a one-step forecast", flush=True)
    online = online_comparison(strong, splits["test"], args)

    meta = {"vrmse": vrmse, "n_regions": args.blocks ** 2,
            "epsilon": args.epsilon, "kappa": args.kappa}

    write_csv(RESULTS / "ext23_manipulation.csv", manipulation)
    write_csv(RESULTS / "ext23_leximin.csv", frontier)
    write_csv(RESULTS / "ext23_attack.csv", attack)
    write_csv(RESULTS / "ext23_online.csv", online["rows"])
    plot(manipulation, frontier, attack, online,
         FIGURES / "ext23_strategic.png")
    print_report(manipulation, frontier, attack, online, meta)
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
