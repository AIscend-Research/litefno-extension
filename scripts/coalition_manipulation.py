r"""ext35: does the manipulation law survive coalitions? (H16)

    python3 scripts/coalition_manipulation.py
    python3 scripts/coalition_manipulation.py --quick

ext23 derived the incentive to misreport, ``|1-alpha|/alpha``, and checked it
against single-region deviations. That is the definition of strategy-proofness
and it is also the *weakest* threat model in the book: it assumes the regions
being allocated to never talk to each other. Regions that share a border, a
budget line or a political interest plainly do, so the derived law is spot-
checked rather than stress-tested until it is run against joint deviations.

This runs it against joint deviations, over every coalition size from 1 to R,
against exhaustively-searched best responses, and on both the alpha-fair family
and the capped leximin mechanism ext23 proposed as its defence.

The prediction being tested
---------------------------
Under ``a_r ∝ ghat_r^beta`` write ``w_r = g_r^beta`` and let coalition ``C``
hold truthful share ``s_C = sum_{r in C} w_r / sum_s w_s``. If every member
scales its report to the profitable corner, each ``w_r`` is multiplied by the
same ``lambda = kappa^|beta|``, so what the coalition captures is

    rho_C = lambda / (1 + (lambda - 1) s_C)                              (*)

which is **the single-region formula with s_r replaced by s_C**. Nothing about
the law changes under collusion; only its argument does. Three things follow
that are worth measuring rather than asserting:

1. ``beta = 0`` gives ``rho_C = 1`` for every coalition, so alpha = 1 is not
   merely strategy-proof but **group** strategy-proof -- the stronger property,
   obtained for the same reason (the rule ignores the state).
2. (*) is *decreasing* in ``s_C``, and ``s_C`` grows with the coalition, so
   colluding regions **dilute each other**. The worst case for the mechanism is
   a lone deviator, not a cartel.
3. The grand coalition captures ``rho = 1`` exactly: if everyone lies by the
   same factor the normalisation cancels and a universal cartel achieves
   nothing.

Point 2 is the one that could plausibly have gone the other way, and it is the
reason a derived law needs stress-testing: "manipulation gets worse when agents
collude" is the intuitive expectation and it is **false here**.

What is measured
----------------
1. **The law under joint deviation.** Measured capture against (*) for every
   coalition size, on real ecosystem gains.
2. **Is the assumed lie actually optimal?** (*) assumes every member goes to its
   own corner. That is verified by exhaustive search over all corner profiles
   and a random interior sample, rather than assumed from the single-agent
   argument.
3. **Ratio versus damage.** A cartel's *ratio* falls with size while the
   *budget it moves* rises. Both are reported, because they answer different
   questions and only reporting the first would understate the harm.
4. **Does the capacity defence survive?** ext23's bound is per region. A
   coalition of k regions pools k capacities, so the bound could fail to
   constrain the group. Measured against capped leximin directly.

Outputs
-------
``results/extensions/ext35_coalition_law.csv``     capture vs the closed form
``results/extensions/ext35_best_response.csv``     is the corner profile optimal
``results/extensions/ext35_leximin_cap.csv``       does the cap bound a cartel
``figures/extensions/ext35_coalition.png``
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from litefno.allocation import (                                # noqa: E402
    allocation_exponent, alpha_fair_allocation, outcomes, region_gains,
    welfare_ce)
from litefno.mechanism import (                                 # noqa: E402
    leximin_allocation, report_bounds)
from litefno.systems import lambda_omega                        # noqa: E402

RESULTS = _ROOT / "results" / "extensions"
FIGURES = _ROOT / "figures" / "extensions"

# identical to ext22/ext23 so the three sit on one ecosystem
ECOSYSTEM = dict(diffusion=0.4, omega=0.6, perturbation=0.8, max_mode=4,
                 spinup=20)

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
KAPPA = 1.5


# --------------------------------------------------------------------------
# coalition mechanics
# --------------------------------------------------------------------------


def corner_report(gains: np.ndarray, coalition, alpha: float,
                  kappa: float) -> np.ndarray:
    """Every member at its own profitable corner, others truthful.

    The direction is the sign of beta and is shared by all members, because the
    exponent is a property of the rule and not of the region: near max-
    efficiency everyone wants to look big, under egalitarian rules everyone
    wants to look needy. At beta = 0 the corner is the truth.
    """
    g = np.asarray(gains, dtype=float)
    low, high = report_bounds(g, kappa=kappa)
    beta = allocation_exponent(alpha)
    reported = g.copy()
    if beta == 0:
        return reported
    source = high if (beta > 0 or not np.isfinite(beta)) else low
    for r in coalition:
        reported[r] = source[r]
    return reported


def coalition_capture(gains: np.ndarray, coalition, alpha: float,
                      kappa: float, budget: float = 1.0) -> dict:
    """What the cartel wins, in ratio and in absolute budget.

    The two are reported together on purpose. The ratio is what the members care
    about and it *falls* as the cartel grows; the absolute transfer is what
    everyone else loses and it *rises*. Reading either alone gives the wrong
    answer about whether collusion matters.
    """
    g = np.asarray(gains, dtype=float)
    coalition = list(coalition)
    honest = alpha_fair_allocation(g, alpha=alpha, budget=budget)
    lied = alpha_fair_allocation(corner_report(g, coalition, alpha, kappa),
                                 alpha=alpha, budget=budget)
    inside = honest[coalition].sum()
    got = lied[coalition].sum()

    # a coalition that truthfully holds nothing and then captures something has
    # an unbounded ratio, not a huge finite one: flooring the denominator would
    # turn "infinitely profitable" into a number that averages like any other.
    # This is the alpha = 0 argmax case, where the allocation is discontinuous.
    if inside <= 1e-12:
        ratio = 1.0 if got <= 1e-12 else float("inf")
    else:
        ratio = float(got / inside)

    outsiders = [r for r in range(len(g)) if r not in set(coalition)]
    ce_honest = float(welfare_ce(outcomes(g, honest), alpha))
    ce_lied = float(welfare_ce(outcomes(g, lied), alpha))
    return {
        "ratio": ratio,
        "captured": float(got - inside),
        "truthful_share": float(inside / budget),
        "outsider_loss": (float(1.0 - lied[outsiders].sum()
                                / max(honest[outsiders].sum(), 1e-300))
                          if outsiders else 0.0),
        "welfare_loss": float(1.0 - ce_lied / max(ce_honest, 1e-300)),
    }


def capture_formula(gains: np.ndarray, coalition, alpha: float,
                    kappa: float) -> float:
    """Equation (*): the single-region law evaluated at the pooled share.

    Returns NaN at alpha = 0, where the rule is a discontinuous argmax and no
    elasticity describes it -- the same exclusion ext23 makes, kept explicit so
    the comparison table cannot quietly compare against a wrong number.
    """
    beta = allocation_exponent(alpha)
    if not np.isfinite(beta):
        return float("nan")
    if beta == 0:
        return 1.0
    w = np.asarray(gains, dtype=float) ** beta
    share = float(w[list(coalition)].sum() / w.sum())
    scale = kappa ** abs(beta)
    return float(scale / (1 + (scale - 1) * share))


def greedy_coalition(gains: np.ndarray, alpha: float, size: int) -> list:
    """The cartel that captures most at this size: the smallest-share members.

    (*) is decreasing in the coalition's truthful share, so the best coalition
    of a given size is the one holding the least -- which is a consequence of
    the law rather than a search heuristic, and is checked against exhaustive
    enumeration at small sizes in the tests.
    """
    beta = allocation_exponent(alpha)
    if beta == 0 or not np.isfinite(beta):
        # no ordering is meaningful: every coalition captures the same amount
        return list(range(size))
    w = np.asarray(gains, dtype=float) ** beta
    return sorted(np.argsort(w)[:size].tolist())


# --------------------------------------------------------------------------
# 1. the law under joint deviation
# --------------------------------------------------------------------------


def coalition_law(gains: np.ndarray, kappa: float, sample: int = 48
                  ) -> list[dict]:
    rows = []
    picked = gains[:sample]
    n = picked.shape[-1]
    for alpha in ALPHAS:
        for size in range(1, n + 1):
            meas, pred, caps, outs, welf, shares, solo = ([], [], [], [],
                                                          [], [], [])
            for g in picked:
                C = greedy_coalition(g, alpha, size)
                got = coalition_capture(g, C, alpha, kappa)
                # what the same regions would win deviating one at a time,
                # each against truthful others: the superadditivity baseline
                solo.append(sum(coalition_capture(g, [r], alpha, kappa)
                                ["captured"] for r in C))
                meas.append(got["ratio"])
                caps.append(got["captured"])
                outs.append(got["outsider_loss"])
                welf.append(got["welfare_loss"])
                shares.append(got["truthful_share"])
                pred.append(capture_formula(g, C, alpha, kappa))
            finite = [m for m in meas if np.isfinite(m)]
            rows.append({
                "unbounded_fraction": float(np.mean(
                    [not np.isfinite(m) for m in meas])),
                "alpha": alpha,
                "exponent": allocation_exponent(alpha),
                "coalition_size": size,
                "kappa": kappa,
                "truthful_share": float(np.mean(shares)),
                "capture_ratio": (float(np.mean(finite)) if finite
                                  else float("inf")),
                "capture_ratio_formula": float(np.mean(pred)),
                "budget_captured": float(np.mean(caps)),
                "budget_captured_solo_sum": float(np.mean(solo)),
                "superadditive": bool(np.mean(caps) > np.mean(solo) * (1 + 1e-9)),
                "outsider_loss": float(np.mean(outs)),
                "welfare_loss": float(np.mean(welf)),
            })
    return rows


# --------------------------------------------------------------------------
# 2. is the corner profile actually the coalition's best response
# --------------------------------------------------------------------------


def best_response_check(gains: np.ndarray, kappa: float, sizes=(2, 3, 4),
                        sample: int = 12, interior: int = 200,
                        seed: int = 0) -> list[dict]:
    """Exhaustive over corners, sampled over the interior.

    The single-agent corner argument is monotonicity in one's own report, and it
    does not automatically extend: a coalition maximising its *total* could in
    principle do better with a member sacrificing. It cannot here -- each
    ``a_r`` is monotone in ``w_r`` with a denominator common to all, so every
    member's corner is dominant *within* the coalition too -- but that is the
    kind of argument that should be checked against a search rather than
    believed.
    """
    rng = np.random.default_rng(seed)
    rows = []
    picked = gains[:sample]
    for alpha in ALPHAS:
        if alpha == 0:
            continue                       # argmax: handled by direct measure
        for size in sizes:
            beaten, margins = 0, []
            for g in picked:
                C = greedy_coalition(g, alpha, size)
                low, high = report_bounds(g, kappa=kappa)
                base = alpha_fair_allocation(
                    corner_report(g, C, alpha, kappa), alpha=alpha)[C].sum()
                best = base
                for bits in itertools.product((0, 1), repeat=size):
                    rep = g.copy()
                    for r, b in zip(C, bits):
                        rep[r] = high[r] if b else low[r]
                    best = max(best, alpha_fair_allocation(
                        rep, alpha=alpha)[C].sum())
                for _ in range(interior):
                    rep = g.copy()
                    for r in C:
                        rep[r] = rng.uniform(low[r], high[r])
                    best = max(best, alpha_fair_allocation(
                        rep, alpha=alpha)[C].sum())
                margins.append(float(best / base - 1.0))
                if best > base * (1 + 1e-9):
                    beaten += 1
            rows.append({"alpha": alpha, "coalition_size": size,
                         "kappa": kappa, "n_states": len(picked),
                         "profiles_searched": 2 ** size + interior,
                         "corner_beaten": beaten,
                         "max_margin": float(np.max(margins))})
    return rows


# --------------------------------------------------------------------------
# 3. does the capacity defence bound a cartel
# --------------------------------------------------------------------------


def leximin_cap_check(gains: np.ndarray, multipliers, kappas,
                      sizes=(1, 2, 4, 8), sample: int = 16) -> list[dict]:
    """ext23's cap, attacked by a group instead of an individual.

    The per-region bound is ``c_r / a_r``. A coalition of k regions is bounded
    by ``sum_C c_r / sum_C a_r``, which is a *weaker* constraint per unit of
    resource than any single member's -- so the question is not whether the cap
    holds (it must, it is a feasibility constraint) but whether it still bites
    once k capacities are pooled.

    The coalition's joint best response is searched over corners, since leximin
    is not a power rule and has no exponent to read a direction off.
    """
    rows = []
    picked = gains[:sample]
    n = picked.shape[-1]
    for multiplier in multipliers:
        caps = (None if not np.isfinite(multiplier)
                else np.full(n, multiplier / n))
        for kappa in kappas:
          for size in sizes:
            ratios, bounds = [], []
            for g in picked:
                truthful = leximin_allocation(g, 1.0, caps)
                # least-served members have the most room to gain
                C = sorted(np.argsort(truthful)[:size].tolist())
                low, high = report_bounds(g, kappa=kappa)
                base = truthful[C].sum()
                best = base
                for bits in itertools.product((0, 1), repeat=size):
                    rep = g.copy()
                    for r, b in zip(C, bits):
                        rep[r] = high[r] if b else low[r]
                    best = max(best,
                               leximin_allocation(rep, 1.0, caps)[C].sum())
                ratios.append(best / max(base, 1e-300))
                if caps is not None:
                    bounds.append(float(caps[C].sum() / max(base, 1e-300)))
            rows.append({
                "cap_multiplier": multiplier, "coalition_size": size,
                "kappa": kappa,
                "max_capture_ratio": float(np.mean(ratios)),
                "structural_bound": (float(np.mean(bounds)) if bounds
                                     else float("inf")),
                "cap_binds": bool(bounds and
                                  np.mean(bounds) <= np.mean(ratios) + 1e-9),
            })
    return rows


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.relative_to(_ROOT)}")


def plot(law, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                # pragma: no cover
        print(f"  (no figure: {exc})")
        return
    shown = [a for a in ALPHAS if a > 0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for alpha in shown:
        sub = [r for r in law if r["alpha"] == alpha]
        sizes = [r["coalition_size"] for r in sub]
        axes[0].plot(sizes, [r["capture_ratio"] for r in sub], "o-", ms=3,
                     label=f"alpha={alpha:g}")
        axes[0].plot(sizes, [r["capture_ratio_formula"] for r in sub], "k--",
                     lw=0.8, alpha=0.5)
        axes[1].plot(sizes, [r["budget_captured"] for r in sub], "o-", ms=3,
                     label=f"alpha={alpha:g}")
        axes[2].plot(sizes, [r["welfare_loss"] for r in sub], "o-", ms=3,
                     label=f"alpha={alpha:g}")
    axes[0].set(xlabel="coalition size", ylabel="capture ratio",
                title="what the cartel wins (dashed: closed form)")
    axes[0].axhline(1.0, color="k", lw=0.6)
    axes[1].set(xlabel="coalition size", ylabel="budget captured",
                title="what everyone else loses")
    axes[2].set(xlabel="coalition size", ylabel="relative welfare loss",
                title="damage")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png.relative_to(_ROOT)}")


def print_report(law, response, cap) -> None:
    print("\n=== Does the manipulation law survive coalitions? (H16) ===\n")

    print("1. capture vs the closed form (*)")
    print(f"    {'alpha':>6} {'|C|':>4} {'share':>7} {'measured':>10} "
          f"{'formula':>10} {'unbdd':>6} {'budget':>9} {'welfare':>9}")
    for r in law:
        if r["coalition_size"] in (1, 2, 4, 8, 16):
            print(f"    {r['alpha']:6g} {r['coalition_size']:4d} "
                  f"{r['truthful_share']:7.3f} {r['capture_ratio']:10.5f} "
                  f"{r['capture_ratio_formula']:10.5f} "
                  f"{r['unbounded_fraction']:6.2f} "
                  f"{r['budget_captured']:9.5f} {r['welfare_loss']:9.5f}")
    print("    (alpha=0 is a discontinuous argmax: 'measured' averages only the"
          " finite\n     cells, and 'unbdd' is the fraction where a coalition"
          " holding nothing\n     truthfully captured something -- an"
          " unbounded ratio, not a large one.)")

    finite = [r for r in law if np.isfinite(r["capture_ratio"])
              and np.isfinite(r["capture_ratio_formula"])]
    err = max(abs(r["capture_ratio"] - r["capture_ratio_formula"])
              for r in finite)
    print(f"\n    max |measured - formula| = {err:.3e} over {len(finite)} cells")

    ones = [r for r in law if r["alpha"] == 1.0]
    worst = max(abs(r["capture_ratio"] - 1.0) for r in ones)
    print(f"    alpha=1, every coalition size: max |ratio - 1| = {worst:.3e}"
          "  -> group strategy-proof")

    grand = [r for r in law if r["coalition_size"] == 16 and r["alpha"] > 0]
    gworst = max(abs(r["capture_ratio"] - 1.0) for r in grand)
    print(f"    grand coalition, every alpha>0: max |ratio - 1| = {gworst:.3e}"
          "  -> a universal cartel wins nothing")

    print("\n    dilution: capture ratio by size, per alpha")
    for alpha in [a for a in ALPHAS if a > 0]:
        sub = sorted([r for r in law if r["alpha"] == alpha],
                     key=lambda r: r["coalition_size"])
        if allocation_exponent(alpha) == 0:
            continue
        seq = [r["capture_ratio"] for r in sub]
        mono = all(b <= a + 1e-12 for a, b in zip(seq, seq[1:]))
        print(f"      alpha={alpha:4g}  {seq[0]:.4f} -> {seq[-1]:.4f}   "
              f"monotone non-increasing: {mono}")

    print("\n2. is the all-corner profile the coalition's best response")
    beaten = sum(r["corner_beaten"] for r in response)
    total = sum(r["n_states"] for r in response)
    searched = sum(r["profiles_searched"] * r["n_states"] for r in response)
    print(f"    beaten in {beaten} of {total} cells "
          f"({searched} alternative profiles searched)")
    print(f"    largest margin any alternative achieved: "
          f"{max(r['max_margin'] for r in response):.3e}")

    print("\n3. does the capacity cap still bind a cartel")
    print(f"    {'cap':>6} {'kappa':>6} {'|C|':>4} {'ratio':>9} {'bound':>9}"
          f" {'binds':>6}")
    for r in cap:
        print(f"    {r['cap_multiplier']:6g} {r['kappa']:6g} "
              f"{r['coalition_size']:4d} {r['max_capture_ratio']:9.4f} "
              f"{r['structural_bound']:9.4f} {str(r['cap_binds']):>6}")

    print("\n4. superadditivity: joint capture vs the sum of solo deviations")
    print(f"    {'alpha':>6} {'|C|':>4} {'joint':>10} {'sum solo':>10} "
          f"{'joint/solo':>11}")
    for r in law:
        if r["alpha"] > 0 and r["coalition_size"] in (2, 4, 8, 16):
            ratio = (r["budget_captured"] / r["budget_captured_solo_sum"]
                     if abs(r["budget_captured_solo_sum"]) > 1e-12
                     else float("nan"))
            print(f"    {r['alpha']:6g} {r['coalition_size']:4d} "
                  f"{r['budget_captured']:10.5f} "
                  f"{r['budget_captured_solo_sum']:10.5f} {ratio:11.4f}")
    supers = [r for r in law if r["alpha"] > 0 and r["coalition_size"] > 1
              and r["superadditive"]]
    print(f"    superadditive in {len(supers)} of "
          f"{len([r for r in law if r['alpha'] > 0 and r['coalition_size'] > 1])}"
          " cells")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-traj", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=24)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--sample", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--fig-dir", type=Path, default=FIGURES)
    args = ap.parse_args()
    if args.quick:
        args.n_traj, args.n_steps, args.sample = 6, 8, 8

    start = time.time()
    field = lambda_omega(n_traj=args.n_traj, n_steps=args.n_steps,
                         size=args.size, seed=args.seed, **ECOSYSTEM)
    gains = region_gains(field, blocks=args.blocks)
    gains = gains.reshape(-1, gains.shape[-1])
    print(f"{len(gains)} states x {gains.shape[-1]} regions, kappa="
          f"{args.kappa}")

    law = coalition_law(gains, args.kappa, sample=args.sample)
    response = best_response_check(gains, args.kappa,
                                   sample=4 if args.quick else 12)
    cap = leximin_cap_check(gains, [np.inf, 4.0, 2.0, 1.5],
                            [args.kappa, 4.0, 16.0],
                            sample=4 if args.quick else 16)

    write_csv(args.out_dir / "ext35_coalition_law.csv", law)
    write_csv(args.out_dir / "ext35_best_response.csv", response)
    write_csv(args.out_dir / "ext35_leximin_cap.csv", cap)
    plot(law, args.fig_dir / "ext35_coalition.png")
    print_report(law, response, cap)
    print(f"\ndone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
