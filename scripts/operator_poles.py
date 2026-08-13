r"""ext19: read the poles out of a trained operator, and score them.

SpecScope steps 1-3. Steps 1 and 2 extract a per-mode transfer function from a
trained model and fit poles to it; step 3 is the part that makes the first two
worth anything -- checking the extracted poles against an answer known in closed
form, in a system where "what the network should have learned" is not a matter
of opinion.

    python3 scripts/operator_poles.py                      # the full study
    python3 scripts/operator_poles.py --quick              # a few minutes
    python3 scripts/operator_poles.py --checkpoint <path>  # a real checkpoint

Three arms, chosen so that a pass and a failure are both informative:

``rotating``    ``A_t = D lap A + i omega A``. Every mode has an exactly known
                complex pole, near-neutral at low wavenumber and strongly damped
                at high. If the extractor works, this is where it shows.
``advection``   ``u_t + c.grad u = nu lap u``. Also exact, but with *no*
                oscillation anywhere. This is the negative control: an extractor
                that reports resonant modes here is finding structure that is
                not in the system, and a method that only ever gets run on
                oscillatory data would never catch that.
``lambda``      the nonlinear limit cycle. Only its frequency is known, so only
                the frequency is scored -- see ``litefno.systems``.

What is being claimed, and what is not
--------------------------------------
The claim is about the *ranking and classification* of modes, not about the
absolute pole magnitudes. The composed operator carries a linearization gain
that rescales every magnitude by the same factor (``gelu_gain ** n_layers``), so
an absolute stability call from the analytic route alone is not trustworthy and
the report says so. The empirical route has no such factor, which is why both
are run and their disagreement is a reported number rather than a footnote.

Outputs
-------
``results/extensions/ext19_ground_truth.csv``   per-mode extracted vs exact
``results/extensions/ext19_route_agreement.csv``  analytic vs empirical
``results/extensions/ext19_summary.csv``        one row per arm
``figures/extensions/ext19_operator_poles.png``
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
    analytic_mode_operators, classify_operator_modes, compare_operators,
    empirical_mode_operators, linear_pde_propagator, operator_poles,
    probe_convergence)
from litefno.specscope import fit, one_step_vrmse              # noqa: E402
from litefno.systems import (                                  # noqa: E402
    advection_diffusion, lambda_omega, lambda_omega_frequency,
    rotating_diffusion, rotating_diffusion_pole, split_trajectories)

RESULTS = _ROOT / "results" / "extensions"
FIGURES = _ROOT / "figures" / "extensions"

# one place for the physics, so the data and the analytic comparison cannot
# drift apart the way they would if each read its own constants
PHYSICS = dict(diffusion=0.4, omega=0.6, dt=0.25, length=32.0)
ADVECTION = dict(nu=0.02, velocity=(0.0, 1.0), dt=0.5, length=32.0)


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def make_data(arm: str, n_traj: int, n_steps: int, size: int, seed: int):
    if arm == "rotating":
        return rotating_diffusion(n_traj=n_traj, n_steps=n_steps, size=size,
                                  seed=seed, **PHYSICS)
    if arm == "advection":
        return advection_diffusion(n_traj=n_traj, n_steps=n_steps, size=size,
                                   seed=seed, **ADVECTION)
    if arm == "lambda":
        return lambda_omega(n_traj=n_traj, n_steps=n_steps, size=size,
                            seed=seed, **PHYSICS)
    raise ValueError(f"unknown arm {arm!r}")


def exact_pole(arm: str, ky: float, kx: float):
    """The analytic pole of one mode, or None where there is no closed form."""
    if arm == "rotating":
        return rotating_diffusion_pole(np.hypot(ky, kx), PHYSICS["diffusion"],
                                       PHYSICS["omega"], PHYSICS["dt"],
                                       PHYSICS["length"])
    if arm == "advection":
        grid = linear_pde_propagator(np.array([ky]), np.array([kx]),
                                     dt=ADVECTION["dt"], nu=ADVECTION["nu"],
                                     velocity=ADVECTION["velocity"],
                                     height=int(ADVECTION["length"]),
                                     width=int(ADVECTION["length"]))
        return grid[0, 0]
    return None                       # lambda-omega: frequency only


# --------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------


def run_arm(arm: str, args, device: str) -> dict:
    print(f"\n=== {arm} ===", flush=True)
    traj = make_data(arm, args.n_traj, args.n_steps, args.size, seed=0)
    splits = split_trajectories(traj, seed=0)
    channels = traj.shape[-1]

    # Seed before constructing, not only before fitting. The initialization
    # draws from the global RNG, so seeding inside fit() alone leaves the
    # weights different on every run -- and this study reads its numbers off
    # the weights. Two identical invocations disagreed on H1's partial
    # correlation by enough to flip the conclusion before this was pinned.
    torch.manual_seed(args.seed)
    model = HarmonicLiteFNO(channels, channels, width=args.width,
                            modes=args.modes, layers=args.layers,
                            rank=args.rank)
    t0 = time.time()
    fitted = fit(model, splits["train"], epochs=args.epochs, lr=args.lr,
                 device=device, seed=args.seed)
    vrmse = one_step_vrmse(model, splits["test"], device)
    print(f"  trained in {time.time() - t0:.0f}s   test one-step VRMSE {vrmse:.5f}",
          flush=True)

    base = splits["test"][0, 0].transpose(2, 0, 1)
    analytic = analytic_mode_operators(model, gelu_gain=args.gelu_gain)
    empirical = empirical_mode_operators(model, base, max_mode=args.max_mode,
                                         eps=args.eps, device=device)
    agreement = compare_operators(analytic, empirical)

    poles = operator_poles(empirical["operators"])
    labels = classify_operator_modes(poles)

    rows = []
    for i, (ky, kx) in enumerate(zip(empirical["ky"], empirical["kx"])):
        lead = int(np.argmax(poles["sigma"][i]))
        exact = exact_pole(arm, ky, kx)
        row = {
            "arm": arm, "ky": int(ky), "kx": int(kx),
            "radius": float(np.hypot(ky, kx)),
            "extracted_magnitude": float(poles["magnitude"][i, lead]),
            "extracted_freq": float(poles["freq"][i, lead]),
            "extracted_sigma": float(poles["sigma"][i, lead]),
            "label": str(labels[i]),
        }
        if exact is not None:
            row.update(
                exact_magnitude=float(abs(exact)),
                exact_freq=float(abs(np.angle(exact)) / (2 * np.pi)),
                magnitude_error=float(poles["magnitude"][i, lead] - abs(exact)),
                freq_error=float(poles["freq"][i, lead]
                                 - abs(np.angle(exact)) / (2 * np.pi)))
        rows.append(row)

    # Is the composed route wrong, or wrong by a constant? The linearization
    # gain multiplies every mode's magnitude by the same factor, so a large
    # disagreement with a *tight* spread and a high rank correlation means the
    # composed route still ranks modes correctly and only its absolute scale is
    # unusable. Those are very different conclusions and the report separates
    # them rather than printing one relative difference and leaving it open.
    gaps = np.array([a["log_magnitude_gap"] for a in agreement])
    summary = {"arm": arm, "test_vrmse": round(vrmse, 6),
               "n_modes": len(rows), "epochs": args.epochs,
               "params": int(sum(p.numel() for p in model.parameters())),
               "route_rel_diff_median": round(float(np.median(
                   [a["rel_norm_diff"] for a in agreement])), 6),
               "route_rel_diff_max": round(float(np.max(
                   [a["rel_norm_diff"] for a in agreement])), 6),
               "route_log_gap_mean": round(float(gaps.mean()), 4),
               "route_log_gap_std": round(float(gaps.std()), 4),
               "route_magnitude_spearman": round(spearman(
                   [a["analytic_lead_magnitude"] for a in agreement],
                   [a["empirical_lead_magnitude"] for a in agreement]), 4)}

    scored = [r for r in rows if "exact_magnitude" in r]
    if scored:
        mag_err = np.abs([r["magnitude_error"] for r in scored])
        freq_err = np.abs([r["freq_error"] for r in scored])
        extracted = [r["extracted_magnitude"] for r in scored]
        exact_mag = [r["exact_magnitude"] for r in scored]
        summary.update(
            magnitude_mae=round(float(mag_err.mean()), 6),
            magnitude_max_error=round(float(mag_err.max()), 6),
            freq_mae=round(float(freq_err.mean()), 6),
            # the ranking is the claim; the absolute value carries the
            # linearization gain and the correlation does not
            magnitude_spearman=round(spearman(extracted, exact_mag), 4))
        summary.update(label_agreement(scored, args.neutral_tol))
    if arm == "lambda":
        want = lambda_omega_frequency(PHYSICS["omega"], 0.0, PHYSICS["dt"])
        osc = [r["extracted_freq"] for r in rows if r["label"] == "resonant"]
        # the least-damped mode is reported whatever its label: if the neutral
        # band is too tight to call anything resonant, the frequency the
        # operator does carry at its slowest-decaying mode is still the number
        # to compare against the known one, and suppressing it when the label
        # count is zero would hide the answer behind a threshold
        least = max(rows, key=lambda r: r["extracted_sigma"])
        summary.update(
            known_frequency=round(want, 6),
            least_damped_frequency=round(least["extracted_freq"], 6),
            median_resonant_frequency=round(float(np.median(osc)), 6)
            if osc else float("nan"),
            n_resonant=len(osc))

    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    summary.update({f"n_{k}": v for k, v in counts.items()})
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
    return {"rows": rows, "agreement": [dict(a, arm=arm) for a in agreement],
            "summary": summary, "model": model, "base": base}


def label_agreement(scored: list[dict], neutral_tol: float) -> dict:
    """Does the extracted classification match the one the closed form implies?

    Counting extracted labels alone cannot fail: a run that calls every mode
    damped looks tidy and says nothing. What matters is whether a mode the
    *system* makes near-neutral is called near-neutral, so both sides are
    labelled by the same rule and compared as a confusion matrix.

    This is where the method's resolution shows up. The neutral band is
    +/- ``neutral_tol`` in log magnitude, so when the extraction error is
    comparable to ``neutral_tol`` the labels cannot be right even though the
    poles are: a mean absolute error of 0.0066 against a band of 0.005 puts
    every near-neutral mode on the wrong side of the line. Reporting the
    smallest tolerance at which the labels do agree turns that from a silent
    failure into the method's stated resolution.
    """
    def near(values, tol):
        return np.abs(np.log(np.maximum(values, 1e-300))) <= tol

    extracted = np.array([r["extracted_magnitude"] for r in scored])
    exact = np.array([r["exact_magnitude"] for r in scored])
    got, want = near(extracted, neutral_tol), near(exact, neutral_tol)

    resolved = float("nan")
    for tol in np.geomspace(neutral_tol, 1.0, 40):
        if np.array_equal(near(extracted, tol), near(exact, tol)):
            resolved = float(tol)
            break
    return {"neutral_tol": neutral_tol,
            "n_exact_neutral": int(want.sum()),
            "n_extracted_neutral": int(got.sum()),
            "label_accuracy": round(float((got == want).mean()), 4),
            "tol_for_exact_labels": round(resolved, 5)
            if np.isfinite(resolved) else float("nan")}


def spearman(x, y) -> float:
    """Rank correlation without a scipy dependency."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(len(v), dtype=float)
    # average ties, or a plateau of equal values biases the correlation
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            sel = inverse == i
            ranks[sel] = ranks[sel].mean()
    return ranks


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(_ROOT)}")


def plot(results: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for res in results:
        scored = [r for r in res["rows"] if "exact_magnitude" in r]
        if not scored:
            continue
        ax.scatter([r["exact_magnitude"] for r in scored],
                   [r["extracted_magnitude"] for r in scored],
                   s=18, alpha=0.75, label=res["summary"]["arm"])
    lim = [0, 1.15]
    ax.plot(lim, lim, "k--", lw=1, label="exact")
    ax.set(xlim=lim, ylim=lim, xlabel="exact |z|",
           ylabel="extracted |z|", title="pole magnitude vs closed form")
    ax.legend(fontsize=8)

    ax = axes[1]
    for res in results:
        rows = sorted(res["rows"], key=lambda r: r["radius"])
        ax.plot([r["radius"] for r in rows], [r["extracted_freq"] for r in rows],
                "o", ms=3, alpha=0.6, label=res["summary"]["arm"])
    ax.axhline(PHYSICS["omega"] * PHYSICS["dt"] / (2 * np.pi), color="k",
               ls="--", lw=1, label="omega dt / 2pi")
    ax.set(xlabel="radial wavenumber |k|", ylabel="extracted frequency (cycles/step)",
           title="does the operator oscillate where the system does?")
    ax.legend(fontsize=8)

    ax = axes[2]
    for res in results:
        rel = [a["rel_norm_diff"] for a in res["agreement"]]
        ax.semilogy(sorted(rel), np.linspace(0, 1, len(rel)),
                    label=res["summary"]["arm"])
    ax.set(xlabel="relative difference, composed vs probed",
           ylabel="fraction of modes",
           title="do the two extraction routes agree?")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png.relative_to(_ROOT)}")


def print_report(summaries: list[dict]) -> None:
    print("\n" + "=" * 74)
    print("ext19  operator poles vs closed form")
    print("=" * 74)
    for s in summaries:
        print(f"\n{s['arm']}  (test VRMSE {s['test_vrmse']}, {s['n_modes']} modes)")
        if "magnitude_mae" in s:
            print(f"  |z| mean abs error   {s['magnitude_mae']:.5f}"
                  f"   (max {s['magnitude_max_error']:.5f})")
            print(f"  |z| rank correlation {s['magnitude_spearman']:+.4f}")
            print(f"  frequency mean error {s['freq_mae']:.5f}")
        if "label_accuracy" in s:
            print(f"  neutral labels       {s['label_accuracy']:.2%} correct at "
                  f"tol {s['neutral_tol']} "
                  f"({s['n_extracted_neutral']} found / {s['n_exact_neutral']} real)")
            print(f"  labels agree from    tol {s['tol_for_exact_labels']}")
        if "known_frequency" in s:
            print(f"  known frequency      {s['known_frequency']:.5f}")
            print(f"  least-damped mode    {s['least_damped_frequency']:.5f}")
            print(f"  median resonant      {s['median_resonant_frequency']:.5f}"
                  f"  over {s['n_resonant']} modes")
        print(f"  route agreement      median {s['route_rel_diff_median']:.2e}"
              f"  max {s['route_rel_diff_max']:.2e}")
        print(f"  route log-gap        {s['route_log_gap_mean']:+.3f}"
              f" +/- {s['route_log_gap_std']:.3f}"
              f"   rank corr {s['route_magnitude_spearman']:+.4f}")
        classes = {k[2:]: v for k, v in s.items() if k.startswith("n_")
                   and k not in ("n_modes", "n_resonant")}
        print(f"  classified           {classes}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arms", nargs="+",
                   default=["rotating", "advection", "lambda"])
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
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--gelu-gain", type=float, default=0.5)
    p.add_argument("--neutral-tol", type=float, default=5e-3,
                   help="half-width of the neutral band, in log magnitude")
    p.add_argument("--quick", action="store_true",
                   help="small model, few epochs; for checking the plumbing, "
                        "not for reading numbers off")
    p.add_argument("--device", default="cpu")
    p.add_argument("--probe-check", action="store_true",
                   help="report the finite-difference step-size plateau")
    args = p.parse_args()

    if args.quick:
        args.n_traj, args.n_steps, args.epochs = 8, 16, 8
        args.width, args.layers, args.max_mode = 16, 2, 4

    results = [run_arm(arm, args, args.device) for arm in args.arms]

    if args.probe_check:
        print("\nfinite-difference step size, rotating arm, mode (1, 1):")
        for step in probe_convergence(results[0]["model"], results[0]["base"],
                                      mode=(1, 1), device=args.device):
            print(f"  eps={step['eps']:<8g} |M|={step['norm']:.6f} "
                  f"rel change {step['rel_change']:.2e}")

    write_csv(RESULTS / "ext19_ground_truth.csv",
              [r for res in results for r in res["rows"]])
    write_csv(RESULTS / "ext19_route_agreement.csv",
              [a for res in results for a in res["agreement"]])
    write_csv(RESULTS / "ext19_summary.csv", [r["summary"] for r in results])
    plot(results, FIGURES / "ext19_operator_poles.png")
    print_report([r["summary"] for r in results])


if __name__ == "__main__":
    main()
