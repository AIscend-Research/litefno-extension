r"""Ablate the instrument: does the pole readout need a trained operator? (ext33)

ext19 extracts a per-mode transfer function from a trained model, fits poles to
it, and scores those poles against an answer known in closed form. It reports a
rank correlation of 0.987 against the exact magnitudes and a frequency error of
1.5e-4, and ext20 and ext21 build on that readout.

None of that establishes the poles come from *learning*. A spectral operator has
mode-indexed structure before it is trained at all: the truncation, the CP
factorization and the initialization are all functions of wavenumber, and ext20
already found a wavenumber-only baseline reproduces the resonance-risk
correlation (+0.78 against the readout's -0.78, leaving -0.14 after partialling).
If the pole structure survives the removal of training, then what ext19 measures
is the architecture, not what the network learned, and the SpecScope line rests
on an instrument that reads its own construction.

This runs the identical readout on weights that never learned anything.

The arms
--------
Every arm goes through the *same* extraction, the same probes, the same pole
fit and the same scoring. Only the weights differ.

``trained``     the ext19 arm: initialise, fit, extract.
``untrained``   identical architecture and identical seed, **zero** optimizer
                steps. Isolates learning from architecture-plus-initialization,
                because the weights are exactly the ones training started from.
``shuffled``    take the *trained* weights and randomly permute the entries
                within each parameter tensor. This is the sharper control: it
                destroys every learned relationship while preserving each
                tensor's exact multiset of values, so the arm matches the
                trained model's weight distribution, scale and sparsity to the
                last element and differs only in arrangement.
``resampled``   fresh weights drawn from a Gaussian matched to each trained
                tensor's mean and standard deviation. Preserves the first two
                moments but not the multiset, which separates "the distribution
                carries the structure" from "the particular values do".

What each outcome would mean
----------------------------
If ``trained`` scores well and the three controls do not, the readout measures
learning and ext19's validation stands.

If the controls score comparably, the pole structure is an artifact of the
architecture. That would not make ext19's numbers wrong -- the extractor really
does recover the exact poles -- but it would make them uninformative about the
trained operator, and every claim downstream of the readout would inherit that.

The quantity that decides it is the one ext19 leads with: the rank correlation
between extracted and exact pole magnitude, computed identically here.

A control the comparison needs
------------------------------
Rank correlation against the exact poles is not enough on its own, because the
exact poles are themselves smooth functions of wavenumber. An arm could score
well by reproducing *any* monotone function of ``|k|`` without carrying pole
information at all -- which is exactly the trap ext20 fell into and had to
partial out.

So each arm is also scored against wavenumber alone, and the partial correlation
of extracted-with-exact given radius is reported next to the raw one. An arm
whose raw correlation is high and whose partial correlation is near zero is
reading the grid, not the operator.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_OP = Path(__file__).resolve().parent / "operator_poles.py"


def _load_ext19():
    spec = importlib.util.spec_from_file_location("operator_poles", _OP)
    m = importlib.util.module_from_spec(spec)
    sys.modules["operator_poles"] = m
    spec.loader.exec_module(m)
    return m


op19 = _load_ext19()
import torch                                                    # noqa: E402

from litefno.models.harmonic import HarmonicLiteFNO             # noqa: E402
from litefno.operator import (                                  # noqa: E402
    analytic_mode_operators, classify_operator_modes, compare_operators,
    empirical_mode_operators, operator_poles)
from litefno.specscope import fit, one_step_vrmse               # noqa: E402
from litefno.systems import split_trajectories                  # noqa: E402

ARMS = ("trained", "untrained", "shuffled", "resampled")


# --------------------------------------------------------------------------
# the weight interventions
# --------------------------------------------------------------------------


def shuffle_weights(model, seed: int):
    """Permute entries within each parameter tensor.

    Preserves every tensor's exact multiset of values -- mean, variance, extremes
    and sparsity are untouched -- and destroys only the arrangement. Scalars and
    single-element tensors are left alone: there is nothing to permute, and
    silently "shuffling" them would overstate how much of the model was
    scrambled.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() < 2:
                continue
            flat = p.reshape(-1)
            perm = torch.randperm(flat.numel(), generator=g)
            p.copy_(flat[perm].reshape(p.shape))
    return model


def resample_weights(model, seed: int):
    """Redraw each tensor from a Gaussian matched to its mean and std.

    The CP spectral weights are **complex**, and a complex tensor has no single
    real mean to match. Real and imaginary parts are therefore matched
    separately, which preserves the marginal moments of both without assuming
    the two are exchangeable -- a spectral weight whose imaginary part carries
    the phase is not the same object with real and imaginary swapped.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() < 2:
                continue
            if torch.is_complex(p):
                re, im = p.real, p.imag
                new_re = (torch.randn(p.shape, generator=g, dtype=re.dtype)
                          * re.std() + re.mean())
                new_im = (torch.randn(p.shape, generator=g, dtype=im.dtype)
                          * im.std() + im.mean())
                p.copy_(torch.complex(new_re, new_im))
            else:
                p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype)
                        * p.std() + p.mean())
    return model


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def partial_spearman(x, y, z) -> float:
    """Spearman of x and y with z partialled out, on ranks.

    ext20's lesson in one function: the raw correlation between a readout and a
    ground truth that are both smooth in wavenumber says little until wavenumber
    is removed from both sides.

    **This returns NaN on these systems, and that is the correct answer rather
    than a failure.** The exact pole of both closed-form systems depends only on
    ``|k|`` -- measured rho(exact magnitude, radius) is -1.000 for rotating and
    -0.9998 for advection, with every radius mapping to a single magnitude. So
    partialling radius out of the ground truth removes the ground truth
    entirely, the residual is identically zero, and no partial correlation
    exists to report. The guard below returns NaN instead of dividing by it.

    That degeneracy is the reason this study controls by ablating the *weights*
    rather than by partialling the wavenumber: the ablation arms hold the entire
    wavenumber structure fixed and vary only what the weights contain.
    """
    x, y, z = (op19._rank(np.asarray(v, dtype=float)) for v in (x, y, z))
    def resid(a, b):
        b = b - b.mean()
        denom = float((b ** 2).sum())
        if denom <= 0:
            return a - a.mean()
        return (a - a.mean()) - b * float((a * b).sum()) / denom
    rx, ry = resid(x, z), resid(y, z)
    dx, dy = float(np.sqrt((rx ** 2).sum())), float(np.sqrt((ry ** 2).sum()))
    if dx <= 0 or dy <= 0:
        return float("nan")
    return float((rx * ry).sum() / (dx * dy))


# --------------------------------------------------------------------------


def run_arm(system: str, weights: str, args, device: str) -> dict:
    """One (system, weights) cell. The readout is ext19's, untouched."""
    traj = op19.make_data(system, args.n_traj, args.n_steps, args.size, seed=0)
    splits = split_trajectories(traj, seed=0)
    channels = traj.shape[-1]

    # seed before construction, exactly as ext19 does: the init draws from the
    # global RNG and this study reads its numbers off the weights
    torch.manual_seed(args.seed)
    model = HarmonicLiteFNO(channels, channels, width=args.width,
                            modes=args.modes, layers=args.layers,
                            rank=args.rank)

    t0 = time.time()
    if weights == "untrained":
        pass                                   # the point of the arm
    else:
        fit(model, splits["train"], epochs=args.epochs, lr=args.lr,
            device=device, seed=args.seed)
        if weights == "shuffled":
            shuffle_weights(model, args.seed + 991)
        elif weights == "resampled":
            resample_weights(model, args.seed + 991)
    vrmse = one_step_vrmse(model, splits["test"], device)

    base = splits["test"][0, 0].transpose(2, 0, 1)
    analytic = analytic_mode_operators(model, gelu_gain=args.gelu_gain)
    empirical = empirical_mode_operators(model, base, max_mode=args.max_mode,
                                         eps=args.eps, device=device)
    agreement = compare_operators(analytic, empirical)
    poles = operator_poles(empirical["operators"])
    labels = classify_operator_modes(poles)

    ext_mag, ext_frq, exa_mag, exa_frq, rad, rows = [], [], [], [], [], []
    for i, (ky, kx) in enumerate(zip(empirical["ky"], empirical["kx"])):
        lead = int(np.argmax(poles["sigma"][i]))
        exact = op19.exact_pole(system, ky, kx)
        r = float(np.hypot(ky, kx))
        row = {"system": system, "weights": weights,
               "ky": int(ky), "kx": int(kx), "radius": r,
               "extracted_magnitude": float(poles["magnitude"][i, lead]),
               "extracted_freq": float(poles["freq"][i, lead]),
               "label": str(labels[i])}
        if exact is not None:
            em, ef = float(abs(exact)), float(abs(np.angle(exact)) / (2 * np.pi))
            row.update(exact_magnitude=em, exact_freq=ef,
                       magnitude_error=row["extracted_magnitude"] - em,
                       freq_error=row["extracted_freq"] - ef)
            ext_mag.append(row["extracted_magnitude"]); exa_mag.append(em)
            ext_frq.append(row["extracted_freq"]); exa_frq.append(ef)
            rad.append(r)
        rows.append(row)

    summary = {"system": system, "weights": weights,
               "test_vrmse": round(float(vrmse), 6),
               "n_modes": len(rows), "seconds": round(time.time() - t0, 1)}
    if ext_mag:
        summary.update(
            rho_magnitude=round(op19.spearman(ext_mag, exa_mag), 4),
            rho_freq=round(op19.spearman(ext_frq, exa_frq), 4),
            rho_magnitude_vs_radius=round(op19.spearman(ext_mag, rad), 4),
            rho_exact_vs_radius=round(op19.spearman(exa_mag, rad), 4),
            partial_rho_magnitude=round(
                partial_spearman(ext_mag, exa_mag, rad), 4),
            median_abs_freq_error=round(float(np.median(
                np.abs(np.array(ext_frq) - np.array(exa_frq)))), 6))
    else:
        summary.update(rho_magnitude=float("nan"), rho_freq=float("nan"),
                       rho_magnitude_vs_radius=float("nan"),
                       rho_exact_vs_radius=float("nan"),
                       partial_rho_magnitude=float("nan"),
                       median_abs_freq_error=float("nan"))
    summary["route_rel_diff_median"] = round(float(np.median(
        [a["rel_norm_diff"] for a in agreement])), 6)
    print(f"    {weights:>10s}  vrmse={vrmse:8.5f}  rho_mag={summary['rho_magnitude']:+.4f}"
          f"  partial={summary['partial_rho_magnitude']:+.4f}"
          f"  freq_err={summary['median_abs_freq_error']:.2e}"
          f"  ({summary['seconds']:.0f}s)", flush=True)
    return {"summary": summary, "rows": rows}


def print_report(summaries: list) -> None:
    systems = list(dict.fromkeys(s["system"] for s in summaries))
    print("\n=== Does the pole readout need a trained operator? ===")
    print("    rho_mag  : Spearman(extracted, exact) -- ext19's headline check")
    print("    r_exact  : Spearman(exact, radius). At -1.000 the ground truth is")
    print("               a pure function of wavenumber, so partialling it out is")
    print("               undefined -- which is why the control is the weight")
    print("               ablation, not a partial correlation.")
    print("    a control that matches trained on rho_mag is reading the "
          "architecture\n")
    hdr = (f"    {'system':>10s} {'weights':>10s} {'vrmse':>9s} {'rho_mag':>8s} "
           f"{'r_exact':>8s} {'rho_freq':>8s} {'freq_err':>10s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for sysname in systems:
        for s in [x for x in summaries if x["system"] == sysname]:
            print(f"    {s['system']:>10s} {s['weights']:>10s} "
                  f"{s['test_vrmse']:>9.5f} {s['rho_magnitude']:>+8.4f} "
                  f"{s['rho_exact_vs_radius']:>+8.4f} "
                  f"{s['rho_freq']:>+8.4f} {s['median_abs_freq_error']:>10.2e}")
        print()

    print("=== verdict per system ===")
    for sysname in systems:
        got = {s["weights"]: s for s in summaries if s["system"] == sysname}
        if "trained" not in got:
            continue
        t = got["trained"]["rho_magnitude"]
        ctrl = {k: v["rho_magnitude"] for k, v in got.items() if k != "trained"}
        if not ctrl or not np.isfinite(t):
            continue
        best = max(ctrl, key=lambda k: abs(ctrl[k]))
        gap = abs(t) - abs(ctrl[best])
        verdict = ("READOUT NEEDS TRAINING" if gap > 0.25 else
                   "ARTIFACT: controls match trained" if gap < 0.05 else
                   "PARTIAL: controls carry much of it")
        print(f"    {sysname:>10s}: trained {t:+.4f}, best control "
              f"{best} {ctrl[best]:+.4f}, gap {gap:+.4f}  -> {verdict}")


def write_csv(path: Path, rows: list) -> None:
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
    p.add_argument("--systems", nargs="+", default=["rotating", "advection"])
    p.add_argument("--weights", nargs="+", default=list(ARMS), choices=list(ARMS))
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    args = p.parse_args()

    summaries, all_rows = [], []
    for sysname in args.systems:
        print(f"\n=== {sysname} ===", flush=True)
        for w in args.weights:
            out = run_arm(sysname, w, args, args.device)
            summaries.append(out["summary"]); all_rows.extend(out["rows"])

    print_report(summaries)
    write_csv(args.out_dir / "ext33_ablation_summary.csv", summaries)
    write_csv(args.out_dir / "ext33_ablation_modes.csv", all_rows)


if __name__ == "__main__":
    sys.exit(main())
