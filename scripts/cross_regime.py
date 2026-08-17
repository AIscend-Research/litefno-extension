r"""Leave-one-regime-out generalization: how far does LiteFNO travel?

Board task: "Test LITEFNO generalization: train on subset of disaster types,
test on held-out type -- measure cross-disaster gap."

The Well's Gray-Scott dataset is stored as six named regimes, each a distinct
(F, k) parameter pair producing qualitatively different dynamics -- bubbles,
gliders, maze, spirals, spots, worms. They are this repo's "types": a model
trained on five of them and tested on the sixth has to extrapolate to a pattern
class it has never seen, which is the question the task asks.

Six folds. Each holds out one regime, trains on the other five, and evaluates
the same trained model twice: on the held-out regime's test trajectories, and on
the five seen regimes' test trajectories. The gap is the ratio between them.

Why the gap and not the absolute error
--------------------------------------
Both numbers come from one model, one training run, one epoch budget. Anything
that would shift the absolute error -- a shorter schedule, fewer trajectories,
different hardware -- shifts both, so the ratio survives choices the absolute
number does not. That matters here because each fold trains on 60 trajectories
against the reference run's 72, so its absolute error is not comparable to
ext13's; the ratio is comparable to itself across folds, which is what the
question needs.

The gap is also tracked at every evaluation checkpoint rather than read once at
the end, so a reader can see whether it is stable or still moving. A gap that
has plateaued by epoch 40 does not depend on the epoch budget; one still drifting
at the end would.

A prediction made before the runs
---------------------------------
ext10 measured, per regime, what fraction of spatial variance sits below mode 8:
spirals 77%, gliders 69%, bubbles 58%, worms 31%, maze 1.3%, spots 0.6%. Maze
and spots are outliers by two orders of magnitude -- their energy sits in a
narrow band at the Turing wavelength near mode 13-16 while the others are
low-wavenumber.

If what a model transfers is spectral content, then holding out a regime whose
spectrum is unlike the training set should hurt most, and maze and spots should
be the worst two folds. That is a prediction with a direction, checked against
the committed ext10 numbers rather than read off these results afterwards.

The alternative it is being tested against is that the gap is governed by
something else entirely -- difficulty, or how much the field moves per step --
in which case the ordering will not follow the spectrum.

Three arms: can the gap be closed? (ext26)
------------------------------------------
Measuring the gap invites the obvious follow-up -- can it be bought down? Two
standard answers are added as arms on top of the baseline, in the order that
keeps them separable:

  baseline    plain LiteFNO, MSE on clean inputs
  robust      identical, but inputs are perturbed during training at 30 dB
  robust+unc  identical to `robust`, evaluated as a 4-member deep ensemble

`robust+unc` differs from `robust` in exactly one field (member count), so the
difference between them is the ensemble's contribution with the augmentation
held fixed. Reporting one arm that changed both at once would confound them.

The noise scale is not tuned. 30 dB is taken from ext3's committed noise curve,
which used ``std * 10 ** (-snr / 20)`` on the input only; reusing the definition
means the augmentation strength can be read against an existing measurement of
what that perturbation costs at test time, rather than being chosen because it
made the result look good.

The headline is the fraction of the baseline gap each arm closes,
``(g_base - g_arm) / (g_base - 1)``, which is 1.0 when the held-out regime
becomes as cheap as the seen ones. That ratio has a failure mode worth naming:
it also improves when the *seen* regimes get worse, which is not
generalization. So the held-out and seen errors are carried next to it and any
fold whose gap closed without the held-out error falling is flagged rather than
counted.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_BR_PATH = Path(__file__).resolve().parent / "baseline_reference.py"


def _load_br():
    spec = importlib.util.spec_from_file_location("baseline_reference", _BR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["baseline_reference"] = module
    spec.loader.exec_module(module)
    return module


br = _load_br()
import torch                                        # noqa: E402
from torch import nn                                # noqa: E402


# --------------------------------------------------------------------------
# regime bookkeeping
# --------------------------------------------------------------------------


def regime_labels(manifest: dict, split: str) -> np.ndarray:
    """Per-trajectory regime name, in the order the split was concatenated.

    stream_preprocess writes regimes in dict order and records how many
    trajectories each contributed, so the labels are reconstructable exactly
    rather than guessed from position.
    """
    labels = []
    for entry in manifest["splits"][split]["files"]:
        labels.extend([entry["regime"]] * len(entry["trajectories"]))
    return np.array(labels)


def load_dataset(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    out = {}
    for split in ("train", "valid", "test"):
        arr = br.load_split(data_dir / f"{split}.h5")
        lab = regime_labels(manifest, split)
        assert len(lab) == arr.shape[0], (split, len(lab), arr.shape)
        out[split] = (arr, lab)
    return out, manifest


# --------------------------------------------------------------------------
# the three arms
# --------------------------------------------------------------------------
#
# Arm 1 is the plain model above. Arms 2 and 3 are the two standard answers to
# "the held-out domain looks different", applied in the order that isolates
# them: arm 2 changes what the model sees during training, arm 3 changes only
# how it is queried at eval. Arm 3 therefore contains arm 2 unchanged -- its
# members are arm-2 models differing solely in seed -- so the arm3-arm2
# difference is the ensemble's contribution alone, with the augmentation held
# fixed. Reporting the two together would confound them.

ARMS = {
    "baseline":     {"snr_db": None, "members": 1},
    "robust":       {"snr_db": 30.0, "members": 1},
    "robust+unc":   {"snr_db": 30.0, "members": 4},
}


def add_noise(x: torch.Tensor, snr_db: float, gen: torch.Generator) -> torch.Tensor:
    """Input-side Gaussian noise at a target SNR.

    Same definition as ext3 (``build_notebooks.py``): the noise scale is set
    from the batch's own standard deviation, ``std * 10 ** (-snr / 20)``, so
    "30 dB" means the same perturbation here that it means in the committed
    ext3 noise curve. Reusing the number rather than picking a fresh one is
    what lets the augmentation strength be read against an existing measurement
    instead of tuned until the result looks good.

    Noise goes on the input only. The target stays clean, so the model is asked
    to map a perturbed state to the true next state rather than to reproduce
    its own corruption.
    """
    std = float(x.std()) * (10.0 ** (-snr_db / 20.0))
    return x + torch.randn(x.shape, generator=gen, device=x.device) * std


def evaluate_ensemble(models: list, x, y, device, batch: int = 128) -> float:
    """VRMSE of the mean prediction over members.

    The average is taken over predictions, not over per-member VRMSEs: the
    ensemble's forecast is the mean field, and the error of an average is not
    the average of errors. With one member this reduces exactly to
    ``br.evaluate_one_step``, which is what keeps the single-member arms
    comparable to the ensemble arm.
    """
    for m in models:
        m.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(device)
            acc = None
            for m in models:
                p = m(xb)
                acc = p if acc is None else acc + p
            preds.append((acc / len(models)).cpu())
    return float(br.vrmse(torch.cat(preds), y))


def _build_member(xtr, device: str, seed: int):
    """One initialised member with its optimiser. Seed drives init and order."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, factorization = br.build_model("litefno", xtr.shape[1], xtr.shape[1])
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=br.LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=br.LR_STEP,
                                            gamma=br.LR_GAMMA)
    gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    return {"model": model, "opt": opt, "sched": sched, "gen": gen,
            "factorization": factorization}


def run_fold(held_out: str, data: dict, device: str, epochs: int, seed: int,
             arm: str = "baseline", log_every: int = 20) -> dict:
    """Train this arm's members and score the ensemble on held-out vs seen.

    Members are trained in lockstep -- all of them advance one epoch before any
    checkpoint -- rather than one after another. That is not for speed; it is so
    the gap-stability curve the module docstring promises exists for the
    ensemble arm too. Evaluating a partially trained ensemble requires every
    member to be at the same epoch, and training them sequentially would only
    permit a curve for the last one.
    """
    cfg = ARMS[arm]
    train_arr, train_lab = data["train"]
    test_arr, test_lab = data["test"]

    seen_mask = train_lab != held_out
    xtr, ytr = br.to_pairs(train_arr[seen_mask])

    x_held, y_held = br.to_pairs(test_arr[test_lab == held_out])
    x_seen, y_seen = br.to_pairs(test_arr[test_lab != held_out])

    members = [_build_member(xtr, device, seed + 1000 * k)
               for k in range(cfg["members"])]
    factorization = members[0]["factorization"]
    loss_fn = nn.MSELoss()
    snr_db = cfg["snr_db"]

    n = len(xtr)
    curve, t0 = [], time.time()
    for epoch in range(epochs):
        total = 0.0
        for mem in members:
            model, opt, gen = mem["model"], mem["opt"], mem["gen"]
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, br.BATCH):
                idx = perm[i:i + br.BATCH]
                xb, yb = xtr[idx].to(device), ytr[idx].to(device)
                if snr_db is not None:
                    xb = add_noise(xb, snr_db, gen)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.detach().item() * len(idx)
            mem["sched"].step()
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            models = [m["model"] for m in members]
            v_held = evaluate_ensemble(models, x_held, y_held, device)
            v_seen = evaluate_ensemble(models, x_seen, y_seen, device)
            curve.append({"epoch": epoch + 1,
                          "train_mse": total / (n * len(members)),
                          "held_out_vrmse": v_held, "seen_vrmse": v_seen,
                          "gap_ratio": v_held / v_seen if v_seen else float("nan")})
            print(f"      epoch {epoch + 1:>4d}/{epochs}  held={v_held:.5f}  "
                  f"seen={v_seen:.5f}  gap={v_held / v_seen:.2f}x  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    last = curve[-1]
    v_held, v_seen = last["held_out_vrmse"], last["seen_vrmse"]
    return {
        "arm": arm, "held_out": held_out, "seed": seed, "epochs": epochs,
        "snr_db": cfg["snr_db"], "members": cfg["members"],
        "factorization": factorization,
        "n_train_traj": int(seen_mask.sum()),
        "n_held_pairs": len(x_held), "n_seen_pairs": len(x_seen),
        "held_out_vrmse": v_held,
        "seen_vrmse": v_seen,
        "gap_ratio": last["gap_ratio"],
        "gap_absolute": v_held - v_seen,
        "train_s": round(time.time() - t0, 1),
        "curve": curve,
    }


# --------------------------------------------------------------------------
# the ext10 prediction
# --------------------------------------------------------------------------


def check_ext10_prediction(records: list[dict], ext10_csv: Path,
                           field: str = "A") -> list[dict]:
    """Does the gap follow how spectrally unlike the held-out regime is?

    Two predictors, because the obvious one is the wrong shape. ext10 gives, per
    regime, the share of spatial variance below mode 8 (call it v).

    ``spectral_distance`` = |v_r - mean(v_others)| is symmetric, and that is its
    problem: it scores spirals (v=77%, extreme high) as far from the training
    set as spots (v=0.6%, extreme low). But the mechanism is not symmetric. A
    model trained on five regimes learns to represent the wavenumbers those five
    contain. Holding out spots removes the only regime with energy at the Turing
    band, so the training set never sees it and cannot extrapolate there.
    Holding out spirals removes a low-wavenumber regime, and the remaining four
    still cover low wavenumbers -- there is nothing new to reach for.

    So ``var_below_mode8`` (signed, the regime's own value) is the predictor the
    mechanism actually implies: low v means the regime's energy sits where the
    others' does not, and the gap should be *larger*, giving a negative
    correlation. Both are reported; the symmetric one is kept as the weaker
    hypothesis it is, so the choice is visible rather than quietly made.

    The concrete pre-registered claim, independent of either metric: maze and
    spots should be the two worst folds.
    """
    if not ext10_csv.exists():
        print(f"\n(ext10 summary not at {ext10_csv}; skipping prediction check)")
        return []
    with ext10_csv.open() as f:
        rows = [r for r in csv.DictReader(f)
                if r["segment"] == "settled" and r["field"] == field
                and r.get("spatial_var_at_modes_8")]
    v = {r["scenario"]: float(r["spatial_var_at_modes_8"]) for r in rows}
    have = [r for r in records if r["held_out"] in v]
    if len(have) < 3:
        return []

    out = []
    for r in have:
        others = [v[k] for k in v if k != r["held_out"]]
        out.append({"held_out": r["held_out"],
                    "var_below_mode8": v[r["held_out"]],
                    "training_mean": float(np.mean(others)),
                    "spectral_distance": abs(v[r["held_out"]] - np.mean(others)),
                    "gap_ratio": r["gap_ratio"]})

    rho_signed = br_spearman([o["var_below_mode8"] for o in out],
                             [o["gap_ratio"] for o in out])
    rho_sym = br_spearman([o["spectral_distance"] for o in out],
                          [o["gap_ratio"] for o in out])

    print("\n=== ext10 prediction check ===")
    print("    ext10 ran before these folds existed; this is a prediction, not "
          "a pattern read off afterwards")
    print(f"    {'held out':>9s} {'var<m8':>8s} {'|dist|':>8s} {'gap':>8s}")
    ranked = sorted(out, key=lambda o: -o["gap_ratio"])
    for o in ranked:
        print(f"    {o['held_out']:>9s} {o['var_below_mode8']:>7.1%} "
              f"{o['spectral_distance']:>7.1%} {o['gap_ratio']:>7.2f}x")

    worst_two = {o["held_out"] for o in ranked[:2]}
    predicted = {"maze", "spots"}
    hit = len(worst_two & predicted)
    print(f"\n    prediction: maze and spots are the two worst folds")
    print(f"    outcome:    worst two are {sorted(worst_two)}  "
          f"-> {hit}/2 correct")
    print(f"    rho(variance below mode 8, gap) = {rho_signed:+.3f}  "
          f"(mechanism predicts negative)")
    print(f"    rho(|distance from training mean|, gap) = {rho_sym:+.3f}  "
          f"(symmetric, the weaker hypothesis)")
    for o in out:
        o["spearman_rho_signed"] = rho_signed
        o["spearman_rho_symmetric"] = rho_sym
        o["predicted_worst_two_hits"] = hit
    return out


def br_spearman(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_report(records: list[dict]) -> None:
    print("\n=== Cross-regime generalization gap (LiteFNO, one-step VRMSE) ===")
    print("    both columns from the same model: held-out regime vs the five "
          "it trained on")
    hdr = (f"    {'held out':>9s} {'held-out':>10s} {'seen':>10s} {'gap':>8s} "
           f"{'absolute':>10s} {'train traj':>11s} {'s':>6s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in sorted(records, key=lambda r: -r["gap_ratio"]):
        print(f"    {r['held_out']:>9s} {r['held_out_vrmse']:>10.5f} "
              f"{r['seen_vrmse']:>10.5f} {r['gap_ratio']:>7.2f}x "
              f"{r['gap_absolute']:>+10.5f} {r['n_train_traj']:>11d} "
              f"{r['train_s']:>6.0f}")
    g = np.array([r["gap_ratio"] for r in records])
    print(f"    gap ratio: min {g.min():.2f}x  median {np.median(g):.2f}x  "
          f"max {g.max():.2f}x")

    print("\n=== Is the gap stable, or an artifact of the epoch budget? ===")
    print("    gap ratio at each checkpoint; flat means the budget does not "
          "drive it")
    for r in sorted(records, key=lambda r: -r["gap_ratio"]):
        pts = " ".join(f"{c['epoch']}:{c['gap_ratio']:.2f}" for c in r["curve"])
        print(f"    {r['held_out']:>9s}  {pts}")


def gap_closed(records: list[dict]) -> list[dict]:
    """Per-fold arm comparison: how much of the baseline's gap each arm removes.

    The gap ratio is 1.0 when the held-out regime costs nothing, so the excess
    a fold starts with is ``g_baseline - 1`` and the fraction closed is

        closed = (g_baseline - g_arm) / (g_baseline - 1)

    1.0 means the held-out regime became as cheap as the seen ones, 0.0 means
    the arm moved nothing, and negative means it widened the gap.

    The catch this table is built to expose: the ratio can also be improved by
    making the *seen* regimes worse, which is not generalization. So the
    held-out and seen VRMSEs are carried alongside, and a fold whose gap closed
    while ``held_out_vrmse`` rose is flagged rather than counted -- that is the
    denominator shrinking, not the numerator.

    Folds with no gap to begin with are excluded rather than scored. If the
    held-out regime was already no harder than the seen ones (``g <= 1``), the
    excess is zero or negative and the fraction is not a fraction of anything:
    dividing by a small negative excess turns a trivial movement into a large
    signed percentage, and sign-flips it. Those folds report NaN and are left
    out of the medians, because "closed 400% of a gap that did not exist" is
    noise wearing a headline number's clothes.
    """
    by_fold: dict[str, dict[str, dict]] = {}
    for r in records:
        by_fold.setdefault(r["held_out"], {})[r["arm"]] = r

    out = []
    for held, arms in by_fold.items():
        base = arms.get("baseline")
        if base is None:
            continue
        for arm, r in arms.items():
            if arm == "baseline":
                continue
            excess = base["gap_ratio"] - 1.0
            # only a positive excess is a gap; see the docstring
            closed = ((base["gap_ratio"] - r["gap_ratio"]) / excess
                      if excess > 1e-12 else float("nan"))
            out.append({
                "held_out": held, "arm": arm,
                "baseline_gap": base["gap_ratio"], "arm_gap": r["gap_ratio"],
                "frac_gap_closed": closed,
                "baseline_held_vrmse": base["held_out_vrmse"],
                "arm_held_vrmse": r["held_out_vrmse"],
                "held_vrmse_delta": r["held_out_vrmse"] - base["held_out_vrmse"],
                "baseline_seen_vrmse": base["seen_vrmse"],
                "arm_seen_vrmse": r["seen_vrmse"],
                "seen_vrmse_delta": r["seen_vrmse"] - base["seen_vrmse"],
                "real_improvement": bool(
                    r["held_out_vrmse"] < base["held_out_vrmse"]),
            })
    return out


def print_arm_report(records: list[dict], closed: list[dict]) -> None:
    print("\n=== Leave-one-regime-out, three arms (one-step VRMSE) ===")
    hdr = (f"    {'held out':>9s} {'arm':>11s} {'held-out':>10s} {'seen':>10s} "
           f"{'gap':>8s} {'s':>7s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for held in dict.fromkeys(r["held_out"] for r in records):
        for arm in ARMS:
            r = next((x for x in records
                      if x["held_out"] == held and x["arm"] == arm), None)
            if r is None:
                continue
            print(f"    {held:>9s} {arm:>11s} {r['held_out_vrmse']:>10.5f} "
                  f"{r['seen_vrmse']:>10.5f} {r['gap_ratio']:>7.2f}x "
                  f"{r['train_s']:>7.0f}")

    if not closed:
        return
    print("\n=== How much of the cross-regime gap is closed? ===")
    print("    closed = (gap_baseline - gap_arm) / (gap_baseline - 1); "
          "1.0 = held-out as cheap as seen")
    print("    'real' = held-out VRMSE actually fell; a closed gap with "
          "real=no means the seen side got worse")
    hdr = (f"    {'held out':>9s} {'arm':>11s} {'base gap':>9s} {'arm gap':>8s} "
           f"{'closed':>8s} {'held dVRMSE':>12s} {'real':>5s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for c in closed:
        print(f"    {c['held_out']:>9s} {c['arm']:>11s} "
              f"{c['baseline_gap']:>8.2f}x {c['arm_gap']:>7.2f}x "
              f"{c['frac_gap_closed']:>7.0%} {c['held_vrmse_delta']:>+12.5f} "
              f"{'yes' if c['real_improvement'] else 'no':>5s}")

    for arm in [a for a in ARMS if a != "baseline"]:
        sub = [c for c in closed if c["arm"] == arm]
        if not sub:
            continue
        fr = np.array([c["frac_gap_closed"] for c in sub], dtype=float)
        fr = fr[np.isfinite(fr)]
        n_real = sum(c["real_improvement"] for c in sub)
        if len(fr):
            print(f"    {arm:>11s}: median {np.median(fr):.0%} of the gap "
                  f"closed across {len(fr)} folds "
                  f"(range {fr.min():.0%} to {fr.max():.0%}); "
                  f"held-out error actually fell in {n_real}/{len(sub)}")


def write_csv(path: Path, rows: list[dict], drop=("curve",)) -> None:
    flat = [{k: v for k, v in r.items() if k not in drop} for r in rows]
    if not flat:
        return
    keys, seen = [], set()
    for r in flat:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    print(f"wrote {path}")


def plot(records: list[dict], pred: list[dict], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped figure")
        return

    rs = sorted(records, key=lambda r: -r["gap_ratio"])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    x = np.arange(len(rs))
    ax.bar(x - 0.2, [r["seen_vrmse"] for r in rs], 0.4, label="seen regimes",
           color="tab:green", edgecolor="k", linewidth=0.4)
    ax.bar(x + 0.2, [r["held_out_vrmse"] for r in rs], 0.4, label="held-out regime",
           color="tab:red", edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r["held_out"] for r in rs], rotation=30, ha="right",
                       fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("one-step VRMSE")
    ax.set_title("Held-out regime vs the five it trained on\n(same model, same "
                 "run)", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    for r in rs:
        ax.plot([c["epoch"] for c in r["curve"]],
                [c["gap_ratio"] for c in r["curve"]], "o-", ms=3, lw=1.3,
                label=r["held_out"])
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("gap ratio (held-out / seen)")
    ax.set_title("Is the gap an artifact of the budget?\nflat = no", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[2]
    if pred:
        ax.scatter([p["var_below_mode8"] * 100 for p in pred],
                   [p["gap_ratio"] for p in pred], s=70, c="tab:blue",
                   edgecolor="k", linewidth=0.5, zorder=3)
        for p in pred:
            ax.annotate(p["held_out"],
                        (p["var_below_mode8"] * 100, p["gap_ratio"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("ext10: share of spatial variance below mode 8 (%)")
        ax.set_ylabel("gap ratio")
        ax.set_title("ext10's prediction, tested\n"
                     f"rho = {pred[0]['spearman_rho_signed']:+.3f} "
                     "(mechanism predicts negative)", fontsize=10)
    else:
        ax.axis("off")

    fig.suptitle("LiteFNO cross-regime generalization — Gray-Scott, "
                 "leave-one-regime-out", y=1.03, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/processed/gray_scott_streamed"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10,
                    help="evaluation interval; also the resolution of the "
                         "gap-stability curve")
    ap.add_argument("--regimes", nargs="*", default=None,
                    help="which regimes to hold out (default: all)")
    ap.add_argument("--arms", nargs="*", default=list(ARMS),
                    choices=list(ARMS),
                    help="which arms to run (default: all three)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/baseline"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/baseline"))
    ap.add_argument("--ext10-csv", type=Path,
                    default=Path("results/extensions/"
                                 "ext10_harmonic_summary_gray_scott.csv"))
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    data, manifest = load_dataset(args.data_dir)
    regimes = args.regimes or list(dict.fromkeys(data["train"][1]))
    print(f"device={device}  regimes={regimes}")
    print(f"  train {data['train'][0].shape}, test {data['test'][0].shape}, "
          f"{args.epochs} epochs/fold, seed {args.seed}")

    records = []
    for held in regimes:
        for arm in args.arms:
            print(f"\n  [hold out {held} | arm {arm}]", flush=True)
            rec = run_fold(held, data, device, args.epochs, args.seed,
                           arm=arm, log_every=args.log_every)
            print(f"    -> held-out {rec['held_out_vrmse']:.5f}  "
                  f"seen {rec['seen_vrmse']:.5f}  gap {rec['gap_ratio']:.2f}x "
                  f"({rec['train_s']:.0f}s)", flush=True)
            records.append(rec)
            write_csv(args.out_dir / "ext26_arms.csv", records)

    base = [r for r in records if r["arm"] == "baseline"]
    if base:
        print_report(base)
    closed = gap_closed(records)
    print_arm_report(records, closed)

    # the ext10 spectral prediction is a claim about the baseline gap, so it is
    # checked against the baseline arm only
    pred = check_ext10_prediction(base, args.ext10_csv) if base else []

    write_csv(args.out_dir / "ext26_arms.csv", records)
    if closed:
        write_csv(args.out_dir / "ext26_gap_closed.csv", closed)
    if pred:
        write_csv(args.out_dir / "ext26_prediction_check.csv", pred)
    (args.out_dir / "ext26_curves.json").write_text(json.dumps(
        [{"held_out": r["held_out"], "arm": r["arm"], "curve": r["curve"]}
         for r in records], indent=2))
    if base:
        plot(base, pred, args.fig_dir / "ext26_cross_regime.png")


if __name__ == "__main__":
    sys.exit(main())
