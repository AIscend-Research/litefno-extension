r"""In-distribution reference number for LiteFNO on Gray-Scott.

Board task: "Baseline: reproduce LITEFNO on ecosystem benchmarks
(in-distribution accuracy) -- reference number."

Why this needs doing at all
---------------------------
The repo has two different things called LiteFNO. ``src/litefno/models/litefno.py``
is a low-rank *CNN* placeholder -- ``docs/reproducibility_findings.md`` says so
plainly, and ``results/extensions/freebieA_repro_audit.json`` records
``repo_litefno_class_is_spectral: false``. Everything in
``results/extensions/logs_reproduction_table.csv`` was produced by that
placeholder and by FNO-S, so despite the filename it contains no LiteFNO number.
The real CP-factorized spectral model exists only inside
``notebooks/headline_3seed.ipynb``, was run on Kaggle, and covers one dataset.

So the in-distribution reference number for the actual architecture rests on a
single un-rerun notebook. This script makes it a command you can run.

Protocol (from notebooks/headline_3seed.ipynb, unchanged)
---------------------------------------------------------
    modes 16, width 64, layers 8, CP factorization at rank 0.02
    200 epochs, batch 64, Adam lr 1e-3, StepLR(step=100, gamma=0.5)
    one-step training; seeds 0, 1, 2

Three arms are trained under identical conditions, because a reference number
for one model is not interpretable on its own:

``litefno``   neuralop FNO with CP-factorized spectral weights -- the paper's
              architecture, and the thing being reproduced
``fno_s``     the repo's own dense spectral FNO (src/litefno/models/fno_s.py)
``cnn``       the repo's low-rank CNN (src/litefno/models/litefno.py), included
              under its honest name; this is the arm that produced the existing
              "litefno" columns

Metrics are VRMSE from ``litefno.metrics``, unchanged: one-step on the test
split, and the autoregressive rollout windows the repo's config already
specifies (``eval_windows: [[6, 12], [13, 30]]``).

What "reference number" means here
----------------------------------
Reported as mean +/- sample standard deviation over seeds, alongside the
committed ``results/seeds/seed_table.csv`` from the original Kaggle run. Two
independent runs of the same protocol on different hardware and a separately
built copy of the data is a replication check, and it is reported as one -- if
the numbers disagree, that is the finding.

The rollout numbers deserve a warning that the one-step numbers do not. A
one-step VRMSE is a stable quantity; a 30-step autoregressive rollout of a
model trained one-step-at-a-time is not, and ext10 gives the reason -- the
60-step training window resolves no temporal structure in this data, so nothing
in training constrains what happens over a long rollout. The seed spread on the
rollout columns of the committed table is 30-40% of the mean. Treat rollout as
an order of magnitude, not a number.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch                                             # noqa: E402
from torch import nn                                     # noqa: E402

from litefno.metrics import vrmse                        # noqa: E402
from litefno.models.fno_s import FNOS                    # noqa: E402
from litefno.models.litefno import LiteFNO               # noqa: E402

# notebooks/headline_3seed.ipynb
MODES, WIDTH, LAYERS, RANK = 16, 64, 8, 0.02
EPOCHS, BATCH, LR, LR_STEP, LR_GAMMA = 200, 64, 1e-3, 100, 0.5
ROLL_WINDOWS = ((6, 12), (13, 30))       # configs/experiments/base_litefno.yaml


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def build_litefno(in_ch: int, out_ch: int, modes: int = MODES, width: int = WIDTH,
                  layers: int = LAYERS, rank: float = RANK):
    """The paper's architecture: spectral convolution with CP-factorized weights.

    Falls back the same way the notebook does, and returns which factorization
    was actually built so the reported number can never silently be a dense FNO
    labelled as CP.
    """
    from neuralop.models import FNO
    base = dict(n_modes=(modes, modes), hidden_channels=width,
                in_channels=in_ch, out_channels=out_ch, n_layers=layers)
    for fac in ("cp", "tucker", None):
        try:
            if fac is None:
                return FNO(**base), "dense"
            return FNO(**base, factorization=fac, rank=rank), fac
        except Exception as exc:
            print(f"    build {fac} failed: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not build FNO")


def build_model(arm: str, in_ch: int, out_ch: int):
    if arm == "litefno":
        return build_litefno(in_ch, out_ch)
    if arm == "fno_s":
        return FNOS(in_ch, out_ch, width=WIDTH, modes=MODES, layers=LAYERS), "dense"
    if arm == "cnn":
        return LiteFNO(in_ch, out_ch, width=WIDTH, rank=32, layers=LAYERS), "n/a"
    raise ValueError(arm)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_split(path: Path, key: str = "data") -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        return np.asarray(f[key], dtype=np.float32)


def to_pairs(traj: np.ndarray):
    """(n_traj, T, H, W, C) -> one-step (x, y) with channels first."""
    x = traj[:, :-1]
    y = traj[:, 1:]
    n, t, h, w, c = x.shape
    x = x.reshape(n * t, h, w, c).transpose(0, 3, 1, 2)
    y = y.reshape(n * t, h, w, c).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(x)), \
        torch.from_numpy(np.ascontiguousarray(y))


# --------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------


def evaluate_one_step(model, x, y, device, batch: int = 128) -> float:
    """VRMSE over the whole split at once, matching litefno.metrics.vrmse.

    Batched only to bound memory; the metric is computed on the concatenated
    predictions, not averaged over batches, because VRMSE normalises by the
    variance of the target and a mean of per-batch ratios is not that.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            preds.append(model(x[i:i + batch].to(device)).cpu())
    return float(vrmse(torch.cat(preds), y))


def evaluate_rollout(model, traj: np.ndarray, device, windows=ROLL_WINDOWS,
                     batch: int = 32) -> dict:
    """Autoregressive rollout from step 0, scored on the repo's windows.

    Feeds the model its own output, which is the setting the config's
    ``eval_windows`` describe. Reported with the caveat in the module docstring.
    """
    model.eval()
    n_traj, n_time = traj.shape[0], traj.shape[1]
    horizon = max(end for _, end in windows) + 1
    horizon = min(horizon, n_time - 1)

    target = torch.from_numpy(
        np.ascontiguousarray(traj[:, 1:horizon + 1].transpose(0, 1, 4, 2, 3)))
    state = torch.from_numpy(
        np.ascontiguousarray(traj[:, 0].transpose(0, 3, 1, 2)))

    outs = []
    with torch.no_grad():
        for i in range(0, n_traj, batch):
            cur = state[i:i + batch].to(device)
            steps = []
            for _ in range(horizon):
                cur = model(cur)
                steps.append(cur.cpu())
            outs.append(torch.stack(steps, dim=1))
    pred = torch.cat(outs)                       # (n_traj, horizon, C, H, W)

    out = {}
    for start, end in windows:
        hi = min(end, horizon)
        if start > hi:
            continue
        out[f"roll_{start}_{end}"] = float(
            vrmse(pred[:, start - 1:hi], target[:, start - 1:hi]))
    return out


def train_arm(arm: str, seed: int, data: dict, device: str, epochs: int,
              log_every: int = 20) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    xtr, ytr = data["train_pairs"]
    xva, yva = data["valid_pairs"]
    xte, yte = data["test_pairs"]
    in_ch = xtr.shape[1]

    model, factorization = build_model(arm, in_ch, in_ch)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP, gamma=LR_GAMMA)
    loss_fn = nn.MSELoss()

    n = len(xtr)
    curve, t0 = [], time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)
        sched.step()
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            va = evaluate_one_step(model, xva, yva, device)
            curve.append({"epoch": epoch + 1, "train_mse": total / n,
                          "valid_vrmse": va})
            print(f"      epoch {epoch + 1:>4d}/{epochs}  train_mse={total / n:.3e} "
                  f"valid_vrmse={va:.5f}  ({time.time() - t0:.0f}s)", flush=True)

    train_s = time.time() - t0
    result = {
        "arm": arm, "seed": seed, "factorization": factorization,
        "params": n_params, "epochs": epochs, "train_s": round(train_s, 1),
        "onestep_test_vrmse": evaluate_one_step(model, xte, yte, device),
        "onestep_valid_vrmse": evaluate_one_step(model, xva, yva, device),
    }
    result.update(evaluate_rollout(model, data["test_traj"], device))
    result["curve"] = curve
    return result


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def summarise(records: list[dict]) -> list[dict]:
    keys = [k for k in records[0]
            if k.startswith(("onestep_", "roll_")) or k == "train_s"]
    out = []
    for arm in dict.fromkeys(r["arm"] for r in records):
        rs = [r for r in records if r["arm"] == arm]
        row = {"arm": arm, "n_seeds": len(rs), "params": rs[0]["params"],
               "factorization": rs[0]["factorization"], "epochs": rs[0]["epochs"]}
        for k in keys:
            v = np.array([r[k] for r in rs], dtype=float)
            row[f"{k}_mean"] = float(v.mean())
            row[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    return out


def compare_to_committed(summary: list[dict], seed_table: Path) -> list[dict]:
    """Replication check against the original Kaggle run.

    The committed table's ``litefno`` rows are the same architecture and
    protocol; ``cnn`` is the low-rank CNN. Different hardware, different build
    of the data, so agreement is evidence and disagreement is a finding.
    """
    if not seed_table.exists():
        print(f"\n(no committed seed table at {seed_table}; skipping check)")
        return []
    with seed_table.open() as f:
        rows = list(csv.DictReader(f))
    out = []
    print(f"\n=== Replication check against {seed_table} (Kaggle run) ===")
    print(f"    {'arm':>8s} {'metric':>14s} {'this run':>18s} {'committed':>18s} "
          f"{'ratio':>7s}")
    mapping = {"onestep_test_vrmse": "onestep", "roll_6_12": "roll_6_12",
               "roll_13_30": "roll_13_30"}
    for row in summary:
        arm = row["arm"]
        ref = [r for r in rows if r["model"] == arm]
        if not ref:
            continue
        for mine, theirs in mapping.items():
            if f"{mine}_mean" not in row or theirs not in ref[0]:
                continue
            got = row[f"{mine}_mean"]
            got_sd = row[f"{mine}_std"]
            v = np.array([float(r[theirs]) for r in ref], dtype=float)
            print(f"    {arm:>8s} {mine:>14s} "
                  f"{got:>10.5f}+-{got_sd:<7.5f} "
                  f"{v.mean():>10.5f}+-{v.std(ddof=1):<7.5f} "
                  f"{got / v.mean():>7.2f}")
            out.append({"arm": arm, "metric": mine, "this_mean": got,
                        "this_std": got_sd, "committed_mean": float(v.mean()),
                        "committed_std": float(v.std(ddof=1)),
                        "ratio": got / v.mean()})
    return out


def print_report(summary: list[dict]) -> None:
    print("\n=== In-distribution reference numbers, Gray-Scott "
          "(mean +/- sd over seeds) ===")
    hdr = (f"    {'arm':>8s} {'factzn':>7s} {'params':>9s} {'one-step VRMSE':>22s} "
           f"{'roll 6-12':>18s} {'roll 13-30':>18s} {'train s':>9s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in summary:
        def cell(k, w=10):
            if f"{k}_mean" not in r:
                return " " * 18
            return f"{r[f'{k}_mean']:>{w}.5f}+-{r[f'{k}_std']:<7.5f}"
        print(f"    {r['arm']:>8s} {r['factorization']:>7s} {r['params']:>9,d} "
              f"{cell('onestep_test_vrmse')} {cell('roll_6_12')} "
              f"{cell('roll_13_30')} {r['train_s_mean']:>9.0f}")
    print("    one-step is the reference number; rollout is order-of-magnitude "
          "only (see docstring)")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def plot(records: list[dict], summary: list[dict], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped figure")
        return
    colours = {"litefno": "tab:blue", "fno_s": "tab:orange", "cnn": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    for r in records:
        if not r["curve"]:
            continue
        e = [c["epoch"] for c in r["curve"]]
        v = [c["valid_vrmse"] for c in r["curve"]]
        ax.semilogy(e, v, color=colours.get(r["arm"], "0.5"), lw=1.2, alpha=0.8,
                    label=r["arm"] if r["seed"] == records[0]["seed"] else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation VRMSE")
    ax.set_title("Convergence (all seeds)\nreference number is read at the end",
                 fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    arms = [r["arm"] for r in summary]
    x = np.arange(len(arms))
    means = [r["onestep_test_vrmse_mean"] for r in summary]
    sds = [r["onestep_test_vrmse_std"] for r in summary]
    ax.bar(x, means, yerr=sds, capsize=5,
           color=[colours.get(a, "0.5") for a in arms], edgecolor="k", linewidth=0.5)
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, m + s + max(means) * 0.03, f"{m:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("one-step test VRMSE")
    ax.set_title("In-distribution reference number\n(lower is better)", fontsize=10)

    fig.suptitle("LiteFNO in-distribution baseline — Gray-Scott, 32x32, "
                 "one-step", y=1.02, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/processed/gray_scott_streamed"))
    ap.add_argument("--arms", nargs="*", default=["litefno", "fno_s", "cnn"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/baseline"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/baseline"))
    ap.add_argument("--seed-table", type=Path,
                    default=Path("results/seeds/seed_table.csv"))
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  torch={torch.__version__}")

    splits = {s: load_split(args.data_dir / f"{s}.h5") for s in
              ("train", "valid", "test")}
    for s, a in splits.items():
        print(f"  {s}: {a.shape}")
    data = {
        "train_pairs": to_pairs(splits["train"]),
        "valid_pairs": to_pairs(splits["valid"]),
        "test_pairs": to_pairs(splits["test"]),
        "test_traj": splits["test"],
    }
    print(f"  one-step pairs: train {len(data['train_pairs'][0])}, "
          f"test {len(data['test_pairs'][0])}")

    records = []
    for arm in args.arms:
        for seed in args.seeds:
            print(f"\n  [{arm} seed {seed}]", flush=True)
            rec = train_arm(arm, seed, data, device, args.epochs)
            print(f"    -> one-step test VRMSE {rec['onestep_test_vrmse']:.5f} "
                  f"({rec['train_s']:.0f}s, {rec['params']:,} params, "
                  f"factorization={rec['factorization']})", flush=True)
            records.append(rec)
            # write as we go so a long run is never lost
            write_csv(args.out_dir / "ext13_baseline_seeds.csv",
                      [{k: v for k, v in r.items() if k != "curve"}
                       for r in records])

    summary = summarise(records)
    print_report(summary)
    check = compare_to_committed(summary, args.seed_table)

    write_csv(args.out_dir / "ext13_baseline_summary.csv", summary)
    if check:
        write_csv(args.out_dir / "ext13_replication_check.csv", check)
    (args.out_dir / "ext13_curves.json").write_text(json.dumps(
        [{"arm": r["arm"], "seed": r["seed"], "curve": r["curve"]}
         for r in records], indent=2))
    plot(records, summary, args.fig_dir / "ext13_baseline.png")


if __name__ == "__main__":
    sys.exit(main())
