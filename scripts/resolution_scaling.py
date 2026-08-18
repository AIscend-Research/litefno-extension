r"""Does the Fourier advantage appear above 32x32? (ext36)

The reproduction's negative result is explicitly scoped:

    "on Gray-Scott at 32x32 across three seeds, a parameter-matched low-rank CNN
    matches or outperforms LiteFNO on one-step VRMSE and on autoregressive
    rollout, so there is no consistent evidence for a Fourier inductive-bias
    advantage **at that scale**."

That scoping is honest but it is also an open door: a spectral prior is supposed
to earn its keep on smooth, multi-scale fields, and 32x32 is a 4x downsample of
The Well's native 128x128 Gray-Scott. ext10 measured where the variance lives --
maze and spots keep ~99% of theirs *above* mode 8 -- and a 32x32 grid keeps every
available mode after truncation, so at that resolution the spectral layer is not
actually truncating anything. The advantage it is supposed to confer has no room
to appear.

This closes the scope by sweeping resolution.

One fetch, three resolutions
----------------------------
The data is streamed **once** at native 128x128 and downsampled locally to 64 and
32 with the repo's own ``downsample_spatial``. Fetching three times would cost
three times the transfer for identical bytes, and -- more importantly -- would
risk the three resolutions differing by which trajectories were drawn rather than
by resolution. Here they are literally the same fields.

The arms are the reproduction's
-------------------------------
``cnn``       ``models/litefno.py``: a low-rank CNN with **no FFT in it**
``litefno``   the CP-factorized spectral operator
``fno_s``     a dense spectral FNO

``cnn`` versus the two spectral arms is the Fourier comparison. ``fno_s`` is
carried because if an advantage appears only for the dense arm, that is about CP
factorization rather than about Fourier structure, and the two should not be
conflated.

What would count as the advantage appearing
-------------------------------------------
Not "a spectral arm wins at 128". The reproduction's claim is comparative and so
is its refutation: the quantity is the **gap** between the best spectral arm and
the CNN, and the question is whether that gap *moves with resolution*. A constant
gap at every resolution says the reproduction's result generalises and the 32x32
scoping was unnecessary caution. A gap that opens as resolution rises says the
spectral prior needs scale to show, and says where.

Modes are the confound to watch
-------------------------------
The repo trains with ``MODES = min(16, H // 2)``. At 32x32 that is 16 -- every
available mode, so no truncation at all. At 128x128 it is 16 of 64, so the
spectral arms genuinely truncate. That means resolution changes two things at
once: the field gets finer *and* the spectral layer starts discarding modes.

Both are reported. ``--modes-policy repo`` follows ``min(16, H//2)`` as the repo
does; ``--modes-policy proportional`` holds the *fraction* of retained modes
fixed at H/4, so the spectral arm truncates equally at every resolution. If the
gap moves under one policy and not the other, the effect is about truncation
rather than about scale, and the two must not be reported as the same finding.
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

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_BR = Path(__file__).resolve().parent / "baseline_reference.py"


def _load_br():
    spec = importlib.util.spec_from_file_location("baseline_reference", _BR)
    m = importlib.util.module_from_spec(spec)
    sys.modules["baseline_reference"] = m
    spec.loader.exec_module(m)
    return m


br = _load_br()
import torch                                                    # noqa: E402
from torch import nn                                            # noqa: E402

from litefno.preprocess import downsample_spatial               # noqa: E402
from litefno.models.fno_s import FNOS                           # noqa: E402
from litefno.models.litefno import LiteFNO                       # noqa: E402

ARMS = ("cnn", "litefno", "fno_s")


def load_native(data_dir: Path):
    """Native-resolution splits plus the regime labels, from one fetch."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    out = {}
    for split in ("train", "test"):
        out[split] = br.load_split(data_dir / f"{split}.h5")
    return out, manifest


def at_resolution(arr: np.ndarray, native: int, target: int) -> np.ndarray:
    """Downsample the *same* fields to a target grid.

    Uses the repo's own ``downsample_spatial`` so the 32x32 arm here is produced
    by the identical reduction the reproduction used, rather than by a different
    interpolation that could shift the comparison on its own.
    """
    if target == native:
        return arr
    if native % target != 0:
        raise ValueError(f"{native} is not divisible by {target}")
    return downsample_spatial(arr, native // target)


def modes_for(size: int, policy: str, cap: int) -> int:
    if policy == "repo":
        return min(cap, size // 2)
    if policy == "proportional":
        return max(4, size // 4)
    raise ValueError(policy)


def build(arm: str, in_ch: int, size: int, modes: int, width: int, layers: int):
    """The reproduction's three arms, with modes made explicit.

    ``br.build_model`` hard-codes the repo's module-level MODES, which is what
    makes the truncation confound invisible; here the mode count is passed in so
    the two policies can be compared.
    """
    if arm == "cnn":
        return LiteFNO(in_ch, in_ch, width=width, rank=32, layers=layers), "n/a"
    if arm == "fno_s":
        return FNOS(in_ch, in_ch, width=width, modes=modes, layers=layers), "dense"
    if arm == "litefno":
        from neuralop.models import FNO
        m = FNO(n_modes=(modes, modes), hidden_channels=width,
                in_channels=in_ch, out_channels=in_ch, n_layers=layers,
                factorization="cp", rank=0.05)
        return m, "cp"
    raise ValueError(arm)


def train_eval(arm: str, xtr, ytr, xte, yte, size: int, modes: int,
               args, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, fac = build(arm, xtr.shape[1], size, modes, args.width, args.layers)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=br.LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=br.LR_STEP,
                                            gamma=br.LR_GAMMA)
    loss_fn = nn.MSELoss()
    n, t0 = len(xtr), time.time()
    batch = max(1, args.batch)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
    v = br.evaluate_one_step(model, xte, yte, device, batch=max(8, batch // 4))
    return {"arm": arm, "factorization": fac, "params": int(params),
            "modes": modes, "size": size, "seed": seed,
            "test_vrmse": float(v), "train_s": round(time.time() - t0, 1)}


def summarise(rows: list) -> list:
    """Per (size, policy): the spectral-minus-CNN gap that the claim is about."""
    out = []
    for size in sorted({r["size"] for r in rows}):
        for pol in sorted({r["policy"] for r in rows}):
            cell = [r for r in rows if r["size"] == size and r["policy"] == pol]
            if not cell:
                continue
            by = {a: np.array([r["test_vrmse"] for r in cell if r["arm"] == a])
                  for a in ARMS}
            if any(len(v) == 0 for v in by.values()):
                continue
            cnn = by["cnn"].mean()
            best_spec = min(by["litefno"].mean(), by["fno_s"].mean())
            best_name = ("litefno" if by["litefno"].mean() <= by["fno_s"].mean()
                         else "fno_s")
            out.append({
                "size": size, "policy": pol,
                "modes": int(cell[0]["modes"]),
                "n_seeds": int(len(by["cnn"])),
                "cnn": float(cnn), "cnn_sd": float(by["cnn"].std()),
                "litefno": float(by["litefno"].mean()),
                "fno_s": float(by["fno_s"].mean()),
                "best_spectral": best_name,
                # positive = spectral better, the direction the Fourier prior predicts
                "gap_rel": float((cnn - best_spec) / cnn) if cnn > 0 else float("nan"),
                "spectral_wins": int(best_spec < cnn),
            })
    return out


def print_report(summary: list) -> None:
    print("\n=== Does the Fourier advantage appear above 32x32? ===")
    print("    gap_rel > 0 means the best spectral arm beat the CNN -- the")
    print("    direction a Fourier inductive bias predicts. The reproduction")
    print("    found gap <= 0 at 32x32 and scoped its claim to that resolution.\n")
    hdr = (f"    {'size':>5s} {'policy':>13s} {'modes':>6s} {'cnn':>9s} "
           f"{'litefno':>9s} {'fno_s':>9s} {'best':>8s} {'gap_rel':>9s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for s in summary:
        print(f"    {s['size']:>5d} {s['policy']:>13s} {s['modes']:>6d} "
              f"{s['cnn']:>9.5f} {s['litefno']:>9.5f} {s['fno_s']:>9.5f} "
              f"{s['best_spectral']:>8s} {s['gap_rel']:>+8.1%}")

    for pol in sorted({s["policy"] for s in summary}):
        sub = [s for s in summary if s["policy"] == pol]
        if len(sub) < 2:
            continue
        g = np.array([s["gap_rel"] for s in sub])
        sizes = np.array([s["size"] for s in sub], dtype=float)
        trend = "opens with resolution" if g[-1] > g[0] + 0.02 else (
            "closes with resolution" if g[-1] < g[0] - 0.02 else "flat")
        print(f"\n    [{pol}] gap at {int(sizes[0])} = {g[0]:+.1%}, "
              f"at {int(sizes[-1])} = {g[-1]:+.1%}  -> {trend}")
        print(f"    spectral arm wins at {sum(s['spectral_wins'] for s in sub)}"
              f"/{len(sub)} resolutions")


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
    p.add_argument("--data-dir", type=Path,
                   default=Path("data/processed/gs_native"))
    p.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--modes-policy", nargs="+", default=["repo"],
                   choices=["repo", "proportional"])
    p.add_argument("--mode-cap", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    data, manifest = load_native(args.data_dir)
    native = data["train"].shape[2]
    print(f"device={device}  native grid {native}x{native}  "
          f"train {data['train'].shape}")

    rows = []
    for size in args.sizes:
        tr = at_resolution(data["train"], native, size)
        te = at_resolution(data["test"], native, size)
        xtr, ytr = br.to_pairs(tr)
        xte, yte = br.to_pairs(te)
        for pol in args.modes_policy:
            modes = modes_for(size, pol, args.mode_cap)
            print(f"\n  [{size}x{size} | {pol} | modes={modes} | "
                  f"{len(xtr)} pairs]", flush=True)
            for arm in ARMS:
                for seed in args.seeds:
                    try:
                        r = train_eval(arm, xtr, ytr, xte, yte, size, modes,
                                       args, seed, device)
                    except Exception as exc:
                        print(f"      {arm} seed {seed} FAILED: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                        continue
                    r["policy"] = pol
                    rows.append(r)
                got = [x["test_vrmse"] for x in rows
                       if x["size"] == size and x["policy"] == pol
                       and x["arm"] == arm]
                if got:
                    print(f"      {arm:>8s} vrmse={np.mean(got):.5f} "
                          f"params={rows[-1]['params']:,}", flush=True)
                write_csv(args.out_dir / "ext36_resolution_cells.csv", rows)

    summary = summarise(rows)
    print_report(summary)
    write_csv(args.out_dir / "ext36_resolution_cells.csv", rows)
    write_csv(args.out_dir / "ext36_resolution_summary.csv", summary)


if __name__ == "__main__":
    sys.exit(main())
