r"""Does spending the rank budget by mode class beat spreading it evenly?

Board task: classify Fourier modes as primary / resonant / damped, then "weight
the low-rank factorization to preserve primary + resonant modes, compress damped
modes". The classifier and the banded layer are in
``src/litefno/models/bands.py``; this is the measurement.

Two arms at **matched parameter count**:

    uniform   one CP factorization of rank R shared across every retained mode
    banded    three CP factorizations, ranks r_primary + r_resonant + r_damped
              = R, each confined to its class's shells

Because the ranks sum to R, both models have the same number of parameters -- a
test in ``tests/test_bands.py`` asserts the equality. So this is a test of
*where* capacity goes, not how much of it there is. That distinction is the
whole experiment; without it a win could just be a bigger model, which is the
trap the harmonic-bias work (ext15) fell into twice before the parameter test
caught it.

The classification is measured, not assumed
-------------------------------------------
Shell classes come from the committed ext10 radial spectra via
``classify_modes``, which finds the dominant peak and the contiguous band around
it. On Gray-Scott that splits the regimes in two: maze, spots and worms climb to
an interior Turing peak (rise 12x to 2501x) and get a resonant band; bubbles,
gliders and spirals decay from the largest scale (rise 2.0x to 6.7x) and get
none.

That split is worth stating because it sets up the prediction. Compressing the
damped tail should cost little everywhere -- it is the part of the spectrum with
almost no variance in it. Protecting a resonant band can only help the regimes
that *have* one. So the expected result is a per-regime pattern, not a uniform
gain, and the readout is per-regime.

A uniform improvement would again be the suspicious outcome: with parameter
count matched, there is no reason for banding to help a regime whose spectrum
the uniform model already covers.

Which spectrum to classify on
-----------------------------
The model trains on all six regimes at once, so it needs one classification, not
six. The obvious way to get it -- sum the per-regime spectra and classify that --
does not work, and failing quietly is what makes it worth recording. The pooled
spectrum has a rise of 6.4x, under the threshold, so it reports **no resonance
at all**, even though maze, spots and worms individually rise by 2501x, 1222x
and 12.4x. Bubbles, gliders and spirals hold most of their variance at the
largest scales, which inflates P(k=1) in the sum and crushes the ratio. The
pooled spectrum represents none of the six.

The default is therefore a union: classify each regime, then take the union of
the resonant bands. A model that has to serve all six must be able to represent
all of their peaks, so this is the right answer on its own terms and not merely
a workaround. ``--classify-on pooled`` and ``--classify-on <regime>`` remain
available for inspection.
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
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_br():
    spec = importlib.util.spec_from_file_location("baseline_reference", _BR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["baseline_reference"] = module
    spec.loader.exec_module(module)
    return module


br = _load_br()

import torch                                             # noqa: E402
from torch import nn                                     # noqa: E402

from litefno.models.bands import (                       # noqa: E402
    CLASSES, BandedLiteFNO, classify_modes, uniform_ranks)


def regime_labels(manifest: dict, split: str) -> np.ndarray:
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


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------
# classification from the committed spectra
# --------------------------------------------------------------------------


def load_radial(csv_path: Path, field: str = "A", segment: str = "settled"):
    with csv_path.open() as f:
        rows = [r for r in csv.DictReader(f)
                if r["binning"] == "radial" and r["field"] == field
                and r["segment"] == segment]
    by_scen = {}
    for r in rows:
        by_scen.setdefault(r["scenario"], []).append(
            (int(r["k"]), float(r["share"])))
    return {s: sorted(v) for s, v in by_scen.items()}


def classify_for_training(radial: dict, scale: int, native: int = 128,
                          on: str = "union") -> tuple[dict, dict]:
    """Classify, then map native shells onto the model's mode grid.

    ``on="union"`` classifies each regime separately and unions the resonant
    bands. That is the default because the alternative does not work: summing
    the regimes' spectra first gives a rise of 6.4x, below the threshold, so the
    pooled spectrum reports *no resonance at all* even though maze, spots and
    worms individually rise by 2501x, 1222x and 12.4x. Bubbles, gliders and
    spirals put most of their variance at the largest scales, which inflates
    P(k=1) in the sum and crushes the ratio. The pooled spectrum is not
    representative of any regime in the mix.

    A model trained on all six has to represent all of their peaks, so the union
    is also the right answer on its own terms, not just a workaround.

    ext10 measured on the native 128x128 field; the model runs on the 32x32 the
    pipeline produces, so native shell k maps to k * scale / native here. The
    mapping happens after classification -- classifying an already-downsampled
    spectrum would fold the Turing peak into the truncation and lose it.
    """
    per_regime = {s: classify_modes(*zip(*v)) for s, v in radial.items()}

    if on == "union":
        resonant_native = sorted({k for g in per_regime.values()
                                  if g["has_resonance"] for k in g["resonant"]})
        primary_native = sorted({k for g in per_regime.values()
                                 for k in g["primary"]} - set(resonant_native))
        damped_native = sorted({k for g in per_regime.values()
                                for k in g["damped"]}
                               - set(resonant_native) - set(primary_native))
        got = {"primary": primary_native, "resonant": resonant_native,
               "damped": damped_native,
               "peak_k": None, "rise": float("nan"),
               "has_resonance": bool(resonant_native),
               "per_regime": {s: g["has_resonance"]
                              for s, g in per_regime.items()}}
    elif on == "pooled":
        ks = sorted({k for v in radial.values() for k, _ in v})
        power = [sum(dict(v).get(k, 0.0) for v in radial.values()) for k in ks]
        got = classify_modes(ks, power)
    else:
        got = per_regime[on]

    ratio = scale / native
    mapped = {c: sorted({int(round(k * ratio)) for k in got[c]}) for c in CLASSES}
    # rounding can collide two native shells onto one model shell; keep the
    # precedence resonant > primary > damped so a mode is never double-claimed
    seen: set[int] = set()
    out = {}
    for c in ("resonant", "primary", "damped"):
        out[c] = [k for k in mapped[c] if k not in seen]
        seen.update(out[c])
    return out, got


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def per_regime_vrmse(model, arr, labels, device) -> dict:
    return {r: br.evaluate_one_step(model, *br.to_pairs(arr[labels == r]), device)
            for r in dict.fromkeys(labels)}


def train_arm(arm: str, cfg: dict, classes: dict, data: dict,
              device: str) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train_arr, _ = data["train"]
    test_arr, test_lab = data["test"]
    xtr, ytr = br.to_pairs(train_arr)
    xte, yte = br.to_pairs(test_arr)

    if arm == "uniform":
        ranks, use_classes = uniform_ranks(cfg["total_rank"], cfg["modes"] * 2)
    else:
        ranks, use_classes = cfg["banded_ranks"], classes

    model = BandedLiteFNO(xtr.shape[1], xtr.shape[1], width=cfg["width"],
                          modes=cfg["modes"], layers=cfg["layers"],
                          ranks=ranks, classes=use_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=br.LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=br.LR_STEP,
                                            gamma=br.LR_GAMMA)
    loss_fn = nn.MSELoss()

    n, t0, curve = len(xtr), time.time(), []
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, br.BATCH):
            idx = perm[i:i + br.BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx].to(device)), ytr[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)
        sched.step()
        if (epoch + 1) % cfg["log_every"] == 0 or epoch == cfg["epochs"] - 1:
            v = br.evaluate_one_step(model, xte, yte, device)
            curve.append({"epoch": epoch + 1, "train_mse": total / n,
                          "test_vrmse": v})
            print(f"      epoch {epoch + 1:>4d}/{cfg['epochs']}  "
                  f"test_vrmse={v:.5f}  ({time.time() - t0:.0f}s)", flush=True)

    rec = {"arm": arm, "seed": cfg["seed"], "params": n_params,
           "ranks": json.dumps(ranks),
           "modes_per_class": json.dumps(model.modes_per_class()),
           "epochs": cfg["epochs"], "train_s": round(time.time() - t0, 1),
           "test_vrmse": br.evaluate_one_step(model, xte, yte, device)}
    rec.update({f"vrmse_{k}": v for k, v in
                per_regime_vrmse(model, test_arr, test_lab, device).items()})
    rec["curve"] = curve
    return rec


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def summarise(records, regimes):
    out = []
    for arm in ("uniform", "banded"):
        rs = [r for r in records if r["arm"] == arm]
        if not rs:
            continue
        row = {"arm": arm, "n_seeds": len(rs), "params": rs[0]["params"],
               "ranks": rs[0]["ranks"],
               "modes_per_class": rs[0]["modes_per_class"]}
        for key in ["test_vrmse"] + [f"vrmse_{r}" for r in regimes]:
            v = np.array([r[key] for r in rs], dtype=float)
            row[f"{key}_mean"] = float(v.mean())
            row[f"{key}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    return out


def print_report(summary, regimes, native_classes, resonant_regimes):
    if len(summary) < 2:
        print("\n(need both arms)")
        return []
    uni = next(s for s in summary if s["arm"] == "uniform")
    ban = next(s for s in summary if s["arm"] == "banded")

    print("\n=== Banded rank allocation vs uniform (one-step test VRMSE) ===")
    print(f"    uniform {uni['params']:,} params | banded {ban['params']:,} "
          f"params  ({'MATCHED' if uni['params'] == ban['params'] else 'MISMATCHED'})")
    print(f"    banded ranks {ban['ranks']}, modes per class "
          f"{ban['modes_per_class']}")
    print(f"    aggregate: {uni['test_vrmse_mean']:.5f} -> "
          f"{ban['test_vrmse_mean']:.5f} "
          f"({100 * (ban['test_vrmse_mean'] / uni['test_vrmse_mean'] - 1):+.1f}%)"
          "   [context only]")

    print("\n    per regime (the readout):")
    hdr = (f"    {'regime':>9s} {'resonance?':>11s} {'uniform':>10s} "
           f"{'banded':>10s} {'change':>9s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    rows = []
    for r in regimes:
        u = uni[f"vrmse_{r}_mean"]
        b = ban[f"vrmse_{r}_mean"]
        rel = 100 * (b / u - 1) if u else float("nan")
        has = r in resonant_regimes
        print(f"    {r:>9s} {str(has):>11s} {u:>10.5f} {b:>10.5f} {rel:>+8.1f}%")
        rows.append({"regime": r, "has_resonance": has, "uniform_vrmse": u,
                     "banded_vrmse": b, "relative_change_pct": rel})

    with_res = [r["relative_change_pct"] for r in rows if r["has_resonance"]]
    without = [r["relative_change_pct"] for r in rows if not r["has_resonance"]]
    if with_res and without:
        print(f"\n    prediction: protecting a resonant band helps the regimes "
              f"that have one")
        print(f"    mean change, regimes with a resonance   : "
              f"{np.mean(with_res):+.1f}%")
        print(f"    mean change, regimes without            : "
              f"{np.mean(without):+.1f}%")
        print("    (negative is better; the prediction is the first being "
              "more negative)")
    return rows


def write_csv(path: Path, rows, drop=("curve",)):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/processed/gray_scott_streamed"))
    ap.add_argument("--spectrum-csv", type=Path,
                    default=Path("results/extensions/"
                                 "ext10_spatial_spectrum_gray_scott.csv"))
    ap.add_argument("--classify-on", default="union",
                    help="union | pooled | <regime name>")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rank-primary", type=int, default=4)
    ap.add_argument("--rank-resonant", type=int, default=3)
    ap.add_argument("--rank-damped", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/baseline"))
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    data, _ = load_dataset(args.data_dir)
    regimes = list(dict.fromkeys(data["test"][1]))

    radial = load_radial(args.spectrum_csv)
    grid = data["train"][0].shape[2]
    classes, native = classify_for_training(radial, scale=grid,
                                            on=args.classify_on)
    resonant_regimes = {s for s, v in radial.items()
                        if classify_modes(*zip(*v))["has_resonance"]}

    banded_ranks = {"primary": args.rank_primary,
                    "resonant": args.rank_resonant,
                    "damped": args.rank_damped}
    total = sum(banded_ranks.values())

    print(f"device={device}  grid={grid}  regimes={regimes}")
    if native.get("peak_k") is not None:
        print(f"  classified on '{args.classify_on}': peak k={native['peak_k']} "
              f"(native), rise={native['rise']:.1f}x, "
              f"resonance={native['has_resonance']}")
    else:
        print(f"  classified on '{args.classify_on}': resonant native shells "
              f"{native['resonant']}, per-regime resonance "
              f"{native.get('per_regime')}")
    print(f"  model shells -> {classes}")
    print(f"  regimes with their own resonance: {sorted(resonant_regimes)}")
    print(f"  banded ranks {banded_ranks} (total {total}) vs uniform rank {total}")

    records = []
    for seed in args.seeds:
        cfg = {"seed": seed, "epochs": args.epochs, "log_every": args.log_every,
               "width": args.width, "modes": args.modes, "layers": args.layers,
               "total_rank": total, "banded_ranks": banded_ranks}
        for arm in ("uniform", "banded"):
            print(f"\n  [{arm} seed {seed}]", flush=True)
            rec = train_arm(arm, cfg, classes, data, device)
            print(f"    -> test VRMSE {rec['test_vrmse']:.5f} "
                  f"({rec['params']:,} params, {rec['train_s']:.0f}s)",
                  flush=True)
            records.append(rec)
            write_csv(args.out_dir / "ext16_banded_rank_seeds.csv", records)

    summary = summarise(records, regimes)
    rows = print_report(summary, regimes, classes, resonant_regimes)
    write_csv(args.out_dir / "ext16_banded_rank_summary.csv", summary)
    if rows:
        write_csv(args.out_dir / "ext16_banded_rank_per_regime.csv", rows)


if __name__ == "__main__":
    sys.exit(main())
