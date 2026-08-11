#!/usr/bin/env python3
"""Driver for the LiteFNO extension experiments on the proxy Gray-Scott dataset.

Runs the width / rank / seed sweeps (LiteFNO) and the Fourier-mode ablation
(FNO-S) through the *canonical* ``src/litefno`` training code, saving a config,
JSONL log, learning-curve CSV, checkpoints and a one-row summary for each run,
plus a combined ``experiments/all_results.csv``.

All runs share one proxy dataset and the same training budget; only the swept
hyper-parameter changes. The width64/rank32/seed1337 LiteFNO run is the shared
baseline referenced by the width, rank and seed groups.

Usage:
    python3 experiments/run_extensions.py [--only NAME ...] [--epochs N] [--device mps]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import yaml

import litefno  # noqa: F401  (ensures canonical package is importable)
from litefno.train import run_training

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = "data/processed/gray_scott_proxy"
DEFAULT_EPOCHS = 50
DEFAULT_DEVICE = "mps"
BATCH_SIZE = 512


def base_config(model: dict, training: dict, metrics_path: str, ckpt_dir: str) -> dict:
    return {
        "model": model,
        "dataset": {
            "name": "gray_scott_proxy",
            "processed_dir": DATA_DIR,
            "splits": ["train", "valid", "test"],
            "dataset_key": "data",
            "input_steps": 1,
            "output_steps": 1,
            "fields": 2,
            "stride": 1,
            "cache": "memory",
        },
        "training": {
            "epochs": training.get("epochs", DEFAULT_EPOCHS),
            "batch_size": BATCH_SIZE,
            "lr": 1e-3,
            "lr_step": 100,
            "lr_gamma": 0.5,
            "device": training.get("device", DEFAULT_DEVICE),
            "seed": training.get("seed", 1337),
            "deterministic": False,
            "eval_every": 1,
            "eval_splits": training.get("eval_splits", ["valid"]),
            "eval_windows": [],
            "test_at_end": True,
            "checkpoint_dir": ckpt_dir,
            "checkpoint_every": 0,
            "checkpoint_best_metric": "valid_vrmse",
            "resume_from": None,
            "num_workers": 0,
            "pin_memory": False,
            "amp": False,
            "cudnn_benchmark": False,
        },
        "logging": {"metrics_path": metrics_path},
    }


def make_specs(epochs: int, device: str) -> list[dict]:
    """Return experiment specs. Each: group, name, model, training-overrides."""
    specs: list[dict] = []

    def lite(width, rank, seed):
        return {"name": "litefno", "layers": 8, "width": width, "rank": rank}

    def fnos(modes):
        return {"name": "fno_s", "layers": 8, "width": 64, "modes": modes}

    # --- shared baseline (width64 / rank32 / seed1337); eval train too for curves ---
    specs.append({"group": "baseline", "name": "litefno_w64_r32_s1337",
                  "model": lite(64, 32, 1337),
                  "training": {"seed": 1337, "eval_splits": ["train", "valid"]}})

    # --- width sweep (rank32, seed1337) ---
    specs.append({"group": "width_sweep", "name": "litefno_w32",
                  "model": lite(32, 32, 1337), "training": {"seed": 1337}})
    specs.append({"group": "width_sweep", "name": "litefno_w128",
                  "model": lite(128, 32, 1337), "training": {"seed": 1337}})

    # --- rank sweep (width64, seed1337) ---
    specs.append({"group": "rank_sweep", "name": "litefno_r16",
                  "model": lite(64, 16, 1337), "training": {"seed": 1337}})
    specs.append({"group": "rank_sweep", "name": "litefno_r64",
                  "model": lite(64, 64, 1337), "training": {"seed": 1337}})

    # --- seed study (width64, rank32) ---
    specs.append({"group": "seeds", "name": "litefno_s2025",
                  "model": lite(64, 32, 2025), "training": {"seed": 2025}})
    specs.append({"group": "seeds", "name": "litefno_s42",
                  "model": lite(64, 32, 42), "training": {"seed": 42}})

    # --- Fourier-mode ablation on FNO-S (baseline modes=12) ---
    specs.append({"group": "modes", "name": "fno_s_m6",
                  "model": fnos(6), "training": {"seed": 1337}})
    specs.append({"group": "modes", "name": "fno_s_m12",
                  "model": fnos(12), "training": {"seed": 1337}})
    specs.append({"group": "modes", "name": "fno_s_m18",
                  "model": fnos(18), "training": {"seed": 1337}})

    for s in specs:
        s["training"]["epochs"] = epochs
        s["training"]["device"] = device
    return specs


def summarize(jsonl_path: Path) -> dict:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    epochs = [r for r in rows if "valid_vrmse" in r and r.get("step", -2) >= 0]
    test = next((r for r in rows if "test_vrmse" in r), {})
    best = min(epochs, key=lambda r: r["valid_vrmse"]) if epochs else {}
    final = epochs[-1] if epochs else {}
    return {
        "params": epochs[0].get("params") if epochs else None,
        "n_epochs": len(epochs),
        "best_epoch": best.get("step"),
        "best_valid_rmse": best.get("valid_rmse"),
        "best_valid_vrmse": best.get("valid_vrmse"),
        "final_valid_rmse": final.get("valid_rmse"),
        "final_valid_vrmse": final.get("valid_vrmse"),
        "test_rmse": test.get("test_rmse"),
        "test_vrmse": test.get("test_vrmse"),
    }


def write_curve_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("step", -2) >= 0 and "valid_vrmse" in r]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_rmse", "valid_rmse", "train_vrmse", "valid_vrmse"])
        for r in rows:
            w.writerow([r["step"], r.get("train_rmse"), r.get("valid_rmse"),
                        r.get("train_vrmse"), r.get("valid_vrmse")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="run only these spec names")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    args = ap.parse_args()

    import os
    os.chdir(ROOT)
    specs = make_specs(args.epochs, args.device)
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]

    all_rows = []
    summary_csv = ROOT / "experiments" / "all_results.csv"
    for i, spec in enumerate(specs, 1):
        outdir = ROOT / "experiments" / spec["group"] / spec["name"]
        outdir.mkdir(parents=True, exist_ok=True)
        metrics_path = f"experiments/{spec['group']}/{spec['name']}/metrics.jsonl"
        ckpt_dir = f"experiments/{spec['group']}/{spec['name']}/checkpoints"
        cfg = base_config(spec["model"], spec["training"], metrics_path, ckpt_dir)
        cfg_path = outdir / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        Path(ROOT / metrics_path).unlink(missing_ok=True)

        print(f"\n=== [{i}/{len(specs)}] {spec['group']}/{spec['name']} "
              f"model={spec['model']} epochs={spec['training']['epochs']} ===", flush=True)
        t0 = time.perf_counter()
        run_training(cfg_path)
        dt = time.perf_counter() - t0

        s = summarize(ROOT / metrics_path)
        write_curve_csv(ROOT / metrics_path, outdir / "metrics.csv")
        row = {"group": spec["group"], "name": spec["name"],
               "model": spec["model"]["name"],
               "width": spec["model"].get("width"),
               "rank": spec["model"].get("rank"),
               "modes": spec["model"].get("modes"),
               "seed": spec["training"]["seed"],
               "epochs": spec["training"]["epochs"],
               "train_time_s": round(dt, 1), **s}
        (outdir / "summary.json").write_text(json.dumps(row, indent=2))
        all_rows.append(row)
        print(f"--- done in {dt:.1f}s | params={s['params']} "
              f"best_valid_vrmse={s['best_valid_vrmse']:.5f}@{s['best_epoch']} "
              f"test_vrmse={s['test_vrmse']:.5f}", flush=True)

    # merge into combined CSV (preserve any rows for specs we didn't run this time)
    existing = {}
    if summary_csv.exists():
        for r in csv.DictReader(summary_csv.open()):
            existing[r["name"]] = r
    for r in all_rows:
        existing[r["name"]] = r
    fieldnames = ["group", "name", "model", "width", "rank", "modes", "seed", "epochs",
                  "train_time_s", "params", "n_epochs", "best_epoch",
                  "best_valid_rmse", "best_valid_vrmse", "final_valid_rmse",
                  "final_valid_vrmse", "test_rmse", "test_vrmse"]
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for name in existing:
            w.writerow({k: existing[name].get(k) for k in fieldnames})
    print(f"\nWrote {summary_csv}")


if __name__ == "__main__":
    main()
