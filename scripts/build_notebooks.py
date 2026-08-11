"""Builds the two Kaggle notebooks for the GS generalization study.

    python scripts/build_notebooks.py

Emits:
  notebooks/phase2_train_real_litefno.ipynb   - implements + trains the REAL
      spectral CP-factorized LiteFNO (via neuraloperator) on Gray-Scott, saves
      a checkpoint + JSONL log.
  notebooks/phase3_extensions.ipynb           - loads every available arm
      (CNN from repo, real LiteFNO from Phase 2, optional FNO-S) and runs all
      extensions per arm + the logs-only 8-dataset analysis.

Both notebooks clone https://github.com/AIscend-Research/litefno-repro and are
self-contained (the real-LiteFNO builder is defined inline in BOTH so they stay
in sync without needing repo changes).
"""
from __future__ import annotations

import json
from pathlib import Path

# A builder string reused verbatim in BOTH notebooks so the architecture (and
# therefore the state_dict keys) match exactly between train and load.
BUILD_FN = r'''
def build_real_litefno(in_ch, out_ch, modes, width=64, layers=8, rank=0.5, factorization="cp"):
    """Real spectral LiteFNO: a CP/Tucker-factorized FNO (neuraloperator).

    Tries the requested factorization, then tucker, then a dense FNO, so the
    notebook still produces a genuine *spectral* operator even if a particular
    factorization API is unavailable. Returns (model, kind_used).
    """
    from neuralop.models import FNO
    base = dict(n_modes=(modes, modes), hidden_channels=width,
                in_channels=in_ch, out_channels=out_ch, n_layers=layers)
    fac = None if factorization in (None, "dense") else factorization
    attempts = [(fac, rank), ("tucker", rank), (None, None)]
    last = None
    for f, r in attempts:
        try:
            if f is None:
                return FNO(**base), "dense"
            return FNO(**base, factorization=f, rank=r), f
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [build] factorization={f} failed: {e}")
    raise RuntimeError(f"could not construct FNO: {last}")
'''.strip("\n")


def make_notebook(cells):
    out = []
    for kind, src in cells:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": source})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ===========================================================================
# PHASE 2: train the real spectral LiteFNO
# ===========================================================================
phase2 = []
M = phase2.append

M(("md", """# Phase 2: Train the REAL spectral LiteFNO (Gray-Scott)

This notebook implements the **actual** LiteFNO architecture the paper describes:
a **CP-factorized spectral FNO** (via `neuraloperator`), and trains it on
Gray-Scott. The repo's existing `LiteFNO` class is a CNN and is *not* used here.

Output (for Phase 3):
- `litefno_real_best.pt` / `litefno_real_last.pt`: checkpoints
- `gray_scott_litefno_real.jsonl`: per-epoch metrics

**After it finishes**, save the notebook so `/kaggle/working/extensions/` becomes
the notebook output, then in Phase 3 add this notebook's output as an input
dataset so Phase 3 can load `litefno_real_best.pt`.

**Setup:** Kaggle Settings -> Internet ON, Accelerator = GPU."""))

M(("code", """import os, subprocess, sys
REPO_URL = "https://github.com/AIscend-Research/litefno-repro"
REPO_DIR = "litefno-repro"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
os.chdir(REPO_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuraloperator", "the_well", "thop"], check=False)
print("cwd:", os.getcwd())"""))

M(("code", """import json, time, math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from litefno.data import DatasetConfig, H5SequenceDataset
from litefno.train import flatten_time
from litefno.metrics import rmse, vrmse
from litefno.download import download_dataset
from litefno.preprocess import preprocess_well_split

torch.manual_seed(1337); np.random.seed(1337)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":   # P100 (sm_60) is incompatible with new torch; verify it works
    try:
        _ = (torch.zeros(1, device=DEVICE) + 1).item()
    except Exception as e:
        print("CUDA unusable (use T4, not P100):", e); DEVICE = torch.device("cpu")
print("device:", DEVICE)
if DEVICE.type != "cuda":
    print("WARNING: not on GPU -> use Settings > Accelerator > GPU T4 x2 (NOT P100), "
          "otherwise training is far too slow.")

OUT = Path("/kaggle/working/extensions") if Path("/kaggle/working").exists() else Path("extensions_out")
OUT.mkdir(parents=True, exist_ok=True)

# --- Gray-Scott config (matches the repo's GS preprocessing) ---
DATASET="gray_scott_reaction_diffusion"; KEY="data"; FIELDS=2
DOWNSAMPLE=4; MAX_TRAJ=1000; MAX_STEPS=60; SEED=0
RAW=Path("/kaggle/temp/gs_raw") if Path("/kaggle").exists() else Path("data/raw/gs_ext"); PROC=Path("data/processed/gs_ext")
# If you uploaded the preprocessed train/valid/test.h5 as a Kaggle Dataset, set this
# to its mount path; Phase 2 will use it and SKIP the 19.5 GB download entirely.
PROC_INPUT = Path("/kaggle/input/gs-processed")

# --- training protocol (matched to the repo CNN runs for a fair comparison) ---
WIDTH=64; LAYERS=8; RANK=0.02; FACT="cp"  # rank 0.02 -> ~180K params (matches paper LiteFNO ~187K); 0.5 would be ~2.5M
EPOCHS=200; BATCH=64; LR=1e-3; LR_STEP=100; LR_GAMMA=0.5
# (paper-faithful would be EPOCHS=500 + a transduction stage; see note at end)"""))

M(("md", "## Download + preprocess Gray-Scott (train / valid / test)"))
M(("code", """import glob as _glob
_hits = sorted(_glob.glob("/kaggle/input/**/train.h5", recursive=True))
if (PROC_INPUT / "train.h5").exists():
    PROC = PROC_INPUT
    print("Using PRE-UPLOADED processed data:", PROC, "(skipping download)")
elif _hits:
    PROC = Path(_hits[0]).parent
    print("Auto-found pre-uploaded data under /kaggle/input:", PROC, "(skipping download)")
else:
    print("No pre-uploaded data found; downloading raw from The Well (large).")
    RAW.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        out_h5 = PROC / f"{split}.h5"
        if out_h5.exists():
            continue
        download_dataset(DATASET, split, RAW)
        PROC.mkdir(parents=True, exist_ok=True)
        preprocess_well_split(RAW, out_h5, DATASET, split, KEY, DOWNSAMPLE, MAX_TRAJ, MAX_STEPS, random_seed=SEED)
        import shutil  # free disk immediately: raw split is ~19.5 GB, processed is tiny
        raw_split = RAW / "datasets" / DATASET / "data" / split
        if raw_split.exists():
            shutil.rmtree(raw_split); print(f"  freed raw '{split}'")

def make_loader(split, bs, shuffle):
    cfg = DatasetConfig(path=PROC / f"{split}.h5", dataset_key=KEY,
                        input_steps=1, output_steps=1, stride=1, cache="memory")
    return DataLoader(H5SequenceDataset(cfg), batch_size=bs, shuffle=shuffle)

train_loader = make_loader("train", BATCH, True)
valid_loader = make_loader("valid", BATCH, False)
test_loader  = make_loader("test",  BATCH, False)

import h5py
with h5py.File(PROC / "train.h5", "r") as f:
    H, W = f[KEY].shape[2], f[KEY].shape[3]
MODES = min(16, H // 2)
print(f"resolution {H}x{W}  modes={MODES}  train batches={len(train_loader)}")"""))

M(("md", "## Build the real spectral LiteFNO"))
M(("code", BUILD_FN + """

model, KIND = build_real_litefno(FIELDS, FIELDS, MODES, WIDTH, LAYERS, RANK, FACT)
model = model.to(DEVICE)
PARAMS = sum(p.numel() for p in model.parameters())
print(f"Built real LiteFNO  factorization={KIND}  params={PARAMS:,}")
BUILD = {"modes": MODES, "width": WIDTH, "layers": LAYERS, "rank": RANK,
         "factorization": KIND, "in_ch": FIELDS, "out_ch": FIELDS}"""))

M(("md", "## Helpers: train + eval one epoch"))
M(("code", """def to_xy(batch):
    xb, yb = batch                     # (B,1,H,W,2)
    return flatten_time(xb).to(DEVICE), flatten_time(yb).to(DEVICE)  # (B,2,H,W)

scaler = None  # AMP GradScaler doesn't support the FNO's ComplexFloat spectral weights -> train in fp32

def train_epoch(opt):
    model.train(); tot = 0.0
    for batch in train_loader:
        x, y = to_xy(batch)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.mse_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward(); opt.step()
        tot += loss.item()
    return tot / max(1, len(train_loader))

@torch.no_grad()
def eval_loader(loader):
    model.eval(); rs = []; vs = []
    for batch in loader:
        x, y = to_xy(batch)
        p = model(x)
        rs.append(rmse(p, y).item()); vs.append(vrmse(p, y).item())
    return float(np.mean(rs)), float(np.mean(vs))"""))

M(("md", "## Timing check (3 epochs): estimate full run before committing"))
M(("code", """opt = torch.optim.AdamW(model.parameters(), lr=LR)
t0 = time.time()
for _ in range(3):
    train_epoch(opt)
per_epoch = (time.time() - t0) / 3
print(f"~{per_epoch:.1f} s/epoch  ->  full {EPOCHS} epochs ~= {per_epoch*EPOCHS/3600:.1f} GPU-hours")
print("If that's too long, lower EPOCHS or use a smaller MODES/WIDTH and re-run from the build cell.")"""))

M(("md", "## Full training (fresh model + optimizer)"))
M(("code", """# rebuild fresh so the 3 probe epochs don't count
model, KIND = build_real_litefno(FIELDS, FIELDS, MODES, WIDTH, LAYERS, RANK, FACT)
model = model.to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP, gamma=LR_GAMMA)

log_path = OUT / "gray_scott_litefno_real.jsonl"
best_v = float("inf")
with open(log_path, "w") as logf:
    for epoch in range(EPOCHS):
        loss = train_epoch(opt)
        tr_r, tr_v = eval_loader(train_loader)
        va_r, va_v = eval_loader(valid_loader)
        rec = {"step": epoch, "loss": loss, "params": PARAMS,
               "train_rmse": tr_r, "train_vrmse": tr_v,
               "valid_rmse": va_r, "valid_vrmse": va_v}
        logf.write(json.dumps(rec) + "\\n"); logf.flush()
        if va_v < best_v:
            best_v = va_v
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "build": BUILD},
                       OUT / "litefno_real_best.pt")
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:>3}  loss={loss:.4e}  valid_vrmse={va_v:.5f}  (best={best_v:.5f})")
        sched.step()
torch.save({"epoch": EPOCHS, "model_state": model.state_dict(), "build": BUILD},
           OUT / "litefno_real_last.pt")
print("saved checkpoints to", OUT)"""))

M(("md", "## Test evaluation"))
M(("code", """ck = torch.load(OUT / "litefno_real_best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ck["model_state"]); model.to(DEVICE)
te_r, te_v = eval_loader(test_loader)
print(f"REAL LiteFNO  test RMSE={te_r:.6f}  test VRMSE={te_v:.6f}  params={PARAMS:,}  factorization={KIND}")
with open(log_path, "a") as logf:
    logf.write(json.dumps({"step": EPOCHS, "test_rmse": te_r, "test_vrmse": te_v}) + "\\n")
print("Compare against: repo CNN GS test_vrmse=0.0227 ; paper LiteFNO GS one-step VRMSE=0.0098")"""))

M(("md", """## Handoff to Phase 3 + notes

- This notebook's `/kaggle/working/extensions/litefno_real_best.pt` is the real
  LiteFNO arm. **Save the notebook**, then in Phase 3 use *Add Input -> Notebook
  Output* to mount it; set `REAL_CKPT` there to its path.
- **Deviations from the paper (document these):** matched-protocol training
  (EPOCHS=200, MSE loss, batch 64, single-step) for a fair head-to-head with the
  CNN arm, rather than the paper's 500 epochs + relative-L2 + transduction
  fine-tuning. The CP `rank` here is `neuraloperator`'s fractional rank, not the
  paper's integer rank {32,48}. For a paper-faithful run, raise EPOCHS to 500 and
  add a transduction (spatio-temporal, S=3) fine-tuning stage."""))


# ===========================================================================
# PHASE 3: extensions across all available arms
# ===========================================================================
phase3 = []
P = phase3.append

P(("md", """# Phase 3: Extensions across arms (Gray-Scott)

Loads every arm that's available and runs all extensions on each, with
comparison plots:
- **CNN** ("LiteFNO" class in the repo): from the committed checkpoint (always)
- **Real LiteFNO**: from Phase 2's `litefno_real_best.pt` (if mounted)
- **FNO-S**: optional; only if you point `FNOS_CKPT` at one

Checkpoint-based extensions: sanity, quantization, noise robustness,
autoregressive rollout (+windowed VRMSE), input spectral sensitivity, energy
spectrum, error maps, inference benchmark. Plus the **logs-only 8-dataset
analysis** (no checkpoint needed).

Run it now for the CNN arm (parallel with Phase 2); re-run after Phase 2 with the
real-LiteFNO checkpoint mounted to fill in that arm.

**Setup:** Internet ON; GPU optional."""))

P(("code", """import os, subprocess, sys
REPO_URL = "https://github.com/AIscend-Research/litefno-repro"
REPO_DIR = "litefno-repro"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
os.chdir(REPO_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuraloperator", "the_well", "thop"], check=False)
print("cwd:", os.getcwd())"""))

P(("code", """import json, time, copy, csv
from pathlib import Path
import numpy as np
import torch, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":   # some Kaggle GPUs (P100, sm_60) are incompatible with new torch
    try:
        _ = (torch.zeros(1, device=DEVICE) + 1).item()
    except Exception as e:
        print("CUDA unusable, falling back to CPU:", e); DEVICE = torch.device("cpu")
OUT = Path("/kaggle/working/extensions") if Path("/kaggle/working").exists() else Path("extensions_out")
OUT.mkdir(parents=True, exist_ok=True)
print("device:", DEVICE, "| out:", OUT.resolve())

DATASET="gray_scott_reaction_diffusion"; KEY="data"; FIELDS=2
DOWNSAMPLE=4; MAX_TRAJ=1000; MAX_STEPS=60; SEED=0
RAW=Path("/kaggle/temp/gs_raw") if Path("/kaggle").exists() else Path("data/raw/gs_ext"); PROC=Path("data/processed/gs_ext"); TEST_H5=PROC/"test.h5"

CNN_CKPT  = Path("outputs/checkpoints/gray_scott_reaction_diffusion/litefno/best.pt")
# Point these at mounted Kaggle inputs after Phase 2 (edit as needed):
REAL_CKPT = Path("/kaggle/input/phase2-output/litefno_real_best.pt")
FNOS_CKPT = None   # optional: set to a checkpoint path string to include FNO-S in figures"""))

P(("md", "## Download + preprocess GS test split"))
P(("code", """from litefno.download import download_dataset
from litefno.preprocess import preprocess_well_split
import glob as _g
# Prefer a mounted preprocessed test.h5 (matches Phase 2's training distribution -> fair eval).
_test_hits = sorted(_g.glob("/kaggle/input/**/test.h5", recursive=True))
DATA_OK = TEST_H5.exists()
if _test_hits and not DATA_OK:
    TEST_H5 = Path(_test_hits[0])
    print("Using mounted test set (matches Phase 2 training regimes):", TEST_H5)
    DATA_OK = True
if not DATA_OK:
    try:
        RAW.mkdir(parents=True, exist_ok=True)
        download_dataset(DATASET, "test", RAW)
        PROC.mkdir(parents=True, exist_ok=True)
        preprocess_well_split(RAW, TEST_H5, DATASET, "test", KEY, DOWNSAMPLE, MAX_TRAJ, MAX_STEPS, random_seed=SEED)
        import shutil
        raw_split = RAW / "datasets" / DATASET / "data" / "test"
        if raw_split.exists(): shutil.rmtree(raw_split)
        DATA_OK = True
    except Exception as e:
        print("!! GS test download/preprocess failed:", repr(e)); DATA_OK = False
if DATA_OK:
    with h5py.File(TEST_H5, "r") as f:
        TEST = f[KEY][...].astype(np.float32)
    H, W = TEST.shape[2], TEST.shape[3]
    print("test:", TEST.shape)"""))

P(("md", "## Load every available arm\n\nAll arms share the same `(B,C,H,W) -> (B,C,H,W)` interface, so the extension code is identical across them."))
P(("code", BUILD_FN + """

from litefno.train import build_model, load_checkpoint, count_parameters

ARMS = {}      # name -> model
PARAMS = {}    # name -> param count

if DATA_OK:
    # CNN arm (repo)
    if CNN_CKPT.exists():
        m = build_model({"name": "litefno", "layers": 8, "width": 64, "rank": 32}, FIELDS, FIELDS)
        load_checkpoint(CNN_CKPT, m, device=DEVICE); m.to(DEVICE).eval()
        ARMS["cnn"] = m; PARAMS["cnn"] = count_parameters(m)

    # Real spectral LiteFNO arm (Phase 2)
    if REAL_CKPT.exists():
        ck = torch.load(REAL_CKPT, map_location=DEVICE, weights_only=False); b = ck["build"]
        m, _ = build_real_litefno(b["in_ch"], b["out_ch"], b["modes"], b["width"], b["layers"], b["rank"], b["factorization"])
        m.load_state_dict(ck["model_state"]); m.to(DEVICE).eval()
        ARMS["litefno_real"] = m; PARAMS["litefno_real"] = sum(p.numel() for p in m.parameters())

    # Optional FNO-S arm
    if FNOS_CKPT and Path(FNOS_CKPT).exists():
        m = build_model({"name": "fno_s", "layers": 8, "width": 64, "modes": 12}, FIELDS, FIELDS)
        load_checkpoint(FNOS_CKPT, m, device=DEVICE); m.to(DEVICE).eval()
        ARMS["fno_s"] = m; PARAMS["fno_s"] = count_parameters(m)

print("arms:", {k: f"{v:,}" for k, v in PARAMS.items()} or "NONE (need data + checkpoints)")"""))

P(("md", "## Shared eval helpers"))
P(("code", """@torch.no_grad()
def step_predict(m, state):           # state (B,H,W,C) -> (B,H,W,C)
    return m(state.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

@torch.no_grad()
def eval_one_step(m, data, bs=256, transform=None):
    N, S = data.shape[0], data.shape[1]; rs = []; vs = []
    for t in range(S - 1):
        x = data[:, t]; y = data[:, t + 1]
        if transform is not None: x = transform(x)
        xt = torch.from_numpy(np.ascontiguousarray(x)).float()
        yt = torch.from_numpy(np.ascontiguousarray(y)).float()
        for i in range(0, N, bs):
            p = step_predict(m, xt[i:i+bs].to(DEVICE)); yb = yt[i:i+bs].to(DEVICE)
            rs.append(rmse(p, yb).item()); vs.append(vrmse(p, yb).item())
    return float(np.mean(rs)), float(np.mean(vs))

from litefno.metrics import rmse, vrmse, window_vrmse
def savecsv(name, rows):
    if not rows: return
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)"""))

P(("md", "## Extension 1: Sanity check (per arm)"))
P(("code", """if ARMS:
    rows = []
    for name, m in ARMS.items():
        r, v = eval_one_step(m, TEST)
        rows.append({"arm": name, "params": PARAMS[name], "test_rmse": r, "test_vrmse": v})
        print(f"{name:>13}: RMSE={r:.5f}  VRMSE={v:.5f}  params={PARAMS[name]:,}")
    savecsv("ext1_sanity.csv", rows)"""))

P(("md", "## Extension 2: Precision / quantization sweep (per arm)"))
P(("code", """def q_real(w, bits):
    if bits == 16: return w.half().float()
    qmax = 2 ** (bits - 1) - 1; s = w.abs().max() / qmax
    return w if s == 0 else torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s

def quantize(m, bits):
    mq = copy.deepcopy(m)
    with torch.no_grad():
        for p in mq.parameters():
            if bits >= 32: continue
            if p.is_complex():
                p.data = torch.complex(q_real(p.data.real, bits), q_real(p.data.imag, bits))
            else:
                p.data = q_real(p.data, bits)
    return mq

if ARMS:
    rows = []; fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in ARMS.items():
        ser = []
        for bits in [32, 16, 8, 6, 4, 3, 2]:
            _, v = eval_one_step(quantize(m, bits), TEST)
            rows.append({"arm": name, "bits": bits, "vrmse": v,
                         "size_mb": PARAMS[name] * bits / 8 / 1e6})
            ser.append((bits, v))
        ax.plot([b for b, _ in ser], [v for _, v in ser], "o-", label=name)
    savecsv("ext2_quantization.csv", rows)
    ax.invert_xaxis(); ax.set_xlabel("weight bit-width"); ax.set_ylabel("test VRMSE")
    ax.set_title("Accuracy vs precision"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "ext2_quantization.png", dpi=150); plt.close(fig)
    print("saved ext2_quantization.{csv,png}")"""))

P(("md", "## Extension 3: Gaussian-noise robustness (per arm)"))
P(("code", """if ARMS:
    rng = np.random.default_rng(0)
    def noisy(snr):
        def t(x):
            std = x.std() * (10 ** (-snr / 20.0))
            return x + rng.normal(0, std, x.shape).astype(np.float32)
        return t
    rows = []; fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in ARMS.items():
        ser = []
        for snr in [40, 30, 20, 10, 5, 0]:
            _, v = eval_one_step(m, TEST, transform=noisy(snr))
            rows.append({"arm": name, "snr_db": snr, "vrmse": v}); ser.append((snr, v))
        ax.plot([s for s, _ in ser], [v for _, v in ser], "o-", label=name)
    savecsv("ext3_noise.csv", rows)
    ax.invert_xaxis(); ax.set_xlabel("input SNR (dB)"); ax.set_ylabel("test VRMSE")
    ax.set_title("Robustness to input noise"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "ext3_noise.png", dpi=150); plt.close(fig)
    print("saved ext3_noise.{csv,png}")"""))

P(("md", "## Extension 4: Autoregressive rollout + windowed VRMSE (per arm)"))
P(("code", """@torch.no_grad()
def rollout(m, data, steps, bs=256):
    N, S = data.shape[0], data.shape[1]; steps = min(steps, S - 1)
    preds = torch.empty((N, steps) + tuple(data.shape[2:]), dtype=torch.float32)
    init = torch.from_numpy(data[:, 0]).float()
    for i in range(0, N, bs):
        s = init[i:i+bs].to(DEVICE)
        for k in range(steps):
            s = step_predict(m, s); preds[i:i+bs, k] = s.cpu()
    gt = torch.from_numpy(data[:, 1:steps + 1]).float()
    return preds, gt

if ARMS:
    STEPS = min(30, TEST.shape[1] - 1); rows = []; wins = []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, m in ARMS.items():
        preds, gt = rollout(m, TEST, STEPS)
        per = [vrmse(preds[:, k], gt[:, k]).item() for k in range(STEPS)]
        for k, v in enumerate(per, 1): rows.append({"arm": name, "step": k, "vrmse": v})
        def win(a, b):
            a, b = min(a, STEPS), min(b, STEPS)
            return float("nan") if b <= a else window_vrmse(preds, gt, a, b, time_dim=1).item()
        wins.append({"arm": name, "vrmse_6_12": win(6, 12), "vrmse_13_30": win(13, 30)})
        ax.plot(range(1, STEPS + 1), per, "o-", ms=3, label=name)
    savecsv("ext4_rollout.csv", rows); savecsv("ext4_windows.csv", wins)
    for w in wins: print(w)
    ax.axvspan(6, 12, alpha=0.1, color="C1"); ax.axvspan(13, 30, alpha=0.1, color="C2")
    ax.set_xlabel("rollout step"); ax.set_ylabel("VRMSE"); ax.set_title("Error accumulation"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "ext4_rollout.png", dpi=150); plt.close(fig)
    print("saved ext4_rollout.{csv,png}, ext4_windows.csv")"""))

P(("md", "## Extension 5: Input spectral sensitivity (per arm)"))
P(("code", """if ARMS:
    yy, xx = np.ogrid[:H, :W]; cy, cx = H / 2.0, W / 2.0
    rr = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    def lowpass(frac):
        mask = (rr <= frac)[None, ..., None]
        def t(x):
            F = np.fft.fftshift(np.fft.fft2(x, axes=(1, 2)), axes=(1, 2)) * mask
            return np.real(np.fft.ifft2(np.fft.ifftshift(F, axes=(1, 2)), axes=(1, 2))).astype(np.float32)
        return t
    rows = []; fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in ARMS.items():
        ser = []
        for frac in [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]:
            _, v = eval_one_step(m, TEST, transform=lowpass(frac))
            rows.append({"arm": name, "retained_frac": frac, "vrmse": v}); ser.append((frac, v))
        ax.plot([f for f, _ in ser], [v for _, v in ser], "o-", label=name)
    savecsv("ext5_spectral_sensitivity.csv", rows)
    ax.set_xlabel("retained input freq fraction"); ax.set_ylabel("test VRMSE")
    ax.set_title("Input frequency sensitivity"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "ext5_spectral_sensitivity.png", dpi=150); plt.close(fig)
    print("saved ext5_spectral_sensitivity.{csv,png}")"""))

P(("md", "## Extension 6: Energy spectrum (per arm vs truth)"))
P(("code", """def radial_psd(field):
    F = np.fft.fftshift(np.fft.fft2(field, axes=(1, 2)), axes=(1, 2))
    Pw = (np.abs(F) ** 2).mean(0); h, w = Pw.shape
    yy, xx = np.indices((h, w)); r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2).astype(int)
    return np.bincount(r.ravel(), Pw.ravel()) / np.maximum(np.bincount(r.ravel()), 1)

if ARMS:
    fig, ax = plt.subplots(figsize=(6, 4))
    ps_true = radial_psd(TEST[:, 1][..., 0]); k = np.arange(1, len(ps_true))
    ax.loglog(k, ps_true[1:], "k-", lw=2, label="ground truth")
    rows = [{"k": int(i), "psd_true": ps_true[i]} for i in k]
    for name, m in ARMS.items():
        xb = torch.from_numpy(TEST[:, 0]).float().to(DEVICE)
        pb = step_predict(m, xb).cpu().numpy()
        ps = radial_psd(pb[..., 0]); ax.loglog(k, ps[1:], label=name)
        for j, i in enumerate(k): rows[j][f"psd_{name}"] = ps[i]
    savecsv("ext6_energy_spectrum.csv", rows)
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("power"); ax.set_title("Energy spectrum (field 0)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "ext6_energy_spectrum.png", dpi=150); plt.close(fig)
    print("saved ext6_energy_spectrum.{csv,png}")"""))

P(("md", "## Extension 7: Spatial error maps (per arm)"))
P(("code", """if ARMS:
    t_show = min(10, TEST.shape[1] - 2); n_arm = len(ARMS)
    fig, axes = plt.subplots(n_arm, 3, figsize=(9, 3 * n_arm), squeeze=False)
    xb = torch.from_numpy(TEST[:1, t_show]).float().to(DEVICE)
    true = TEST[0, t_show + 1, ..., 0]
    for r, (name, m) in enumerate(ARMS.items()):
        pred = step_predict(m, xb).cpu().numpy()[0, ..., 0]
        for c, (img, ttl) in enumerate([(true, "truth"), (pred, f"{name}"), (np.abs(true - pred), "|err|")]):
            ax = axes[r][c]; im = ax.imshow(img, cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(ttl)
            if c == 1: ax.set_ylabel(name)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Spatial error (t={t_show}->{t_show+1}, field 0)")
    fig.tight_layout(); fig.savefig(OUT / "ext7_error_maps.png", dpi=150); plt.close(fig)
    print("saved ext7_error_maps.png")"""))

P(("md", "## Extension 8: Inference benchmark (per arm, CPU + GPU)"))
P(("code", """def bench_model(m, devname, bs_list=(1, 4, 16, 64), reps=20):
    dev = torch.device(devname); m = m.to(dev).eval(); res = []
    with torch.no_grad():
        for bs in bs_list:
            x = torch.randn(bs, FIELDS, H, W, device=dev)
            for _ in range(5): m(x)
            if devname == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(reps): m(x)
            if devname == "cuda": torch.cuda.synchronize()
            dt = (time.time() - t0) / reps
            res.append({"arm": "?", "device": devname, "batch": bs, "ms_per_batch": dt * 1e3, "samples_per_s": bs / dt})
    return res

if ARMS:
    rows = []; fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, m in ARMS.items():
        devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dn in devs:
            try:
                r = bench_model(m, dn)
            except Exception as e:
                print(f"  benchmark {name}/{dn} skipped: {e}"); continue
            for x in r: x["arm"] = name
            rows += r
            d = [x for x in r if x["device"] == dn]
            ax.plot([x["batch"] for x in d], [x["samples_per_s"] for x in d], "o-", label=f"{name}/{dn}")
        m.to(DEVICE)
    savecsv("ext8_benchmark.csv", rows)
    ax.set_xscale("log", base=2); ax.set_xlabel("batch size"); ax.set_ylabel("samples/s")
    ax.set_title("Inference throughput"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "ext8_benchmark.png", dpi=150); plt.close(fig)
    print("saved ext8_benchmark.{csv,png}")"""))

P(("md", """## Freebie A - Reproducibility audit: released code vs. paper

A zero-compute reproducibility finding (the kind MLRC explicitly values): we
programmatically check whether the released implementations contain spectral
(FFT) operations, and contrast that with the paper's described architecture."""))
P(("code", """import inspect
from litefno.models import litefno as _cnn_mod, fno_s as _fnos_mod

def has_spectral(mod):
    s = inspect.getsource(mod).lower()
    return ("fft" in s) or ("rfft" in s) or ("spectralconv" in s)

finding = {
    "paper_litefno_architecture": "spectral FFT + CP low-rank + transduction",
    "repo_litefno_class_is_spectral": has_spectral(_cnn_mod),
    "repo_fno_s_class_is_spectral": has_spectral(_fnos_mod),
}
print("Reproducibility finding")
print("  Paper LiteFNO  : spectral (FFT) + CP low-rank factorization + transduction")
print("  Repo 'LiteFNO' : spectral?", finding["repo_litefno_class_is_spectral"])
print("  Repo  FNO-S    : spectral?", finding["repo_fno_s_class_is_spectral"])
if not finding["repo_litefno_class_is_spectral"]:
    print("  => MISMATCH: the released 'LiteFNO' is a CNN (no FFT), not the spectral")
    print("     architecture the paper describes. This motivates our generalization study:")
    print("     does the spectral machinery actually earn its keep vs a plain low-rank CNN?")
with open(OUT / "freebieA_repro_audit.json", "w") as f:
    json.dump(finding, f, indent=2)
print("saved freebieA_repro_audit.json")"""))

P(("md", "## Freebie B - FLOPs / compute-accuracy trade-off (per arm)"))
P(("code", """try:
    from thop import profile
    HAVE_THOP = True
except Exception:
    HAVE_THOP = False
    print("thop unavailable (pip install thop) - skipping FLOPs")

if ARMS and HAVE_THOP:
    rows = []
    x = torch.randn(1, FIELDS, H, W).to(DEVICE)
    for name, m in ARMS.items():
        flops = None
        try:
            macs, _ = profile(copy.deepcopy(m).to(DEVICE), inputs=(x,), verbose=False)
            flops = 2 * macs
        except Exception as e:
            print(name, "FLOPs failed:", e)
        _, v = eval_one_step(m, TEST)
        rows.append({"arm": name, "params": PARAMS[name], "flops": flops, "test_vrmse": v})
        print(f"{name:>13}: params={PARAMS[name]:,}  flops={flops}  vrmse={v:.5f}")
    savecsv("freebieB_flops.csv", rows)
    pts = [r for r in rows if r["flops"]]
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in pts:
            ax.scatter(r["flops"], r["test_vrmse"])
            ax.annotate(r["arm"], (r["flops"], r["test_vrmse"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("FLOPs / forward (1 sample)"); ax.set_ylabel("test VRMSE")
        ax.set_title("Compute-accuracy trade-off")
        fig.tight_layout(); fig.savefig(OUT / "freebieB_flops.png", dpi=150); plt.close(fig)
    print("saved freebieB_flops.{csv,png}")"""))

P(("md", "## Logs-only analysis (8 datasets, no checkpoint needed)"))
P(("code", """DATASETS = ["gray_scott_reaction_diffusion","euler_multi_quadrants_openBC","euler_multi_quadrants_periodicBC",
            "acoustic_scattering_discontinuous","active_matter","rayleigh_benard",
            "turbulent_radiative_layer_2D","viscoelastic_instability"]
SHORT = {"gray_scott_reaction_diffusion":"GS","euler_multi_quadrants_openBC":"EMQ-O","euler_multi_quadrants_periodicBC":"EMQ-P",
         "acoustic_scattering_discontinuous":"AS-SD","active_matter":"AM","rayleigh_benard":"RB",
         "turbulent_radiative_layer_2D":"TRL-2D","viscoelastic_instability":"VI"}
LOGDIR = Path("outputs/logs")
def read_log(ds, model):
    p = LOGDIR / f"{ds}_{model}.jsonl"
    if not p.exists(): return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    tr = [r for r in rows if "valid_vrmse" in r]; te = [r for r in rows if "test_vrmse" in r]
    if not tr: return None
    return {"params": tr[0].get("params"), "best_valid_vrmse": min(r["valid_vrmse"] for r in tr),
            "final_train_vrmse": tr[-1].get("train_vrmse"), "final_valid_vrmse": tr[-1].get("valid_vrmse"),
            "test_vrmse": te[-1]["test_vrmse"] if te else None,
            "curve": [(i, r["valid_vrmse"]) for i, r in enumerate(tr)]}
CNN = {d: read_log(d, "litefno") for d in DATASETS}
FNOS = {d: read_log(d, "fno_s") for d in DATASETS}

table = []
for d in DATASETS:
    l, f = CNN[d], FNOS[d]
    if not l: continue
    imp = (f["test_vrmse"] - l["test_vrmse"]) / f["test_vrmse"] * 100 if (f and f["test_vrmse"]) else None
    table.append({"dataset": SHORT[d], "cnn_params": l["params"], "cnn_test_vrmse": l["test_vrmse"],
                  "fnos_test_vrmse": f["test_vrmse"] if f else None, "improvement_pct": imp})
savecsv("logs_reproduction_table.csv", table)
for r in table: print(r)"""))

P(("code", """# improvement bar + efficiency frontier + convergence + generalization gap
imp = [r for r in table if r["improvement_pct"] is not None]
if imp:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["dataset"] for r in imp], [r["improvement_pct"] for r in imp]); ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("CNN vs FNO-S VRMSE improvement (%)"); ax.set_title("Logs: CNN arm vs FNO-S")
    fig.tight_layout(); fig.savefig(OUT / "logs_improvement.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for d in DATASETS:
    l, f = CNN[d], FNOS[d]
    if l and l["test_vrmse"]:
        ax.scatter(l["params"], l["test_vrmse"], color="C0")
        ax.annotate(SHORT[d], (l["params"], l["test_vrmse"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    if f and f["test_vrmse"]: ax.scatter(f["params"], f["test_vrmse"], color="C1", marker="^")
ax.scatter([], [], color="C0", label="CNN"); ax.scatter([], [], color="C1", marker="^", label="FNO-S")
ax.set_xlabel("params"); ax.set_ylabel("test VRMSE"); ax.set_yscale("log"); ax.set_title("Efficiency frontier"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "logs_frontier.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
for d in DATASETS:
    l = CNN[d]
    if not l: continue
    ax.plot([i for i, _ in l["curve"]], [v for _, v in l["curve"]], lw=1, label=SHORT[d])
ax.set_xlabel("epoch"); ax.set_ylabel("valid VRMSE"); ax.set_yscale("log"); ax.set_title("CNN convergence"); ax.legend(ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "logs_convergence.png", dpi=150); plt.close(fig)
print("saved logs_*.png")"""))

P(("code", """print("Artifacts in", OUT.resolve())
for p in sorted(OUT.iterdir()): print("  ", p.name)"""))


# ===========================================================================
out_dir = Path(__file__).resolve().parent.parent / "notebooks"
out_dir.mkdir(parents=True, exist_ok=True)
for name, cells in [("phase2_train_real_litefno", phase2), ("phase3_extensions", phase3)]:
    path = out_dir / f"{name}.ipynb"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(make_notebook(cells), fh, indent=1)
    print("wrote", path)
