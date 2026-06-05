"""Builds notebooks/spectral_regularizer.ipynb — the spectral-consistency
regularizer experiment (CNN with vs without the loss, multi-seed) on Gray-Scott.

Run: python scripts/build_regularizer_notebook.py
"""
from __future__ import annotations
import json
from pathlib import Path

cells = []
def md(s): cells.append(("md", s))
def code(s): cells.append(("code", s))

md(r"""# Spectral-Consistency Regularizer (Gray-Scott)

A small **novel contribution** on top of the LiteFNO study: we add a
differentiable **spectral-consistency loss** that penalizes the mismatch between
the predicted and ground-truth Fourier magnitude spectra, and test whether it
gives a plain low-rank **CNN** better high-frequency / energy-spectrum fidelity.

Loss:  `L = MSE(pred, target) + λ · MSE(|F(pred)|, |F(target)|)`

It trains the CNN **with vs. without** the regularizer across several seeds and
reports: one-step VRMSE, a high-wavenumber PSD error metric, the predicted-vs-true
energy spectrum, and (near-free bonus) autoregressive rollout windows.

**Setup:** Add Input → your `gs-processed` dataset (train/valid/test.h5).
Accelerator = **GPU T4**. Internet ON (for the one-time git clone).""")

code('''import os, subprocess, sys
REPO = "litefno-repro"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/AIscend-Research/litefno-repro"], check=True)
sys.path.insert(0, os.path.abspath(REPO))   # import litefno without pip (deps already on Kaggle)

import json, time, glob, copy, csv
from pathlib import Path
import numpy as np, torch, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from litefno.train import build_model, flatten_time, set_seed
from litefno.data import DatasetConfig, H5SequenceDataset
from litefno.metrics import rmse, vrmse, window_vrmse

torch.manual_seed(0); np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    try: _ = (torch.zeros(1, device=DEVICE) + 1).item()
    except Exception as e: print("CUDA unusable -> CPU:", e); DEVICE = torch.device("cpu")
print("device:", DEVICE)
OUT = Path("/kaggle/working/spectral_reg") if Path("/kaggle/working").exists() else Path("spectral_reg_out")
OUT.mkdir(parents=True, exist_ok=True)''')

md("## Config + data (auto-finds the mounted gs-processed dataset)")
code('''KEY = "data"; FIELDS = 2
SEEDS  = [0, 1, 2]      # set to [0] for a quick 2-run smoke test
EPOCHS = 200
BATCH  = 64
LR     = 1e-3; LR_STEP = 100; LR_GAMMA = 0.5
LAMBDA = 0.1           # spectral-consistency weight (tune: try 0.03, 0.1, 0.3)

def find(split):
    h = sorted(glob.glob(f"/kaggle/input/**/{split}.h5", recursive=True))
    return Path(h[0]) if h else None
TRAIN_H5, VALID_H5, TEST_H5 = find("train"), find("valid"), find("test")
print("train:", TRAIN_H5, "| test:", TEST_H5)
assert TRAIN_H5 and TEST_H5, "Mount your gs-processed dataset via Add Input!"

def make_loader(p, bs, shuffle):
    cfg = DatasetConfig(path=p, dataset_key=KEY, input_steps=1, output_steps=1, stride=1, cache="memory")
    return DataLoader(H5SequenceDataset(cfg), batch_size=bs, shuffle=shuffle)
train_loader = make_loader(TRAIN_H5, BATCH, True)

with h5py.File(TRAIN_H5, "r") as f: H, W = f[KEY].shape[2], f[KEY].shape[3]
with h5py.File(TEST_H5, "r") as f: TEST = f[KEY][...].astype(np.float32)
print(f"resolution {H}x{W}  train batches={len(train_loader)}  test={TEST.shape}")''')

md("## The differentiable spectral-consistency loss + training")
code('''def spectral_loss(pred, target):
    # pred, target: (B, C, H, W). Penalize Fourier-magnitude mismatch (differentiable).
    Fp = torch.fft.rfft2(pred,  norm="ortho")
    Ft = torch.fft.rfft2(target, norm="ortho")
    return torch.mean((Fp.abs() - Ft.abs()) ** 2)

def to_xy(batch):
    xb, yb = batch                       # each (B,1,H,W,2)
    return flatten_time(xb).to(DEVICE), flatten_time(yb).to(DEVICE)   # (B,2,H,W)

def train_cnn(lam, seed):
    set_seed(seed)
    model = build_model({"name": "litefno", "layers": 8, "width": 64, "rank": 32}, FIELDS, FIELDS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, LR_STEP, LR_GAMMA)
    for ep in range(EPOCHS):
        model.train()
        for batch in train_loader:
            x, y = to_xy(batch)
            opt.zero_grad(set_to_none=True)
            p = model(x)
            loss = torch.nn.functional.mse_loss(p, y) + lam * spectral_loss(p, y)
            loss.backward(); opt.step()
        sch.step()
    return model.eval()''')

md("## Eval helpers (one-step VRMSE, radial PSD, high-wavenumber error)")
code('''@torch.no_grad()
def eval_vrmse(model, data, bs=256):
    N, S = data.shape[0], data.shape[1]; vs = []
    for t in range(S - 1):
        x = torch.from_numpy(data[:, t]).float(); y = torch.from_numpy(data[:, t + 1]).float()
        for i in range(0, N, bs):
            xb = x[i:i+bs].permute(0, 3, 1, 2).to(DEVICE)
            yb = y[i:i+bs].permute(0, 3, 1, 2).to(DEVICE)
            vs.append(vrmse(model(xb), yb).item())
    return float(np.mean(vs))

def radial_psd(field):  # (B,H,W) -> 1D radial power spectrum
    F = np.fft.fftshift(np.fft.fft2(field, axes=(1, 2)), axes=(1, 2))
    P = (np.abs(F) ** 2).mean(0); h, w = P.shape
    yy, xx = np.indices((h, w)); r = np.sqrt((yy - h/2) ** 2 + (xx - w/2) ** 2).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)

@torch.no_grad()
def pred_field0(model, data):   # one-step preds (t0->t1), field 0
    x = torch.from_numpy(data[:, 0]).permute(0, 3, 1, 2).float().to(DEVICE)
    return model(x).permute(0, 2, 3, 1).cpu().numpy()[..., 0]

def highfreq_relerr(model, data):  # relative PSD error over upper-half wavenumbers
    ps_p = radial_psd(pred_field0(model, data)); ps_t = radial_psd(data[:, 1][..., 0])
    k = np.arange(1, len(ps_t)); hi = k[len(k)//2:]
    return float(np.mean(np.abs(ps_p[hi] - ps_t[hi]) / (ps_t[hi] + 1e-12)))''')

md("## Run: CNN baseline vs CNN + spectral reg, across seeds")
code('''rows = []; models = {"baseline": [], "reg": []}
for seed in SEEDS:
    for tag, lam in [("baseline", 0.0), ("reg", LAMBDA)]:
        t0 = time.time(); m = train_cnn(lam, seed); dt = time.time() - t0
        v = eval_vrmse(m, TEST); hf = highfreq_relerr(m, TEST)
        torch.save({"model_state": m.state_dict(), "lam": lam, "seed": seed},
                   OUT / f"cnn_{tag}_seed{seed}.pt")
        rows.append({"config": tag, "lambda": lam, "seed": seed,
                     "test_vrmse": v, "highfreq_psd_relerr": hf, "train_s": round(dt)})
        models[tag].append(m)
        print(f"{tag:>8} seed{seed}: VRMSE={v:.5f}  HF-PSD-relerr={hf:.4f}  ({dt:.0f}s)")
with open(OUT / "reg_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved reg_results.csv")''')

md("## Summary (mean ± std over seeds)")
code('''for tag in ["baseline", "reg"]:
    vs = [r["test_vrmse"] for r in rows if r["config"] == tag]
    hf = [r["highfreq_psd_relerr"] for r in rows if r["config"] == tag]
    print(f"{tag:>8}:  VRMSE {np.mean(vs):.5f} +/- {np.std(vs):.5f}   "
          f"|  HF-PSD relerr {np.mean(hf):.4f} +/- {np.std(hf):.4f}")
print("\\nInterpretation: if 'reg' has LOWER HF-PSD-relerr at similar/better VRMSE,")
print("the spectral-consistency loss improves high-frequency fidelity -> the contribution holds.")''')

md("## Energy-spectrum figure (truth vs baseline vs reg, averaged over seeds)")
code('''ps_true = radial_psd(TEST[:, 1][..., 0]); k = np.arange(1, len(ps_true))
def avg_ps(ms): return np.mean([radial_psd(pred_field0(m, TEST)) for m in ms], axis=0)
ps_b, ps_r = avg_ps(models["baseline"]), avg_ps(models["reg"])

with open(OUT / "reg_energy_spectrum.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["k", "truth", "baseline", "reg"])
    for i in k: w.writerow([int(i), ps_true[i], ps_b[i], ps_r[i]])

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(k, ps_true[1:], "k-", lw=2, label="ground truth")
ax.loglog(k, ps_b[1:], label="CNN baseline")
ax.loglog(k, ps_r[1:], label="CNN + spectral reg")
ax.set_xlabel("wavenumber k"); ax.set_ylabel("power")
ax.set_title("Energy spectrum (field 0): effect of spectral-consistency reg"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "reg_energy_spectrum.png", dpi=150); plt.close(fig)
print("saved reg_energy_spectrum.{csv,png}")''')

md("## Bonus (near-free): autoregressive rollout windows")
code('''@torch.no_grad()
def rollout_windows(model, data, steps=30, bs=256):
    N, S = data.shape[0], data.shape[1]; steps = min(steps, S - 1)
    preds = torch.empty((N, steps) + tuple(data.shape[2:]), dtype=torch.float32)
    init = torch.from_numpy(data[:, 0]).float()
    for i in range(0, N, bs):
        s = init[i:i+bs].to(DEVICE)
        for kk in range(steps):
            s = model(s.permute(0, 3, 1, 2)).permute(0, 2, 3, 1); preds[i:i+bs, kk] = s.cpu()
    gt = torch.from_numpy(data[:, 1:steps + 1]).float()
    def win(a, b):
        a, b = min(a, steps), min(b, steps)
        return float("nan") if b <= a else window_vrmse(preds, gt, a, b, time_dim=1).item()
    return win(6, 12), win(13, 30)

rrows = []
for tag in ["baseline", "reg"]:
    ws = [rollout_windows(m, TEST) for m in models[tag]]
    a = float(np.nanmean([w[0] for w in ws])); b = float(np.nanmean([w[1] for w in ws]))
    rrows.append({"config": tag, "vrmse_6_12": a, "vrmse_13_30": b})
    print(f"{tag:>8} rollout: 6:12={a:.4f}  13:30={b:.4f}")
with open(OUT / "reg_rollout.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "vrmse_6_12", "vrmse_13_30"]); w.writeheader(); w.writerows(rrows)
print("saved reg_rollout.csv")''')

md("## Outputs")
code('''print("Artifacts in", OUT.resolve())
for p in sorted(OUT.iterdir()): print("  ", p.name)''')

# ---- emit notebook ----
def mk(cells):
    out = []
    for kind, src in cells:
        lines = src.split("\n"); source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source})
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}

p = Path(__file__).resolve().parent.parent / "notebooks" / "spectral_regularizer.ipynb"
p.parent.mkdir(parents=True, exist_ok=True)
with p.open("w", encoding="utf-8") as fh: json.dump(mk(cells), fh, indent=1)
print("wrote", p)
