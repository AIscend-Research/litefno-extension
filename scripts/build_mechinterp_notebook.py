"""Builds notebooks/mechinterp_day1.ipynb — Day-1 mechanistic-interpretability
validation for the trained spectral LiteFNO checkpoint:
  (1) dead-mode analysis (per-Fourier-mode weight magnitude),
  (2) CP-rank utilization (effective rank of the factorization),
  (3) causal mode ablation -> one-step VRMSE AND rollout windows,
  (4) extended rollout curve.
Goal: see if the thesis holds ("modes mostly dead; active low-freq modes drive
rollout stability") BEFORE spending the rigor budget.

Run: python scripts/build_mechinterp_notebook.py
"""
from __future__ import annotations
import json
from pathlib import Path

cells = []
def md(s): cells.append(("md", s))
def code(s): cells.append(("code", s))

BUILD = r'''
def build_real_litefno(in_ch, out_ch, modes, width=64, layers=8, rank=0.5, factorization="cp"):
    from neuralop.models import FNO
    base = dict(n_modes=(modes, modes), hidden_channels=width,
                in_channels=in_ch, out_channels=out_ch, n_layers=layers)
    fac = None if factorization in (None, "dense") else factorization
    for f, r in [(fac, rank), ("tucker", rank), (None, None)]:
        try:
            return (FNO(**base), "dense") if f is None else (FNO(**base, factorization=f, rank=r), f)
        except Exception as e:
            print("  build", f, "failed:", e)
    raise RuntimeError("could not build FNO")
'''.strip("\n")

md(r"""# Day-1 Mech-Interp Validation — Spectral LiteFNO

Cheap (~minutes, eval-only) check of the proposed thesis **before** investing the
multi-seed/multi-dataset rigor budget:

> *LiteFNO's spectral capacity is mostly **dead**; the few active **low-frequency**
> modes are what causally drive its **rollout stability**.*

We run, on your trained LiteFNO checkpoint:
1. **Dead-mode analysis** — per-Fourier-mode weight magnitude (are high modes ~0?)
2. **CP-rank utilization** — effective rank of the CP factorization
3. **Causal mode ablation** — keep only the lowest-f fraction of modes, measure
   one-step VRMSE **and** rollout windows (do high modes matter? do low modes
   matter *more* for rollout?)
4. **Extended rollout** curve for the full model

**GO/NO-GO:** if high modes are NOT dead, or ablating them hurts as much as low
modes, the thesis doesn't hold — stop here and pivot before spending GPU.

**Setup:** Add Input → `gs-processed` (test.h5) **and** your LiteFNO checkpoint
dataset. GPU optional (eval is light). Internet ON for the clone.""")

code('''import os, subprocess, sys
REPO = "litefno-repro"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/AIscend-Research/litefno-repro"], check=True)
sys.path.insert(0, os.path.abspath(REPO))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuraloperator"], check=False)

import json, glob, copy, csv
from pathlib import Path
import numpy as np, torch, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from litefno.metrics import vrmse, window_vrmse
from litefno.train import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    try: _ = (torch.zeros(1, device=DEVICE) + 1).item()
    except Exception as e: print("CUDA unusable -> CPU:", e); DEVICE = torch.device("cpu")
print("device:", DEVICE)
OUT = Path("/kaggle/working/mechinterp") if Path("/kaggle/working").exists() else Path("mechinterp_out")
OUT.mkdir(parents=True, exist_ok=True)
KEY = "data"; FIELDS = 2''')

md("## Load the trained LiteFNO checkpoint + test data (auto-found)")
code(BUILD + '''

ckpts = sorted(glob.glob("/kaggle/input/**/litefno_real_best.pt", recursive=True))
tests = sorted(glob.glob("/kaggle/input/**/test.h5", recursive=True))
assert ckpts, "Mount your LiteFNO checkpoint dataset (litefno_real_best.pt)!"
assert tests, "Mount gs-processed (test.h5)!"
CKPT, TEST_H5 = Path(ckpts[0]), Path(tests[0])
print("ckpt:", CKPT, "\\ntest:", TEST_H5)

ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
b = ck["build"]
model, kind = build_real_litefno(b["in_ch"], b["out_ch"], b["modes"], b["width"], b["layers"], b["rank"], b["factorization"])
model.load_state_dict(ck["model_state"]); model.to(DEVICE).eval()
PARAMS = sum(p.numel() for p in model.parameters())
print(f"loaded LiteFNO: factorization={kind}, params={PARAMS:,}, modes={b['modes']}")

with h5py.File(TEST_H5, "r") as f: TEST = f[KEY][...].astype(np.float32)
H, W = TEST.shape[2], TEST.shape[3]
print("test:", TEST.shape)

# Matched CNN checkpoint (trained on the SAME gs-processed data -> fair comparison).
cnns = (sorted(glob.glob("/kaggle/input/**/cnn_baseline*.pt", recursive=True))
        or sorted(glob.glob("/kaggle/input/**/cnn_*.pt", recursive=True)))
CNN = None
if cnns:
    CNN = build_model({"name": "litefno", "layers": 8, "width": 64, "rank": 32}, FIELDS, FIELDS)
    st = torch.load(cnns[0], map_location=DEVICE, weights_only=False)
    CNN.load_state_dict(st["model_state"]); CNN.to(DEVICE).eval()
    print("loaded matched CNN:", cnns[0])
else:
    print("No CNN checkpoint mounted -> comparative cell will be skipped (mount cnn_baseline_seed0.pt).")''')


md("## Eval helpers")
code('''@torch.no_grad()
def step_predict(m, state):                       # (B,H,W,C)->(B,H,W,C)
    return m(state.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

@torch.no_grad()
def eval_vrmse(m, data, bs=256):
    N, S = data.shape[0], data.shape[1]; vs = []
    for t in range(S - 1):
        x = torch.from_numpy(data[:, t]).float(); y = torch.from_numpy(data[:, t + 1]).float()
        for i in range(0, N, bs):
            p = step_predict(m, x[i:i+bs].to(DEVICE)); yb = y[i:i+bs].to(DEVICE)
            vs.append(vrmse(p, yb).item())
    return float(np.mean(vs))

@torch.no_grad()
def rollout(m, data, steps=30, bs=256):
    N, S = data.shape[0], data.shape[1]; steps = min(steps, S - 1)
    preds = torch.empty((N, steps) + tuple(data.shape[2:]), dtype=torch.float32)
    init = torch.from_numpy(data[:, 0]).float()
    for i in range(0, N, bs):
        s = init[i:i+bs].to(DEVICE)
        for kk in range(steps):
            s = step_predict(m, s); preds[i:i+bs, kk] = s.cpu()
    gt = torch.from_numpy(data[:, 1:steps + 1]).float()
    per = [vrmse(preds[:, kk], gt[:, kk]).item() for kk in range(steps)]
    def win(a, b):
        a, b = min(a, steps), min(b, steps)
        return float("nan") if b <= a else window_vrmse(preds, gt, a, b, time_dim=1).item()
    return per, win(6, 12), win(13, 30)

@torch.no_grad()
def rollout_full(m, data, steps, bs=256):       # returns (preds, gt)
    N, S = data.shape[0], data.shape[1]; steps = min(steps, S - 1)
    preds = torch.empty((N, steps) + tuple(data.shape[2:]), dtype=torch.float32)
    init = torch.from_numpy(data[:, 0]).float()
    for i in range(0, N, bs):
        s = init[i:i+bs].to(DEVICE)
        for kk in range(steps):
            s = step_predict(m, s); preds[i:i+bs, kk] = s.cpu()
    return preds, torch.from_numpy(data[:, 1:steps + 1]).float()

def radial_psd(field):                          # (B,H,W) -> 1D radial power
    F = np.fft.fftshift(np.fft.fft2(field, axes=(1, 2)), axes=(1, 2))
    P = (np.abs(F) ** 2).mean(0); h, w = P.shape
    yy, xx = np.indices((h, w)); r = np.sqrt((yy - h/2) ** 2 + (xx - w/2) ** 2).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)''')

md("""## (1) Dead-mode analysis

For each spectral layer, the CP factors `factor_2` (mode-dim0) and `factor_3`
(mode-dim1) weight each Fourier mode. Per-mode magnitude = row norm across rank,
averaged over layers. If high-index modes are ~0, the model's spectral capacity
is mostly dead.""")
code('''def layer_factors(m):
    d = {}
    for n, p in m.named_parameters():
        if ".weight.factors.factor_" in n:
            L = int(n.split(".")[2]); f = n.split(".")[-1]
            d.setdefault(L, {})[f] = p.detach().cpu()
    return [d[k] for k in sorted(d)]

LF = layer_factors(model)
m0 = np.mean([LF[l]["factor_2"].abs().norm(dim=1).numpy() for l in range(len(LF))], axis=0)
m1 = np.mean([LF[l]["factor_3"].abs().norm(dim=1).numpy() for l in range(len(LF))], axis=0)
m0n, m1n = m0 / m0.max(), m1 / m1.max()
dead0 = float((m0n < 0.1).mean()); dead1 = float((m1n < 0.1).mean())
print("dim0 per-mode (norm):", np.round(m0n, 2))
print("dim1 per-mode (norm):", np.round(m1n, 2))
print(f"fraction of modes with <10% of peak magnitude:  dim0={dead0:.0%}  dim1={dead1:.0%}")

with open(OUT / "deadmode.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["axis", "mode_idx", "norm_mag"])
    for i, v in enumerate(m0n): w.writerow(["dim0", i, v])
    for i, v in enumerate(m1n): w.writerow(["dim1", i, v])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(m0n)), m0n, "o-", label="mode dim0")
ax.plot(range(len(m1n)), m1n, "s-", label="mode dim1")
ax.axhline(0.1, ls="--", color="gray", lw=0.8, label="10% threshold")
ax.set_xlabel("Fourier mode index"); ax.set_ylabel("normalized weight magnitude")
ax.set_title("Per-mode spectral weight magnitude (dead-mode analysis)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "deadmode.png", dpi=150); plt.close(fig)
print("saved deadmode.{csv,png}")''')

md("## (2) CP-rank utilization")
code('''def rank_components(fac):
    return (fac["factor_0"].abs().norm(dim=0) * fac["factor_1"].abs().norm(dim=0) *
            fac["factor_2"].abs().norm(dim=0) * fac["factor_3"].abs().norm(dim=0)).numpy()

R = LF[0]["factor_0"].shape[1]
comp = np.mean([rank_components(LF[l]) for l in range(len(LF))], axis=0)
comp = np.sort(comp)[::-1]; cum = np.cumsum(comp) / comp.sum()
eff90 = int((cum < 0.9).sum()) + 1
print(f"CP rank R={R}; rank components carrying 90% of energy: {eff90}  ({eff90/R:.0%} of rank)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, R + 1), cum, "o-")
ax.axhline(0.9, ls="--", color="gray", lw=0.8)
ax.set_xlabel("# CP rank components (sorted)"); ax.set_ylabel("cumulative energy")
ax.set_title(f"CP-rank utilization (effective rank {eff90}/{R})")
fig.tight_layout(); fig.savefig(OUT / "cprank.png", dpi=150); plt.close(fig)
print("saved cprank.png")''')

md("""## (3) Causal mode ablation — one-step VRMSE *and* rollout

Keep only the lowest fraction `f` of Fourier modes (zero the CP mode-factor rows
above the cutoff), then measure both one-step VRMSE and rollout windows. The
thesis predicts: cutting **high** modes barely hurts (dead), but accuracy —
especially **rollout** — collapses once you cut into the **low** modes.""")
code('''def ablate_keep(m, frac):
    mq = copy.deepcopy(m)
    for n, p in mq.named_parameters():
        if "factor_2" in n:
            k = max(1, int(round(p.shape[0] * frac))); p.data[k:] = 0
        if "factor_3" in n:
            k = max(1, int(round(p.shape[0] * frac))); p.data[k:] = 0
    return mq.eval()

rows = []
for frac in [1.0, 0.75, 0.5, 0.25, 0.1]:
    mq = ablate_keep(model, frac)
    v = eval_vrmse(mq, TEST)
    _, w612, w1330 = rollout(mq, TEST, steps=min(30, TEST.shape[1] - 1))
    rows.append({"keep_frac": frac, "onestep_vrmse": v, "rollout_6_12": w612, "rollout_13_30": w1330})
    print(f"keep {frac:>4} of modes:  one-step={v:.4f}  rollout6:12={w612:.4f}  rollout13:30={w1330:.4f}")
with open(OUT / "mode_ablation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(figsize=(6.5, 4))
fr = [r["keep_frac"] for r in rows]
ax.plot(fr, [r["onestep_vrmse"] for r in rows], "o-", label="one-step VRMSE")
ax.plot(fr, [r["rollout_6_12"] for r in rows], "s-", label="rollout 6:12")
ax.plot(fr, [r["rollout_13_30"] for r in rows], "^-", label="rollout 13:30")
ax.set_xlabel("fraction of (lowest) modes kept"); ax.set_ylabel("VRMSE")
ax.set_title("Causal mode ablation: which modes are load-bearing?"); ax.legend()
ax.invert_xaxis(); fig.tight_layout(); fig.savefig(OUT / "mode_ablation.png", dpi=150); plt.close(fig)
print("saved mode_ablation.{csv,png}")''')

md("## (4) Extended rollout curve (full model)")
code('''STEPS = min(30, TEST.shape[1] - 1)
per, w612, w1330 = rollout(model, TEST, steps=STEPS)
with open(OUT / "rollout_curve.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["step", "vrmse"])
    for i, v in enumerate(per, 1): w.writerow([i, v])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, STEPS + 1), per, "o-", ms=3)
ax.axvspan(6, 12, alpha=0.1, color="C1"); ax.axvspan(13, 30, alpha=0.1, color="C2")
ax.set_xlabel("rollout step"); ax.set_ylabel("VRMSE")
ax.set_title(f"Full-model rollout (6:12={w612:.3f}, 13:30={w1330:.3f})")
fig.tight_layout(); fig.savefig(OUT / "rollout_curve.png", dpi=150); plt.close(fig)
print("saved rollout_curve.{csv,png}")''')

md("""## (5) Comparative spectral drift — the "why LiteFNO beats the CNN" figure

Roll out **both** the CNN and LiteFNO and track, per step, the predicted
high-wavenumber energy relative to ground truth. If the CNN's high-freq energy
ratio drifts away from 1 (over-smooths or blows up) while LiteFNO stays near 1,
that mechanistically explains LiteFNO's rollout advantage — and connects to the
mode-ablation (LiteFNO leans on the rollout-critical low modes).""")
code('''if CNN is not None:
    STEPS = min(30, TEST.shape[1] - 1)
    pr_lf, gt = rollout_full(model, TEST, STEPS)
    pr_cnn, _ = rollout_full(CNN, TEST, STEPS)

    def per_vrmse(pr): return [vrmse(pr[:, k], gt[:, k]).item() for k in range(STEPS)]
    def per_hf(pr):
        out = []
        for k in range(STEPS):
            ps_p = radial_psd(pr[:, k, ..., 0].numpy()); ps_t = radial_psd(gt[:, k, ..., 0].numpy())
            kk = np.arange(1, len(ps_t)); hi = kk[len(kk)//2:]
            out.append(float(ps_p[hi].sum() / (ps_t[hi].sum() + 1e-12)))
        return out

    v_lf, v_cnn = per_vrmse(pr_lf), per_vrmse(pr_cnn)
    hf_lf, hf_cnn = per_hf(pr_lf), per_hf(pr_cnn)

    with open(OUT / "spectral_drift.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "vrmse_cnn", "vrmse_litefno", "hf_ratio_cnn", "hf_ratio_litefno"])
        for k in range(STEPS): w.writerow([k + 1, v_cnn[k], v_lf[k], hf_cnn[k], hf_lf[k]])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    s = range(1, STEPS + 1)
    ax[0].plot(s, v_cnn, "o-", ms=3, label="CNN"); ax[0].plot(s, v_lf, "s-", ms=3, label="LiteFNO")
    ax[0].set_xlabel("rollout step"); ax[0].set_ylabel("VRMSE"); ax[0].set_title("Rollout error"); ax[0].legend()
    ax[1].axhline(1.0, ls="--", color="k", lw=0.8, label="ground truth")
    ax[1].plot(s, hf_cnn, "o-", ms=3, label="CNN"); ax[1].plot(s, hf_lf, "s-", ms=3, label="LiteFNO")
    ax[1].set_xlabel("rollout step"); ax[1].set_ylabel("high-freq energy / truth")
    ax[1].set_title("Spectral drift under rollout"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "spectral_drift.png", dpi=150); plt.close(fig)
    print("saved spectral_drift.{csv,png}")
    print(f"final-step high-freq ratio  CNN={hf_cnn[-1]:.3f}  LiteFNO={hf_lf[-1]:.3f}  (1.0 = matches truth)")
else:
    print("CNN not mounted - skipped comparative spectral drift.")''')

md("## GO / NO-GO verdict")
code('''print("THESIS CHECKLIST (eyeball these):")
print(f"  1. Are high modes dead?            dim0 {dead0:.0%} / dim1 {dead1:.0%} below 10% peak  -> want YES (large %)")
print(f"  2. Is CP rank under-utilized?      {eff90}/{R} components for 90% energy  -> want eff << R")
print( "  3. Mode ablation (mode_ablation.csv): does keeping only low modes (e.g. 0.5)")
print( "     barely change one-step VRMSE, while ROLLOUT degrades faster? -> the key signal")
print()
print("If 1-3 hold, the thesis is supported -> proceed to multi-seed/multi-dataset rigor.")
print("If high modes are NOT dead and ablation hurts uniformly -> pivot before spending GPU.")
print("\\nArtifacts in", OUT.resolve())
for p in sorted(OUT.iterdir()): print("  ", p.name)''')

# ---- emit ----
def mk(cells):
    out = []
    for kind, src in cells:
        lines = src.split("\n"); source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        out.append({"cell_type": "markdown", "metadata": {}, "source": source} if kind == "md"
                   else {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source})
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}

p = Path(__file__).resolve().parent.parent / "notebooks" / "mechinterp_day1.ipynb"
with p.open("w", encoding="utf-8") as fh: json.dump(mk(cells), fh, indent=1)
print("wrote", p)
