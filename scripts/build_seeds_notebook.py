"""Builds notebooks/headline_3seed.ipynb -- the "minimum flip": makes the
headline finding (rollout stability + opposite spectral failure modes) rigorous
by retraining CNN and spectral LiteFNO on the matched gs-processed data across
3 seeds, then reporting rollout VRMSE and high-frequency spectral drift with
mean +/- std error bands.

Run: python scripts/build_seeds_notebook.py
"""
from __future__ import annotations
import json
from pathlib import Path

cells = []
def md(s): cells.append(("md", s))
def code(s): cells.append(("code", s))

BUILD = r'''
def build_real_litefno(in_ch, out_ch, modes, width=64, layers=8, rank=0.02, factorization="cp"):
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

md(r"""# Headline finding, 3 seeds (the "minimum flip")

Makes the paper's headline rigorous: retrain a parameter-matched **CNN** and a
**spectral LiteFNO** on the matched Gray-Scott data across **3 seeds**, then report
with **mean ± std**:
- autoregressive **rollout VRMSE** (CNN vs LiteFNO)
- **spectral drift** (predicted high-freq energy / truth) over rollout — the
  "opposite failure modes" finding (CNN over-smooths, LiteFNO over-sharpens)
- one-step VRMSE + windowed rollout VRMSE (6:12, 13:30)

This removes the single-seed objection: the figures get error bands.

**Setup:** Add Input -> `gs-processed` (train/valid/test.h5). Accelerator = **GPU T4**
(NOT P100). Internet ON. Run as **Save & Run All (Commit)** -- it trains 6 models
(~5-8 GPU-hr).""")

code('''import os, subprocess, sys
REPO = "litefno-repro"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/AIscend-Research/litefno-repro"], check=True)
sys.path.insert(0, os.path.join(os.path.abspath(REPO), "src"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuraloperator"], check=False)

import json, glob, time, csv
from pathlib import Path
import numpy as np, torch, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from litefno.train import build_model, flatten_time, set_seed
from litefno.data import DatasetConfig, H5SequenceDataset
from litefno.metrics import vrmse, window_vrmse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    try: _ = (torch.zeros(1, device=DEVICE) + 1).item()
    except Exception as e: print("CUDA unusable -> CPU (use T4):", e); DEVICE = torch.device("cpu")
print("device:", DEVICE)
OUT = Path("/kaggle/working/seeds") if Path("/kaggle/working").exists() else Path("seeds_out")
OUT.mkdir(parents=True, exist_ok=True)''')

md("## Config + data (auto-finds mounted gs-processed)")
code('''KEY = "data"; FIELDS = 2
SEEDS  = [0, 1, 2]
EPOCHS = 200; BATCH = 64; LR = 1e-3; LR_STEP = 100; LR_GAMMA = 0.5
MODES  = 16; WIDTH = 64; LAYERS = 8; RANK = 0.02
ROLL_STEPS = 30

def find(split):
    h = sorted(glob.glob(f"/kaggle/input/**/{split}.h5", recursive=True))
    return Path(h[0]) if h else None
TRAIN_H5, TEST_H5 = find("train"), find("test")
assert TRAIN_H5 and TEST_H5, "Mount gs-processed (Add Input)!"
print("train:", TRAIN_H5, "| test:", TEST_H5)

def make_loader(p, bs, shuffle):
    cfg = DatasetConfig(path=p, dataset_key=KEY, input_steps=1, output_steps=1, stride=1, cache="memory")
    return DataLoader(H5SequenceDataset(cfg), batch_size=bs, shuffle=shuffle)
train_loader = make_loader(TRAIN_H5, BATCH, True)
with h5py.File(TRAIN_H5, "r") as f: H, W = f[KEY].shape[2], f[KEY].shape[3]
with h5py.File(TEST_H5, "r") as f: TEST = f[KEY][...].astype(np.float32)
print(f"resolution {H}x{W}  test {TEST.shape}")''')

md("## Training (CNN and spectral LiteFNO, matched protocol, fp32)")
code(BUILD + '''

def to_xy(batch):
    xb, yb = batch
    return flatten_time(xb).to(DEVICE), flatten_time(yb).to(DEVICE)

def train_model(kind, seed):
    set_seed(seed)
    if kind == "cnn":
        m = build_model({"name": "litefno", "layers": LAYERS, "width": WIDTH, "rank": 32}, FIELDS, FIELDS)
    else:
        m, _ = build_real_litefno(FIELDS, FIELDS, MODES, WIDTH, LAYERS, RANK, "cp")
    m = m.to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, LR_STEP, LR_GAMMA)
    for ep in range(EPOCHS):
        m.train()
        for batch in train_loader:
            x, y = to_xy(batch)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(m(x), y)
            loss.backward(); opt.step()
        sch.step()
    return m.eval()''')

md("## Eval helpers")
code('''@torch.no_grad()
def step_predict(m, s): return m(s.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

@torch.no_grad()
def eval_onestep(m, data, bs=256):
    N, S = data.shape[0], data.shape[1]; vs = []
    for t in range(S - 1):
        x = torch.from_numpy(data[:, t]).float(); y = torch.from_numpy(data[:, t + 1]).float()
        for i in range(0, N, bs):
            vs.append(vrmse(step_predict(m, x[i:i+bs].to(DEVICE)), y[i:i+bs].to(DEVICE)).item())
    return float(np.mean(vs))

@torch.no_grad()
def rollout_full(m, data, steps, bs=256):
    N, S = data.shape[0], data.shape[1]; steps = min(steps, S - 1)
    preds = torch.empty((N, steps) + tuple(data.shape[2:]), dtype=torch.float32)
    init = torch.from_numpy(data[:, 0]).float()
    for i in range(0, N, bs):
        s = init[i:i+bs].to(DEVICE)
        for kk in range(steps):
            s = step_predict(m, s); preds[i:i+bs, kk] = s.cpu()
    return preds, torch.from_numpy(data[:, 1:steps + 1]).float()

def radial_psd(field):
    F = np.fft.fftshift(np.fft.fft2(field, axes=(1, 2)), axes=(1, 2))
    P = (np.abs(F) ** 2).mean(0); h, w = P.shape
    yy, xx = np.indices((h, w)); r = np.sqrt((yy - h/2) ** 2 + (xx - w/2) ** 2).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)

def per_step_metrics(preds, gt, steps):
    vr, hf = [], []
    for k in range(steps):
        vr.append(vrmse(preds[:, k], gt[:, k]).item())
        ps_p = radial_psd(preds[:, k, ..., 0].numpy()); ps_t = radial_psd(gt[:, k, ..., 0].numpy())
        kk = np.arange(1, len(ps_t)); hi = kk[len(kk)//2:]
        hf.append(float(ps_p[hi].sum() / (ps_t[hi].sum() + 1e-12)))
    def win(a, b):
        a, b = min(a, steps), min(b, steps)
        return float("nan") if b <= a else window_vrmse(preds, gt, a, b, time_dim=1).item()
    return np.array(vr), np.array(hf), win(6, 12), win(13, 30)''')

md("## Run all seeds")
code('''STEPS = min(ROLL_STEPS, TEST.shape[1] - 1)
res = {"cnn": {"vr": [], "hf": [], "one": [], "w612": [], "w1330": []},
       "litefno": {"vr": [], "hf": [], "one": [], "w612": [], "w1330": []}}
table = []
for seed in SEEDS:
    for kind in ["cnn", "litefno"]:
        t0 = time.time(); m = train_model(kind, seed); dt = time.time() - t0
        one = eval_onestep(m, TEST)
        preds, gt = rollout_full(m, TEST, STEPS)
        vr, hf, w612, w1330 = per_step_metrics(preds, gt, STEPS)
        res[kind]["vr"].append(vr); res[kind]["hf"].append(hf)
        res[kind]["one"].append(one); res[kind]["w612"].append(w612); res[kind]["w1330"].append(w1330)
        torch.save({"model_state": m.state_dict(), "kind": kind, "seed": seed}, OUT / f"{kind}_seed{seed}.pt")
        table.append({"model": kind, "seed": seed, "onestep": one, "roll_6_12": w612, "roll_13_30": w1330,
                      "hf_final": float(hf[-1]), "train_s": round(dt)})
        print(f"{kind:>8} seed{seed}: one-step={one:.4f}  6:12={w612:.4f}  13:30={w1330:.4f}  hf_final={hf[-1]:.3f}  ({dt:.0f}s)")
with open(OUT / "seed_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
print("saved seed_table.csv")''')

md("## Summary (mean ± std over 3 seeds)")
code('''def ms(xs): return float(np.mean(xs)), float(np.std(xs))
print(f"{'metric':<14}{'CNN (mean±std)':<26}{'LiteFNO (mean±std)':<26}")
for key, name in [("one", "one-step VRMSE"), ("w612", "rollout 6:12"), ("w1330", "rollout 13:30")]:
    cm, cs = ms(res["cnn"][key]); lm, ls = ms(res["litefno"][key])
    print(f"{name:<14}{cm:.4f} ± {cs:.4f}        {lm:.4f} ± {ls:.4f}")
chm, chs = ms([h[-1] for h in res["cnn"]["hf"]]); lhm, lhs = ms([h[-1] for h in res["litefno"]["hf"]])
print(f"{'hf-ratio@end':<14}{chm:.3f} ± {chs:.3f}          {lhm:.3f} ± {lhs:.3f}   (1.0 = matches truth)")''')

md("## Figure: rollout VRMSE + spectral drift, with error bands (3 seeds)")
code('''s = np.arange(1, STEPS + 1)
def band(ax, arrs, label):
    A = np.stack(arrs); mu, sd = A.mean(0), A.std(0)
    ax.plot(s, mu, "-", label=label); ax.fill_between(s, mu - sd, mu + sd, alpha=0.2)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
band(ax[0], res["cnn"]["vr"], "CNN"); band(ax[0], res["litefno"]["vr"], "LiteFNO")
ax[0].set_xlabel("rollout step"); ax[0].set_ylabel("VRMSE")
ax[0].set_title("Rollout error (mean ± std, 3 seeds)"); ax[0].legend()
ax[1].axhline(1.0, ls="--", color="k", lw=0.8, label="ground truth")
band(ax[1], res["cnn"]["hf"], "CNN"); band(ax[1], res["litefno"]["hf"], "LiteFNO")
ax[1].set_xlabel("rollout step"); ax[1].set_ylabel("high-freq energy / truth")
ax[1].set_title("Spectral drift (mean ± std, 3 seeds)"); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT / "headline_3seed.png", dpi=150); plt.close(fig)

# also dump the per-step mean/std for the paper
with open(OUT / "headline_curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "cnn_vr_mean", "cnn_vr_std", "lf_vr_mean", "lf_vr_std",
                "cnn_hf_mean", "cnn_hf_std", "lf_hf_mean", "lf_hf_std"])
    cvr, lvr = np.stack(res["cnn"]["vr"]), np.stack(res["litefno"]["vr"])
    chf, lhf = np.stack(res["cnn"]["hf"]), np.stack(res["litefno"]["hf"])
    for i in range(STEPS):
        w.writerow([i + 1, cvr[:, i].mean(), cvr[:, i].std(), lvr[:, i].mean(), lvr[:, i].std(),
                    chf[:, i].mean(), chf[:, i].std(), lhf[:, i].mean(), lhf[:, i].std()])
print("saved headline_3seed.png, headline_curves.csv")''')

md("## Verdict")
code('''cm, cs = ms(res["cnn"]["w1330"]); lm, ls = ms(res["litefno"]["w1330"])
sep = (cm - lm) / (np.sqrt(cs**2 + ls**2) + 1e-9)
print(f"rollout 13:30 -- CNN {cm:.3f}±{cs:.3f}  vs  LiteFNO {lm:.3f}±{ls:.3f}")
print(f"separation (gap / pooled std) = {sep:.1f}")
print("If LiteFNO rollout < CNN beyond the error bands (sep >~ 2), the finding is robust to seeds.")
print("If the spectral-drift directions are consistent (CNN<1, LiteFNO>1) across seeds -> headline holds.")
print("\\nArtifacts in", OUT.resolve())
for p in sorted(OUT.iterdir()): print("  ", p.name)''')

def mk(cells):
    out = []
    for kind, src in cells:
        lines = src.split("\n"); source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        out.append({"cell_type": "markdown", "metadata": {}, "source": source} if kind == "md"
                   else {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source})
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}

p = Path(__file__).resolve().parent.parent / "notebooks" / "headline_3seed.ipynb"
with p.open("w", encoding="utf-8") as fh: json.dump(mk(cells), fh, indent=1)
print("wrote", p)
