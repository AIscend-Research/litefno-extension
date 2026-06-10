"""Builds notebooks/mechinterp_3seed.ipynb — mechanistic-interpretability of the
spectral LiteFNO across 3 seeds, with error bands:
  (1) dead-mode analysis (per-Fourier-mode weight magnitude, mean +/- std),
  (2) CP-rank utilization (effective rank, mean +/- std),
  (3) causal mode ablation -> one-step VRMSE + rollout windows (mean +/- std).

Loads litefno_seed{0,1,2}.pt (from the headline 3-seed run); falls back to a
single litefno_real_best.pt if the seed checkpoints aren't mounted.

Run: python scripts/build_mechinterp_notebook.py
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

md(r"""# Mechanistic interpretability of spectral LiteFNO (3 seeds)

Runs the mech-interp analyses across **all 3 LiteFNO seeds** with **error bands**,
so the findings are seed-robust:
1. **Dead-mode analysis** — per-Fourier-mode weight magnitude (mean ± std).
2. **CP-rank utilization** — effective rank of the factorization (mean ± std).
3. **Causal mode ablation** — keep only the lowest-f fraction of modes; one-step
   VRMSE + rollout windows (mean ± std).

**Setup — attach two datasets via Add Input:**
- your **LiteFNO seed checkpoints** (`litefno_seed0.pt`, `litefno_seed1.pt`, `litefno_seed2.pt`)
- **gs-processed** (for `test.h5`)

GPU optional (eval is light). Internet ON for the clone.""")

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    try: _ = (torch.zeros(1, device=DEVICE) + 1).item()
    except Exception as e: print("CUDA unusable -> CPU:", e); DEVICE = torch.device("cpu")
print("device:", DEVICE)
OUT = Path("/kaggle/working/mechinterp") if Path("/kaggle/working").exists() else Path("mechinterp_out")
OUT.mkdir(parents=True, exist_ok=True)
KEY = "data"; FIELDS = 2''')

md("## Load all LiteFNO seed checkpoints + test data")
code(BUILD + '''

# default config for seed checkpoints (which carry no "build" dict)
DEF = dict(in_ch=FIELDS, out_ch=FIELDS, modes=16, width=64, layers=8, rank=0.02, factorization="cp")

def load_litefno(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    b = ck.get("build", DEF)
    m, kind = build_real_litefno(b["in_ch"], b["out_ch"], b["modes"], b["width"], b["layers"], b["rank"], b["factorization"])
    m.load_state_dict(ck["model_state"]); m.to(DEVICE).eval()
    return m

lf_paths = (sorted(glob.glob("/kaggle/input/**/litefno_seed*.pt", recursive=True))
            or sorted(glob.glob("/kaggle/input/**/litefno_real_best.pt", recursive=True)))
tests = sorted(glob.glob("/kaggle/input/**/test.h5", recursive=True))
assert lf_paths, "Mount the LiteFNO seed checkpoints (litefno_seed*.pt)!"
assert tests, "Mount gs-processed (test.h5)!"
print("LiteFNO checkpoints (%d):" % len(lf_paths)); [print("  ", p) for p in lf_paths]

MODELS = [load_litefno(p) for p in lf_paths]
N_SEED = len(MODELS)
PARAMS = sum(p.numel() for p in MODELS[0].parameters())
print(f"loaded {N_SEED} seed(s); params={PARAMS:,}")

TEST_H5 = Path(tests[0])
with h5py.File(TEST_H5, "r") as f: TEST = f[KEY][...].astype(np.float32)
H, W = TEST.shape[2], TEST.shape[3]
print("test:", TEST.shape)''')

md("## Eval helpers")
code('''@torch.no_grad()
def step_predict(m, s): return m(s.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

@torch.no_grad()
def eval_vrmse(m, data, bs=256):
    N, S = data.shape[0], data.shape[1]; vs = []
    for t in range(S - 1):
        x = torch.from_numpy(data[:, t]).float(); y = torch.from_numpy(data[:, t + 1]).float()
        for i in range(0, N, bs):
            vs.append(vrmse(step_predict(m, x[i:i+bs].to(DEVICE)), y[i:i+bs].to(DEVICE)).item())
    return float(np.mean(vs))

@torch.no_grad()
def rollout(m, data, steps, bs=256):
    N, S = data.shape[0], data.shape[1]; steps = min(steps, S - 1)
    preds = torch.empty((N, steps) + tuple(data.shape[2:]), dtype=torch.float32)
    init = torch.from_numpy(data[:, 0]).float()
    for i in range(0, N, bs):
        s = init[i:i+bs].to(DEVICE)
        for kk in range(steps):
            s = step_predict(m, s); preds[i:i+bs, kk] = s.cpu()
    gt = torch.from_numpy(data[:, 1:steps + 1]).float()
    def win(a, b):
        a, b = min(a, steps), min(b, steps)
        return float("nan") if b <= a else window_vrmse(preds, gt, a, b, time_dim=1).item()
    return win(6, 12), win(13, 30)

def layer_factors(m):
    d = {}
    for n, p in m.named_parameters():
        if ".weight.factors.factor_" in n:
            L = int(n.split(".")[2]); f = n.split(".")[-1]
            d.setdefault(L, {})[f] = p.detach().cpu()
    return [d[k] for k in sorted(d)]

def band(ax, A, label, x=None):       # A: (seeds, n) -> mean line + std band
    A = np.asarray(A); mu, sd = A.mean(0), A.std(0)
    xs = np.arange(len(mu)) if x is None else x
    ax.plot(xs, mu, "o-", ms=3, label=label); ax.fill_between(xs, mu - sd, mu + sd, alpha=0.2)''')

md("## (1) Dead-mode analysis (mean ± std over seeds)")
code('''def deadmode(m):
    LF = layer_factors(m)
    m0 = np.mean([LF[l]["factor_2"].abs().norm(dim=1).numpy() for l in range(len(LF))], axis=0)
    m1 = np.mean([LF[l]["factor_3"].abs().norm(dim=1).numpy() for l in range(len(LF))], axis=0)
    return m0 / m0.max(), m1 / m1.max()

D0 = np.stack([deadmode(m)[0] for m in MODELS])   # (seeds, n_modes0)
D1 = np.stack([deadmode(m)[1] for m in MODELS])
dead0 = float((D0.mean(0) < 0.1).mean()); dead1 = float((D1.mean(0) < 0.1).mean())
print(f"fraction of modes <10% of peak (mean over seeds):  dim0={dead0:.0%}  dim1={dead1:.0%}")

with open(OUT / "deadmode.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["axis", "mode_idx", "norm_mag_mean", "norm_mag_std"])
    for i in range(D0.shape[1]): w.writerow(["dim0", i, D0[:, i].mean(), D0[:, i].std()])
    for i in range(D1.shape[1]): w.writerow(["dim1", i, D1[:, i].mean(), D1[:, i].std()])

fig, ax = plt.subplots(figsize=(6, 4))
band(ax, D0, "mode dim0"); band(ax, D1, "mode dim1", x=np.arange(D1.shape[1]))
ax.axhline(0.1, ls="--", color="gray", lw=0.8, label="10% threshold")
ax.set_xlabel("Fourier mode index"); ax.set_ylabel("normalized weight magnitude")
ax.set_title(f"Per-mode spectral magnitude (mean ± std, {N_SEED} seeds)"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "deadmode.png", dpi=150); plt.close(fig)
print("saved deadmode.{csv,png}")''')

md("## (2) CP-rank utilization (mean ± std over seeds)")
code('''def eff_rank(m):
    LF = layer_factors(m)
    def comp(fac): return (fac["factor_0"].abs().norm(dim=0) * fac["factor_1"].abs().norm(dim=0) *
                           fac["factor_2"].abs().norm(dim=0) * fac["factor_3"].abs().norm(dim=0)).numpy()
    c = np.mean([comp(LF[l]) for l in range(len(LF))], axis=0)
    c = np.sort(c)[::-1]; cum = np.cumsum(c) / c.sum()
    return cum, int((cum < 0.9).sum()) + 1, len(c)

cums, e90s, R = [], [], None
for m in MODELS:
    cum, e90, R = eff_rank(m); cums.append(cum); e90s.append(e90)
print(f"CP rank R={R}; components for 90% energy: {np.mean(e90s):.1f} ± {np.std(e90s):.1f}  ({np.mean(e90s)/R:.0%} of rank)")
fig, ax = plt.subplots(figsize=(6, 4))
band(ax, np.stack(cums), "cumulative energy", x=np.arange(1, R + 1))
ax.axhline(0.9, ls="--", color="gray", lw=0.8)
ax.set_xlabel("# CP rank components (sorted)"); ax.set_ylabel("cumulative energy")
ax.set_title(f"CP-rank utilization (eff {np.mean(e90s):.0f}/{R}, {N_SEED} seeds)")
fig.tight_layout(); fig.savefig(OUT / "cprank.png", dpi=150); plt.close(fig)
print("saved cprank.png")''')

md("## (3) Causal mode ablation (mean ± std over seeds)")
code('''def ablate_keep(m, frac):
    mq = copy.deepcopy(m)
    for n, p in mq.named_parameters():
        if "factor_2" in n or "factor_3" in n:
            k = max(1, int(round(p.shape[0] * frac))); p.data[k:] = 0
    return mq.eval()

FRACS = [1.0, 0.75, 0.5, 0.25, 0.1]
one = np.zeros((N_SEED, len(FRACS))); r612 = np.zeros_like(one); r1330 = np.zeros_like(one)
STEPS = min(30, TEST.shape[1] - 1)
for si, m in enumerate(MODELS):
    for fi, fr in enumerate(FRACS):
        mq = ablate_keep(m, fr)
        one[si, fi] = eval_vrmse(mq, TEST)
        r612[si, fi], r1330[si, fi] = rollout(mq, TEST, STEPS)
    print(f"seed {si} done")

with open(OUT / "mode_ablation.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["keep_frac", "onestep_mean", "onestep_std",
                                   "roll6_12_mean", "roll6_12_std", "roll13_30_mean", "roll13_30_std"])
    for fi, fr in enumerate(FRACS):
        w.writerow([fr, one[:, fi].mean(), one[:, fi].std(), r612[:, fi].mean(), r612[:, fi].std(),
                    r1330[:, fi].mean(), r1330[:, fi].std()])
for fi, fr in enumerate(FRACS):
    print(f"keep {fr:>4}: one-step={one[:,fi].mean():.4f}±{one[:,fi].std():.4f}  "
          f"roll6:12={r612[:,fi].mean():.3f}±{r612[:,fi].std():.3f}  "
          f"roll13:30={r1330[:,fi].mean():.3f}±{r1330[:,fi].std():.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4))
band(ax, one, "one-step VRMSE", x=FRACS); band(ax, r612, "rollout 6:12", x=FRACS)
band(ax, r1330, "rollout 13:30", x=FRACS)
ax.set_xlabel("fraction of (lowest) modes kept"); ax.set_ylabel("VRMSE")
ax.set_title(f"Causal mode ablation (mean ± std, {N_SEED} seeds)"); ax.legend(); ax.invert_xaxis()
fig.tight_layout(); fig.savefig(OUT / "mode_ablation.png", dpi=150); plt.close(fig)
print("saved mode_ablation.{csv,png}")''')

md("## Verdict")
code('''print(f"Seeds analyzed: {N_SEED}")
print(f"1. Dead modes? dim0 {dead0:.0%} / dim1 {dead1:.0%} below 10% of peak  (want large % for 'dead')")
print(f"2. CP eff-rank: {np.mean(e90s):.0f}/{R}  (want << R for under-utilization)")
print( "3. Ablation: does keeping only low modes preserve one-step while rollout degrades? (see mode_ablation.csv)")
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

p = Path(__file__).resolve().parent.parent / "notebooks" / "mechinterp_3seed.ipynb"
with p.open("w", encoding="utf-8") as fh: json.dump(mk(cells), fh, indent=1)
print("wrote", p)
