r"""Accuracy under increasing input degradation: does training for it help? (ext27)

Board task: "Report accuracy at increasing degradation severity (0% artifacts ->
100% synthetic smartphone degradation): show robustness-trained closes the gap
vs. baseline."

The task is phrased for image capture -- a clinical photograph re-shot on a
phone picks up optical blur, sensor noise, an illumination shift, and encoding
artifacts. This repo's testbed is a Gray-Scott field on a periodic 32x32 grid,
so "smartphone degradation" is reproduced as a synthetic analog of that capture
chain applied to the model's *input* state, with one severity knob s in [0, 1]:
s = 0 is the clean field (0% artifacts) and s = 1 is the full corruption.

This is a synthetic analog, not a photograph of anything. The claim it supports
is about a model's response to a parameterised input corruption, not about
phones.

The chain, in physical capture order
------------------------------------
  illumination   per-sample gain and offset       (exposure / white balance)
  optics         periodic Gaussian blur           (lens defocus)
  sensor         additive Gaussian noise          (read noise)
  encoding       uniform quantisation             (compression)

Every component is exactly the identity at s = 0, so the s = 0 column is the
clean number rather than a nearly-clean one. That is asserted in the tests, not
assumed.

The blur is done in Fourier space because the domain really is periodic -- ext24
established that on this grid the Fourier modes are the lattice Laplacian's
eigenvectors -- so a wraparound blur is the physically right one and is exact at
sigma = 0 rather than approximate.

Only the noise component has a committed input-side precedent: ext3 swept input
SNR with ``std * 10 ** (-snr / 20)``, and the noise here is the same
standard-deviation-relative construction. ext2's sweep is *weight* precision,
not input encoding, so the quantisation step here is a new axis rather than a
reuse of that number.

The two arms
------------
  baseline    trained on clean inputs
  robust      trained on the same chain at a severity drawn uniformly per batch

`robust` therefore sees the corruption family it is later tested on. That is
what "robustness-trained" means and it is the intended comparison, but on its
own it cannot distinguish a model that got genuinely steadier from one that
memorised the augmentation. So the sweep is run twice: once on the matched
chain, and once on a corruption the training never showed -- pixel dropout,
which is structurally unlike blur, noise, gain or quantisation. An arm that
closes the gap in-family and not out-of-family has learned the augmentation, not
robustness.

The number
----------
At each severity the degradation-induced excess is the rise over that arm's own
clean error, ``v(s) - v(0)``, and the fraction closed is

    closed(s) = (excess_baseline(s) - excess_robust(s)) / excess_baseline(s)

Measuring each arm against its own clean error is deliberate: augmentation
usually costs something on clean inputs, and charging the robust arm for that
cost twice -- once as a worse clean number, once as a smaller closed fraction --
would double-count it. The clean errors are reported side by side so the tax is
visible on its own.
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
# the degradation chain
# --------------------------------------------------------------------------
#
# Amplitudes at s = 1. These set what "100% degradation" means and are the
# knobs a reader would challenge, so they are named constants rather than
# literals buried in the functions.

GAIN_MAX = 0.30        # +/- 30% multiplicative gain
BIAS_MAX = 0.20        # +/- 0.20 sigma additive offset
BLUR_MAX = 1.20        # Gaussian blur sigma, in pixels
NOISE_MAX = 0.30       # noise std as a fraction of the field's std (~10.5 dB)
QUANT_MAX = 0.40       # quantiser step as a fraction of the field's std
DROP_MAX = 0.30        # held-out corruption: fraction of pixels zeroed


def _periodic_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur with wraparound, applied in Fourier space.

    Exact at sigma = 0 (the transfer function is identically 1), which is what
    lets s = 0 be the clean field rather than an almost-clean one.
    """
    if sigma <= 0:
        return x
    h, w = x.shape[-2], x.shape[-1]
    fy = torch.fft.fftfreq(h, device=x.device).view(-1, 1)
    fx = torch.fft.fftfreq(w, device=x.device).view(1, -1)
    # Fourier transform of a Gaussian: exp(-2 pi^2 sigma^2 |f|^2)
    transfer = torch.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fy ** 2 + fx ** 2))
    return torch.fft.ifft2(torch.fft.fft2(x) * transfer).real


def degrade(x: torch.Tensor, s: float, gen: torch.Generator) -> torch.Tensor:
    """The synthetic smartphone chain at severity ``s`` in [0, 1].

    Returns ``x`` unchanged at s = 0. Scales are relative to the field's own
    standard deviation, so the chain means the same thing on regimes whose
    amplitudes differ by orders of magnitude.
    """
    if s <= 0:
        return x
    std = float(x.std())

    # illumination: one gain and offset per sample, not per pixel
    n = x.shape[0]
    shape = (n,) + (1,) * (x.dim() - 1)
    gain = 1.0 + s * GAIN_MAX * (
        torch.rand(shape, generator=gen, device=x.device) * 2 - 1)
    bias = s * BIAS_MAX * std * (
        torch.rand(shape, generator=gen, device=x.device) * 2 - 1)
    out = x * gain + bias

    # optics
    out = _periodic_blur(out, s * BLUR_MAX)

    # sensor
    out = out + torch.randn(out.shape, generator=gen, device=out.device) * (
        s * NOISE_MAX * std)

    # encoding
    step = s * QUANT_MAX * std
    if step > 0:
        out = torch.round(out / step) * step
    return out


def drop_pixels(x: torch.Tensor, s: float, gen: torch.Generator) -> torch.Tensor:
    """Held-out corruption: zero a fraction of pixels. Identity at s = 0.

    Structurally unlike every component of the training chain -- it is neither
    smooth, nor additive, nor a monotone map of the value -- so an arm that only
    learned the training chain has no reason to be steadier under it.
    """
    if s <= 0:
        return x
    keep = torch.rand(x.shape, generator=gen, device=x.device) >= (s * DROP_MAX)
    return x * keep


CORRUPTIONS = {"smartphone": degrade, "dropout": drop_pixels}


# --------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------


def train_arm(xtr, ytr, device: str, epochs: int, seed: int, augment: bool,
              log_every: int = 10):
    """One LiteFNO. ``augment`` draws a fresh severity per batch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 10_000)

    model, factorization = br.build_model("litefno", xtr.shape[1], xtr.shape[1])
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=br.LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=br.LR_STEP,
                                            gamma=br.LR_GAMMA)
    loss_fn = nn.MSELoss()

    n, t0 = len(xtr), time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, br.BATCH):
            idx = perm[i:i + br.BATCH]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            if augment:
                # severity uniform on [0, 1]: the model must handle the whole
                # range, not one operating point
                s = float(torch.rand(1, generator=gen, device=device).item())
                xb = degrade(xb, s, gen)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)
        sched.step()
        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            print(f"      epoch {epoch + 1:>4d}/{epochs}  train_mse="
                  f"{total / n:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    return model, factorization, round(time.time() - t0, 1)


def evaluate_at(model, x, y, s: float, kind: str, device: str, seed: int,
                batch: int = 128) -> float:
    """VRMSE with the input degraded at severity ``s``; target stays clean.

    The generator is reseeded per call so every arm meets the identical
    corruption draw at a given severity. Without that, an arm could look better
    purely from a luckier noise sample.
    """
    fn = CORRUPTIONS[kind]
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            gen = torch.Generator(device=device).manual_seed(seed + i)
            xb = fn(x[i:i + batch].to(device), s, gen)
            preds.append(model(xb).cpu())
    return float(br.vrmse(torch.cat(preds), y))


def gap_closed(rows: list[dict]) -> list[dict]:
    """Fraction of the degradation-induced error rise that `robust` removes.

    Each arm's excess is measured against its own clean error, so the clean-side
    cost of augmentation is not charged twice. At s = 0 the excess is zero for
    both arms and the fraction is undefined, reported as NaN rather than 0.
    """
    by = {}
    for r in rows:
        by.setdefault((r["corruption"], r["severity"]), {})[r["arm"]] = r["vrmse"]
    # the s = 0 row is the reference every excess is measured against; without
    # it there is no defined baseline to subtract, so that corruption is skipped
    # rather than silently rebased on its mildest available severity
    clean = {c: by[(c, 0.0)] for c, s in by if s == 0.0}

    out = []
    for (corr, s), arms in sorted(by.items()):
        if "baseline" not in arms or "robust" not in arms or corr not in clean:
            continue
        eb = arms["baseline"] - clean[corr]["baseline"]
        er = arms["robust"] - clean[corr]["robust"]
        out.append({
            "corruption": corr, "severity": s,
            "baseline_vrmse": arms["baseline"], "robust_vrmse": arms["robust"],
            "baseline_excess": eb, "robust_excess": er,
            "frac_gap_closed": (eb - er) / eb if abs(eb) > 1e-12 else float("nan"),
            "robust_better_absolute": bool(arms["robust"] < arms["baseline"]),
        })
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_report(rows: list[dict], closed: list[dict]) -> None:
    for corr in CORRUPTIONS:
        sub = [c for c in closed if c["corruption"] == corr]
        if not sub:
            continue
        label = ("matched: the chain robust trained on" if corr == "smartphone"
                 else "held out: never seen in training")
        print(f"\n=== {corr} ({label}) ===")
        hdr = (f"    {'severity':>9s} {'baseline':>10s} {'robust':>10s} "
               f"{'base rise':>10s} {'rob rise':>10s} {'closed':>8s} {'rob<base':>9s}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for c in sub:
            frac = ("     n/a" if not np.isfinite(c["frac_gap_closed"])
                    else f"{c['frac_gap_closed']:>7.0%}")
            print(f"    {c['severity']:>8.0%} {c['baseline_vrmse']:>10.5f} "
                  f"{c['robust_vrmse']:>10.5f} {c['baseline_excess']:>+10.5f} "
                  f"{c['robust_excess']:>+10.5f} {frac} "
                  f"{'yes' if c['robust_better_absolute'] else 'no':>9s}")
        fr = np.array([c["frac_gap_closed"] for c in sub], dtype=float)
        fr = fr[np.isfinite(fr)]
        if len(fr):
            print(f"    median {np.median(fr):.0%} of the degradation-induced "
                  f"rise closed (range {fr.min():.0%} to {fr.max():.0%})")

    base0 = next((c["baseline_vrmse"] for c in closed
                  if c["corruption"] == "smartphone" and c["severity"] == 0.0), None)
    rob0 = next((c["robust_vrmse"] for c in closed
                 if c["corruption"] == "smartphone" and c["severity"] == 0.0), None)
    if base0 is not None:
        tax = (rob0 - base0) / base0 if base0 else float("nan")
        print(f"\n=== The clean-input cost of augmenting ===")
        print(f"    at 0% artifacts: baseline {base0:.5f}, robust {rob0:.5f} "
              f"({tax:+.1%})")
        print("    a positive number is the robustness tax: what the robust arm "
              "gives up\n    on undegraded inputs to buy its steadiness later")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def plot(closed: list[dict], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping plot: {exc}")
        return
    corrs = [c for c in CORRUPTIONS if any(x["corruption"] == c for x in closed)]
    fig, axes = plt.subplots(1, len(corrs), figsize=(6 * len(corrs), 4.2),
                             squeeze=False)
    for ax, corr in zip(axes[0], corrs):
        sub = sorted([c for c in closed if c["corruption"] == corr],
                     key=lambda c: c["severity"])
        sev = [c["severity"] * 100 for c in sub]
        ax.plot(sev, [c["baseline_vrmse"] for c in sub], "o-", label="baseline")
        ax.plot(sev, [c["robust_vrmse"] for c in sub], "s-", label="robust")
        ax.set_xlabel("degradation severity (%)")
        ax.set_ylabel("one-step VRMSE")
        ax.set_title(f"{corr}"
                     + (" (matched)" if corr == "smartphone" else " (held out)"))
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("data/processed/gray_scott_streamed"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--severities", type=float, nargs="*",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/extensions"))
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    xtr, ytr = br.to_pairs(br.load_split(args.data_dir / "train.h5"))
    xte, yte = br.to_pairs(br.load_split(args.data_dir / "test.h5"))
    print(f"device={device}  train pairs {len(xtr)}  test pairs {len(xte)}")
    print(f"  {args.epochs} epochs/arm, seed {args.seed}, "
          f"severities {args.severities}")

    rows, models = [], {}
    for arm, augment in (("baseline", False), ("robust", True)):
        print(f"\n  [{arm}]  augment={augment}", flush=True)
        model, fac, secs = train_arm(xtr, ytr, device, args.epochs, args.seed,
                                     augment)
        models[arm] = model
        for corr in CORRUPTIONS:
            for s in args.severities:
                v = evaluate_at(model, xte, yte, s, corr, device, args.seed)
                rows.append({"arm": arm, "corruption": corr, "severity": s,
                             "vrmse": v, "factorization": fac,
                             "epochs": args.epochs, "seed": args.seed,
                             "train_s": secs})
                print(f"      {corr:>10s} s={s:>4.0%}  vrmse={v:.5f}", flush=True)

    closed = gap_closed(rows)
    print_report(rows, closed)
    write_csv(args.out_dir / "ext27_sweep.csv", rows)
    write_csv(args.out_dir / "ext27_gap_closed.csv", closed)
    plot(closed, args.fig_dir / "ext27_degradation.png")


if __name__ == "__main__":
    sys.exit(main())
