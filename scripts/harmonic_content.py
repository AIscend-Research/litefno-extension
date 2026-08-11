r"""Harmonic-content decomposition of the ground-truth time series, per scenario.

Board task: "run FFT on time series, measure how much variance is in
low-frequency modes vs. high-frequency (noise, fast fluctuations) -- this tells
you how much harmonic structure exists to exploit".

This is the *temporal* companion to ``spectral_variance_decomposition.py``
(ext9), which decomposed variance over *spatial* wavenumber for a single
Gray-Scott field. Here every scenario in a dataset family is decomposed, and the
spatial axis is redone per scenario so the two can be read side by side.

A "scenario" is one parameter setting of a dataset family, which is exactly how
The Well stores them -- one HDF5 file per setting:

  gray_scott_reaction_diffusion   6 scenarios (bubbles/gliders/maze/spirals/
                                  spots/worms), each a distinct (F, k) pair
  turbulent_radiative_layer_2D    8 scenarios, cooling time t_cool = 0.03..1.78

Files are read over HTTP range requests straight from HuggingFace (the full
Gray-Scott family is 140 GB), so nothing is written to disk.

Method
------
For a field u(t, y, x) the total variance over (t, y, x) splits exactly, by the
law of total variance, into a part carried by the time-averaged pattern and a
part carried by the fluctuation about it:

    Var(u) = Var_{y,x}( mean_t u )  +  mean_{y,x}( Var_t u )
             \_______ static _______/  \_______ dynamic _______/

Only the dynamic part has a temporal spectrum. We remove the per-pixel temporal
mean, take an rFFT along time, and accumulate |X_f|^2 over every pixel and
trajectory. Parseval normalisation (weight 2 on the interior bins, 1 on DC and
Nyquist, divide by T^2) makes the summed spectrum equal the dynamic variance --
asserted numerically, not assumed.

Bands are defined as fractions of the Nyquist frequency so they mean the same
thing across families with different T:

    LOW    f <= 0.10 f_Nyq     slow, coherent, few-harmonic structure
    MID    0.10 < f <= 0.50
    HIGH   f > 0.50 f_Nyq      fast fluctuation / noise floor

Three traps this script guards against
--------------------------------------
1. The spin-up transient. Over a full Gray-Scott trajectory the pattern forms
   once and then sits there. That one-time ramp is a step function in time, and
   a step has a 1/f spectrum, so "almost all variance is low-frequency" comes
   out true no matter what the settled dynamics look like. Every scenario is
   therefore decomposed over three time segments -- the full run, the 60-step
   window the repo actually trains on, and the settled second half -- and the
   settled segment is the one that answers the question.
2. Red noise masquerading as harmonic structure. A smoothly decaying spectrum
   is low-frequency dominated without containing a single exploitable harmonic.
   Each spectrum is therefore compared against an AR(1) null fitted to its own
   lag-1 autocorrelation (recovered from the spectrum by Wiener-Khinchin). What
   sits *above* that null is line-like structure; what tracks it is just
   correlated noise.
3. Spectral leakage, and mean-vs-total power per shell. Trajectories are not
   periodic in time, so a boxcar FFT smears low-frequency power upward and
   would understate the low-frequency share; every temporal metric is computed
   under both boxcar and Hann windows and both are reported. On the spatial
   side, averaging power over a radial shell under-weights high wavenumbers,
   which contain many more modes -- the spatial pass sums energy per shell (the
   same correction ext9 applies).

The spatial pass runs only for families with doubly-periodic boundaries
(Gray-Scott). Turbulent-radiative-layer is not periodic in y, so an unwindowed
2-D FFT there would manufacture high-wavenumber power at the boundary; it is
reported as temporal-only.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# scenario inventory
# --------------------------------------------------------------------------

GRAY_SCOTT = "polymathic-ai/gray_scott_reaction_diffusion"
TRL = "polymathic-ai/turbulent_radiative_layer_2D"


@dataclass
class Family:
    repo: str
    label: str
    fields: dict          # display name -> HDF5 path
    scenarios: dict       # scenario name -> file path in repo
    periodic: bool        # doubly periodic => spatial FFT is meaningful
    dt: float
    notes: str = ""


def gs_file(name: str, f: float, k: float) -> str:
    return f"data/test/gray_scott_reaction_diffusion_{name}_F_{f}_k_{k}.hdf5"


FAMILIES = {
    "gray_scott": Family(
        repo=GRAY_SCOTT,
        label="Gray-Scott reaction-diffusion",
        fields={"A": "t0_fields/A", "B": "t0_fields/B"},
        scenarios={
            "bubbles": gs_file("bubbles", 0.098, 0.057),
            "gliders": gs_file("gliders", 0.014, 0.054),
            "maze": gs_file("maze", 0.029, 0.057),
            "spirals": gs_file("spirals", 0.018, 0.051),
            "spots": gs_file("spots", 0.03, 0.062),
            "worms": gs_file("worms", 0.058, 0.065),
        },
        periodic=True,
        dt=10.0,
    ),
    "trl": Family(
        repo=TRL,
        label="Turbulent radiative layer 2D",
        fields={"density": "t0_fields/density", "pressure": "t0_fields/pressure"},
        scenarios={
            f"tcool_{v}": f"data/test/turbulent_radiative_layer_tcool_{v}.hdf5"
            for v in ["0.03", "0.06", "0.10", "0.18", "0.32", "0.56", "1.00", "1.78"]
        },
        periodic=False,
        dt=1.597033,
        notes=("not periodic in y, so temporal analysis only. T=101 is too short "
               "for a separate settled segment, but these runs are already in a "
               "statistically steady turbulent state, so 'full' is the settled "
               "case -- there is no formation transient to exclude."),
    ),
}

# --------------------------------------------------------------------------
# spectra
# --------------------------------------------------------------------------


def parseval_weights(n_time: int) -> np.ndarray:
    """Weights making sum(w * |rfft|^2) / T^2 equal the mean square of the series."""
    n_bins = n_time // 2 + 1
    w = np.full(n_bins, 2.0)
    w[0] = 1.0
    if n_time % 2 == 0:
        w[-1] = 1.0
    return w


def temporal_power(u: np.ndarray, window: str) -> np.ndarray:
    """Summed temporal power spectrum over all pixels of one trajectory.

    ``u`` is (T, ...). The per-pixel temporal mean is removed first, so bin 0 is
    empty by construction and the spectrum decomposes the dynamic variance only.
    Returns an array of length T//2+1 whose sum is the total (summed over
    pixels) time variance.
    """
    n_time = u.shape[0]
    x = np.asarray(u, dtype=np.float64).reshape(n_time, -1)
    x = x - x.mean(axis=0, keepdims=True)

    if window == "hann":
        w = np.hanning(n_time)
        # preserve variance: a Hann window removes power by its mean square
        w = w / np.sqrt((w ** 2).mean())
        x = x * w[:, None]
    elif window != "boxcar":
        raise ValueError(window)

    spec = np.fft.rfft(x, axis=0)
    power = parseval_weights(n_time)[:, None] * np.abs(spec) ** 2 / n_time ** 2
    return power.sum(axis=1)


def spatial_power(u: np.ndarray, stride: int = 1):
    """Radial-shell and FNO-box spatial energy, summed over the sampled frames.

    Returns ``(radial, box)`` where ``radial[k]`` is the energy in integer
    wavenumber shell k and ``box[m]`` is the energy at Chebyshev radius
    m = max(|kx|, |ky|). The box form is what an FNO's ``n_modes=(m, m)``
    truncation actually keeps, so its cumulative curve reads directly as
    "variance retained at this mode count".
    """
    frames = np.asarray(u[::stride], dtype=np.float64)
    n_frames, h, w = frames.shape
    frames = frames - frames.mean(axis=(1, 2), keepdims=True)  # drop the DC mode

    spec = np.fft.fft2(frames, axes=(1, 2))
    power = (np.abs(spec) ** 2).sum(axis=0) / (h * w) ** 2

    ky = np.fft.fftfreq(h, d=1.0 / h)
    kx = np.fft.fftfreq(w, d=1.0 / w)
    kyy, kxx = np.meshgrid(ky, kx, indexing="ij")

    r = np.sqrt(kyy ** 2 + kxx ** 2).astype(int)
    radial = np.bincount(r.ravel(), weights=power.ravel())

    cheb = np.maximum(np.abs(kyy), np.abs(kxx)).astype(int)
    box = np.bincount(cheb.ravel(), weights=power.ravel())

    # Parseval again: binned energy must equal the summed per-frame spatial
    # variance. This is what catches a mean-vs-total-per-shell mix-up.
    ref = frames.var(axis=(1, 2)).sum()
    if ref > 0:
        for binned in (radial, box):
            rel_err = abs(binned.sum() - ref) / ref
            assert rel_err < 1e-9, f"spatial Parseval failed: {rel_err:.2e}"
    return radial, box


# --------------------------------------------------------------------------
# band metrics
# --------------------------------------------------------------------------

LOW_EDGE = 0.10
HIGH_EDGE = 0.50

TRAIN_STEPS = 60  # configs/datasets/*.yaml: max_steps


def segments_for(n_time: int) -> dict:
    """Time windows each scenario is decomposed over.

    ``full``    the whole run, transient included -- low-frequency by
                construction, reported so the bias is visible rather than hidden
    ``train``   the first ``TRAIN_STEPS`` steps, the window the repo preprocesses
                down to and the only one the model is ever shown
    ``settled`` the last half of the run, after pattern formation has finished;
                this is the segment where a low-frequency share means something
    """
    seg = {"full": slice(None)}
    if n_time > TRAIN_STEPS:
        seg["train"] = slice(0, TRAIN_STEPS)
    if n_time >= 2 * TRAIN_STEPS:
        seg["settled"] = slice(n_time // 2, None)
    return seg


def ar1_null(power: np.ndarray, n_time: int) -> tuple[np.ndarray, float]:
    """AR(1) power spectrum matched to ``power``'s own lag-1 autocorrelation.

    ``power`` carries the Parseval weights from :func:`temporal_power`, so the
    weights are divided out to recover the raw one-sided |X_f|^2 before the
    two-sided PSD is rebuilt and inverted. The autocovariance is recovered from
    the spectrum by Wiener-Khinchin rather than re-read from the series, so the
    null is fitted to exactly the quantity it is compared against. Returns the
    null renormalised to the same non-DC total, and the fitted phi.
    """
    p = np.asarray(power, dtype=np.float64) / parseval_weights(n_time)
    two_sided = np.concatenate([p, p[-2:0:-1]]) if n_time % 2 == 0 else \
        np.concatenate([p, p[-1:0:-1]])
    acov = np.real(np.fft.ifft(two_sided))
    if acov[0] <= 0:
        return np.zeros(len(power)), 0.0
    phi = float(np.clip(acov[1] / acov[0], -0.999, 0.999))

    f = np.arange(len(p)) / n_time
    null = 1.0 / (1.0 - 2.0 * phi * np.cos(2.0 * np.pi * f) + phi ** 2)
    null = null * parseval_weights(n_time)   # back onto the input's scale
    null[0] = 0.0
    tail = null[1:].sum()
    if tail > 0:
        null = null * (np.asarray(power)[1:].sum() / tail)
    return null, phi


def harmonic_excess(power: np.ndarray, n_time: int) -> dict:
    """How much of the spectrum sits above a matched red-noise null.

    ``excess_share`` is the fraction of non-DC variance by which bins exceed the
    AR(1) null. A pure AR(1) process scores near zero however red it is; a
    travelling wave or an oscillation puts a line above the null and scores
    high. This is the metric that separates "low-frequency" from "harmonic".
    """
    p = np.asarray(power, dtype=np.float64)
    total = p[1:].sum()
    if total <= 0:
        return {"ar1_phi": 0.0, "excess_share": 0.0, "peak_bin": 0,
                "peak_over_null": 0.0, "peak_rel_nyquist": 0.0}
    null, phi = ar1_null(p, n_time)
    excess = np.clip(p[1:] - null[1:], 0, None)
    ratio = np.divide(p[1:], null[1:], out=np.zeros_like(p[1:]), where=null[1:] > 0)
    peak = int(np.argmax(ratio)) + 1
    return {
        "ar1_phi": phi,
        "excess_share": float(excess.sum() / total),
        "peak_bin": peak,
        "peak_over_null": float(ratio[peak - 1]),
        "peak_rel_nyquist": float(peak / (len(p) - 1)),
    }


def band_metrics(power: np.ndarray) -> dict:
    """Low/mid/high shares plus concentration measures of a 1-D spectrum.

    ``power[0]`` (DC) is dropped: the static component is accounted for
    separately by the static/dynamic split, and including it here would let a
    large constant offset masquerade as low-frequency structure.
    """
    p = np.asarray(power, dtype=np.float64)[1:]
    total = p.sum()
    if total <= 0:
        return {}
    frac = p / total
    idx = np.arange(1, len(p) + 1)
    nyq = len(p)                      # index of the highest resolved bin
    rel = idx / nyq

    cum = np.cumsum(frac)

    def modes_for(q: float) -> int:
        return int(idx[np.searchsorted(cum, q)]) if cum[-1] >= q else int(idx[-1])

    return {
        "low_share": float(frac[rel <= LOW_EDGE].sum()),
        "mid_share": float(frac[(rel > LOW_EDGE) & (rel <= HIGH_EDGE)].sum()),
        "high_share": float(frac[rel > HIGH_EDGE].sum()),
        "share_bin1": float(frac[0]),
        "share_le_2": float(cum[min(1, len(cum) - 1)]),
        "share_le_4": float(cum[min(3, len(cum) - 1)]),
        "share_le_8": float(cum[min(7, len(cum) - 1)]),
        "modes_50": modes_for(0.50),
        "modes_90": modes_for(0.90),
        "modes_99": modes_for(0.99),
        # participation ratio: how many bins the variance is effectively spread over
        "effective_modes": float(total ** 2 / (p ** 2).sum()),
        "centroid_rel": float((frac * rel).sum()),
    }


# --------------------------------------------------------------------------
# per-scenario driver
# --------------------------------------------------------------------------


def open_remote(repo: str, path: str, block_size: int = 2 ** 23):
    """Open a Well HDF5 file over HTTP range requests.

    ``block_size`` is the fsspec read-ahead unit and must match the access
    pattern. Reading whole trajectories wants a large block; picking scattered
    single frames out of a 65 MB contiguous trajectory wants a small one, or
    every 65 KB frame drags a full block across the wire.
    """
    import fsspec
    import h5py
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, path, repo_type="dataset")
    fs = fsspec.filesystem("http", block_size=block_size)
    return h5py.File(fs.open(url, "rb"), "r")


def analyse_segment(u: np.ndarray, periodic: bool, spatial_stride: int) -> dict:
    """Full decomposition of one (T, H, W) block. Every identity is asserted.

    The cast to float64 is load-bearing, not hygiene. Several scenarios sit at a
    large constant offset with a tiny fluctuation on top (Gray-Scott A hovers
    near 0.98 with a variance of ~1e-3), and a float32 ``var`` loses most of its
    significant digits to cancellation there -- enough to fail the Parseval
    check below by ~3e-3 purely from rounding.
    """
    u = np.asarray(u, dtype=np.float64)
    n_time = u.shape[0]
    static_var = float(u.mean(axis=0).var())
    dynamic_var = float(u.var(axis=0).mean())
    total_var = float(u.var())

    spec = {win: temporal_power(u, win) for win in ("boxcar", "hann")}

    # Parseval: the boxcar spectrum must sum to the dynamic variance
    if dynamic_var > 0:
        summed = spec["boxcar"].sum() / u[0].size
        rel_err = abs(summed - dynamic_var) / dynamic_var
        assert rel_err < 1e-6, f"Parseval failed: {rel_err:.2e}"
    # law of total variance: static + dynamic must close on the total
    if total_var > 0:
        split_err = abs(static_var + dynamic_var - total_var) / total_var
        assert split_err < 1e-6, f"variance split failed: {split_err:.2e}"

    entry = {
        "n_time": n_time,
        "static_var": static_var,
        "dynamic_var": dynamic_var,
        "total_var": total_var,
        "spec": spec,
    }
    if periodic:
        entry["radial"], entry["box"] = spatial_power(u, spatial_stride)
    return entry


def analyse_scenario(fam: Family, scenario: str, n_traj: int, spatial_stride: int,
                     verbose: bool = True) -> list[dict]:
    rows = []
    h5 = open_remote(fam.repo, fam.scenarios[scenario])
    try:
        for fname, fpath in fam.fields.items():
            dset = h5[fpath]
            take = min(n_traj, dset.shape[0])
            n_time = dset.shape[1]
            segs = segments_for(n_time)

            per_traj = []
            for i in range(take):
                u = np.asarray(dset[i])                       # (T, H, W)
                if verbose:
                    print(f"    {scenario:>10s} {fname:>9s} traj {i}  {u.shape}",
                          flush=True)
                per_traj.append({
                    name: analyse_segment(u[sl], fam.periodic, spatial_stride)
                    for name, sl in segs.items()
                })

            rows.append({
                "scenario": scenario,
                "field": fname,
                "n_traj": take,
                "n_time": n_time,
                "segments": list(segs),
                "shape": tuple(dset.shape[2:]),
                "per_traj": per_traj,
            })
    finally:
        h5.close()
    return rows


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def summarise(rows: list[dict], fam: Family) -> list[dict]:
    out = []
    for r in rows:
        for seg in r["segments"]:
            pt = [e[seg] for e in r["per_traj"]]
            n_time = pt[0]["n_time"]
            static = np.array([e["static_var"] for e in pt])
            dynamic = np.array([e["dynamic_var"] for e in pt])
            total = np.array([e["total_var"] for e in pt])

            rec = {
                "family": fam.label,
                "scenario": r["scenario"],
                "field": r["field"],
                "segment": seg,
                "n_traj": r["n_traj"],
                "n_time": n_time,
                "dt": fam.dt,
                "grid": "x".join(str(d) for d in r["shape"]),
                # share of total variance frozen in the time-averaged pattern
                "static_share": float(static.sum() / max(total.sum(), 1e-30)),
                "dynamic_share": float(dynamic.sum() / max(total.sum(), 1e-30)),
            }

            for win in ("boxcar", "hann"):
                stacked = np.stack([e["spec"][win] for e in pt]).sum(axis=0)
                for k, v in band_metrics(stacked).items():
                    rec[f"{win}_{k}"] = v
                for k, v in harmonic_excess(stacked, n_time).items():
                    rec[f"{win}_{k}"] = v
                # per-trajectory spread on the two headline numbers
                per_low = [band_metrics(e["spec"][win])["low_share"] for e in pt]
                per_exc = [harmonic_excess(e["spec"][win], n_time)["excess_share"]
                           for e in pt]
                rec[f"{win}_low_share_sd"] = float(np.std(per_low))
                rec[f"{win}_excess_share_sd"] = float(np.std(per_exc))

            if fam.periodic:
                box = np.stack([e["box"] for e in pt]).sum(axis=0)
                radial = np.stack([e["radial"] for e in pt]).sum(axis=0)
                for k, v in band_metrics(box).items():
                    rec[f"spatial_box_{k}"] = v
                for k, v in band_metrics(radial).items():
                    rec[f"spatial_radial_{k}"] = v
                cum = np.cumsum(box[1:]) / box[1:].sum()
                for m_cut in (4, 8, 12, 16):
                    if m_cut <= len(cum):
                        rec[f"spatial_var_at_modes_{m_cut}"] = float(cum[m_cut - 1])

            out.append(rec)
    return out


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    keys, seen = [], set()
    for r in records:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)
    print(f"wrote {path}")


def write_spectra(path: Path, rows: list[dict], fam: Family) -> None:
    recs = []
    for r in rows:
        for seg in r["segments"]:
            pt = [e[seg] for e in r["per_traj"]]
            n_time = pt[0]["n_time"]
            for win in ("boxcar", "hann"):
                p = np.stack([e["spec"][win] for e in pt]).sum(axis=0)
                tot = p[1:].sum()
                if tot <= 0:
                    continue
                null, _ = ar1_null(p, n_time)
                cum = np.cumsum(p[1:]) / tot
                nyq = len(p) - 1
                for i in range(1, len(p)):
                    recs.append({
                        "family": fam.label,
                        "scenario": r["scenario"],
                        "field": r["field"],
                        "segment": seg,
                        "window": win,
                        "bin": i,
                        "freq_rel_nyquist": i / nyq,
                        "period_steps": n_time / i,
                        "period_time_units": n_time * fam.dt / i,
                        "share": float(p[i] / tot),
                        "cumulative_share": float(cum[i - 1]),
                        "ar1_null_share": float(null[i] / tot),
                    })
    write_csv(path, recs)


def write_spatial(path: Path, rows: list[dict], fam: Family) -> None:
    if not fam.periodic:
        return
    recs = []
    for r in rows:
        for seg in r["segments"]:
            pt = [e[seg] for e in r["per_traj"]]
            for kind in ("radial", "box"):
                p = np.stack([e[kind] for e in pt]).sum(axis=0)
                tot = p[1:].sum()
                if tot <= 0:
                    continue
                cum = np.cumsum(p[1:]) / tot
                for i in range(1, len(p)):
                    recs.append({
                        "family": fam.label,
                        "scenario": r["scenario"],
                        "field": r["field"],
                        "segment": seg,
                        "binning": kind,
                        "k": i,
                        "share": float(p[i] / tot),
                        "cumulative_share": float(cum[i - 1]),
                    })
    write_csv(path, recs)


def plot(rows: list[dict], summary: list[dict], fam: Family, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped figure")
        return

    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    field0 = rows[0]["field"]
    keep = [r for r in rows if r["field"] == field0]
    cmap = plt.get_cmap("viridis")
    colors = {s: cmap(0.08 + 0.84 * i / max(len(scenarios) - 1, 1))
              for i, s in enumerate(scenarios)}

    seg_main = "settled" if "settled" in keep[0]["segments"] else "full"
    panels = ["full", seg_main] if seg_main != "full" else ["full"]
    ncol = len(panels) + 1 + (1 if fam.periodic else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(4.9 * ncol, 4.3))
    axes = np.atleast_1d(axes)

    # spectrum panels, one per segment, each against its own AR(1) null
    for ax, seg in zip(axes, panels):
        for r in keep:
            pt = [e[seg] for e in r["per_traj"]]
            p = np.stack([e["spec"]["hann"] for e in pt]).sum(axis=0)
            null, _ = ar1_null(p, pt[0]["n_time"])
            tot = p[1:].sum()
            f = np.arange(1, len(p)) / (len(p) - 1)
            ax.loglog(f, p[1:] / tot, color=colors[r["scenario"]], lw=1.5,
                      label=r["scenario"])
            ax.loglog(f, null[1:] / tot, color=colors[r["scenario"]], lw=0.8,
                      ls=":", alpha=0.7)
        ax.axvline(LOW_EDGE, ls="--", c="0.6", lw=0.8)
        ax.axvline(HIGH_EDGE, ls="--", c="0.6", lw=0.8)
        ax.set_xlabel("temporal frequency / Nyquist")
        ax.set_ylabel("variance share per bin")
        ax.set_title(f"Temporal spectrum — {seg}\n(dotted = matched AR(1) null)",
                     fontsize=10)
        if seg == panels[0]:
            ax.legend(fontsize=7, loc="lower left")

    # low-frequency share vs harmonic excess: the point of the whole exercise
    ax = axes[len(panels)]
    marks = {"full": "o", "train": "s", "settled": "^"}
    for r in summary:
        if r["field"] != field0:
            continue
        ax.scatter(r["hann_low_share"] * 100, r["hann_excess_share"] * 100,
                   color=colors[r["scenario"]], s=64,
                   marker=marks.get(r["segment"], "o"),
                   edgecolor="k", linewidth=0.5, zorder=3)
    for seg, mk in marks.items():
        if seg in keep[0]["segments"]:
            ax.scatter([], [], marker=mk, c="0.5", edgecolor="k",
                       linewidth=0.5, label=seg)
    ax.set_xlabel("low-frequency share (%)")
    ax.set_ylabel("harmonic excess over AR(1) null (%)")
    ax.set_xlim(0, 102)
    # Fold the headline into the title rather than annotating inside the axes:
    # in the Gray-Scott case every point piles against the right edge, so there
    # is no collision-free spot for a text box.
    lows = [r["hann_low_share"] for r in summary if r["field"] == field0]
    verdict = ("x barely varies — it does not discriminate"
               if max(lows) - min(lows) < 0.10 else
               "x does spread here — unlike Gray-Scott")
    ax.set_title("Low-frequency ≠ harmonic (up = exploitable)\n"
                 f"x spans {min(lows):.0%}–{max(lows):.0%}: {verdict}", fontsize=9.5)
    ax.legend(fontsize=7, title="segment", title_fontsize=7, loc="lower left")

    if fam.periodic:
        ax = axes[-1]
        for r in keep:
            pt = [e[seg_main] for e in r["per_traj"]]
            p = np.stack([e["box"] for e in pt]).sum(axis=0)[1:]
            cum = np.cumsum(p) / p.sum()
            ax.plot(np.arange(1, len(p) + 1), cum * 100,
                    color=colors[r["scenario"]], lw=1.6)
        ax.axvline(16, ls="--", c="crimson", lw=1)
        ax.text(16.5, 30, "modes=16\n(native grid)", color="crimson", fontsize=8)
        ax.set_xlabel("FNO mode cutoff  m = max(|kx|,|ky|)")
        ax.set_ylabel("cumulative variance (%)")
        ax.set_xlim(1, 64)
        ax.set_ylim(0, 101)
        ax.set_title(f"Spatial variance retained\nby truncation — {seg_main}",
                     fontsize=10)

    fig.suptitle(f"Harmonic content by scenario — {fam.label} "
                 f"(field {field0}, Hann window)", y=1.04, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


def rows_from_csv(out_dir: Path, tag: str, fam: Family) -> tuple[list[dict], list[dict]]:
    """Rebuild the structures ``plot`` needs from the committed CSVs.

    The CSVs carry every per-bin number the figure draws, so ``--replot`` can
    regenerate figures offline instead of re-downloading gigabytes to move a
    label. Spectra are read back already summed over trajectories, so the
    per-trajectory list is reconstructed with a single synthetic entry.
    """
    def read(name):
        path = out_dir / f"ext10_{name}_{tag}.csv"
        if not path.exists():
            return []
        with path.open() as f:
            return list(csv.DictReader(f))

    summary = []
    for r in read("harmonic_summary"):
        rec = dict(r)
        for k, v in r.items():
            if k in ("family", "scenario", "field", "segment", "grid"):
                continue
            try:
                rec[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                rec[k] = v
        summary.append(rec)

    temporal, spatial = read("temporal_spectrum"), read("spatial_spectrum")
    keys = list(dict.fromkeys((r["scenario"], r["field"]) for r in temporal))
    rows = []
    for scenario, fld in keys:
        segs = list(dict.fromkeys(
            r["segment"] for r in temporal
            if (r["scenario"], r["field"]) == (scenario, fld)))
        entry = {}
        for seg in segs:
            e = {}
            for win in ("boxcar", "hann"):
                sel = [r for r in temporal
                       if (r["scenario"], r["field"], r["segment"], r["window"])
                       == (scenario, fld, seg, win)]
                sel.sort(key=lambda r: int(r["bin"]))
                # bin 0 was dropped on write; restore it as the empty DC bin
                e.setdefault("spec", {})[win] = np.array(
                    [0.0] + [float(r["share"]) for r in sel])
                e["n_time"] = int(round(float(sel[0]["period_steps"])))
            for kind in ("radial", "box"):
                sel = [r for r in spatial
                       if (r["scenario"], r["field"], r["segment"], r["binning"])
                       == (scenario, fld, seg, kind)]
                sel.sort(key=lambda r: int(r["k"]))
                if sel:
                    e[kind] = np.array([0.0] + [float(r["share"]) for r in sel])
            entry[seg] = e
        rows.append({
            "scenario": scenario, "field": fld, "n_traj": 0,
            "n_time": entry[segs[0]]["n_time"], "segments": segs,
            "shape": (), "per_traj": [entry],
        })
    return rows, summary


def print_table(summary: list[dict], fam: Family) -> None:
    segs = list(dict.fromkeys(r["segment"] for r in summary))
    for seg in segs:
        sub = [r for r in summary if r["segment"] == seg]
        print(f"\n=== {fam.label} — segment '{seg}' "
              f"(T={sub[0]['n_time']}; low = f <= {LOW_EDGE:.0%} Nyq, "
              f"high = f > {HIGH_EDGE:.0%} Nyq) ===")
        hdr = (f"{'scenario':>12s} {'field':>9s} {'static':>7s} | {'low':>6s} "
               f"{'mid':>6s} {'high':>6s} {'low|box':>8s} | {'n90':>5s} "
               f"{'eff':>6s} | {'phi':>6s} {'excess':>7s} {'pk/null':>8s} "
               f"{'pk per':>7s}")
        print(hdr)
        print("-" * len(hdr))
        for r in sub:
            per = r["n_time"] / r["hann_peak_bin"] if r["hann_peak_bin"] else float("nan")
            print(f"{r['scenario']:>12s} {r['field']:>9s} {r['static_share']:>6.1%} | "
                  f"{r['hann_low_share']:>5.1%} {r['hann_mid_share']:>5.1%} "
                  f"{r['hann_high_share']:>5.1%} {r['boxcar_low_share']:>7.1%} | "
                  f"{r['hann_modes_90']:>5d} {r['hann_effective_modes']:>6.1f} | "
                  f"{r['hann_ar1_phi']:>+6.3f} {r['hann_excess_share']:>6.1%} "
                  f"{r['hann_peak_over_null']:>8.1f} {per:>7.1f}")
        print("  excess = variance above a matched AR(1) null; pk/null ~1 means "
              "no line, just red noise")

    if fam.periodic:
        seg = "settled" if "settled" in segs else segs[0]
        print(f"\n=== {fam.label} — spatial variance retained by FNO mode "
              f"cutoff (segment '{seg}', native {summary[0]['grid']} grid) ===")
        hdr = (f"{'scenario':>12s} {'field':>9s} {'m<=4':>8s} {'m<=8':>8s} "
               f"{'m<=12':>8s} {'m<=16':>8s} {'k for 99%':>10s}")
        print(hdr)
        print("-" * len(hdr))
        for r in summary:
            if r["segment"] != seg:
                continue
            print(f"{r['scenario']:>12s} {r['field']:>9s} "
                  f"{r.get('spatial_var_at_modes_4', float('nan')):>7.1%} "
                  f"{r.get('spatial_var_at_modes_8', float('nan')):>7.1%} "
                  f"{r.get('spatial_var_at_modes_12', float('nan')):>7.1%} "
                  f"{r.get('spatial_var_at_modes_16', float('nan')):>7.1%} "
                  f"{r.get('spatial_box_modes_99', -1):>10d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="gray_scott", choices=sorted(FAMILIES))
    ap.add_argument("--n-traj", type=int, default=3,
                    help="trajectories per scenario (Gray-Scott test files hold 20)")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--spatial-stride", type=int, default=4,
                    help="use every Nth frame for the spatial FFT")
    ap.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/extensions"))
    ap.add_argument("--tag", default=None, help="output filename tag (default: family)")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild the figure from the committed CSVs, no network")
    args = ap.parse_args()

    fam = FAMILIES[args.family]
    tag = args.tag or args.family

    if args.replot:
        rows, summary = rows_from_csv(args.out_dir, tag, fam)
        if not rows:
            raise SystemExit(f"no CSVs for tag '{tag}' under {args.out_dir}")
        print_table(summary, fam)
        plot(rows, summary, fam, args.fig_dir / f"ext10_harmonic_content_{tag}.png")
        return

    ca = _certifi_path()
    if ca:
        # the venv python has no system CA bundle; without this the
        # HTTPS range reads fail on certificate verification
        os.environ.setdefault("SSL_CERT_FILE", ca)

    names = args.scenarios or list(fam.scenarios)

    print(f"{fam.label}: {len(names)} scenarios, {args.n_traj} trajectory(ies) each")
    if fam.notes:
        print(f"  note: {fam.notes}")

    rows = []
    for s in names:
        print(f"  [{s}]", flush=True)
        rows.extend(analyse_scenario(fam, s, args.n_traj, args.spatial_stride))

    summary = summarise(rows, fam)
    print_table(summary, fam)

    write_csv(args.out_dir / f"ext10_harmonic_summary_{tag}.csv", summary)
    write_spectra(args.out_dir / f"ext10_temporal_spectrum_{tag}.csv", rows, fam)
    write_spatial(args.out_dir / f"ext10_spatial_spectrum_{tag}.csv", rows, fam)
    plot(rows, summary, fam, args.fig_dir / f"ext10_harmonic_content_{tag}.png")


def _certifi_path() -> str:
    try:
        import certifi
        return certifi.where()
    except ImportError:
        return ""


if __name__ == "__main__":
    sys.exit(main())
