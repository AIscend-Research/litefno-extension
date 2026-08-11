r"""Field recovery under thin sensor coverage, per scenario.

Board task: "downsampled/masked versions of the simulations to simulate 'thin
sensor coverage' or 'incomplete surveys' typical of low-resource regions".

The Well's simulations are fully observed on a dense grid. A low-resource
deployment is not: you get a handful of stations, a regular but coarse sensor
lattice, or a survey with swath-shaped holes in it. This script masks the
ground-truth fields to those regimes and asks how much of the field can still be
recovered, scenario by scenario.

ext10 (`harmonic_content.py`) makes this predictive rather than exploratory. It
found that Gray-Scott's scenarios put their spatial variance in very different
places -- spirals holds 77% of its variance below mode 8 while maze and spots
hold ~1%, because their energy sits in a narrow band at the Turing wavelength
near mode 13-16. Sparse sampling constrains a limited number of Fourier modes,
so the prediction is that maze and spots collapse under thin coverage while
spirals degrades gently. That prediction is tested here.

Sampling regimes
----------------
``random``    i.i.d. Bernoulli pixels. Incoherent and unbiased -- the best case,
              and the one compressed-sensing results assume. Rarely available.
``grid``      a regular lattice at stride s. A fixed sensor network or a
              constant-cadence survey. Cheap to deploy and it aliases: any mode
              above the lattice Nyquist folds down onto a lower one.
``blocks``    contiguous 8x8 patches observed, the rest missing. Swath gaps,
              cloud cover, terrain access.
``stations``  a few point sites with a 3x3 footprint each. The extreme
              low-resource case: most of the domain is never seen.

Reconstruction
--------------
Band-limited least squares: fit the coefficients of every Fourier mode inside
the box |kx| <= M, |ky| <= M to the observed pixels, then evaluate on the full
grid. This is the right model class for the question because it is exactly what
an FNO's ``n_modes=(M, M)`` truncation represents -- the reconstruction succeeds
only if the sensors can constrain the modes the operator would use.

The fit is truncated-SVD least squares: the problem is solved only in the
subspace the observations actually determine. The truncation level is swept
(1e-6 .. 1e-1 of the largest singular value) and each scenario is given its own
best level, chosen in hindsight against the very field being reconstructed --
an estimator far too generous to be a real method, so that any configuration
which still fails has failed for reasons no tuning can fix. ``rcond_spread``
records how far the answer moved across those levels: near zero for `random`
and `grid`, large for the clustered regimes, where the data does not determine
the reconstruction at all. Underdetermined is not a failure mode to avoid here,
it is the regime of interest -- a box of M modes has (2M+1)^2 real degrees of
freedom, so 1% coverage of a 128x128 grid (164 pixels) cannot constrain M=8 (289
dof) no matter how good the solver is. Both the observations-per-dof ratio and
the number of singular directions actually retained (``n_modes_constrained``)
are reported alongside every number; the latter is the effective size of the
model the sensor layout can support.

The control that makes the numbers mean something
-------------------------------------------------
Reconstruction error has two separable causes: modes outside the box are
discarded (band-limiting), and the modes inside it are poorly constrained
(sparsity). Reporting only the total conflates them, and at small M the
band-limiting term dominates so completely that every sampling regime looks
equally bad.

So every configuration is run against an **oracle floor**: the same M-mode fit
using *all* pixels, which is the best any M-mode model could do. The reported
sparsity penalty is the excess over that floor. A penalty near zero means the
sensors recovered everything their model class allows -- the coverage is
sufficient, whatever the absolute error.

VRMSE matches `litefno.metrics.vrmse` (asserted in the tests).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

_HC_PATH = Path(__file__).resolve().parent / "harmonic_content.py"


def _load_hc():
    spec = importlib.util.spec_from_file_location("harmonic_content", _HC_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["harmonic_content"] = module
    spec.loader.exec_module(module)
    return module


hc = _load_hc()

DEFAULT_COVERAGE = (50.0, 25.0, 10.0, 5.0, 2.0, 1.0)
DEFAULT_MODES = (4, 8, 12, 16)
BLOCK = 8          # block side for the `blocks` regime
FOOTPRINT = 3      # station footprint side for the `stations` regime


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def vrmse(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Variance-scaled RMSE, matching litefno.metrics.vrmse.

    float32 in the reference implementation; computed here in float64 and cast,
    since the fields sit at a large offset with a small fluctuation (see the
    cancellation note in harmonic_content.analyse_segment).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = ((pred - target) ** 2).mean()
    return float(np.sqrt(mse / (target.var() + eps)))


# --------------------------------------------------------------------------
# sampling regimes
# --------------------------------------------------------------------------


def mask_random(shape, coverage: float, rng) -> np.ndarray:
    h, w = shape
    n_keep = max(1, int(round(coverage * h * w)))
    flat = np.zeros(h * w, dtype=bool)
    flat[rng.choice(h * w, size=n_keep, replace=False)] = True
    return flat.reshape(h, w)


def mask_grid(shape, coverage: float, rng) -> np.ndarray:
    """Regular lattice. Stride is rounded, so realised coverage differs."""
    h, w = shape
    stride = max(1, int(round(1.0 / np.sqrt(coverage))))
    off_y, off_x = (int(rng.integers(stride)), int(rng.integers(stride))) \
        if stride > 1 else (0, 0)
    m = np.zeros((h, w), dtype=bool)
    m[off_y::stride, off_x::stride] = True
    return m


def mask_blocks(shape, coverage: float, rng) -> np.ndarray:
    """Contiguous BLOCK x BLOCK patches observed; the rest is a gap."""
    h, w = shape
    ny, nx = h // BLOCK, w // BLOCK
    n_blocks = max(1, int(round(coverage * ny * nx)))
    m = np.zeros((ny, nx), dtype=bool)
    m.ravel()[rng.choice(ny * nx, size=n_blocks, replace=False)] = True
    return np.kron(m, np.ones((BLOCK, BLOCK), dtype=bool))[:h, :w]


def mask_stations(shape, coverage: float, rng) -> np.ndarray:
    """A few point sites, each seeing a FOOTPRINT x FOOTPRINT patch."""
    h, w = shape
    per_site = FOOTPRINT ** 2
    n_sites = max(1, int(round(coverage * h * w / per_site)))
    m = np.zeros((h, w), dtype=bool)
    r = FOOTPRINT // 2
    cy = rng.integers(0, h, size=n_sites)
    cx = rng.integers(0, w, size=n_sites)
    for y, x in zip(cy, cx):
        ys = (np.arange(y - r, y + r + 1)) % h        # periodic domain
        xs = (np.arange(x - r, x + r + 1)) % w
        m[np.ix_(ys, xs)] = True
    return m


REGIMES = {
    "random": mask_random,
    "grid": mask_grid,
    "blocks": mask_blocks,
    "stations": mask_stations,
}


def mask_seed(regime: str, coverage: float, seed: int) -> int:
    """Deterministic seed for a mask configuration.

    Explicitly *not* built from ``hash()``: Python salts string hashing per
    process, so a hash-derived seed silently draws different masks on every run
    and the results stop being reproducible. Derived from a stable digest
    instead.
    """
    key = f"{regime}|{coverage:.6f}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 63)


# --------------------------------------------------------------------------
# band-limited basis
# --------------------------------------------------------------------------


def fourier_basis(shape, modes: int) -> np.ndarray:
    """Real design matrix (H*W, (2M+1)^2) for the box |kx|,|ky| <= M.

    One constant column plus a cos/sin pair per conjugate mode pair, so the span
    is exactly the set of real fields whose spectrum is confined to the box --
    the same set an FNO with ``n_modes=(M, M)`` can represent.
    """
    h, w = shape
    yy, xx = np.indices((h, w))
    cols = [np.ones(h * w)]
    seen = set()
    for ky in range(-modes, modes + 1):
        for kx in range(-modes, modes + 1):
            if (ky, kx) == (0, 0) or (-ky, -kx) in seen:
                continue
            seen.add((ky, kx))
            theta = 2 * np.pi * (ky * yy / h + kx * xx / w)
            cols.append(np.cos(theta).ravel())
            cols.append(np.sin(theta).ravel())
    basis = np.stack(cols, axis=1)
    assert basis.shape[1] == (2 * modes + 1) ** 2, basis.shape
    return basis


RCONDS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
RCOND = 1e-6


def fit_over_rconds(basis: np.ndarray, obs_idx: np.ndarray, fields: np.ndarray,
                    rconds=RCONDS) -> dict:
    """Reconstruct at several truncation levels, sharing one SVD.

    Needed because the truncation level is not innocuous for clustered masks.
    For `random` and `grid` the reconstruction is identical across five orders
    of magnitude of ``rcond``; for `blocks` and `stations` it moves by three
    orders (VRMSE 1189 at 1e-6 down to 1.5 at 1e-2 on spirals at 25% coverage).
    Where a result depends that strongly on a regularisation constant, the data
    is not determining it, and picking one constant would be picking an answer.

    So the caller is given every level and takes the best per scenario -- an
    estimator tuned in hindsight, on the test field, to the most favourable
    setting. It is deliberately far too generous to be a real method. That is
    the point: a configuration that still fails under it has failed for reasons
    no amount of tuning will fix.

    Returns ``{rcond: (reconstruction, n_singular_values_kept)}``.
    """
    design = basis[obs_idx]
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    observed = fields[obs_idx]
    out = {}
    for rcond in rconds:
        keep = s > rcond * s[0] if s.size and s[0] > 0 else np.zeros(0, dtype=bool)
        n_kept = int(keep.sum())
        if n_kept == 0:
            out[rcond] = (np.zeros_like(fields), 0)
            continue
        coeffs = vt[keep].T @ ((u[:, keep].T @ observed) / s[keep][:, None])
        out[rcond] = (basis @ coeffs, n_kept)
    return out


def fit_and_reconstruct(basis: np.ndarray, obs_idx: np.ndarray,
                        fields: np.ndarray, rcond: float = RCOND):
    """Truncated-SVD least-squares fit on observed pixels, evaluated everywhere.

    ``fields`` is (n_pixels, n_frames) and the returned array matches it. The
    decomposition depends only on the mask and the basis, so every frame and
    scenario sharing a mask is solved in one call. Also returns how many
    singular values survived the cut.

    Why not plain ``lstsq``: with a clustered mask (blocks, stations) the design
    matrix is very close to singular, because reconstructing the field outside a
    contiguous observed patch is extrapolation of a band-limited function --
    exponentially ill-conditioned. Default-``rcond`` min-norm least squares
    keeps those near-null directions and returns coefficients large enough to
    produce VRMSE around 1e8. That number is a property of the solver, not of
    the sensor network, and reporting it would be meaningless.

    Truncating at ``rcond`` times the largest singular value instead solves the
    problem only in the subspace the observations actually determine. The count
    of retained singular values is the useful quantity in its own right: it is
    the effective number of modes this sensor layout can constrain, which is at
    most the mode budget and often far less. It is carried through to the CSV as
    ``n_modes_constrained``.
    """
    design = basis[obs_idx]
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    keep = s > rcond * s[0] if s.size and s[0] > 0 else np.zeros(0, dtype=bool)
    n_kept = int(keep.sum())
    if n_kept == 0:
        return np.zeros_like(fields), 0
    coeffs = vt[keep].T @ ((u[:, keep].T @ fields[obs_idx]) / s[keep][:, None])
    return basis @ coeffs, n_kept


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def build_cache(fam, cache_path: Path, n_traj: int, n_frames: int) -> dict:
    """Fetch settled-segment frames once and cache them locally.

    Only the settled half of each trajectory is used, for the reason ext10
    established: the spin-up transient is a one-time ramp and is not
    representative of what a deployed sensor network would be observing.
    """
    if cache_path.exists():
        with np.load(cache_path) as z:
            return {k: z[k] for k in z.files}

    cache = {}
    for scenario, path in fam.scenarios.items():
        print(f"  fetching {scenario}", flush=True)
        h5 = hc.open_remote(fam.repo, path, block_size=2 ** 20)
        try:
            for fname, fpath in fam.fields.items():
                dset = h5[fpath]
                take = min(n_traj, dset.shape[0])
                n_time = dset.shape[1]

                # One CONTIGUOUS run of frames per trajectory. These reads are
                # latency-bound (~2 s per request regardless of size), so eight
                # scattered single frames cost ~17 s while eight consecutive
                # ones cost ~2 s. Independent realisations come from using
                # separate trajectories; the run start is staggered across them
                # so the frames are not all drawn from the same instant.
                frames = []
                for i in range(take):
                    lo = n_time // 2
                    span = n_time - lo - n_frames
                    start = lo + (span * i) // max(take, 1)
                    frames.append(np.asarray(dset[i, start:start + n_frames]))
                cache[f"{scenario}|{fname}"] = np.concatenate(frames).astype(np.float32)
        finally:
            h5.close()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cache)
    print(f"  cached {cache_path} "
          f"({cache_path.stat().st_size / 1e6:.0f} MB, {len(cache)} arrays)")
    return cache


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run(cache: dict, coverages, mode_list, regimes, n_masks: int,
        shape) -> list[dict]:
    keys = sorted(cache)
    stacked = np.stack([cache[k].reshape(len(cache[k]), -1) for k in keys])
    n_key, n_frame, n_pix = stacked.shape
    flat = np.ascontiguousarray(
        stacked.transpose(2, 0, 1).reshape(n_pix, n_key * n_frame),
        dtype=np.float64)

    records = []
    for modes in mode_list:
        basis = fourier_basis(shape, modes)
        n_dof = basis.shape[1]

        # oracle: the same M-mode fit with every pixel observed
        oracle, _ = fit_and_reconstruct(basis, np.arange(n_pix), flat)
        oracle_v = {}
        for i, key in enumerate(keys):
            sl = slice(i * n_frame, (i + 1) * n_frame)
            oracle_v[key] = vrmse(oracle[:, sl], flat[:, sl])

        for regime in regimes:
            for cov in coverages:
                for seed in range(n_masks):
                    rng = np.random.default_rng(mask_seed(regime, cov, seed))
                    m = REGIMES[regime](shape, cov / 100.0, rng)
                    obs_idx = np.flatnonzero(m.ravel())
                    actual = len(obs_idx) / n_pix

                    fits = fit_over_rconds(basis, obs_idx, flat)
                    for i, key in enumerate(keys):
                        scenario, fld = key.split("|")
                        sl = slice(i * n_frame, (i + 1) * n_frame)
                        # best truncation for THIS scenario, chosen in hindsight
                        scored = [(vrmse(rec[:, sl], flat[:, sl]), rc, nk)
                                  for rc, (rec, nk) in fits.items()]
                        v, best_rcond, n_kept = min(scored)
                        spread = max(x[0] for x in scored) - v
                        records.append({
                            "scenario": scenario,
                            "field": fld,
                            "regime": regime,
                            "coverage_nominal_pct": cov,
                            "coverage_actual_pct": 100 * actual,
                            "seed": seed,
                            "modes": modes,
                            "n_dof": n_dof,
                            "n_obs": len(obs_idx),
                            "obs_per_dof": len(obs_idx) / n_dof,
                            "n_modes_constrained": n_kept,
                            "constrained_frac": n_kept / n_dof,
                            "best_rcond": best_rcond,
                            # how much the answer moves across truncation levels;
                            # large => the data is not determining it
                            "rcond_spread": spread,
                            "vrmse": v,
                            "vrmse_oracle": oracle_v[key],
                            "sparsity_penalty": v - oracle_v[key],
                        })
                print(f"    modes={modes:>2d} {regime:>9s} "
                      f"cov={cov:>5.1f}% -> {100 * actual:>5.2f}% actual, "
                      f"{len(obs_idx):>5d} obs / {n_dof:>4d} dof "
                      f"({len(obs_idx) / n_dof:>5.2f}x)", flush=True)
    return records


def aggregate(records: list[dict]) -> list[dict]:
    """Mean over mask seeds, keeping the spread."""
    groups = {}
    for r in records:
        k = (r["scenario"], r["field"], r["regime"],
             r["coverage_nominal_pct"], r["modes"])
        groups.setdefault(k, []).append(r)
    out = []
    for k, rs in groups.items():
        v = np.array([r["vrmse"] for r in rs])
        p = np.array([r["sparsity_penalty"] for r in rs])
        base = rs[0]
        out.append({
            "scenario": k[0], "field": k[1], "regime": k[2],
            "coverage_nominal_pct": k[3], "modes": k[4],
            "coverage_actual_pct": base["coverage_actual_pct"],
            "n_obs": base["n_obs"], "n_dof": base["n_dof"],
            "obs_per_dof": base["obs_per_dof"],
            "n_modes_constrained": base["n_modes_constrained"],
            "constrained_frac": base["constrained_frac"],
            "best_rcond": base["best_rcond"],
            "rcond_spread": float(np.mean([r["rcond_spread"] for r in rs])),
            "n_masks": len(rs),
            "vrmse": float(v.mean()), "vrmse_sd": float(v.std()),
            "vrmse_oracle": base["vrmse_oracle"],
            "sparsity_penalty": float(p.mean()),
            "sparsity_penalty_sd": float(p.std()),
        })
    return sorted(out, key=lambda r: (r["modes"], r["regime"],
                                      -r["coverage_nominal_pct"],
                                      r["scenario"], r["field"]))


def print_tables(agg: list[dict], headline_modes: int) -> None:
    scenarios = sorted({r["scenario"] for r in agg})
    print(f"\n=== VRMSE of the reconstructed field (modes={headline_modes}, "
          f"field A, mean over mask seeds) ===")
    print("    oracle = same mode budget, every pixel observed (the floor)")
    for regime in sorted({r["regime"] for r in agg}):
        rows = [r for r in agg if r["regime"] == regime
                and r["modes"] == headline_modes and r["field"] == "A"]
        if not rows:
            continue
        covs = sorted({r["coverage_nominal_pct"] for r in rows}, reverse=True)
        print(f"\n  [{regime}]")
        print(f"    {'scenario':>9s} {'oracle':>7s} | "
              + " ".join(f"{c:>6.1f}%" for c in covs))
        print("    " + "-" * (19 + 8 * len(covs)))
        for s in scenarios:
            sub = {r["coverage_nominal_pct"]: r for r in rows if r["scenario"] == s}
            if not sub:
                continue
            oracle = next(iter(sub.values()))["vrmse_oracle"]
            cells = " ".join(f"{sub[c]['vrmse']:>7.3f}" if c in sub else " " * 7
                             for c in covs)
            print(f"    {s:>9s} {oracle:>7.3f} | {cells}")

    print(f"\n=== Sparsity penalty: VRMSE above the oracle floor "
          f"(modes={headline_modes}, field A) ===")
    print("    ~0 means the sensors recovered everything this mode budget allows")
    for regime in sorted({r["regime"] for r in agg}):
        rows = [r for r in agg if r["regime"] == regime
                and r["modes"] == headline_modes and r["field"] == "A"]
        if not rows:
            continue
        covs = sorted({r["coverage_nominal_pct"] for r in rows}, reverse=True)
        print(f"\n  [{regime}]")
        print(f"    {'scenario':>9s} | " + " ".join(f"{c:>6.1f}%" for c in covs))
        print("    " + "-" * (11 + 8 * len(covs)))
        for s in scenarios:
            sub = {r["coverage_nominal_pct"]: r for r in rows if r["scenario"] == s}
            if not sub:
                continue
            cells = " ".join(f"{sub[c]['sparsity_penalty']:>7.3f}" if c in sub
                             else " " * 7 for c in covs)
            print(f"    {s:>9s} | {cells}")


def print_best_budget(agg: list[dict], field: str = "A") -> None:
    """Best reachable error at each coverage, choosing the mode budget freely.

    This is the deployment question. Nobody has to commit to M=16 in advance; if
    the sensors only support M=4, you use M=4. So for each (scenario, regime,
    coverage) take the mode budget that minimises VRMSE and report both the
    error and which budget won. A scenario whose variance sits at high
    wavenumber cannot be rescued by dropping to a budget it can afford -- the
    affordable budgets do not contain its structure.
    """
    scenarios = sorted({r["scenario"] for r in agg})
    covs = sorted({r["coverage_nominal_pct"] for r in agg}, reverse=True)
    for regime in sorted({r["regime"] for r in agg}):
        print(f"\n=== Best achievable VRMSE at each coverage, mode budget free "
              f"(field {field}, {regime}) ===")
        print("    cell = best VRMSE (winning mode budget); "
              "* = worse than predicting the mean")
        hdr = f"    {'scenario':>9s} | " + " ".join(f"{c:>12.1f}%" for c in covs)
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for s in scenarios:
            cells = []
            for c in covs:
                rows = [r for r in agg if r["scenario"] == s and r["field"] == field
                        and r["regime"] == regime and r["coverage_nominal_pct"] == c]
                if not rows:
                    cells.append(" " * 13)
                    continue
                best = min(rows, key=lambda r: r["vrmse"])
                flag = "*" if best["vrmse"] >= 1.0 else " "
                cells.append(f"{best['vrmse']:>8.3f}(M{best['modes']:>2d}){flag}")
            print(f"    {s:>9s} | " + " ".join(cells))


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return float(np.corrcoef(rx, ry)[0, 1])


def check_ext10_prediction(agg: list[dict], ext10_csv: Path,
                           field: str = "A") -> list[dict]:
    """Test ext10's prediction against this experiment's outcome.

    ext10 measured, per scenario, what fraction of spatial variance sits below
    mode 8. If that measurement is the thing that governs recoverability, it
    should rank the scenarios the same way their reconstruction error ranks
    them, in reverse. This scores that rank correlation -- a prediction made
    before these runs existed, checked against them, rather than a pattern read
    off the results afterwards.
    """
    if not ext10_csv.exists():
        print(f"\n(ext10 summary not found at {ext10_csv}; skipping check)")
        return []
    with ext10_csv.open() as f:
        rows = [r for r in csv.DictReader(f)
                if r["segment"] == "settled" and r["field"] == field]
    retained = {r["scenario"]: float(r["spatial_var_at_modes_8"]) for r in rows
                if r.get("spatial_var_at_modes_8")}
    if len(retained) < 3:
        return []

    out = []
    print(f"\n=== ext10 prediction check (field {field}) ===")
    print("    ext10 measured spatial variance below mode 8 per scenario.")
    print("    If that is what governs recovery, it must rank the scenarios")
    print("    inversely to their reconstruction error. rho = -1 is exact.")
    for regime in sorted({r["regime"] for r in agg}):
        for cov in sorted({r["coverage_nominal_pct"] for r in agg}):
            names, best = [], []
            for s in sorted(retained):
                sub = [r["vrmse"] for r in agg
                       if r["scenario"] == s and r["field"] == field
                       and r["regime"] == regime
                       and r["coverage_nominal_pct"] == cov]
                if sub:
                    names.append(s)
                    best.append(min(sub))
            if len(names) < 3:
                continue
            rho = spearman([retained[s] for s in names], best)
            out.append({"regime": regime, "coverage_nominal_pct": cov,
                        "n_scenarios": len(names), "spearman_rho": rho})
            if cov <= 2.0:
                print(f"    {regime:>9s} @ {cov:>4.1f}% coverage: rho = {rho:+.3f}")
    return out


def plot(agg: list[dict], out_png: Path, headline_modes: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped figure")
        return

    scenarios = sorted({r["scenario"] for r in agg})
    regimes = [r for r in ("random", "grid", "blocks", "stations")
               if any(x["regime"] == r for x in agg)]
    cmap = plt.get_cmap("viridis")
    colors = {s: cmap(0.08 + 0.84 * i / max(len(scenarios) - 1, 1))
              for i, s in enumerate(scenarios)}

    fig, axes = plt.subplots(1, len(regimes) + 1,
                             figsize=(4.3 * (len(regimes) + 1), 4.2))

    for ax, regime in zip(axes, regimes):
        for s in scenarios:
            rows = sorted([r for r in agg if r["regime"] == regime
                           and r["scenario"] == s and r["field"] == "A"
                           and r["modes"] == headline_modes],
                          key=lambda r: r["coverage_actual_pct"])
            if not rows:
                continue
            x = [r["coverage_actual_pct"] for r in rows]
            y = [r["vrmse"] for r in rows]
            ax.semilogx(x, y, "o-", color=colors[s], lw=1.5, ms=4, label=s)
            ax.axhline(rows[0]["vrmse_oracle"], color=colors[s], ls=":", lw=0.8)
        ax.axhline(1.0, color="crimson", ls="--", lw=1)
        ax.set_xlabel("coverage (% of pixels observed)")
        ax.set_ylabel("VRMSE of reconstructed field")
        ax.set_title(f"{regime}\n(dotted = oracle floor at modes={headline_modes})",
                     fontsize=10)
        ax.set_ylim(0, 1.6)
        if regime == regimes[0]:
            ax.legend(fontsize=7)
            ax.text(1.05, 1.02, "VRMSE 1.0 = no better\nthan predicting the mean",
                    color="crimson", fontsize=7)

    # coverage needed to get within 0.05 VRMSE of the floor, by scenario
    ax = axes[-1]
    width = 0.8 / max(len(regimes), 1)
    any_finite = False
    for j, regime in enumerate(regimes):
        needed = []
        for s in scenarios:
            rows = sorted([r for r in agg if r["regime"] == regime
                           and r["scenario"] == s and r["field"] == "A"
                           and r["modes"] == headline_modes],
                          key=lambda r: r["coverage_actual_pct"])
            ok = [r["coverage_actual_pct"] for r in rows
                  if r["sparsity_penalty"] < 0.05]
            needed.append(min(ok) if ok else np.nan)
        pos = np.arange(len(scenarios)) + j * width - 0.4 + width / 2
        ax.bar(pos, needed, width=width, label=regime)
        any_finite = any_finite or np.isfinite(needed).any()
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
    if any_finite:
        # log scale raises if every bar is NaN (no config ever reached the floor)
        ax.set_yscale("log")
    ax.set_ylabel("coverage needed (%)")
    ax.set_title("Coverage to get within 0.05 VRMSE\nof the oracle floor "
                 "(missing bar = never)", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle("Field recovery under thin sensor coverage — Gray-Scott "
                 f"(field A, modes={headline_modes})", y=1.03, fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="gray_scott", choices=sorted(hc.FAMILIES))
    ap.add_argument("--n-traj", type=int, default=3)
    ap.add_argument("--n-frames", type=int, default=8,
                    help="settled-segment frames per trajectory")
    ap.add_argument("--n-masks", type=int, default=3, help="mask seeds per config")
    ap.add_argument("--coverage", type=float, nargs="*", default=list(DEFAULT_COVERAGE))
    ap.add_argument("--modes", type=int, nargs="*", default=list(DEFAULT_MODES))
    ap.add_argument("--regimes", nargs="*", default=list(REGIMES))
    ap.add_argument("--headline-modes", type=int, default=16)
    ap.add_argument("--cache", type=Path,
                    default=Path("data/processed/sparsity_cache.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/extensions"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/extensions"))
    args = ap.parse_args()

    ca = hc._certifi_path()
    if ca:
        import os
        os.environ.setdefault("SSL_CERT_FILE", ca)

    fam = hc.FAMILIES[args.family]
    print(f"{fam.label}: {len(fam.scenarios)} scenarios")
    cache = build_cache(fam, args.cache, args.n_traj, args.n_frames)
    shape = next(iter(cache.values())).shape[1:]
    print(f"  {len(cache)} arrays, {next(iter(cache.values())).shape[0]} frames "
          f"each, grid {shape[0]}x{shape[1]}")

    records = run(cache, args.coverage, args.modes, args.regimes,
                  args.n_masks, shape)
    agg = aggregate(records)
    print_tables(agg, args.headline_modes)
    print_best_budget(agg)

    tag = args.family
    check = check_ext10_prediction(
        agg, args.out_dir / f"ext10_harmonic_summary_{tag}.csv")
    write_csv(args.out_dir / f"ext11_sparsity_{tag}.csv", agg)
    if check:
        write_csv(args.out_dir / f"ext11_ext10_prediction_check_{tag}.csv", check)
    plot(agg, args.fig_dir / f"ext11_sparsity_{tag}.png", args.headline_modes)


if __name__ == "__main__":
    sys.exit(main())
