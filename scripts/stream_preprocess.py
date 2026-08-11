r"""Build the repo's processed HDF5 splits by streaming The Well over HTTP.

``litefno download`` pulls a whole Well dataset before ``litefno preprocess``
reduces it -- 44 GB for Gray-Scott, 15 TB for the collection. That is the one
step of this pipeline a low-resource machine cannot do, which is awkward for a
repo whose stated emphasis is low-resource deployment.

Preprocessing caps time at ``max_steps`` (60) and downsamples space by 4x, so
better than 99% of what gets downloaded is discarded. This script reads only the
part that survives: for each trajectory it fetches a contiguous ``max_steps``
prefix -- one range request of about 4 MB against a 21 GB file -- and applies the
reductions immediately. Building the Gray-Scott splits costs under 1 GB of
transfer instead of 44 GB.

The reductions themselves are imported from ``litefno.preprocess`` rather than
reimplemented, so the output is the same function of the raw data that
``litefno preprocess`` would produce. What differs is only which bytes were
moved to compute it. The output is the flat layout the training loader expects:
``<out_dir>/<split>.h5`` holding ``data`` of shape
``(n_traj, n_steps, H, W, n_fields)``.

Trajectory selection is deterministic and recorded: trajectories are taken in
order from each regime file, ``per_scenario`` of them, and the manifest written
alongside the data lists exactly which file and which indices were used.
Sampling randomly would be closer to ``cap_trajectories``, but the point here is
that someone else can rebuild the identical file, and The Well's regimes are not
interchangeable -- ext10 showed the six Gray-Scott scenarios differ by two orders
of magnitude in where they keep their variance, so a balanced draw across
regimes matters more than a random one within them.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_HC_PATH = Path(__file__).resolve().parent / "harmonic_content.py"
_SRC = Path(__file__).resolve().parents[1] / "src"


def _load_hc():
    spec = importlib.util.spec_from_file_location("harmonic_content", _HC_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["harmonic_content"] = module
    spec.loader.exec_module(module)
    return module


hc = _load_hc()

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# the repo's own reductions -- imported, not reimplemented, so this produces the
# same arrays `litefno preprocess` would
from litefno.preprocess import downsample_spatial  # noqa: E402


GRAY_SCOTT = {
    "repo": "polymathic-ai/gray_scott_reaction_diffusion",
    "name": "gray_scott_reaction_diffusion",
    "fields": ("t0_fields/A", "t0_fields/B"),
    "regimes": {
        "bubbles": "bubbles_F_0.098_k_0.057",
        "gliders": "gliders_F_0.014_k_0.054",
        "maze": "maze_F_0.029_k_0.057",
        "spirals": "spirals_F_0.018_k_0.051",
        "spots": "spots_F_0.03_k_0.062",
        "worms": "worms_F_0.058_k_0.065",
    },
}


def fetch_split(spec: dict, split: str, per_scenario: int, max_steps: int,
                factor: int, tries: int = 5):
    """One split, balanced across regimes, reduced on arrival."""
    arrays, manifest = [], []
    for regime, stem in spec["regimes"].items():
        path = f"data/{split}/{spec['name']}_{stem}.hdf5"
        for attempt in range(tries):
            try:
                h5 = hc.open_remote(spec["repo"], path, block_size=2 ** 22)
                dsets = [h5[f] for f in spec["fields"]]
                n_avail = dsets[0].shape[0]
                take = min(per_scenario, n_avail)
                per_traj = []
                for i in range(take):
                    # contiguous max_steps prefix: one small range request per
                    # (trajectory, field) instead of the whole 65 MB trajectory
                    chans = [np.asarray(d[i, :max_steps]) for d in dsets]
                    per_traj.append(np.stack(chans, axis=-1))
                h5.close()
                block = np.stack(per_traj)          # (take, T, H, W, fields)
                arrays.append(downsample_spatial(block, factor))
                manifest.append({"regime": regime, "file": path,
                                 "trajectories": list(range(take)),
                                 "available": int(n_avail)})
                print(f"    {split:>5s} {regime:>8s}: {take} traj -> "
                      f"{arrays[-1].shape}", flush=True)
                break
            except Exception as exc:                 # transient HTTP/SSL
                print(f"    retry {attempt + 1} {regime}: {type(exc).__name__}",
                      flush=True)
        else:
            raise RuntimeError(f"could not read {path}")
    return np.concatenate(arrays).astype(np.float32), manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/processed/gray_scott_streamed"))
    ap.add_argument("--max-steps", type=int, default=60,
                    help="configs/datasets/*.yaml: max_steps")
    ap.add_argument("--downsample", type=int, default=4,
                    help="configs/datasets/*.yaml: downsample_factor")
    ap.add_argument("--train-per-scenario", type=int, default=12)
    ap.add_argument("--valid-per-scenario", type=int, default=4)
    ap.add_argument("--test-per-scenario", type=int, default=4)
    args = ap.parse_args()

    import h5py

    ca = hc._certifi_path()
    if ca:
        import os
        os.environ.setdefault("SSL_CERT_FILE", ca)

    spec = GRAY_SCOTT
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": args.train_per_scenario, "valid": args.valid_per_scenario,
              "test": args.test_per_scenario}
    manifest = {"dataset": spec["name"], "source": spec["repo"],
                "max_steps": args.max_steps, "downsample_factor": args.downsample,
                "fields": list(spec["fields"]), "splits": {}}

    for split, per in counts.items():
        print(f"  [{split}] {per} trajectories per regime", flush=True)
        t0 = time.time()
        data, files = fetch_split(spec, split, per, args.max_steps, args.downsample)
        out = args.out_dir / f"{split}.h5"
        with h5py.File(out, "w") as f:
            f.create_dataset("data", data=data, compression="gzip",
                             compression_opts=4)
        manifest["splits"][split] = {
            "shape": list(data.shape), "files": files,
            "bytes": out.stat().st_size,
        }
        print(f"    wrote {out} {data.shape} "
              f"({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s",
              flush=True)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    sys.exit(main())
