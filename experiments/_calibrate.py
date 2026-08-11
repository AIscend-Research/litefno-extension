import json, time, tempfile, os
from pathlib import Path
import yaml
import litefno
print("using litefno from", litefno.__file__)
from litefno.train import run_training

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

cfg = {
    "model": {"name": "litefno", "layers": 8, "width": 64, "rank": 32},
    "dataset": {
        "name": "gray_scott_proxy",
        "processed_dir": "data/processed/gray_scott_proxy",
        "splits": ["train", "valid", "test"],
        "dataset_key": "data", "input_steps": 1, "output_steps": 1,
        "fields": 2, "stride": 1, "cache": "memory",
    },
    "training": {
        "epochs": 5, "batch_size": 256, "lr": 1e-3, "lr_step": 100, "lr_gamma": 0.5,
        "device": "mps", "seed": 1337, "eval_every": 1,
        "eval_splits": ["train", "valid"], "eval_windows": [],
        "test_at_end": True, "amp": False, "pin_memory": False,
    },
    "logging": {"metrics_path": "experiments/_calib.jsonl"},
}
p = ROOT / "experiments" / "_calib.yaml"
p.write_text(yaml.safe_dump(cfg))
Path("experiments/_calib.jsonl").unlink(missing_ok=True)
t0 = time.perf_counter()
run_training(p)
dt = time.perf_counter() - t0
rows = [json.loads(l) for l in Path("experiments/_calib.jsonl").read_text().splitlines()]
print(f"\nTOTAL {dt:.1f}s for 5 epochs -> {dt/5:.2f}s/epoch")
for r in rows:
    print(r)
