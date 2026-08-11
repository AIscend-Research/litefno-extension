import time, os
from pathlib import Path
import torch
import litefno
from litefno.data import DatasetConfig, H5SequenceDataset
from litefno.models import LiteFNO
from litefno.train import flatten_time, unflatten_time
os.chdir(Path(__file__).resolve().parents[1])

ds = H5SequenceDataset(DatasetConfig(path=Path("data/processed/gray_scott_proxy/train.h5"),
                                     dataset_key="data", input_steps=1, output_steps=1,
                                     stride=1, cache="memory"))
print("dataset windows:", len(ds))

# (1) dataloader iteration cost (no compute)
for bs in (256, 1024):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    t0 = time.perf_counter(); n = 0
    for x, y in dl:
        n += x.shape[0]
    print(f"dataloader-only bs={bs}: {time.perf_counter()-t0:.2f}s for {n} samples")

# (2) compute-only: fixed batch, repeated fwd+bwd
for dev in ("cpu", "mps"):
    d = torch.device(dev)
    m = LiteFNO(2, 2, width=64, rank=32, layers=8).to(d)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = torch.randn(512, 2, 32, 32, device=d)
    y = torch.randn(512, 2, 32, 32, device=d)
    # warmup
    for _ in range(3):
        opt.zero_grad(); out = m(x); loss = ((out - y) ** 2).mean(); loss.backward(); opt.step()
    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        opt.zero_grad(); out = m(x); loss = ((out - y) ** 2).mean(); loss.backward(); opt.step()
    if dev == "mps": torch.mps.synchronize()
    print(f"compute-only {dev}: {(time.perf_counter()-t0)/20*1000:.1f} ms / 512-batch step")
