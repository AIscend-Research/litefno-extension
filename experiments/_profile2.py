import time, os
from pathlib import Path
import torch
import litefno
from litefno.data import DatasetConfig, H5SequenceDataset
from litefno.models import LiteFNO
from litefno.train import flatten_time, unflatten_time
os.chdir(Path(__file__).resolve().parents[1])

d = torch.device("mps")
ds = H5SequenceDataset(DatasetConfig(path=Path("data/processed/gray_scott_proxy/train.h5"),
                                     dataset_key="data", input_steps=1, output_steps=1,
                                     stride=1, cache="memory"))
for BS in (512, 1024, 2808):
    dl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=True, num_workers=0)
    m = LiteFNO(2, 2, width=64, rank=32, layers=8).to(d)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    # 1 warmup epoch
    for x, y in dl:
        x = x.to(d); y = y.to(d); xf = flatten_time(x)
        opt.zero_grad(); out = m(xf); out = unflatten_time(out, 1, 2)
        loss = torch.nn.functional.mse_loss(out, y); loss.backward(); opt.step(); loss.item()
    torch.mps.synchronize()
    t0 = time.perf_counter(); nb = 0
    for x, y in dl:
        x = x.to(d); y = y.to(d); xf = flatten_time(x)
        opt.zero_grad(); out = m(xf); out = unflatten_time(out, 1, 2)
        loss = torch.nn.functional.mse_loss(out, y); loss.backward(); opt.step(); loss.item()
        nb += 1
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    print(f"BS={BS}: {dt:.2f}s/epoch over {nb} batches ({dt/nb*1000:.0f} ms/batch)")
