# Is the low-rank operator actually deployable? (ext25, H6)

`scripts/deployability.py`, `src/litefno/bench.py`

Every low-rank paper reports a parameter count, and the reader is invited to
read it as a deployability claim. This extension states that invitation as a
hypothesis and measures it.

**H6: parameter count predicts deployability — across this model family, a model
with fewer parameters is smaller on disk, cheaper in FLOPs, and faster to run on
the hardware a low-resource scientist actually has.**

The disk half is true by construction and uninteresting. The rest is false, by a
wide margin, and the reason is specific, closed-form, and fixable.

## The headline

```
            arm       params   disk MB   FLOPs b1    ms b1
  fno_s-w64-m16   16,810,818     64.13   1.615e+08    22.45
      cp-w64-r8       43,906      0.21   7.006e+08    30.87
```

The CP-factorized model has **383x fewer parameters**, a **306x smaller
checkpoint**, and it is **37% slower**. Across the nine-arm family the rank
correlation between parameter count and batch-1 latency is **0.067** — parameter
count carries essentially no information about how long a forward pass takes.

## Why: the cost that low-rank factorization adds

`CPSpectralConv2d.forward` calls `self.weight()`, which contracts the CP factors
back into the dense `(in, out, m1, m2)` spectral weight — every forward pass,
every time. In closed form that is

```
8 * rank * in * out * m1 * m2   flops
```

and it **does not depend on the batch size**. CP makes the weight cheap to
*store* and leaves it exactly as expensive to *use*, plus the cost of rebuilding
it. At batch 1 — a scientist stepping one field forward in a notebook, which is
the setting the task names — that reconstruction is the majority of the model's
arithmetic:

```
        arm    fixed share of batch-1 FLOPs
  cp-w32-r2                          41.5%
  cp-w32-r8                          73.9%
 cp-w32-r32                          91.9%
  cp-w64-r8                          76.9%
```

Raising the rank to buy accuracy costs compute **linearly in rank** while buying
parameters far more slowly, so the two knobs the phrase "low-rank" bundles
together point in opposite directions.

## 1. The FLOP model is closed form, and pinned

Nothing above is worth reading if the FLOP counts are guesses, so they are
derived analytically per layer and checked against `torch`'s own tracer:

```
            arm     analytic       tracer    rel err  FFT share  fixed share
    cnn-w32-r16   2.7525e+07   2.7525e+07    0.0e+00       0.0%         0.0%
    cnn-w64-r32   2.1863e+08   2.1863e+08    0.0e+00       0.0%         0.0%
   fno_s-w32-m8   9.1750e+06   9.1750e+06    0.0e+00      37.5%         0.0%
  fno_s-w32-m16   1.0748e+07   1.0748e+07    0.0e+00      27.6%         0.0%
  fno_s-w64-m16   8.4410e+07   8.4410e+07    0.0e+00      16.2%         0.0%
      cp-w32-r2   1.4963e+07   1.4942e+07    1.4e-03      16.1%        41.5%
      cp-w32-r8   2.7607e+07   2.7525e+07    3.0e-03       7.2%        73.9%
     cp-w32-r32   7.8184e+07   7.7857e+07    4.2e-03       2.2%        91.9%
      cp-w64-r8   2.1919e+08   2.1863e+08    2.5e-03       3.7%        76.9%
```

Exact — not approximately, *exactly* — on the two architectures whose operations
the tracer fully covers, and within 0.4% on CP, where the residual is the two
small factor contractions `torch` does not emit as matmuls.

The comparison is made in the tracer's own convention, and that convention is
itself a finding worth stating: `FlopCounterMode` counts a complex
multiply-accumulate as 2 flops rather than the 4 real multiplies and 4 adds it
costs, and **does not count FFTs at all**. An FNO is mostly FFT — 16-38% of the
honest count for the dense arms here — so a traced FLOP number for a Fourier
operator undercounts exactly the term that distinguishes it. Everything this
extension reports uses the honest convention (complex MAC = 8 flops, FFT at
`2.5 N log2 N`); the tracer convention exists only so the closed form can be
checked against something independent.

## 2. H6: does parameter count rank models the way latency does?

```
  batch  threads   rho(params, ms)   rho(FLOPs, ms)   rho(disk, ms)
      1        1             0.067            0.617           0.067
      1        4             0.167            0.567           0.167
     16        1             0.217            0.883           0.217
     16        4             0.433            0.800           0.433
```

**H6 is refuted.** Parameter count is nearly uninformative about latency in the
interactive regime (ρ = 0.067) and never gets above 0.433 even at batch 16 where
the fixed cost has amortised away. FLOPs rank models far better — 0.57 to 0.88 —
which is the useful positive result: *report FLOPs, not parameters, if the claim
is about cost.* Disk size ranks identically to parameters, as it must, since
here it is four bytes times the parameter count.

Note that ρ(FLOPs, latency) is not 1.0 either. FLOPs are the better proxy, not a
good one; §3 measures how much better.

## 3. The fix: fold the CP factors once, at eval time

The reconstruction is a function of the parameters alone, so at inference it is
loop-invariant and can be hoisted out of the forward pass. `fuse_spectral_weights`
does that. The speedup is **predicted from the closed form before it is
measured** — fusing removes exactly the batch-independent term, so the expected
gain is `total(B) / (total(B) - fixed)`:

```
        arm  batch   fixed  predicted  measured  ms before  ms after   |diff|  GF/s cut  GF/s kept
  cp-w32-r8      1   73.9%     x 3.84    x 1.40       3.97      2.84  0.0e+00      59.5        8.4
  cp-w32-r8     64    4.2%     x 1.04    x 1.02      70.51     69.08  0.0e+00      47.4       22.0
 cp-w32-r32      1   91.9%    x 12.35    x 1.79       5.30      2.96  0.0e+00     115.3        8.0
 cp-w32-r32     64   15.1%     x 1.18    x 1.03      70.70     68.35  0.0e+00     115.0       22.2
  cp-w64-r8      1   76.9%     x 4.34    x 1.53      30.43     19.83  0.0e+00      50.8        8.1
  cp-w64-r8     64    5.0%     x 1.05    x 1.04     379.43    364.83  0.0e+00      36.9       28.3
```

**What holds.** The gain is real and free: 1.40-1.79x at batch 1, outputs
**bitwise identical** (`|diff|` is exactly 0.0, not a tolerance — it is the same
tensor computed once instead of once per call), parameter count unchanged, and
the checkpoint unchanged, because the cache is registered as a non-persistent
buffer. It costs RAM instead: +8 MB for the width-32 arms, +64 MB for width-64.
The decay of the speedup with batch size — down to ~1.02 at batch 64 — is the
signature the closed form predicts, and it confirms the diagnosis: the cost being
removed is the batch-independent one.

**What does not hold: the magnitude.** The closed form predicts 12.35x where 1.79x
is measured. The last two columns say why. Divide the removed FLOPs by the time
actually saved and that work was running at **99-115 GFLOP/s**; what remains runs
at **8-22 GFLOP/s**. The reconstruction is a dense einsum near peak arithmetic
throughput, while the rest is FFT-dominated and memory-bound. Equal FLOPs are not
equal time, by an order of magnitude, *within a single model*.

That is the honest summary of the whole extension in one line: **FLOPs rank
models correctly and price them badly**, and parameters do neither.

## 4. Resolution scaling: which model is cheapest depends on the grid

```
            arm       32       64       96      128  exponent
    cnn-w32-r16     0.40     0.97     1.46     2.54      1.29
    cnn-w64-r32     1.26     2.86     6.89    18.22      1.86
   fno_s-w32-m8     1.50     2.69     4.61     7.40      1.13
  fno_s-w32-m16     3.97     5.22     7.18    10.45      0.67
  fno_s-w64-m16    22.45    27.15    36.67    51.50      0.57
      cp-w32-r2     4.14     5.29     7.20    10.15      0.62
      cp-w32-r8     4.01     5.19     7.17    10.00      0.64
     cp-w32-r32     5.09     6.61     9.12    12.70      0.64
      cp-w64-r8    30.65    35.95    46.29    59.24      0.46
```

The CNN approaches the area law (`N^1.86` for the wide one); every spectral model
sits at `N^0.46-0.67`, far below it, because a fixed mode count makes the
contraction constant in `N` and the fixed CP cost is constant in everything. So
**the ranking inverts with resolution**: `cnn-w64-r32` is 3.2x faster than
`fno_s-w32-m16` at 32x32 and 1.7x slower at 128x128. A deployability claim
measured at one resolution does not transfer to another, and this repository
trains at 32x32.

## 5. The envelope

```
              arm  disk MB   ms/step   train h     verdict
      cnn-w32-r16     0.05      0.40       4.3  deployable
      cnn-w64-r32     0.41      1.17      20.2  fails: trains_in_session
     fno_s-w32-m8     2.02      1.45       5.0  deployable
    fno_s-w32-m16     8.02      4.04       5.6  deployable
    fno_s-w64-m16    64.13     22.45      26.0  fails: trains_in_session
        cp-w32-r2     0.02      4.45       5.4  deployable
  cp-w32-r2+fused     0.02      2.86       5.4  deployable
        cp-w32-r8     0.04      4.03       5.5  deployable
  cp-w32-r8+fused     0.04      3.20       5.4  deployable
       cp-w32-r32     0.11      5.46       5.7  deployable
 cp-w32-r32+fused     0.11      2.88       5.8  deployable
        cp-w64-r8     0.21    30.87      25.4  fails: trains_in_session
  cp-w64-r8+fused     0.21    19.46      25.9  fails: trains_in_session
```

Budget: ≤ 100 MB on disk, ≤ 50 ms per interactive step, ≤ 12 h to train 200
epochs × 40,000 samples at batch 64 on 4 CPU threads.

**Every width-64 configuration in this repository — including the two shapes its
own protocol trains — misses a 12-hour CPU session by a factor of two.** Nothing
fails on disk or on interactive latency; the binding constraint is training
time, and it is binding for exactly the models the repo treats as its headline.
The width-32 variants all fit, at 4-6 hours.

The budget is the script's assumption, set by flags, not a measured fact about
any hosted notebook service — free-tier quotas change and a number baked into a
repository would be wrong within a year. What is defensible is the shape: a few
CPU cores, no guaranteed accelerator, a session that ends, and a reader who wants
one field stepped forward without waiting.

## What this rules out, and what it does not

Ruled out: reading a parameter count as a cost claim, for this architecture
family. ρ = 0.067 at batch 1.

Ruled out: that CP factorization is free at inference. It costs
`8·rank·in·out·m1·m2` flops per layer per call, it is 74-92% of the batch-1
budget at the ranks used here, and it grows linearly in the rank you raise to
recover accuracy.

Ruled out: that fusing is a tradeoff. Outputs are bitwise identical and the
checkpoint is unchanged; the only price is resident memory.

Not ruled out: that CP is the right choice anyway. This extension measures cost,
not the accuracy/cost frontier — see the limits below.

Not ruled out: that a different CP implementation avoids the reconstruction
entirely. Contracting the input against the factors in sequence, rather than
rebuilding the dense weight, should be asymptotically cheaper still, and is not
implemented here.

## Honest limits

- **No accuracy is measured.** "Deployable" here means "fits the stated budget",
  not "fits the budget and is as good". The Pareto claim a reader wants —
  accuracy per millisecond — needs training runs this script deliberately does
  not do, and the arms it benchmarks are mostly shapes the repo has never
  trained. Costs are comparable across arms; quality is not established for any
  of them here.
- **One machine.** All timings are from a single Apple-silicon laptop CPU
  (`results/extensions/ext25_host.json` records the host). Absolute
  milliseconds will not transfer. The rank correlations, the FLOP ratios, the
  batch-independence of the CP term and the bitwise equality of fusion will.
- **CPU only.** No GPU numbers. On an accelerator the arithmetic-intensity story
  changes — the dense reconstruction is exactly the kind of work a GPU is good
  at — and the fusion gain would likely be smaller.
- **The training-time projection is an extrapolation.** It is one measured
  optimiser step scaled by an assumed 200 epochs × 40,000 samples; it ignores
  data loading, evaluation and checkpointing, so it is a lower bound on wall
  clock. The samples-per-epoch figure is a flag, and the projection scales
  linearly in it.
- **Nine arms.** A Spearman correlation over nine points has wide error bars.
  The claim rests on the size of the effect — 383x in parameters against a 37%
  latency loss in the wrong direction — rather than on the correlation being
  precisely 0.067.
- **`thop` is not used**, and neither is any external profiler. The counts are
  the closed form plus `torch`'s built-in tracer as a cross-check.

## Reproducing

```bash
python3 scripts/deployability.py           # ~2.5 minutes on CPU
python3 scripts/deployability.py --quick   # plumbing check only
```

Outputs: `results/extensions/ext25_flop_audit.csv`, `ext25_family.csv`,
`ext25_rank.csv`, `ext25_fusion.csv`, `ext25_resolution.csv`,
`ext25_envelope.csv`, `ext25_host.json`, and
`figures/extensions/ext25_deployability.png`.

Budget flags: `--max-disk-mb`, `--max-interactive-ms`, `--max-session-hours`,
`--samples-per-epoch`, `--train-epochs`.
