"""Deployability measurement: size, FLOPs, latency, memory.

Why this module exists rather than a few ``time.perf_counter`` calls
-------------------------------------------------------------------
"Low-rank" is a claim about parameters. Deployability is a claim about wall
clock, resident memory and disk. The repository's headline architecture is
CP-factorized, which makes the first claim true by construction and says nothing
about the second, and the two come apart here by a factor of 280 in one
direction and 1.6 in the other. So the numbers have to be measured rather than
inferred, and the measurement has to be hard to fool.

Three things make timing results wrong more often than not, and each is handled
explicitly rather than hoped away:

* **Warmup.** The first forward pass allocates workspaces, picks FFT plans and
  faults in pages. ``latency`` discards a warmup phase and reports a median over
  repeats, plus the interquartile range, because a mean over a lazily-initialised
  first call is a measurement of allocation.
* **Threads.** A model that parallelises well looks fast on a 10-core laptop and
  ordinary on a 4-vCPU notebook VM. Every timing here takes an explicit thread
  count and restores the previous setting, so a "faster model" is never just a
  better-parallelising one.
* **Batch.** Costs that do not scale with batch size -- and the CP weight
  reconstruction is exactly one -- are invisible in throughput benchmarks and
  dominant at batch 1. Both are reported.

The FLOP model is analytic and closed form
------------------------------------------
``analytic_flops`` derives the count from the architecture rather than tracing
it, for a reason that matters: ``torch.utils.flop_counter`` does not count FFTs,
and an FNO is mostly FFT. A traced count is therefore an undercount of exactly
the term that distinguishes these architectures. The analytic model is checked
against the tracer on the operations the tracer does cover (``flop_audit``), so
it is pinned where pinning is possible and explicit where it is not.

Conventions, stated because FLOP counts are not comparable without them:

* one multiply-accumulate is 2 flops;
* one complex multiply-accumulate is 8 real flops (4 real multiplies, 4 adds);
* a real 2-D FFT of an ``H x W`` field is ``2.5 * H * W * log2(H * W)`` flops,
  the usual halving of the ``5 N log2 N`` complex-FFT figure (Cooley-Tukey).

No network access, no data files.
"""
from __future__ import annotations

import platform
import statistics
import time
from math import log2
from typing import Optional

# `resource` is POSIX-only. It backs exactly one function here
# (`peak_rss_bytes`), so importing it at module scope made the whole module --
# and with it every FLOP and latency helper, none of which need it -- fail to
# import on Windows. Windows gets the equivalent number from psapi instead.
try:
    import resource
except ImportError:  # Windows
    resource = None

import torch
from torch import nn

# 4 real multiplies + 4 real adds for one complex multiply-accumulate
FLOPS_PER_COMPLEX_MAC = 8
# 2 flops per real multiply-accumulate
FLOPS_PER_MAC = 2


# --------------------------------------------------------------------------
# size
# --------------------------------------------------------------------------


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def state_dict_bytes(model: nn.Module) -> int:
    """Bytes a checkpoint of this model occupies.

    Taken from ``state_dict()`` rather than from ``parameters()``, because
    persistent buffers ship with the checkpoint and have to be downloaded --
    a model storing a dense mask or a precomputed propagator is that much bigger
    for whoever fetches it, whether or not the number is called a parameter.

    Non-persistent buffers are excluded, which is what makes the fused model
    below cost nothing extra on disk: it is a runtime transform of a checkpoint
    that still contains only the CP factors.
    """
    return sum(t.numel() * t.element_size()
               for t in model.state_dict().values()
               if isinstance(t, torch.Tensor))


def resident_bytes(model: nn.Module) -> int:
    """Bytes the model's tensors occupy in memory, persistent or not.

    Differs from ``state_dict_bytes`` exactly by the caches a model materialises
    at runtime. That difference is the price of fusing, and it is a price paid
    in RAM rather than on disk.
    """
    seen: dict[int, int] = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor is not None:
            seen[id(tensor)] = tensor.numel() * tensor.element_size()
    return sum(seen.values())


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    ``ru_maxrss`` is in kilobytes on Linux and in bytes on macOS. Getting this
    wrong is a factor of 1024, which is large enough to turn a model that fits a
    16 GB notebook into one that does not, so the unit is resolved by platform
    rather than assumed.

    Windows has no ``getrusage``; ``PeakWorkingSetSize`` from psapi is the same
    quantity (high-water mark of resident pages) and is already in bytes.
    """
    if resource is not None:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(raw) if platform.system() == "Darwin" else int(raw) * 1024

    import ctypes
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    # Signatures must be declared: HANDLE is 64-bit on win64 and ctypes would
    # otherwise marshal the -1 pseudo-handle as a 32-bit int.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]

    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


# --------------------------------------------------------------------------
# the closed-form FLOP model
# --------------------------------------------------------------------------


def fft_flops(height: int, width: int, n_fields: int = 1) -> float:
    """Real 2-D FFT cost under the ``2.5 N log2 N`` convention."""
    n = height * width
    if n <= 1:
        return 0.0
    return 2.5 * n * log2(n) * n_fields


def conv_flops(in_channels: int, out_channels: int, height: int, width: int,
               kernel: int = 1, bias: bool = True) -> float:
    """Dense 2-D convolution, 'same' spatial size."""
    macs = in_channels * out_channels * kernel * kernel * height * width
    total = FLOPS_PER_MAC * macs
    if bias:
        total += out_channels * height * width
    return float(total)


def spectral_contraction_flops(in_channels: int, out_channels: int,
                               modes1: int, modes2: int,
                               mac: int = FLOPS_PER_COMPLEX_MAC) -> float:
    """``bimn,iomn->bomn`` over the retained block, per sample."""
    return float(mac * in_channels * out_channels * modes1 * modes2)


def cp_reconstruction_flops(in_channels: int, out_channels: int, modes1: int,
                            modes2: int, rank: int,
                            mac: int = FLOPS_PER_COMPLEX_MAC) -> float:
    """Cost of rebuilding the dense spectral weight from its CP factors.

    ``einsum("r,ir,or,ar,br->ioab")``: one pass over the full ``(in, out, m1,
    m2)`` output tensor per rank component, plus two much smaller pairwise
    factor contractions. The full-tensor term dominates and is the one that
    matters, because it is *independent of batch size* -- the same work is
    repeated for a batch of 1 and a batch of 64.

    This is the term the word "low-rank" hides. CP makes the weight small to
    store and leaves it exactly as expensive to use, plus the cost of rebuilding
    it every time it is used.

    The two small contractions contribute ``rank * (in*out + m1*m2)`` MACs,
    around 0.4% of the total at the sizes used here. They are kept in the model
    because they are really performed; ``torch``'s tracer does not emit them as
    matmuls, which is the whole of the residual in ``flop_audit``.
    """
    per_component = in_channels * out_channels + modes1 * modes2
    full = in_channels * out_channels * modes1 * modes2
    return float(mac * rank * (per_component + full))


def analytic_flops(model: nn.Module, size: int, batch: int = 1,
                   convention: str = "real") -> dict:
    """Closed-form forward FLOPs, broken down by where they go.

    The breakdown is the point. A single total cannot say why a model with 280x
    fewer parameters is slower, and the split into ``batched`` and ``fixed``
    can: ``fixed`` is the work done once per forward call regardless of how many
    samples are in it.

    ``convention`` selects how the count is taken:

    ``"real"``      the honest one. A complex multiply-accumulate is 8 real
                    flops, bias adds are counted, FFTs are counted.
    ``"counter"``   ``torch.utils.flop_counter``'s. A complex MAC counts 2 (the
                    tracer sees element counts, not the 4 real multiplies each
                    one costs), bias adds and FFTs are not counted at all.

    The second exists only so the closed form can be checked against the tracer
    on identical terms. Everything reported by the experiment uses ``"real"``.
    """
    from litefno.models.fno_s import FNOS, SpectralConv2d
    from litefno.models.harmonic import CPSpectralConv2d, HarmonicLiteFNO
    from litefno.models.litefno import LiteFNO

    if convention not in ("real", "counter"):
        raise ValueError(f"unknown convention: {convention}")
    counter_mode = convention == "counter"
    mac = FLOPS_PER_MAC if counter_mode else FLOPS_PER_COMPLEX_MAC

    parts: dict[str, float] = {"conv": 0.0, "fft": 0.0, "spectral": 0.0}
    fixed = 0.0

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            parts["conv"] += batch * conv_flops(
                module.in_channels, module.out_channels, size, size,
                kernel=module.kernel_size[0],
                bias=module.bias is not None and not counter_mode)
        elif isinstance(module, (SpectralConv2d, CPSpectralConv2d)):
            if not counter_mode:
                parts["fft"] += batch * fft_flops(
                    size, size, module.in_channels + module.out_channels)
            parts["spectral"] += batch * spectral_contraction_flops(
                module.in_channels, module.out_channels,
                module.modes1, module.modes2, mac=mac)
            if isinstance(module, CPSpectralConv2d) and not getattr(
                    module, "is_fused", False):
                fixed += cp_reconstruction_flops(
                    module.in_channels, module.out_channels, module.modes1,
                    module.modes2, module.rank, mac=mac)

    parts["cp_reconstruction"] = fixed
    batched = parts["conv"] + parts["fft"] + parts["spectral"]
    parts["batched"] = batched
    parts["fixed"] = fixed
    parts["total"] = batched + fixed
    parts["fixed_share"] = fixed / parts["total"] if parts["total"] else 0.0
    # kept so a caller can assert the architecture was recognised at all
    parts["family"] = ("cp" if isinstance(model, HarmonicLiteFNO) else
                       "dense_spectral" if isinstance(model, FNOS) else
                       "cnn" if isinstance(model, LiteFNO) else "unknown")
    return parts


def counted_flops(model: nn.Module, example: torch.Tensor) -> Optional[float]:
    """Traced FLOPs from ``torch.utils.flop_counter``, or ``None``.

    Counts matmul- and convolution-shaped operations only. FFTs are *not*
    counted, which is why this is a cross-check on part of the analytic model
    rather than the measurement itself.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:                                    # pragma: no cover
        return None
    model.eval()
    counter = FlopCounterMode(display=False)
    with counter, torch.inference_mode():
        model(example)
    return float(counter.get_total_flops())


def flop_audit(model: nn.Module, size: int, batch: int = 1) -> dict:
    """Analytic model against the tracer, on identical terms.

    The tracer sees convolutions and matmuls -- the spectral contraction and the
    CP reconstruction are einsums, hence matmuls -- and does not see FFTs or
    bias adds, so the comparison is made in the tracer's own convention. Any
    disagreement then means the closed form has a *shape* wrong, which is what
    the check is for.
    """
    example = torch.zeros(batch, _in_channels(model), size, size)
    real = analytic_flops(model, size, batch=batch)
    comparable = analytic_flops(model, size, batch=batch,
                                convention="counter")["total"]
    counted = counted_flops(model, example)
    rel_error = (abs(counted - comparable) / comparable
                 if counted is not None and comparable else float("nan"))
    return {"analytic_total": real["total"],
            "analytic_counter_convention": comparable,
            "counted": counted,
            "rel_error": rel_error,
            "fft_share": real["fft"] / real["total"] if real["total"] else 0.0,
            "fixed_share": real["fixed_share"]}


def _in_channels(model: nn.Module) -> int:
    proj = getattr(model, "input_proj", None)
    if isinstance(proj, nn.Conv2d):
        return proj.in_channels
    for module in model.modules():                         # pragma: no cover
        if isinstance(module, nn.Conv2d):
            return module.in_channels
    raise ValueError("could not infer input channels")


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------


def latency(model: nn.Module, example: torch.Tensor, repeats: int = 30,
            warmup: int = 5, threads: Optional[int] = None) -> dict:
    """Median forward latency in milliseconds, with warmup discarded.

    Reports the median rather than the mean because a single scheduler
    interruption moves a mean and not a median, and the IQR alongside it so a
    noisy measurement is visible rather than averaged into confidence.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    previous = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(example)
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(example)
                samples.append((time.perf_counter() - start) * 1e3)
    finally:
        torch.set_num_threads(previous)

    samples.sort()
    median = statistics.median(samples)
    batch = example.shape[0]
    return {"ms": median,
            "ms_iqr": (samples[int(0.75 * (len(samples) - 1))]
                       - samples[int(0.25 * (len(samples) - 1))]),
            "ms_min": samples[0],
            "ms_per_sample": median / batch,
            "samples_per_s": batch / (median / 1e3),
            "threads": threads if threads is not None else previous,
            "repeats": repeats}


# --------------------------------------------------------------------------
# the fix: fold the CP factors once, at eval time
# --------------------------------------------------------------------------


def fuse_spectral_weights(model: nn.Module) -> int:
    """Precompute each CP layer's dense weight; return how many were fused.

    The reconstruction is a function of the parameters alone, so at inference it
    is loop-invariant and can be hoisted out of the forward pass. Outputs are
    unchanged -- bitwise, not approximately, since it is the same tensor
    computed once instead of every call.

    This trades memory for time in the direction deployment wants. The dense
    weight is materialised inside every forward call anyway, so peak memory is
    barely changed -- only the lifetime of one allocation grows -- and the cache
    is registered non-persistently, so a checkpoint still contains just the CP
    factors and the download stays small.

    **Inference only.** The fused tensor is detached, so the CP factors would
    receive no gradient and training a fused model would silently update
    nothing. ``unfuse_spectral_weights`` restores the trainable path, and
    callers that train must use it.
    """
    from litefno.models.harmonic import CPSpectralConv2d

    fused = 0
    for module in model.modules():
        if isinstance(module, CPSpectralConv2d) and not getattr(
                module, "is_fused", False):
            with torch.no_grad():
                dense = module.weight().detach().clone()
            module.register_buffer("_fused_weight", dense, persistent=False)
            module._cp_weight = module.weight
            module.weight = lambda _m=module: _m._fused_weight
            module.is_fused = True
            fused += 1
    return fused


def unfuse_spectral_weights(model: nn.Module) -> int:
    """Undo ``fuse_spectral_weights``; return how many layers were restored."""
    from litefno.models.harmonic import CPSpectralConv2d

    restored = 0
    for module in model.modules():
        if isinstance(module, CPSpectralConv2d) and getattr(
                module, "is_fused", False):
            module.weight = module._cp_weight
            del module._cp_weight
            if hasattr(module, "_fused_weight"):
                del module._fused_weight
            module.is_fused = False
            restored += 1
    return restored


# --------------------------------------------------------------------------
# rank correlation
# --------------------------------------------------------------------------


def _ranks(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0            # average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman rank correlation, ties averaged.

    Written out rather than imported because the question this extension asks --
    "does parameter count *rank* models the way latency does" -- is a rank
    question, and a Pearson correlation on quantities spanning three orders of
    magnitude would answer a different one.
    """
    if len(a) != len(b):
        raise ValueError("length mismatch")
    if len(a) < 2:
        raise ValueError("need at least two points")
    ra, rb = _ranks(list(a)), _ranks(list(b))
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va == 0 or vb == 0:
        return float("nan")
    return cov / (va * vb) ** 0.5
