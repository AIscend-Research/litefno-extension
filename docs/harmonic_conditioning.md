# Harmonic conditioning: a learnable bias on the Turing shells (ext15)

Board task: *"Add harmonic conditioning: modify LITEFNO's spectral factorization
to add a harmonic-mode bias."*

Two separable pieces. `src/litefno/models/harmonic.py` supplies a genuinely
spectral CP-factorized convolution -- `rfft2`, complex weights, mode truncation,
the weight stored as a rank vector plus one complex factor matrix per tensor mode
-- and then layers a learnable complex bias `b_k` on designated harmonic shells
on top of it. `scripts/harmonic_ab.py` is the measurement.

The factorization came first because the repo did not have one:
`models/litefno.py` is a low-rank CNN with no FFT in it, which
[reproducibility_findings.md](reproducibility_findings.md) records and
`freebieA_repro_audit.json` pins. The only CP-factorized spectral model in the
project arrived via neuralop, inside a notebook.

## Shells, not modes, and why the band is not a free parameter

The bias is added on radial shells -- a fundamental wavenumber and its integer
multiples -- rather than on individual `(kx, ky)` modes, because a Turing pattern
selects a *wavelength*, not an orientation, so its energy is spread around a ring.

The choice of ring is inherited, not tuned. ext10 measured that maze and spots
keep ~99% of their spatial variance *above* mode 8, concentrated in a narrow band
at the Turing wavelength near mode 13-16 natively, which is mode 3-4 after the
pipeline's 4x downsample. That is what the default `--fundamental 4.0` refers to.
Picking the band from the data the intervention is then tested on would make the
result unfalsifiable; picking it from a committed prior measurement does not.

## The two arms

    control       CP-factorized spectral LiteFNO
    conditioned   the same, plus the learnable complex bias on the harmonic shells

They share seeds, initialisation, data order, optimiser and schedule, and at
initialisation they are **bit-identical** -- the bias starts at zero, and a test
in `tests/test_harmonic.py` pins that the two produce equal output before any
training. The bias adds a small, fixed number of parameters -- 220 of them, or
**+2.94%**, at the script's default width of 32 (measured in ext30; the original
commit message said 1-2%, which holds at larger widths) -- so a difference cannot
be attributed to model size.

## The prediction, which is differential rather than uniform

The readout that matters is **per regime**, not the aggregate, and the script
says so before any run. The conditioning targets a specific band that only two
regimes occupy, so the prediction has a shape: harmonic conditioning on the
Turing shells should help **maze and spots** and do close to nothing for
**spirals and gliders**, which are low-wavenumber.

A *uniform* improvement would be evidence **against** the mechanism, not for it
-- it would suggest the extra parameters are helping generically rather than by
supplying structure at the wavenumbers that need it. An aggregate VRMSE would
therefore be the wrong verdict and would probably show nothing either way.

The repo's own prior on the effect size is low, and recording that in advance is
the point of stating it, and it was borne out -- see
[harmonic_verdict.md](harmonic_verdict.md) for where this fits in the claim's
overall record. ext9 killed the spatial harmonic prior in its
low-wavenumber form. ext12 found that even with documented, exactly periodic
forcing, harmonics carry only 5.4% of temporal variance globally. This
intervention differs in targeting a mid-spectrum band two regimes genuinely
concentrate in, which is a narrower claim -- but a small effect is the
expectation going in.

## Status: measured

The A/B was run as specified: three seeds, 100 epochs, `--fundamental 4.0
--n-harmonics 3`, on the streamed regime-balanced Gray-Scott set (318 train /
60 valid / 60 test trajectories, 53 per scenario, 32×32, 60 steps). Shells
resolved to `[4.0, 8.0, 12.0]`, biasing 220 of the retained modes. Results in
`results/extensions/ext15_harmonic_ab_{summary,seeds,per_regime}.csv` and
`figures/extensions/ext15_harmonic_ab.png`.

**The prediction fails.** Not "is unsupported" — fails, in the direction
opposite to the one stated.

### Per regime, which is the verdict

| regime | var < mode 8 | control | conditioned | change |
| --- | --- | --- | --- | --- |
| spots | 0.6% | 0.018057 | 0.018049 | −0.048% |
| maze | 1.3% | 0.030308 | 0.030304 | −0.016% |
| worms | 30.9% | 0.035361 | 0.035359 | −0.006% |
| bubbles | 58.3% | 0.041845 | 0.042001 | **+0.374%** |
| gliders | 69.0% | 0.023149 | 0.023145 | −0.016% |
| spirals | 77.4% | 0.040039 | 0.040018 | −0.051% |

**rho(var below mode 8, relative change) = −0.200.**

The pre-registered prediction was that conditioning helps maze and spots, the
two regimes that hold ~99% of their variance in the biased band, and does close
to nothing for spirals and gliders. What happened instead:

- The largest improvement belongs to **spirals** (−0.051%), the most
  low-wavenumber regime in the set, and the one the shells were explicitly not
  placed for. spots is second at −0.048%.
- maze, one of the two targets, improves by −0.016%, the same amount as
  gliders, which holds 69% of its variance below mode 8.
- The only change large enough to see is **bubbles at +0.374%**, a regression,
  in a regime with 58% of its variance below mode 8.

So the effect appears where the mechanism predicts nothing and is absent where
it predicts something. A negative rho is the sharper reading: the ordering by
band occupancy carries no information about the ordering by benefit, and if
anything runs the wrong way.

### Aggregate, for context only

| arm | mean VRMSE | std over seeds | params |
| --- | --- | --- | --- |
| control | 0.032153 | 0.005463 | 7,490 |
| conditioned | 0.032182 | 0.005449 | 7,710 |

The difference is 0.0000291 against a seed std of 0.005463, or **0.005 σ**. The
control's own spread across three seeds is 0.01003, roughly 345× the gap
between arms. The conditioned arm is worse in **3 of 3 seeds**, consistent with
ext30's 0 of 12.

Per seed:

| seed | control | conditioned |
| --- | --- | --- |
| 0 | 0.029657 | 0.029727 |
| 1 | 0.028383 | 0.028391 |
| 2 | 0.038418 | 0.038427 |

### The fundamental, swept once

`harmonic_verdict.md` names "a fundamental other than the inherited 4.0, which
was never swept" as one of two open qualifications on the negative result. One
point on that sweep now exists.

The same A/B was run at `--fundamental 3.5`, which with `shell_width 0.5` widens
the first annulus to cover radius 3.0–4.0 rather than 3.5–4.5, so it catches
modes 3 and 4 instead of mode 4 alone. Shells resolve to `[3.5, 7.0, 10.5]`,
same 220 biased modes.

| | aggregate control → conditioned | rho | conditioned wins |
| --- | --- | --- | --- |
| fundamental 4.0 (pre-registered) | 0.032153 → 0.032182 | −0.200 | 0 / 3 |
| fundamental 3.5 | 0.032153 → 0.032195 | +0.086 | 2 / 3 |

Both are null. 3.5 is marginally the more favourable of the two — its rho is at
least positive, and two of three seeds land on the conditioned side — and it
still buys nothing: maze −0.14%, spots +0.008%, which is the same absence of a
differential.

This is one alternative band, not a sweep. It does not close the qualification;
it makes the more likely reading of it that the null is not an artifact of where
the ring was placed.

### What this adds, and what it does not

It closes the line in `harmonic_verdict.md`'s evidence table that reads
`ext15 | none -- never run`. The verdict it supports was already reached by
ext30 at pilot scale and by ext31 on genuinely seasonal data; this is the
specified protocol agreeing with them, not a new finding.

The second qualification in that doc stands untouched: a null at 318 training
trajectories does not exclude an effect at much larger training sets.

One observation the numbers invite, offered as a reading rather than a result.
The bias is additive and input-independent — a fixed complex offset on the
selected shells, identical for every sample. It can supply a constant, not a
response. Whether an input-dependent instrument on the same shells (a
multiplicative gate, or CP rank allocated preferentially to those modes) would
behave differently is untested here, and ext16's mode-classified rank allocation
is the nearest thing in the repo to that question.

### Deviations from the run as specified

None in the arguments. Two in the environment, both recorded because they cost
time and will cost the next person the same:

- `pip install -e .` replaces a working CUDA torch with a CPU wheel
  (`2.10.0+cpu`, `cuda.is_available()` False), because `torch` is unpinned under
  `dependencies` in `pyproject.toml`. Every `--device cuda` run then fails.
  Working pattern on Kaggle: `pip install --no-deps neuraloperator tensorly`,
  then `PYTHONPATH=src python scripts/...`.
- The Zenodo bundle (DOI 10.5281/zenodo.20718092) cannot drive this script.
  `load_dataset` reads `manifest.json` for per-trajectory regime labels and
  asserts the count matches the array; the bundle has no manifest and the
  provenance is not recoverable from it. `stream_preprocess.py` is required for
  anything regime-aware.

### Measured runtimes

`harmonic_verdict.md` notes no wall-clock is logged anywhere. On a Kaggle T4:

| step | measured |
| --- | --- |
| `stream_preprocess.py`, 53/10/10 per scenario | 9m24s – 11m55s |
| `harmonic_ab.py`, 3 seeds × 2 arms, 100 epochs | 47m15s (≈475s per arm-seed) |

Reproduced twice at 3.5 — once interactive, once as a batch job — with every
per-seed VRMSE identical to the digit. The pipeline is deterministic under seed.
