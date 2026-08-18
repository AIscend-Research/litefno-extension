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

## Status: not yet measured

**The ext15 A/B itself has not been run.** The model, the measurement script and
its tests were committed together in `c22335c`; no results were produced then and
none exist now. There is no `ext15_*.csv` in `results/`, and this document
deliberately reports no numbers rather than borrowing any.

Its two arms *have* since been measured, in a different experiment.
[ext30](data_efficiency.md) runs them across four training-set sizes at three
seeds and finds no effect: the harmonic arm is very slightly worse at every size,
wins 0 of 12 paired runs, and buys a data multiplier of 1.00x. The differential
prediction below does not appear either -- maze improves 0.1%, spots gets
slightly worse. That is evidence about this conditioning at pilot scale, but it
is not the ext15 A/B, which specifies three seeds at 100 epochs on the full
training pool and a per-regime verdict of its own.

What exists and is checkable today is the machinery and its guarantees: the
bit-identical-at-initialisation property, the parameter-count bound, and the
pre-registered differential prediction above. What is missing is the run.

Running it produces `ext15_harmonic_ab_summary.csv`,
`ext15_harmonic_ab_seeds.csv`, `ext15_harmonic_ab_per_regime.csv` and
`ext15_harmonic_ab.png`:

    python3 scripts/harmonic_ab.py --seeds 0 1 2 --epochs 100 \
        --fundamental 4.0 --n-harmonics 3 \
        --out-dir results/extensions --fig-dir figures/extensions

Three seeds at 100 epochs is the default because the expected effect is small and
a one-seed difference at this scale would not be distinguishable from noise. The
per-regime breakdown, not the aggregate, is the verdict.
