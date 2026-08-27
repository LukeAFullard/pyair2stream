# 07 — Reproducibility, provenance and documentation corrections

**Priority: P2.** Nothing here changes a number, but several items would be raised in
review, and one documented claim about the reference implementation is factually wrong.

## Defect A — calibrations are not reproducible

`main.run_optimizer` (`main.py:25-38`) calls every optimizer without a seed. All five
optimizer entry points take `seed: Optional[int] = None` and default to `None`, so
`np.random.seed` is never called and `differential_evolution(..., seed=None)` draws from
global state. No config key exists for a calibration seed — `forward_options.random_seed`
(`optimization.py:75`) covers only the forward prediction-interval path.

Two `DE` runs with byte-identical config:

```
run 1  NSE 0.99214   0.860551 0.350687 0.398303 -0.317271 11.156396 8.091281 0.554414 1.151017
run 2  NSE 0.99268   0.504040 0.242934 0.293294 -0.750023  3.494572 2.439196 0.558781 0.349264
```

Not merely different in the last decimal — the discharge parameters differ threefold. See
report 09; this is equifinality, not just PRNG noise, but the absence of a seed makes it
impossible to distinguish the two or to reproduce a published result.

### Required change 7.1

Add a top-level `random_seed:` config key. Thread it through `read_calibration` into
`data.random_seed`, and have `run_optimizer` pass it to whichever optimizer it dispatches.
Inside each optimizer, prefer a local `np.random.Generator` over `np.random.seed()`, which
mutates global state and will interfere with any calling script.

`differential_evolution` and `emcee.EnsembleSampler` both accept explicit seeds or
generators. `PSO_mode`'s `ProcessPoolExecutor` workers need per-worker derived seeds
(`SeedSequence.spawn`) if any worker-side randomness is added later; currently the workers
are deterministic given their inputs, so parent-side seeding is sufficient.

Record the seed in the `calibration_metadata.json` introduced in report 01.

### Required change 7.2

Add a test that runs `DE` twice with the same explicit seed and asserts bit-identical
`par_best`, and twice with different seeds and asserts they differ.

## Defect B — the "version 8 parameter zeroing" claim is false

`README.md:245-249` and `CHANGELOG.md` both state:

> the original Fortran had a duplicated `IF (version == 4)` block where the second
> occurrence appears to have been intended for `version == 8`, causing parameters 5-8 to be
> incorrectly zeroed in Version 8 mode.

The upstream source (`fortran/upstream/src/AIR2STREAM_READ.f90:81-87`) reads:

```fortran
    IF (version == 4) THEN          !air2stream with 8 parameters
         parmin(5)=0;    parmax(5)=0;    flag_par(5)=.false.
         ...
    END IF
```

The guard is `version == 4`. The block is byte-identical to the one at lines 67-72 and
**never executes for version 8**. Only the trailing comment is wrong. The Fortran has a
cosmetic copy-paste typo in a comment, not a physics bug, and `io.py`'s corresponding
comment (lines 148-155) describes a fix to something that was never broken.

This matters: it is presented as a discovered bug in the reference implementation, and a
reviewer who checks the source will find it does not hold.

### Required change 7.3

Correct `README.md`, `CHANGELOG.md` and the comment block in `io.py:148-155`. Accurate
wording: the upstream source contains a duplicated `IF (version == 4)` block whose comment
misleadingly says "8 parameters"; the duplication is harmless, and `pyair2stream` omits the
redundant block. Remove it from the "Known deviations" list, since it is not a deviation.

## Defect C — undocumented deviations from the Fortran

Three real deviations are not in the "Known deviations" list:

1. **PSO convergence criterion.** Fortran uses `IF (norm .lt. 0.0)`
   (`AIR2STREAM_RUNMODE.f90:143`), which never fires. `optimization.py:317` uses
   `norm < 1e-4`, so PSO can now terminate early where the Fortran runs to completion.
   Legitimate fix, undocumented behavioural change.
2. **Qmedia definition.** Fortran excludes only `Q == -999`; `io.py:187` also excludes
   `Q <= 0`. Sensible, undocumented.
3. **`tt` construction.** Fortran walks sequential day counts from `year_ini`
   (`AIR2STREAM_READ.f90:195-217`); `io.py:359-369` computes day-of-year from real calendar
   dates. Equivalent when the series starts 1 January and is complete, more robust
   otherwise. Undocumented.

Report 01 adds a fourth (freezing Qmedia across calibration and validation) and report 02
adds a fifth (default integrator change to CRN).

### Required change 7.4

Add all of these to the "Known deviations" section with a one-line rationale each.

## Defect D — version numbers disagree three ways

```
pyproject.toml          version = "0.1.0"
pyair2stream/__init__.py __version__ = "0.1.0"
CHANGELOG.md            ## [1.0.0] - 2026-07-09
CLI banner              prints 0.1.0
```

### Required change 7.5

Pick one. If the intent is a 1.0.0 release, bump `pyproject.toml` and `__init__.py` and
have the latter read from package metadata (`importlib.metadata.version`) rather than
duplicating the string. Given the P0 findings in reports 01 and 02, 1.0.0 is premature —
`0.2.0` after those land is more defensible.

## Defect E — `CITATION.cff` is incomplete

`CITATION.cff` has no `given-names` for the author, no `version`, no `date-released` and no
DOI. `preferred-citation` correctly points at Toffolon & Piccolroaz (2015) but a user
following the file has no way to cite the software itself.

### Required change 7.6

Add `given-names`, `version`, `date-released`, and a software DOI (Zenodo integration is
the usual route). Keep the `preferred-citation` block — pointing users at the original
model paper is correct and should be preserved.

Also add an explicit statement, in both `CITATION.cff` and `README.md`, that
`pyair2stream` is not maintained by or endorsed by the original authors. `README.md:238`
says this already; make sure it survives.

## Defect F — dependency floors are loose for a reproducibility claim

`pyproject.toml` allows `python >= 3.9` with `numpy >= 1.20`, `scipy >= 1.6`,
`pandas >= 1.2`. The CI matrix tests 3.9 and 3.12, which will resolve very different
dependency sets. `poetry.lock` exists but `pip install -e .` in CI ignores it.

### Required change 7.7

Either install from the lock file in CI, or add an upper bound and a documented tested
range. For a package intended for reproducible science, publish the exact environment used
to generate the bundled example outputs.

## Defect G — dataclass field declared implicitly

`io.py:80` assigns `data.uncertainty_options = {...}` but `CommonData` (`config.py`)
declares no such field. It works because the dataclass has no `__slots__`, but it means the
attribute is invisible to type checkers and to anyone reading `config.py`. Several call
sites defensively use `getattr(data, 'uncertainty_options', {})` as a result.

### Required change 7.8

Declare `uncertainty_options: Optional[dict] = None` on `CommonData` and remove the
defensive `getattr` calls (`optimization.py:95`, `160`, `733`; `post_processing.py:316`,
`320`, `379`, `385` and equivalents). Same treatment for `_input_data_path_cal`,
`_input_data_path_val`, `_n_tot_raw` and `_segment_warned`, all currently set by
assignment.

## Acceptance criteria

- Seeded-reproducibility test as described in 7.2.
- A documentation test or manual check confirming no remaining reference to the
  "version 8 parameter zeroing bug".
- `python -c "import pyair2stream; print(pyair2stream.__version__)"` matches
  `pyproject.toml` and the newest `CHANGELOG.md` heading.
- `cffconvert --validate` passes on `CITATION.cff`.
