# 08 — Testing gaps

**Priority: P2.** No confirmed bug here, but the gaps explain why the defects in reports
01-06 survived, and closing them is what makes the validation claim defensible.

## What is already good

The golden-test design is the strongest thing in this repository and should not be
weakened. `tests/fortran_runner.py` pins the upstream Fortran as a submodule, applies a
documented, auditable, in-memory portability patch (`fortran/patches/NOTICE.md`), compiles
with gfortran and compares numerically. CI checks out submodules recursively and installs
gfortran. That is more rigour than most research code has.

The literature validation in `examples/validation/Switzerland/README.md` — reproducing
published NSE values to within 0.005 across three flow regimes by injecting the published
parameters in FORWARD mode — is independently convincing.

## Gap A — the package-vs-Fortran matrix is narrow

Two files do different things, and this is easy to misread:

- `tests/test_golden.py` compares the **package** to Fortran. Coverage:
  `(v8, RK4)`, `(v8, RK2)`, `(v7, RK4)`. That is all.
- `tests/test_physics_golden.py` compares a **hand-written reimplementation inside the test
  file** to Fortran. Coverage: `(v8, RK4)`, `(v8, RK2)`, `(v8, CRN)`, `(v7, RK4)`.

So `CRN` — which report 02 recommends making the default — is **never compared to the
Fortran as implemented in the package**. It is only compared via a test-local copy of the
algorithm, which would not catch a bug in `model_numba.fast_run_integration`.

Versions 3, 4 and 5 are never compared to Fortran in either file. The `EUL` integrator is
never compared at all.

Version 4 in particular relies on a subtlety: the CRN branch at `model_numba.py:239-262`
handles versions 8, 7 and 4 with one code path, and is only correct for version 4 because
`a5`, `a6` and `a8` are forced to zero by `io.py:141-143`, making the extra `theta`
group vanish. Version 7 similarly relies on `a4 = 0` making `theta**a4 == 1`. Both are
correct as written. Neither is tested.

### Required change 8.1

Extend `tests/test_golden.py` to the full cross-product of
`version in {3, 4, 5, 7, 8}` x `integrator in {EUL, RK2, RK4, CRN}` — 20 cases, all cheap.
Parametrise rather than copying the existing test body four more times.

Add `EXP` to the matrix once report 02 lands, comparing against CRN rather than Fortran
(the Fortran has no equivalent scheme).

## Gap B — golden tolerance is loose and the horizon is short

```python
np.testing.assert_allclose(data.Twat_mod[365:], golden_twat_mod, rtol=1e-2, atol=1e-2)
```

(`test_golden.py:72`, `165`, `219`; same in `test_physics_golden.py`.) The comparison window
is **10 days** after warm-up (`n_tot_raw = 10`).

For a deterministic reimplementation of identical arithmetic in double precision, agreement
should be near `1e-12`. At `atol = 1e-2` over 10 days, a slow systematic divergence over a
multi-year run passes unnoticed. This is the single change that would most strengthen the
validation claim.

### Required change 8.2

Tighten to `rtol=1e-9, atol=1e-9` and extend `n_tot_raw` to at least 3 years. If the tests
then fail, that is a finding, not a reason to loosen the tolerance back — investigate the
source of divergence first. Two known candidates worth checking:

- `TTT` is a fixed `1.0/365.0` in the RK stage offsets (`model_numba.py:13`, used at lines
  289, 306, 315) while `data.tt` is leap-aware (`io.py:359-369`). The Fortran uses the same
  fixed `ttt` (`AIR2STREAM_MODULES.f90:10`) with a leap-aware `tt`, so this should match —
  but it is exactly the sort of thing a 1e-2 tolerance hides.
- Float accumulation order in `fast_funcobj` versus the Fortran `funcobj`.

## Gap C — weekly and monthly aggregation are completely untested

**Every test in the suite uses `time_res = '1d'`.** Confirmed by grep across `tests/`.
Coverage reports `model.py` lines 204-277 as unexecuted — the entire `unit == 'w'` and
`unit == 'm'` branches of `aggregation()`.

That code is hand-translated 1-based-to-0-based index arithmetic carrying comments like
`# n_pos-2 in Python because we do n_pos += 1` (`model.py:230`). I probed `1w`, `2w` and
`1m` directly and the window means, target indices and `I_inf` pointers all check out
against manual computation, so **I do not believe there is a live bug**. But it is one edit
away from one.

One genuine behaviour worth a test and a docs note: a trailing partial month is accepted as
a full month, because `prc` is compared against the partial `n_days`
(`model.py:274`). A 4-day fragment then carries the same weight in the objective as a
31-day month. This is Fortran-equivalent, so it is arguably correct-as-ported, but it
should be documented rather than discovered.

Also note `data.time_res` parsing (`model.py:174-179`) only special-cases the exact string
`'1d'`. A value like `'2d'` falls through to `print("Error: variable time_res")`, sets
`n_dat = 0`, and then raises inside `statis` with an unrelated message. A malformed value
of length other than 2 or 3 (for example `'daily'`, used in `tests/test_bugfix.py:31`)
raises `UnboundLocalError` on `unit`.

### Required change 8.3

- Parametrise the aggregation tests over `{'1d', '1w', '2w', '1m'}` and assert window
  means against an independent pandas `resample` computation.
- Validate `time_resolution` in `read_calibration` with an explicit allowed-pattern check
  and a clear error message, rather than failing deep inside `aggregation`.
- Add a test for the trailing-partial-month behaviour asserting the documented outcome.

## Gap D — the coverage number is misleading

Reported totals:

```
pyair2stream/model_numba.py      180    169     6%
pyair2stream/model.py            208     83    60%
TOTAL                           2268    520    77%
```

`model_numba.py` at 6% is a **coverage.py artifact** — Numba JIT-compiled functions are not
traced, so the numerical core is executed but invisible. The practical consequence is that
there is no measurement at all of how much of the integrator code is exercised, and the
77% total is therefore not meaningful.

### Required change 8.4

Set `NUMBA_DISABLE_JIT=1` in a dedicated coverage CI job so the kernels are traced as pure
Python. Keep the normal JIT-enabled job for correctness and speed. Report coverage from the
disabled-JIT job only, and add a comment in `.github/workflows/tests.yml` explaining why.

## Gap E — no end-to-end CLI tests

Reports 05 A and B are both `main()`-level defects that no unit test could catch, because
nothing in the suite invokes `main()` on a realistic config. `main.py` coverage is 73%,
with lines 140-190 — the entire validation block — unexecuted.

### Required change 8.5

Add end-to-end tests that build a config, invoke `main()` (or the console script via
`subprocess`), and assert on the output files. Minimum set:

- `DE` calibration with validation, asserting `1_`, `2_`, `3_` files exist and are parseable.
- `FORWARD` projection with no `T_water` column (report 05 A).
- Calibration with a sub-365-day validation file (report 05 B).
- Gap-tolerant calibration followed by sensitivity analysis (report 06 D).

## Gap F — example scripts are not executed

Two example scripts raise `KeyError` against the current code (report 05 E). CI would have
caught this on the commit that renamed the columns.

### Required change 8.6

Add a CI job that runs the example scripts against small, fast configs. If runtime is a
concern, add a `--smoke` flag to each that reduces `n_run` / `mcmc_steps`.

## Acceptance criteria

- 20-case version x integrator golden matrix passing at `rtol=1e-9` over ≥ 3 years.
- Aggregation tests covering `1d`, `1w`, `2w`, `1m`.
- Coverage job with `NUMBA_DISABLE_JIT=1` reporting > 70% on `model_numba.py`.
- End-to-end CLI tests covering the four scenarios above.
- Example scripts executing in CI.
