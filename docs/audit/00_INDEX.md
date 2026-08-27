# pyair2stream audit — index and working instructions

## Provenance

| | |
|---|---|
| Repository | `https://github.com/LukeAFullard/pyair2stream` |
| Commit audited | `442e7d931383b63bc01f70d22dfc91aecb31af44` |
| Fortran reference | submodule `fortran/upstream` @ `d4834bccf01657c03ab60efb4c18f8a256132c53` |
| Environment | Python 3.12, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, numba 0.67.0, emcee 3.1.6 |
| Test suite result | 57 passed (7 golden tests require `git submodule update --init --recursive` and `gfortran`) |

All line numbers in these reports refer to the commit above. Verify before editing.

## Intended use

Two studies drive the prioritisation:

1. **Water abstraction.** Calibrate on observed temperature and flow, then re-run the fitted model on naturalised flow and report the temperature difference.
2. **Climate projection.** Drive the fitted model with projected air temperature and flow to estimate future water temperature and thermal-stress statistics.

An issue is P0 if it would produce a *silently wrong answer* in either study.

## Report set

| File | Topic | Priority |
|---|---|---|
| `01_qmedia_scenario_invariance.md` | Discharge normalisation destroys the scenario signal | **P0** |
| `02_numerical_integration.md` | Explicit integrators diverge silently under scenario flows | **P0** |
| `03_objective_function_and_masks.md` | Mismatched samples in NSE/KGE/R²; `eval_mask` never set | P1 |
| `04_uncertainty_and_mcmc.md` | Likelihood ignores autocorrelation; ensemble not exposed | P1 |
| `05_cli_and_io_correctness.md` | Projections crash; short validation; warm-up rows in outputs | P1 |
| `06_diagnostics_and_plots.md` | Dotty plots wrong column; PSO metrics never recorded | P2 |
| `07_reproducibility_and_provenance.md` | No seeds; version/doc inconsistencies to correct | P2 |
| `08_testing_gaps.md` | Golden-test matrix, untested aggregation, tolerance | P2 |
| `09_study_design_notes.md` | Non-code guidance for the two studies | — |

## Suggested implementation order

Each report is self-contained, but there are dependencies. Work in this order:

```
01  Qmedia persistence
02  Integrator selection + divergence guard
03  eval_mask unconditional        (04 and 06 depend on this)
05  CLI projection path
04  Ensemble output + sidecar sigma
06  Plot fixes
07  Seed plumbing
08  Test matrix
```

Report 03 must land before 04 and 06: both consume `data.eval_mask`, and fixing them
first would bake in the current wrong behaviour.

## Conventions for the implementing agent

- **Do not change the governing equations or the Fortran-equivalent integrators.** The
  physics is validated against the reference implementation and against published
  parameter values (see `examples/validation/Switzerland/README.md`). Fixes are additive.
- Every change that alters numerical output must be accompanied by a regression test.
- `tests/test_golden.py` and `tests/test_physics_golden.py` must continue to pass
  unchanged. If a change breaks them, the change is wrong.
- Where a report says "add a new option", default it to the current behaviour unless the
  report explicitly says otherwise. Two exceptions are flagged: the default integrator
  (report 02) and Qmedia persistence (report 01).
- Prefer failing loudly over silent fallback. Several issues in this audit exist because
  a bad state produced plausible-looking numbers instead of an exception.

## Terminology

- **θ (theta)** — dimensionless discharge, `Q / Qmedia`.
- **B** — linearised decay rate of the ODE, `(a3 + a8·θ) / θ^a4`, units of 1/day.
- **Warm-up block** — indices `0..364` of every internal array, a verbatim copy of the
  first 365 days of input data, prepended by `read_Tseries`. Real data starts at index 365.
- **Sentinel** — `-999.0` for floats, `-999` for ints. Used throughout instead of NaN.
