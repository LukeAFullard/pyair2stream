# Changelog

## [Unreleased]

### Changed
- **Default integrator changed from `RK4` to `CRN`** (audit report 02). Explicit
  schemes (`RK4`/`RK2`/`EUL`) are only conditionally stable and can diverge silently
  -- with no NaN, no error -- at discharge different from calibration (e.g. scenario
  runs). `CRN` (Crank-Nicolson) is unconditionally stable and matches the new `EXP`
  (exponential/integrating-factor) integrator to well under 0.1 °C on every case
  tested. This changes results for any run that relied on the previous `RK4` default;
  set `integrator: "RK4"` explicitly to keep the old behaviour.
- **Qmedia is no longer silently recomputed for validation or FORWARD runs** (audit
  report 01). `Qmedia` is now frozen across the calibration/validation split within a
  run and persisted to `calibration_metadata.json`; `FORWARD` mode requires an
  explicit `Qmedia:` or `paths.calibration_metadata` instead of recomputing it from
  whatever discharge is loaded, which previously could cancel a scenario's discharge
  signal entirely. This changes validation-period objective values for existing
  non-gap-tolerant configurations that relied on the old (unfitted) recomputed
  `Qmedia`.
- **`eval_mask` is now always built, and the calibration objective/MCMC likelihood
  are aligned with it** (audit report 03). Previously `eval_mask` was only built in
  gap-tolerant mode, so the DE-MCMC likelihood's own daily mask double-counted the
  warm-up block (a verbatim copy of year one) as real observations in every
  non-gap-tolerant run. Separately, in gap-tolerant mode, `statis()` computed
  `mean_obs`/`TSS_obs` over every window with a valid observation while `funcobj()`
  additionally excluded windows outside `eval_mask` (warm-up plus each segment's
  `warmup_drop_days`) -- a different, larger sample than the one actually scored.
  This changes every reported NSE/KGE/R²/AIC/BIC in gap-tolerant mode, and the
  MCMC posterior in every mode.

### Added
- `NumericalDivergenceError`, raised by `main.forward()`, `optimization.forward_mode()`,
  and `sensitivity_analysis()` when a simulated water temperature is non-finite or
  exceeds `max_plausible_twat` (default 60 °C) -- not inside the calibration hot loop.
- `model.stability_report()` / `warn_on_stability()`: a pre-flight screening check that
  warns (or errors above `stability_error_fraction`, default 10%) when the discharge-
  dependent stability coefficient `B` exceeds the current integrator's stability limit.
- New `EXP` integrator option (exponential / integrating-factor step), unconditionally
  stable and exact for piecewise-constant coefficients.
- `calibration_metadata.json`, written by every run, recording `Qmedia`, its source,
  the calibrated theta range, version, integrator, and best-fit parameters.
- New config keys: `max_plausible_twat`, `stability_error_fraction`,
  `paths.calibration_metadata`.

## [1.0.0] - 2026-07-09

### Added
- Python port of the air2stream hybrid model for river water temperature.
- YAML-based configuration instead of fixed-width text files.
- CSV input and output.
- Gap-tolerant mode for handling missing data.
- Modern calibration algorithms (DE, PSO, LATHYP, DE-MCMC).
- Uncertainty quantification via MCMC and AR(1) prediction intervals.
- Leave-one-year-out cross-validation.

### Fixed
- Fixed version 8 parameter zeroing bug from original Fortran implementation.
- Fixed PSO initialization to handle NaN and use `-1e30` instead of zero.
- Addressed stale Italian console strings in documentation.
- Removed dead and unused functions (`_step`, `_get_RK_func`) from `model.py`.
- Corrected `n_runs` parameter naming inconsistency in configuration files to `n_run`.
- Fixed cross-validation data leak and hardcoded initial condition bug by strictly enforcing that the first year of data cannot be a candidate fold.
