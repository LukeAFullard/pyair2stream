# Changelog

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
