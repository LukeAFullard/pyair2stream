# Changelog

## [0.2.0] - 2026-08-27

Version numbers previously disagreed three ways: `pyproject.toml`/`__init__.py`
stayed at `0.1.0` while this file's newest heading said `[1.0.0]` and the CLI
banner printed `0.1.0` (docs/audit/07_reproducibility_and_provenance.md, Defect
D). `pyproject.toml` was in fact never bumped for that `1.0.0` heading, and
given the P0 findings fixed by audit reports 01 and 02 above, a `1.0.0` release
at that point would have been premature regardless. The `[1.0.0]` heading below
has been relabelled `[0.1.0]` to match what was actually shipped, and this
release -- everything above it in this file -- is `0.2.0`. `pyair2stream.__version__`
is now read from installed package metadata (`importlib.metadata.version`)
rather than duplicated as a literal string, so it cannot drift from
`pyproject.toml` again.

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
- **Output CSVs no longer contain the 365-day warm-up block** (audit report 05,
  Defect C). `2_*.csv`, `3_*.csv`, `MCMC_envelopes_*.csv`, and
  `Forward_Prediction_Envelopes_*.csv` previously started with 365 rows of
  `Year=-999` junk (a verbatim copy of year one used internally as a numerical
  spin-up); reading these files directly broke `pd.to_datetime` and row-count
  expectations. Row count now equals the input file's row count.
- **`FORWARD` mode with no `T_water` at all no longer crashes** (audit report 05,
  Defect A). `main()` called `aggregation()`/`statis()` unconditionally before
  dispatching to any run mode; a pure climate-projection run (no observations to
  calibrate against) crashed with `n_dat is 0` before reaching `forward_mode()`'s
  own correct handling of that case.
- **A validation period shorter than one year no longer silently re-scores the
  calibration data as "validation"** (audit report 05, Defect B). `read_Tseries`
  returned before overwriting `data.n_tot` for a too-short validation period, so
  it silently retained the calibration value and passed the length guard in
  `main.forward()`.
- **The DE-MCMC/DE-CV-MCMC likelihood can now account for residual autocorrelation**
  (audit report 04, Defect A). Setting `uncertainty_options.noise_model: "ar1"` now
  also selects an AR(1)-whitened concentrated log-likelihood for the sampler itself
  (previously the AR(1) noise model was applied only to the downstream predictive
  envelope, never to the likelihood the posterior was actually sampled from). Daily
  residuals with `rho` in the 0.8-0.95 range typically understate posterior/interval
  width by a factor of `sqrt((1+rho)/(1-rho))` (~4x at `rho=0.9`) under the unchanged
  `iid` default. `rho` is estimated once at the DE optimum and held fixed for the
  likelihood.
- **`forward_mode()` now falls back to the MCMC sidecar for `residual_sigma`, and
  raises instead of warning if none is available** (audit report 04, Defect C).
  Previously an omitted `residual_sigma` silently defaulted to `0.0` behind a
  `print`, producing a "prediction interval" with parameter uncertainty only and no
  residual term. It now mirrors the existing `rho` sidecar carry-forward, and raises
  `ValueError` if `enable_prediction_intervals` is set with no usable sigma from
  either source.
- **MCMC walker initialisation now reflects off parameter bounds instead of
  clipping to them, and is scaled to each parameter's bound width** (audit report
  04, Defect D). Clipping collapsed the ensemble's spread to a single point in any
  dimension where the DE optimum sat exactly on a bound (observed for `a5`/`a6` on
  some validation datasets), which `emcee`'s stretch move cannot recover from.
  Initialisation now asserts non-degenerate spread and raises rather than proceeding
  silently.
- **MCMC burn-in is now adaptive by default, and convergence diagnostics are
  computed on the post-burn-in chain** (audit report 04, 4.6). Burn-in defaults to
  `max(0.3*mcmc_steps, 5*max(tau))` (previously a flat 30%), overridable via
  `uncertainty_options.burnin_fraction`. Autocorrelation time is now recomputed
  after discarding burn-in rather than on the full chain. A new split-Rhat
  (Gelman-Rubin) diagnostic is reported per parameter and recorded in the sidecar.
  `uncertainty_options.strict_convergence: true` promotes the existing
  chain-too-short warning to a hard `RuntimeError`.
- **`docs/MCMC_uncertainty.md`'s `DE-MCMC` vs. `DE-CV-MCMC` comparison reframed**
  (audit report 04, 4.5) as a non-convergence diagnostic rather than evidence that
  `DE-CV-MCMC` finds a wider/better posterior -- two converged chains sampling the
  same posterior must agree on spread as well as point estimate.
- **Dotty plots select parameter/efficiency columns by name, not position**
  (audit report 06, Defect A). The optimizer history CSV has 12 columns
  (`par_1..par_8, eff_index, NSE, R2, MAE`); the previous positional `[:-1]`/`[-1]`
  split treated `NSE`/`R2` as parameter columns and `MAE` as the plotted
  "efficiency", mislabeling the y-axis and marking the highest-error parameter set
  as best. Visible in the previously-committed
  `examples/Hopelands/output/dottyplots_DE-MCMC_NSE_Hopelands.png` (y-axis labelled
  "NSE" with values 1-4.3, impossible for NSE). Committed example dotty plots have
  not been regenerated in this change -- doing so requires re-running each
  example's full (often multi-hour) calibration; treat any committed dotty plot as
  stale until its example is next re-run.
- **The dotty-plot `par_8` panel is no longer blanked on every run** (audit report
  06, Defect B). A stray `for...else` (same indentation as the plotting loop, not
  inside an `if`) ran unconditionally after normal loop completion and called
  `axes[7].axis('off')`.
- **PSO history now records each particle's own NSE/R2/MAE** (audit report 06,
  Defect C). `eval_particle_worker` previously returned only the scalar objective
  value from its `ProcessPoolExecutor` worker; `data.current_nse`/`current_r2`/
  `current_mae` are set as a side effect inside the child process and never
  crossed back to the parent, so every history row silently recorded the parent's
  own untouched `-999.0` defaults for all three columns regardless of the
  particle's actual fit. `DE_mode`/`LH_mode` run single-threaded and were
  unaffected.
- **Sensitivity index normalization is now configurable, and the console message
  describing it now matches what is actually computed** (audit report 06, Defect
  E). New `sensitivity_perturbation_mode` config key: `"value"` (default,
  unchanged from earlier releases -- perturb by a percentage of the parameter's
  own calibrated value) or `"range"` (perturb by a percentage of the parameter's
  bound width instead, comparable across parameters and immune to the
  near-zero-value problem, but not backward compatible with `"value"`-mode
  numbers). The console message previously said "% of parameter range" while
  always computing "% of parameter value" regardless of setting. A perturbation
  clipped on only one side by a bound (an asymmetric, first-order rather than
  second-order estimate) is now flagged `Status: "Bounded"` in the output CSV
  instead of being silently reported as a plain `"Active"` row, and the plot's
  y-axis label states which normalization was used.
- **Residual ACF plot is now gap-aware** (audit report 06, Defect F).
  `pd.plotting.autocorrelation_plot()` on a NaN-dropped residual series
  concatenates non-adjacent days, so its lag-k is not lag-k in calendar time
  (`estimate_ar1_rho` already handled this correctly for lag-1; the diagnostic
  plot did not, and the two could disagree). `post_processing.gap_aware_acf()`
  instead pairs day `t` with day `t+k` only where both are non-missing, for every
  plotted lag, and the plot now reports the number of valid pairs used.

### Added
- `pyair2stream/scenario.py`: `load_ensemble`, `aggregate`, `exceedance`, and
  `paired_difference` helpers for working with a saved raw MCMC/forward ensemble
  (audit report 04, Defect B / 4.2). Percentile envelope bands alone cannot produce
  aggregate statistics (the p5 of a 7-day rolling mean is not the rolling mean of
  the p5 series), so `uncertainty_options.save_ensemble: true` now additionally
  writes the full `(n_samples, n_days)` noisy-trajectory matrix, post-warm-up, as
  compressed `.npz` (`MCMC_ensemble_*.npz` / `Forward_Prediction_Ensemble_*.npz`).
  `paired_difference` is the row-aligned scenario-comparison function both the
  water-abstraction and climate-projection studies need and which was previously
  unobtainable from the percentile-only output.
- New `uncertainty_options` keys: `save_ensemble` (bool, default `false`),
  `strict_convergence` (bool, default `false`), `burnin_fraction` (optional float
  in `(0, 1)`, default adaptive).
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
- New config key `calendar`: `"standard"` (default), `"noleap"`, or `"360_day"`, for
  GCM output on a non-standard calendar (audit report 05, Defect D). `tt` (the
  seasonal term's phase) is computed from row position against the declared
  calendar rather than from the `Date` column, which a non-standard-calendar file
  padded to pass the old validation would otherwise silently misalign. The
  1-January start requirement is also relaxed for `run_mode: FORWARD`.
- New config keys: `max_plausible_twat`, `stability_error_fraction`,
  `paths.calibration_metadata`, `calendar`.
- New config key `sensitivity_perturbation_mode` (`"value"`, default, or
  `"range"`; audit report 06, Defect E).
- `pyair2stream.post_processing.select_dotty_data()` and `.gap_aware_acf()`,
  factored out for direct testability (audit report 06, Defects A and F).
- New top-level config key `random_seed` (audit report 07, 7.1), threaded through
  `run_optimizer` to whichever optimizer `run_mode` dispatches to (PSO, LATHYP, DE,
  DE-MCMC, DE-CV-MCMC) and recorded in `calibration_metadata.json`. Previously no
  config key seeded calibration at all -- `differential_evolution(..., seed=None)`
  drew from global numpy random state, so two runs of the same config could
  converge to substantially different parameter sets with no way to reproduce a
  published result. `PSO_mode`/`LH_mode` now also draw from a local
  `np.random.Generator` instead of mutating global `numpy.random` state via
  `np.random.seed()`.
- `tests/test_golden.py` now covers the full `version x integrator` cross
  product (5 versions x 4 Fortran-backed integrators = 20 cases, plus `EXP`
  vs `CRN` for each version) over a 3-year horizon at a much tighter
  tolerance, instead of 3 hand-picked combinations over 10 days at
  `rtol=atol=1e-2` (audit report 08, 8.1/8.2).
- `tests/test_report08_aggregation.py`: `model.aggregation()` is now tested
  at `'1w'`/`'2w'`/`'1m'` resolutions against independent pandas computations,
  not just `'1d'` (audit report 08, 8.3).

### Removed
- The `Twat_mod_p5`/`Twat_mod_p95` dual-name fallback in `post_processing.py`
  (audit report 05, Defect E) -- a compatibility shim for a column name the code
  has never actually written (the real columns are `Twat_mod_lower`/
  `Twat_mod_p50`/`Twat_mod_upper`). Two example scripts still referenced the
  removed names and have been fixed.

### Fixed
- `model.aggregation()`'s weekly (`'Nw'`) branch could raise `IndexError` (or,
  in the original Fortran, silently write out of bounds -- `AIR2STREAM_
  SUBROUTINES.f90`'s equivalent `pos_tmp` is equally unguarded) whenever the
  record length was not an exact multiple of the window length, because the
  trailing partial window's "representative position" was computed assuming
  a full-length window. This is the common case for any real dataset, and
  was invisible because every test in the suite used `time_res = '1d'`
  before now. Clamped to the last valid index; only the trailing partial
  window's position is affected (docs/audit/08_testing_gaps.md, 8.3).
- `time_resolution` is now validated in `read_calibration` against the
  patterns `aggregation()` actually understands (`'1d'`, or 1-2 digits plus
  `'w'`/`'m'`), with a clear, actionable error. Previously an invalid value
  either raised an opaque `UnboundLocalError` (e.g. `'daily'`) or silently
  produced zero calibration data with an unrelated downstream error message
  (e.g. `'2d'`) (docs/audit/08_testing_gaps.md, 8.3).
- `tests/fortran_runner.py` (the golden-test harness that compiles and drives
  the upstream Fortran reference) wrote its date column as `day month year`;
  the real air2stream input format is `year month day` (confirmed against
  `fortran/upstream/Switzerland/*_cc.txt`). This fed the day-of-month to
  `AIR2STREAM_READ.f90`'s `year_ini=date(366,1)`, corrupting its leap-year
  block-length bookkeeping (`tt`'s seasonal phase) for every year beyond the
  first -- invisible in every previous golden test (all <=100 days, never
  crossing a year boundary), and only surfaced by extending the golden matrix
  to a 3-year horizon (docs/audit/08_testing_gaps.md, 8.2). `pyair2stream`
  itself was not affected: its own calendar-aware `tt` construction (`io.py`)
  was independently confirmed correct throughout. Test-infrastructure fix
  only; no change to `pyair2stream`'s own code or behaviour.

## [0.1.0] - 2026-07-09

### Added
- Python port of the air2stream hybrid model for river water temperature.
- YAML-based configuration instead of fixed-width text files.
- CSV input and output.
- Gap-tolerant mode for handling missing data.
- Modern calibration algorithms (DE, PSO, LATHYP, DE-MCMC).
- Uncertainty quantification via MCMC and AR(1) prediction intervals.
- Leave-one-year-out cross-validation.

### Fixed
- Fixed PSO initialization to handle NaN and use `-1e30` instead of zero.
- Addressed stale Italian console strings in documentation.
- Removed dead and unused functions (`_step`, `_get_RK_func`) from `model.py`.
- Corrected `n_runs` parameter naming inconsistency in configuration files to `n_run`.
- Fixed cross-validation data leak and hardcoded initial condition bug by strictly enforcing that the first year of data cannot be a candidate fold.
