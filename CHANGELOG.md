# Changelog

## [0.5.1] - 2026-10-09

### Added
- **`summary.html`.** Every run now also writes its summary as a web page, with
  the figures inside it (at screen size), so it opens in any browser and can be
  sent as one file. `summary.md` now also shows the figures and links each
  output file.

### Fixed
- ⚠ **Zero-flow days in gap-tolerant mode are no longer skipped silently.**
  Versions 4, 7 and 8 cannot simulate a day without flow, and gap-tolerant mode
  treated such days as gaps without a message: the output marked them as not
  missing, and `scenario.exceedance` counted them as not above the threshold.
  A scenario in which the river dries up could therefore show fewer warm days
  than the baseline. Now:
  - a calibration or validation run warns, with the number of days and the
    first one, marks them in `Q_gap` and counts them in `gaps_summary.txt`;
  - a FORWARD run stops with an error that explains the choices;
  - with `min_theta_floor` set, the days are simulated (they were gaps even
    then);
  - `scenario.exceedance` and `scenario.aggregate` warn about days without a
    simulated value, and `scenario.paired_difference` refuses two runs that
    simulated different days.
- **Yearly statistics leave out partial years.** `scenario.year_statistics`
  gave a "highest 7-day mean" and a count of warm days for the first and last
  year of a file even when the file covered only part of them (for example a
  record ending in March, whose "yearly peak" was a winter value). Such years
  are now left out, with a warning; `partial_years="keep"` includes them. The
  cross-validation check is unchanged: it keeps a year only if its season was
  measured.
- ⚠ **Any file may start on any date.** The 1 January start, which the
  Fortran assumed (it counted the time of year from the row number), is no
  longer required for calibration and validation files: the time of year comes
  from each row's date, and the warm-up year copies the phase of the rows it
  repeats. Before, a record starting on 2 January had to wait for the next
  1 January, losing almost a year of measurements. Records starting on
  1 January give exactly the same results as before.
- **`calendar: "noleap"` and `"360_day"` take the time of year from the dates.**
  They counted it from the row position with the first row as 1 January, so a
  FORWARD or gap-tolerant file starting on another date (for example
  1 October, a water year) ran with its seasonal term out of phase, without a
  message. A `noleap` file now has real dates without 29 February, checked for
  missing and repeated days like standard dates. In a `360_day` file the first
  date sets the day of the year the file starts on, and the rows are counted on
  from there (METHODS §2).
- **Ensemble draws that are unstable with RK4, RK2 or EUL are excluded.** In
  the DE-MCMC band and FORWARD intervals a draw now also counts as divergent
  when a difference can grow more than `stability_max_growth` times
  (`largest_growth`), as the best-fit run already did; such a run can stay below
  `max_plausible_twat` and still be wrong. `CRN` and `EXP` are not affected.
- **DE-MCMC likelihood: exactly the scored values.** With weekly or monthly
  scoring, `prc` below 1 and gap-tolerant mode, a block whose middle day was
  itself unscored was scored by the objective but left out of the likelihood.
  The likelihood now uses the same blocks as the objective. Daily scoring and
  `prc: 1` are unchanged.
- **LATHYP** started from a best score of −999, so if every sample scored
  lower (NSE can be far below −999), it returned the all-zero start
  parameters. It now keeps the best finite score, and stops with an error if
  no sample has one.
- The warnings about ρ (too few pairs, ρ at its limit of 0.99) are printed like
  every other warning, so they now appear in `summary.md` and
  `RunResult.messages`.

### Documentation
- METHODS §7 and USER_GUIDE §8: the scores in `goodness_of_fit_*.csv` are
  computed on the scored values (daily values, or weekly or monthly means).
- METHODS §10: a gap-tolerant FORWARD run takes the day-of-year averages from
  its own file. METHODS §13: in gap-tolerant mode a paired difference is too
  small for the first days of each segment.

## [0.5.0] - 2026-10-05

### Added
- ⚠ **A stricter stability check for RK4, RK2 and EUL.** Before simulating, the
  package now also works out from the B series how much a difference in the
  simulated temperature (from the start value, rounding or the inputs) can grow
  over a stretch of days, and stops the run if it can grow more than
  `stability_max_growth` times (new setting, default 100). The equation is
  linear in water temperature, so this follows exactly from B:
  `model.step_amplification` gives what one step of each integrator does to a
  difference, and `model.largest_growth` the largest product over a stretch of
  days; `stability_report` returns it (`max_growth`, `growth_stretch`) and the
  warning names the stretch. Why: the share of days above the limit (stop
  above 10%) let through runs that went wrong. With the Swiss rivers' published
  parameters, `RK2` on the Rhône (version 7, 2015 parameters) was 0.99 °C off
  the equation with B above its limit on 4.2% of days (a difference could grow
  800,000 times), and `RK4` with the Mentue's flows doubled 0.50 °C off; the
  0 °C floor kept both below the 60 °C divergence check
  ([example 09](examples/09_integrator_stability/README.md)). `CRN` and `EXP`
  are not affected.
- **`pyair2stream.run(config)`: run from Python** exactly as the command line
  does, with a settings file or a dict, and get the results back: the best
  parameters, the scores, the warnings and notes, the output folder and its
  files (USER_GUIDE §7.2). `verbose=False` runs silently; errors raise
  exceptions. The command line now calls it.
- **`summary.md` in every output folder**: one page with the settings, the
  data used, the scores, the parameters (flagging any on a bound), the
  uncertainty, every warning and note the run printed, and what each output
  file is.
- **`pyair2stream.plots`: ready-made figures** for simulation ensembles
  (USER_GUIDE §12). `prediction_range` draws the median and range of the
  simulations day by day, optionally as 7-day means (computed in each
  simulation first), with the measurements and a limit. `change` draws the
  difference between scenarios by day, month or year. `yearly_statistic` draws
  a yearly statistic year by year against a limit, with the chance of
  exceeding it. Examples 03, 04 and 08 now make their figures with them.
- **`filled_water_temperature_<period>.csv`**: the measured water temperature
  with the model's values on the days without a measurement, a `source` column
  saying which, and the prediction range after DE-MCMC or a FORWARD run with
  intervals.
- ⚠ **A cross-validated check of yearly statistics**, written by every
  cross-validation run (`cv_yearly_statistics.csv`,
  `cv_yearly_statistics_summary.csv`; docs/METHODS.md §11). For each held-out
  year it simulates the highest daily mean, the highest 7-day mean and the
  number of days above a threshold 1,000 times, from that fold's calibration and
  error model, and records where the measured value fell. The summary gives how
  often the 50% and 90% ranges held and the model's mean error in each
  statistic, with a 95% interval. New options: `cross_validation.threshold` and
  `cross_validation.season_months`. Why: in years not used for calibration, the
  model's error on the hottest days is not always its typical error (on the
  Mentue, version 8's simulated yearly peaks were 0.6–0.8 °C too high), and
  uncorrected 90% ranges for yearly statistics held in only 73–92% of 48 held-out
  years per version on the Swiss rivers (validation V11).
- `scenario.correct_statistic`: corrects a simulated yearly statistic by the mean
  cross-validated error, with that mean's uncertainty (docs/METHODS.md §13). With
  it, the 90% ranges held in 85–94% of held-out years (V11). Example 03 now uses
  the check and the correction.
- `scenario.year_statistics` (the three yearly statistics, defined as in the
  check), `scenario.pit`, `scenario.central_range` and `scenario.inside_range`.
- **Any interval level, checked at every level.** `uncertainty_options.prediction_interval`
  (default 90) now also sets the ranges of the cross-validation check, whose
  summary reports coverage at 50%, 80%, 90% and 95% as well;
  `cv_interval_coverage.csv` gives the share of held-out days and 7-day means
  inside the interval at each of these levels. New
  `uncertainty_options.parameter_interval` (default 90) sets the level of the
  parameter intervals: the jackknife rows of `cv_results.csv` (90% before) and
  the MCMC summary `parameter_significance_*.csv` (⚠ 95% before; its columns are
  now named after the level). Its "differs from zero" column stays a test at the
  5% level. The validation tests coverage at 50%, 80%, 90%, 95% and 99% (V4, V5,
  V9, V10, V11): on synthetic data intervals hold at every level; on the Swiss
  rivers daily intervals hold from 50% to 95% over 48 held-out years per version,
  but ⚠ 99% daily intervals held only 98.0–98.2% of days (real errors have
  heavier tails than the normal distribution assumed), and in the later
  validation years 95% intervals held 91–95% (V5).
- Validation V11 (the check and the correction over 48 held-out years per
  version), and part C of V9 (the correction from cross-validation of the
  calibration years, applied to the later years).
- [docs/UNCERTAINTY.md](docs/UNCERTAINTY.md): every uncertainty statistic and
  test explained for water quality scientists (what, when, why, how, what it
  rests on in the literature, and how to defend it), with figures drawn by
  `docs/figures/make_uncertainty_figures.py`. A test checks that every link in
  the documentation resolves.
- Validation V12: does the cross-validation's design flatter the model? It
  hides a held-out year's water temperatures and predicts the year from its own
  air temperature and discharge, the standard design (docs/METHODS.md §11 now
  explains why, with references). Over 96 held-out river-years, hiding the
  inputs from the calibration too changed no year's RMSE by more than
  0.007 °C (against gap-tolerant mode with the inputs kept), and a 60-day
  buffer around the year by no more than 0.008 °C.
  Calibrating on earlier years only was as accurate, but its 90% intervals held
  on 87.1% of days against 89.6%: for predictions of future years, quote the
  later-years tests (V5, V10) as well. Jackknife parameter intervals depend a
  little on the optimizer: another seed alone changed their typical width by a
  factor of 0.65–1.06 (0.35–1.55 before the DE stopping rule was fixed); check
  poorly determined parameters with a second `random_seed`. `validation/run_all.py --only` now keeps the other
  checks in `REPORT.md`.
- Validation V13: the parameters and errors of the first air2stream paper
  (Toffolon and Piccolroaz, 2015), transcribed to
  `data/switzerland/published/Toffolon_Piccolroaz_ERL2015.csv`. With RK4 all
  30 published errors are reproduced within their rounding (with
  Crank-Nicolson they are missed by up to 0.32 °C: the 2015 results used RK4,
  the 2016 results Crank-Nicolson), and DE calibration with RK4 returns the
  published parameters in all 15 cases. The paper's stated Rhône periods
  (1984–2003 / 2004–2013) are a misprint for the distributed split.
- Validation V15 and `data/british_columbia/`: the dataset of Callahan and
  Moore (2025, *Hydrological Processes* 39(1), e70033; Zenodo,
  https://doi.org/10.5281/zenodo.14502248, CC BY 4.0), with the parameters,
  inputs and simulated water temperatures of air2stream for 23 rivers in
  British Columbia. Given their parameters and inputs, pyair2stream computes 45
  of the 46 published simulated series to within 0.00013 °C on every day, and
  its calibration fits every station at least as well. The 46th, a
  calibration record starting on 1 November 2012, was run by its authors as if
  it started on 1 January (the original program reads records by row); read
  that way, it is reproduced to 0.00001 °C. The published parameters are not
  where pyair2stream's calibration lands (same values at none of 23 stations),
  but where its DE-MCMC sampler converged (16 stations, with chains of up to
  100,000 steps), 115 of 128 published values lie inside its 90% intervals.
  The exceptions are stations 08HA002 and 08KH006, whose published
  calibrations are far from the best fit: recalibrated, they predict
  2021–2022 with RMSE 0.75 °C instead of 1.09 °C, and 1.10 °C instead of
  1.18 °C.
- Validation V16: how many years of data a calibration needs. Each Swiss
  river was calibrated on 1, 2, 3, 5 and 10 consecutive years (up to six
  placements of each) and on its whole record, and every calibration predicted
  the same later years. With 3 or more years, version 8's median RMSE was
  within 0.034 °C of the whole record's; a single unusual year cost up to
  0.17 °C.
- Validation V17: air2stream against four regressions of water temperature on
  air temperature (the day's, averaged, an S-curve, and averaged with
  discharge), all fitted on the same years, on the 3 Swiss and 23 British
  Columbia rivers. In held-out years, version 8's daily RMSE was below the
  best regression's on 23 of 26 rivers (medians 0.74 against 0.89 °C, and 0.96
  against 1.17 °C). On yearly peaks, the hottest days and the 2021 heat dome
  its lead was smaller (better on 14 of 23 British Columbia rivers).
- [docs/PUBLISHED_RESULTS.md](docs/PUBLISHED_RESULTS.md): the published
  air2stream results pyair2stream reproduces (Toffolon and Piccolroaz, 2015;
  Piccolroaz et al., 2016; Callahan and Moore, 2025), the errors found in each
  publication, and the conventions of the original program behind them.
- Validation V14: prediction intervals on the hottest days of held-out years.
  On the 10% of days predicted to be hottest, version 8's 90% intervals held
  on 91% of days; ⚠ version 5's on 84%, because it predicted those days about
  0.5 °C too warm. Checking intervals only on the days with the highest
  measurements is a biased test: there even a correct 90% interval is exceeded
  on 14.5–21.4% of days.

### Changed
- ⚠ A FORWARD run with prediction intervals refuses a DE-MCMC chain fitted with
  another model version, integrator or `Qmedia`. DE-MCMC now records them in the
  chain's `_meta.json`; chains from 0.4.2 or earlier cannot be checked, and a note
  says so.
- The README, the User Guide and the example READMEs are rewritten in plain
  language for new users, with shorter sentences and a clear path: prepare the
  data, write the settings, run, read the results. New: what to check first in
  the outputs (USER_GUIDE §8), how to choose calibration and validation years
  and how to build a daily file from raw data (§5), and where to get help. The
  README's summary of the validation is shorter; the evidence is unchanged in
  `validation/REPORT.md`.
- The weekly error persistence (`rho_timescale: "weekly"`) is documented in
  full: what it is, why it is used, how it is estimated and used, what it
  changes, the evidence, when to use the alternative, and answers to likely
  questions and criticisms (docs/UNCERTAINTY.md §5; docs/METHODS.md §12, "Error
  persistence"). Five new figures (`docs/figures/U15`–`U19`, drawn by
  `make_uncertainty_figures.py --rho`) show how the two estimates are made, the
  conversion from week-to-week correlation to ρ, simulated errors with each ρ,
  how four single-ρ choices reproduce the real errors from 1 to 90 days on the
  three Swiss rivers, and the validation evidence. The later figures of
  UNCERTAINTY.md are renumbered. METHODS no longer quotes a fast/slow split of
  the error variance that a reproducible fit did not confirm for version 5 on
  two rivers; it quotes the measured week-to-week correlation instead (0.52–0.81,
  against 0.25–0.51 implied by the daily ρ).
- Continuous integration also tests Python 3.14, the newest stable release
  (3.9, 3.12 and 3.14).
- ⚠ **Data checks: one checker for every file, and the validation file before
  calibrating.** `pyair2stream.data_checks.check_table` now checks every file a
  run reads (the calibration file, a FORWARD scenario file and the validation
  file), and the pre-analysis report uses it too. The validation file is checked
  right after the calibration file, before any calibration: a problem in it used
  to stop the run only after the calibration and MCMC had finished (103 s in a
  test with example 02; hours with long chains), with no plots written. Every
  message names the file, the column and the first line concerned: a date with
  no row, a repeated date, dates out of order, a blank or unreadable date, text
  that is not a number (with a hint for decimal commas), a column name with a
  space, and the first missing day of a gap. New errors: a calibration or
  validation file without a `T_water` column or without any measurement (before:
  "n_dat is 0 after aggregation", for validation only after the calibration). A
  FORWARD file shorter than a year is now named as a scenario file (the 365-day
  minimum is now documented). New warnings: measured validation days that are
  also measured calibration days (the validation score is then not independent),
  and a validation file in a cross-validation run, which does not use it.
- `analyze_timeseries` makes exactly the checks a run makes, with the same
  settings, and its report starts with "A run would accept this data" or "A run
  would STOP on this data" and the reasons. It counts dates with no row as
  missing days (it used to report "0 missing" for a file with 10 rows deleted).
  Its default is now `gap_tolerant=False`, as in a run. New arguments: `period`,
  `calendar`, `min_theta_floor`, `source`; new summary keys: `errors`,
  `warnings`, `run_would_stop`, `missing_dates`.
- `merge_timeseries` and `read_and_resample` report what they cannot use: rows
  whose time cannot be read (they used to be dropped silently), values of `-999`
  (now treated as missing; they used to be averaged in), and days with fewer than
  half the usual number of readings. Text that is not a number stops them with a
  message naming the file, column and line. New options: `na_values` (per file)
  and `min_readings_per_day` (days with fewer readings are left blank).
- Gap-tolerant runs now also say when `warmup_drop_days` is much longer than
  the fitted model needs. The note gives a warm-up of about three relaxation
  times, and how many more measured days it would score (USER_GUIDE §10).
- **Examples.** Example 05 (gaps) is expanded, with six figures: checking the
  data first; missing water temperature filled by the model (July-August 2006
  hidden: RMSE 0.44 °C against the hidden measurements); interpolation across a
  three-week air temperature gap; and the stretches gap-tolerant mode scores.
  Its new `gap_study.py` tests when gap-tolerant mode works (about 130
  calibrations). For gaps of a month to a year, gap-tolerant mode changed the
  calibrated model least (at most 0.07 °C in the predictions for other years,
  against up to 1.5 °C for a straight-line fill and 0.16 °C for the seasonal
  average). For scattered one-day gaps, a warm-up of about three relaxation
  times (4 days on the Mentue, 2–11 days on 26 rivers) doubles the days scored
  with 5–10% of days missing; a 0-day warm-up flatters the reported fit. New
  example 07 prepares input files from
  raw logger files (merge, check, fill, split, run). New example 08 projects a
  warmer climate (+2 °C air, with and without 20% less summer flow), with paired
  changes and checked and corrected yearly peaks. Example 06 adds a sensitivity
  analysis. The examples index maps tasks to examples.
- **Example 09: integrator stability.** The B series (`compute_B_series`) of
  every model version with the 30 published parameter sets of the three Swiss
  rivers; each integrator's amplification factor, measured through the package;
  the B-series (Butcher trees) behind the limits 2 and 2.785; every integrator
  with every parameter set against a fine-step solution, with the package's
  checks; scenario flows from 0.1 to 3 times the record. It shows that a run
  can be unstable without blowing up (the 0 °C floor holds it, and only the B
  check stops it), that some wrong runs are only warned about, that the RK4
  calibration of the Dischmabach's version 5 (2015) relies on RK4's behaviour
  just under its limit, and how to compute, from the B series, how much a
  difference can grow over a stretch of days.

### Fixed
- ⚠ **DE calibration could stop a third of the way through, on a worse fit.**
  It used SciPy's default stopping rule: stop once the population's scores
  agree to within 1% of their mean. For NSE (about 0.98) that allows a spread of
  0.01, ten times the difference between a good and a poor fit. The local polish
  then ended at the best member's nearest optimum, which could be a worse fit
  with very different parameters. In 125 calibrations of the Mentue with gaps,
  4 ended worse by 0.0009–0.0029 in NSE, with `a5` = 5–11 instead of 2.5,
  after about 30 of 100 generations. On the 23 British Columbia rivers of
  validation V15 it cost more: 12 calibrations had ended on a worse fit, by up
  to 0.09 °C in RMSE. Fold calibrations that stopped early at scattered points
  had also made the cross-validation (jackknife) parameter intervals about
  twice as wide as they are. The rule is now `optimization.tol` (default
  0.001), which found the best fit in all 125 Mentue calibrations.
  Calibrations take about twice as long. DE-MCMC (its starting fit) and
  cross-validation folds use the same rule. In the eight examples, which had
  converged, best-fit parameters moved by at most 0.007 and simulated
  temperatures by at most 0.008 °C. Results drawn from MCMC samples moved
  slightly more, because the chain starts from the fit: example 03's
  corrected probabilities went from 0.59/0.33/0.12 to 0.57/0.34/0.14. The
  examples and their figures are rerun. The whole validation suite is rerun
  with the fix: the same 11 checks pass and the same 4 (V5, V9, V10, V14)
  miss some criteria, for the same reasons. Most numbers moved by a few
  tenths of a percentage point. The exceptions show how much the early stop
  had cost: in V12, the design differences it had seemed to show were mostly
  optimizer scatter (hiding a held-out year's inputs changed no year's RMSE by
  more than 0.007 °C, against 0.034 °C before), and in V15 the jackknife
  intervals are half as wide. The full suite now takes several hours; run it
  in parts with `--only` where jobs are limited in length.
- Monthly scoring (`time_resolution: "1m"`) of a 360-day-calendar record longer
  than about 60 years stopped with `IndexError`.
- Documentation errors: example 02 gave the parameter ranges as 95% ranges
  with old values (they are 90% ranges: `a5` 2.02 to 4.24); the code in
  examples 03 and 04 and in USER_GUIDE §12 used paths and names that did not
  run as written; `full_simulation_*` outputs were described as covering the
  whole record (they cover the calibration file, on every day);
  docs/METHODS.md §6 and USER_GUIDE §9.1 called all published parameters
  Crank–Nicolson parameters (those of Toffolon and Piccolroaz, 2015, are RK4
  parameters); example run times are updated.

## [0.4.2] - 2026-10-05

### Fixed
- ⚠ DE-MCMC with weekly or monthly scoring (`time_resolution` `Nw` or `1m`) and
  the least-squares likelihood (the default since 0.4.1) applied the daily
  effective sample size n(1 − ρ)/(1 + ρ) to the weekly or monthly means. Means
  of blocks of days are much less correlated from block to block than days are,
  so parameter intervals and prediction bands were too wide (parameter
  intervals by a factor of 1.6–4.4 for ρ = 0.5–0.95). The likelihood now uses
  the exact factor for block means of AR(1) errors, n/n_eff = 1 + 2·r_b/(1 − ρ^m)
  (docs/METHODS.md §12), recorded as `likelihood_variance_factor` in the
  chain's `_meta.json`. Daily scoring is unchanged. The methods documentation
  said ρ played no part with weekly or monthly scoring; that held only for the
  exact AR(1) likelihood, which now warns that it treats block errors as
  independent.
- A configuration section left empty, for example `uncertainty_options:` with
  every line under it commented out, stopped with `AttributeError`. It now
  means the defaults, as when the section is absent.

### Changed
- ⚠ **ρ, the error persistence behind DE-MCMC intervals and FORWARD
  probabilities, is now estimated from week-to-week persistence by default**
  (`uncertainty_options.rho_timescale: "weekly"`). The previous estimate, the
  correlation of consecutive days' errors, is still available as
  `rho_timescale: "daily"`. Model errors have a fast part (days) and a slow part
  (weeks to a season); matched to consecutive days, ρ ignores the slow part, so
  bands for weekly to seasonal quantities (a 7-day mean, a yearly peak, the
  probability a limit was exceeded) were too narrow and the interval for the
  seasonal timing parameter `a7` was about 1.5 times too narrow in the
  known-truth test. On the Swiss rivers, in years not used for calibration,
  90% bands for 7-day means now held 89–94% of observed values (83–88% before),
  and version 8's 90% ranges for yearly statistics 73–93% of river-years
  (67–87% before). The weekly estimate is the larger of the consecutive-day ρ and the ρ
  that reproduces the correlation of 7-day mean errors one week apart. It falls
  back to the consecutive-day ρ when fewer than 20 weeks of scored errors are
  available. The reasons, in theory and in the validation, are in
  docs/METHODS.md §12.
  Daily bands barely change. Bands for weekly and longer quantities, and
  intervals for the parameters, widen where the errors have a slow part; the
  intervals of the fast-acting parameters (`a2`, `a3`) are wider than they need
  to be, as they were with the daily ρ. Prediction intervals and probabilities
  from DE-MCMC and FORWARD runs change; rerun them. A FORWARD run that reuses a
  0.4.1 chain keeps that chain's ρ, recorded in its `_meta.json`.
- The exact AR(1) likelihood (`likelihood: "exact"`) always uses the
  consecutive-day ρ, which is what it models; `rho_timescale` then sets only the
  ρ of the error added to predictions. The chain's `_meta.json` records both
  (`rho`, `rho_likelihood`) and `rho_timescale`.

### Added
- ⚠ A warning when a simulation is physically implausible in a way its score
  may not show: the relaxation rate B is negative on some day, or the daily
  simulation zigzags from one day to the next (checked after calibration,
  before DE-MCMC sampling and in FORWARD runs). Weekly or monthly means cannot
  see a zigzag: with the authors' bounds, which allow a negative `a2` and `a3`,
  9 of 30 weekly-scored calibrations of version 5 on synthetic data ended on
  such a parameter set, with daily errors of about 11 °C and weekly means that
  fit (validation V4, case J). **If you calibrated on weekly or monthly means,
  check your parameters**, and set the minimum of `a2` and `a3` to 0
  (docs/METHODS.md §7).
- A warning when ρ reaches its limit of 0.99, which usually means a systematic
  error such as a seasonal bias; and a note when a FORWARD run reuses a chain
  whose ρ was estimated at another time scale than the run's `rho_timescale`
  (for example a 0.4.1 chain, whose ρ came from consecutive days).
- `bias_by_month_<period>_*.csv`/`.png` (calibration and validation) and
  `cv_bias_by_month.csv`/`.png` (cross-validation, held-out years): the mean
  error by calendar month, season and year with a 95% interval from the
  year-to-year spread, to show a bias in one season that a whole-year score
  hides (docs/METHODS.md §7). Cross-validation folds now keep their dates
  (`FoldResult.dates_held_out`), and `cross_validate(..., return_folds=True)`
  also returns the folds.
- Validation V1 part B: the calibration score (RMS, NSE, KGE) and the weekly and
  monthly averages it uses agree with the original Fortran, with and without
  gaps in the record and for two values of `prc`.
- Validation V9: probabilities that a yearly statistic exceeded a limit. They
  come true as often as stated on synthetic data; on the Swiss rivers version 8's
  beat going by past years, but their 90% ranges held in only 73–93% of years,
  and version 5's did no better than past years for the yearly peaks.
- Validation V10: calibrating on the coolest (or highest-flow) years and
  predicting the warmest (or lowest-flow) ones cost at most 0.07 °C of RMSE;
  version 5 does not suit the Rhône whichever years it is calibrated on.

## [0.4.1] - 2026-10-04

### Fixed
- Weekly or monthly scoring (`time_resolution` `Nw` or `1m`) with `prc: 0` stopped
  with `ZeroDivisionError` when a week or month had no observation. `prc` must
  now be above 0 and at most 1, and a block without observations is never scored.
- Gap-tolerant cross-validation scored the first `warmup_drop_days` of a segment
  that starts inside a held-out year (after a real gap in the forcing), which
  calibration and validation do not score. These days are now excluded, and
  `n_obs_held_out` counts the days actually scored. Runs without gaps are
  unchanged.

### Changed
- ⚠ **DE-MCMC now uses the least-squares likelihood by default**
  (`uncertainty_options.likelihood: "least_squares"`), widened for the
  autocorrelation of the errors by the effective sample size n(1 − ρ)/(1 + ρ).
  The previous exact AR(1) likelihood is still available as `likelihood:
  "exact"`. On real rivers the exact AR(1) likelihood moved the parameters away
  from the best fit, so its prediction bands were centred on slightly worse
  predictions (validation V5: band centres up to 0.11 °C from the best fit's,
  against 0.01 °C with the new default). The new default stays on the best
  fit, its parameter intervals contained the truth at least 90% of the time for
  versions 5 and 8 in the known-truth test (version 8: about 75% before), and
  its 7-day mean bands held better on real rivers (83–88% against 76–88%).
  Prediction intervals and probabilities from DE-MCMC and FORWARD runs change;
  rerun them.

### Added
- A warning when `T_air` is outside −60 to 60 °C or `T_water` outside −2 to
  50 °C, which usually means a missing-value code other than `-999` (such as
  `-99` or `-9999`) or a unit error. The values are still used as given.
- A clearer message when a gap-tolerant run's file has no water temperatures.
- Validation: one full report per check (`validation/reports/`), with figures;
  V2 compares the published parameters with 90% intervals around the same
  least-squares estimator (part F); V4 and V5 test both likelihoods.

## [0.4.0] - 2026-09-25

A correctness review found several defects that gave wrong results **without
any error**. If you used an earlier version, check the items marked ⚠ against
your runs.

### Fixed
- ⚠ **`-999` in `T_air` was used as an air temperature of −999 °C** in the
  default (non-gap-tolerant) mode, pulling water temperature to 0 °C. `-999` is
  now treated as missing everywhere, like an empty cell (so an incomplete `T_air`
  or `Discharge` now stops with an error, as documented).
- ⚠ **FORWARD runs used non-zero values of unused parameters** in
  `parameters_forward` with the default `CRN` integrator (e.g. `a6` for version
  3), changing results by up to several °C. Unused parameters are now always zero.
- ⚠ **Validation plots showed the calibration period's prediction band**,
  titled as a prediction interval. Bands are now matched by date and drawn only
  where they were computed; a FORWARD run uses its own band.
- ⚠ **`R2` in `goodness_of_fit_*.csv` and plot titles was actually NSE.** Both
  are now reported: `NSE`, and `R2` as the squared correlation.
- ⚠ **FORWARD runs not starting on 1 January** began with a wrongly phased
  warm-up year (2–3 °C error in the first days, fading over weeks).
- ⚠ **Gap-tolerant cross-validation** computed the NSE denominator from slightly
  more days than it scored, inflating NSE.
- ⚠ **Prediction intervals with weekly/monthly `time_resolution`** added daily
  noise with the (smaller) spread of weekly/monthly means. They now use the
  daily residual spread (no change at `1d`).
- **DE-MCMC was not reproducible** with `random_seed` set (the sampler's random
  generator was not seeded).
- DE calibration no longer prints `RuntimeWarning: overflow encountered in
  square`. It came from SciPy's convergence test when some trial parameters make
  the simulation run away; the search itself is unchanged.
- Saved ensembles (`save_ensemble`) held values near −999 on gap days; now NaN.
- A FORWARD run overwrote `calibration_metadata.json` when it shared the
  calibration's output folder; FORWARD runs no longer write this file.
- The sensitivity plot's unit label was wrong by a factor of 100 (the index is
  °C per 100% change in the parameter).
- DE now keeps its own best result if the L-BFGS-B polish ends worse.
- Invalid `version`, `run_mode`, `integrator`, `objective_function`, bounds that
  are not 8 values or have min > max, and `time_resolution` values such as `2m`
  (silently treated as monthly) are now rejected with a clear message. The
  optimizers refuse to run when no parameter is free (e.g. bounds missing).
- Messages no longer point to documents that do not exist.
- ⚠ **Paired scenario differences carried two independent sets of random
  error**, one from each run, so the band of a difference was much too wide
  (for two identical scenarios on the Mentue, which should differ by exactly
  zero, the difference had a standard deviation of about 0.7 °C). Each parameter draw now gets the same error in both runs, and it
  cancels.
- In AR(1) FORWARD runs, ρ was re-estimated from the observations being
  predicted when there were any. It now comes from the calibration, like σ.
- A `validation_data` path that does not exist (for example a typo) silently
  skipped validation; it now stops with an error.

### Added
- Prediction-interval **coverage check**: the share of observed days inside the
  band is printed and saved in the `_meta.json` sidecar.
- Gap-tolerant runs warn when `warmup_drop_days` is too short for the calibrated
  model to forget each segment's approximate start value.
- `goodness_of_fit_*.csv` now also reports `N` and `NSE`.
- A `DE-MCMC` run with `cross_validation.enabled` now warns that the block is ignored.
- **Validation suite** (`validation/run_all.py`, results in
  `validation/REPORT.md`): Fortran equivalence on real inputs, the published
  results and parameters of Piccolroaz et al. (2016) (including runs of the
  original program's own calibration, with Crank–Nicolson and with RK4),
  known-truth recovery, interval
  calibration, real-river prediction, numerical accuracy, gaps, and exact
  workflow and scenario answers.
- **Parameter confidence intervals from cross-validation**: `cv_results.csv`
  now includes delete-one-year jackknife rows (`jackknife_se`,
  `jackknife_90_lower`, `jackknife_90_upper`). In validation (V4) they contained
  the true parameters 83–94% of the time for every model version; the plain
  spread between folds (`std`), by contrast, is far too small to use as an
  uncertainty (35–56%).
- FORWARD runs without `parameters_forward` use the calibrated parameters in
  `paths.calibration_metadata`, so they need not be copied by hand.
- The FORWARD ensemble's `_meta.json` records the noise model, σ and ρ used.
- A note is printed when `RK4`, `RK2` or `EUL` is selected: with a one-day step
  they can be inaccurate even when stable.
- The Fortran comparison tests now include discharge that varies from day to day.

### Removed
- **`run_mode: "DE-CV-MCMC"`.** It ran a cross-validation only to choose where
  the MCMC sampler starts, and gave the same parameter and prediction intervals
  as `DE-MCMC` (to within 1.5% of their width on the Mentue; validation report
  at commit b48b6dc) while taking longer. A config that asks for it now stops
  with a message pointing to `DE-MCMC`. Cross-validation itself is unchanged;
  note that the spread of its parameters is not a confidence interval.

### Changed
- **PSO calibration is about 4 times faster.** The data were sent to a worker
  process with every particle evaluation, which cost more than the model run
  itself; each worker now receives them once. Results are unchanged. A
  500-particle, 500-iteration calibration of the Mentue takes about 30 s on 4
  cores (the original Fortran: about 40–70 s on one core, depending on how it
  is compiled).
- ⚠ **The default `noise_model` is now `"ar1"`** (was `"iid"`). Real model
  errors persist from day to day; with `"iid"`, 90% intervals for 7-day means
  contained only 39–62% of observed values on the Swiss rivers, against 76–88%
  with `"ar1"` (validation V5). Daily intervals are about the same width
  either way. Set `noise_model: "iid"` to get the previous behaviour.
- ⚠ **DE-MCMC sampling.** The sampler uses the differential-evolution move
  instead of emcee's default stretch move (3–4 times faster on air2stream's
  correlated parameters). It runs until converged (at least 50 autocorrelation
  times and split-R̂ below 1.01) or until `mcmc_steps`, which is now a maximum
  with default 20,000 (was a fixed 1,000, usually far too short). If it has not
  converged it stops with an error (`strict_convergence: true`, new default)
  instead of warning. The saved chain is thinned. Intervals from earlier
  versions' default runs were often based on unconverged chains.
- **Examples replaced** by six worked examples on the Mentue (quickstart,
  uncertainty, compliance, scenario, gaps, cross-validation). The old examples
  were removed: several no longer matched the code, and the Hopelands data had
  no recorded source or licence.
- The Swiss river data moved to `data/switzerland/`, with their source, licence
  and checksums.
- Documentation rewritten for clarity: README, USER_GUIDE, and a single
  step-by-step methods description, `docs/METHODS.md`, which replaces the five
  separate topic documents in `docs/`.

Entries below cite review reports under `docs/audit/`, which have since been
removed from the repository (they remain in the git history).

## [0.3.0] - 2026-08-28

Three defects found during a follow-up audit targeting the water-abstraction and
climate-projection scenario studies: running a fitted model on discharge different
from calibration, and propagating parameter uncertainty into a paired scenario
comparison.

### Fixed
- **`Q == 0` no longer crashes or silently mis-simulates, in every integrator**
  (`model.py`, `model_numba.py`, `io.py`, `config.py`). For versions 4, 7, and 8,
  discharge only enters the ODE through `theta = Q/Qmedia`, with `theta ** a4` as a
  divisor. At `Q == 0` this either raised a bare, uncaught `ZeroDivisionError`
  (`a4 > 0`) or silently evaluated to `inf`, collapsing that day's (and every
  subsequent day's, in the same segment) simulated temperature towards zero with no
  error, no NaN, and no warning (`a4 < 0`, the sign optimizers empirically tend to
  select) -- undetectable by `check_numerical_divergence`, since the resulting value
  stays inside the plausible temperature range. This affected both a naturally-
  occurring zero-flow day in the calibration record (crashing DE calibration the
  first time it sampled a positive `a4`) and a `FORWARD`-mode scenario discharge
  file (naturalised flow, climate projection). `read_Tseries` now raises a clear
  `ValueError` naming the offending index/date and count for both the calibration
  and scenario record, before any parameter vector or integrator is involved. A new
  opt-in `min_theta_floor` config key clamps `theta` away from zero by a small,
  documented epsilon instead, applied consistently across every integrator
  (`CRN`/`RK2`/`RK4`/`EUL`/`EXP`) and between calibration and `FORWARD`/scenario
  runs. Versions 3/5 (which never evaluate `theta`) are unaffected; no new
  restriction was added for them. Gap-tolerant mode's existing (heavier) handling
  of `Q <= 0` days -- excluding them from every segment -- is unchanged.
- **Posterior/prediction-interval ensemble loops are now covered by a per-draw
  divergence guard** (`optimization.py`, `model.py`). `check_numerical_divergence`
  previously ran only on the single deterministic best-fit simulation --
  `optimization.forward_mode()`'s prediction-interval loop and
  `optimization._run_mcmc_uncertainty()`'s envelope loop (used by
  `DE-MCMC`/`DE-CV-MCMC`) each call `call_model()` once per posterior draw
  (hundreds to ~1000 times) with no divergence check at all. A single bad draw
  either crashed the whole batch with no context, or -- if it stayed finite -- was
  silently written into the percentile envelope CSV and the raw `.npz` ensemble
  that `scenario.paired_difference` consumes, with nothing flagging it. A new
  lightweight per-draw check (`model.is_numerically_divergent`) now runs inside
  both loops: by default (`uncertainty_options.on_divergent_draw: "drop"`) a
  divergent draw is excluded from the ensemble and the exclusion is reported on the
  console and in the run's sidecar metadata (`MCMC_chain_*_meta.json` /
  `Forward_Prediction_Ensemble_*_meta.json`, a new sidecar for `FORWARD` mode);
  setting `on_divergent_draw: "raise"` fails loudly on the first divergent draw
  instead. If the excluded fraction exceeds `uncertainty_options.
  max_divergent_fraction` (default 10%, mirroring `stability_error_fraction`), or
  every draw diverges, the run raises rather than silently proceeding with a
  depleted (or empty) ensemble. The saved `.npz` ensemble already excludes flagged
  rows (rather than requiring a separate validity mask), so it always matches the
  reported percentile envelope. Existing well-behaved `forward_mode`/`DE_MCMC_mode`/
  `DE_CV_MCMC_mode` runs are unaffected.
- **Paired scenario ensembles now require matched posterior draws, enforced rather
  than merely documented** (`optimization.py`, `scenario.py`).
  `scenario.paired_difference(ens_a, ens_b)` is documented as requiring both
  ensembles to have been generated from the SAME parameter draws in the SAME order,
  but only ever checked `.shape` -- two ensembles built from unrelated draws (e.g.
  because `forward_options.random_seed` was omitted, or set differently in two
  config files) passed silently and produced a plausible-shaped but statistically
  meaningless "paired" difference. `forward_mode()` (and `_run_mcmc_uncertainty()`,
  for consistency) now persists the resolved seed, requested sample count, a
  content hash + row count identifying the source chain, the drawn
  `sample_indices`, and the `valid_draw_indices` that actually survived per-draw
  divergence filtering, into the ensemble's sidecar JSON. A new
  `forward_options.reuse_sample_indices_from: <path to prior sidecar>` config key
  lets a second `forward_mode()` call reuse a prior run's exact indices --
  byte-identical regardless of global random state -- instead of drawing fresh
  ones, cross-checking the source chain and erroring on mismatch. A new
  `scenario.paired_difference_from_files(path_a, path_b)` loads both ensembles'
  provenance and raises `ValueError` if the source chain, requested sample count,
  requested indices, or surviving indices do not match exactly; this is now the
  documented recommended path. The existing shape-only `paired_difference()`
  remains available unchanged for advanced/same-process callers.

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
- CI `coverage` job: runs the suite with `NUMBA_DISABLE_JIT=1` so `model_numba.py`'s
  `@njit` kernels are traced by `coverage.py` instead of running as invisible
  compiled code (previously reported as ~6% covered regardless of actual
  exercise, making the overall number meaningless; audit report 08, 8.4).
- `tests/test_report08_e2e_cli.py`: end-to-end tests invoking the real CLI
  entry point (`main()`) for a `DE` calibration with a full validation period
  (asserting `1_`/`2_`/`3_` output files all exist and parse) and a
  gap-tolerant calibration followed by sensitivity analysis. Previously
  nothing in the suite invoked `main()` on a config exercising either path in
  full (audit report 08, Gap E / 8.5); the other two scenarios in the report's
  minimum set were already covered end-to-end in `tests/test_cli_and_io.py`.
- CI `examples-smoke` job: actually *runs* the `quickstart` and
  `forward_prediction_intervals` example scripts (the two that are
  self-contained -- checked-in or self-generated input data), instead of
  only parsing them. `examples/forward_prediction_intervals/run_example.py`
  gained a `--smoke` flag that cuts the DE population/MCMC chain down to the
  minimum needed to exercise every code path, so the job runs in seconds
  instead of minutes (audit report 08, Gap F / 8.6). This is exactly the kind
  of regression `tests/test_examples_smoke.py`'s static parse-only check
  could not catch (report 05, Defect E). The remaining example scripts read
  real station data or the Piccolroaz et al. (2016) supplementary dataset
  that is intentionally not vendored in this repository, so there is no
  small/fast config that makes them runnable in CI.

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
