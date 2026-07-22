# Date-Based Block Cross-Validation in pyair2stream

## 1. Overview

`pyair2stream` is a Python reimplementation of the Fortran `air2stream` model (Toffolon and Piccolroaz, 2015), which simulates daily river water temperature from air temperature and, optionally, discharge. The original Fortran implementation offers no mechanism for assessing out-of-sample predictive performance: a single calibration/validation split is performed, and there is no built-in procedure for testing whether a calibrated parameter set generalizes across different years or hydrological regimes.

`pyair2stream` adds a leave-one-year-out (and, more generally, leave-N-years-out) block cross-validation (CV) capability that does not exist in the Fortran reference implementation. This document describes the design rationale, the algorithm, its implementation, its configuration, its outputs, and empirical results from its application to a real river dataset. It is intended to serve as a citable technical description of this feature for use in derivative scientific work.

## 2. Motivation

Standard leave-one-out cross-validation (LOOCV), in which individual observations are withheld and predicted one at a time, is poorly suited to `air2stream`. The model is a stateful ordinary differential equation (ODE) integrator: each simulated day depends on the integrator's state carried forward from previous days, and the fitted objective functions (NSE, KGE, RMSE) are computed over temporally autocorrelated residuals. Withholding individual, scattered days would leak information from adjacent, temporally correlated in-sample days into the assessment of any withheld day, giving an overly optimistic and largely uninformative measure of generalization.

`pyair2stream` instead implements a grouped, date-based block CV scheme, withholding one full year (or a fixed number of consecutive years) of `T_water` observations at a time. This design:

- Tests generalization across distinct hydrological years (e.g., wet years, drought years, heat waves) rather than across arbitrary, autocorrelated days.
- Avoids information leakage between the training and held-out sets, since an entire seasonal cycle is excluded from calibration at once.
- Produces diagnostics on both predictive skill (per-fold and pooled NSE/KGE/RMSE) and parameter stability (variability of the calibrated parameter vector across folds), the latter serving as a practical diagnostic for equifinality and overparameterization.

## 3. Design Principles

The implementation (`pyair2stream/cross_validation.py`) is built around four design constraints, stated in the module's own documentation and enforced by its test suite:

1. **Folds are built from calendar dates, never row position.** Fold membership is derived from the year (and month, for water-year offsets) recorded in `data.date`, not from row indices. This keeps fold boundaries aligned with leap years, with gap-tolerant segment boundaries, and with the phase of the model's seasonal cosine term, regardless of any missing-data structure in the record.
2. **No changes to the ODE integrator or optimizer internals.** Cross-validation is implemented entirely as an outer masking/orchestration layer around the existing calibration pathway (`model.py`, `optimization.py`). It reuses the pre-existing missing-observation convention (`Twat_obs == -999.0`, already consumed by `aggregation()`/`funcobj()`) to hide a fold's `T_water` targets from the calibration objective, while leaving `T_air`/discharge forcing untouched in the default (non-gap-tolerant) mode. The ODE therefore continues to integrate through and beyond the held-out window using real forcing data, without requiring a re-spin-up.
3. **Minimal, reversible state mutation.** Only `data.Twat_obs` (and, in gap-tolerant mode, `data.Tair`/`data.Q`) are ever mutated, and only transiently for the duration of a single fold. Original values are restored via `try/finally` both after each fold and on any exception, so a failed or interrupted CV run cannot leave the shared `CommonData` state corrupted for subsequent use.
4. **The first eligible year can never be held out.** Because the model requires at least one prior year of continuous forcing data to spin up its state, the earliest calendar (or water) year in the record is always excluded as a candidate fold. If a user disables the default skip behaviour, they must instead configure a positive `min_train_years`; attempting to disable both raises a configuration error before any calibration is attempted.

## 4. Algorithm

### 4.1 Fold Construction

Fold membership is computed by `assign_year_groups()` and `build_folds()`:

- **Year labelling.** Every row is labelled with a "water year" derived from `data.date`. With the default `water_year_start_month = 1`, this recovers plain calendar years. For any other value (e.g., 10), rows in or after that month are relabelled as belonging to the following year — e.g. with `water_year_start_month = 10`, the period October 2013–September 2014 is labelled entirely as water year "2014". This lets a user align fold boundaries with a hydrologically quiet point in the annual cycle (e.g., a winter low-flow trough) rather than splitting a seasonal cycle mid-summer.
- **Eligibility window.** The earliest `min_train_years + int(skip_first_year)` labelled years are dropped from consideration entirely — they exist only to provide training/spin-up data and are never eligible to be held out.
- **Fold granularity.** With `unit = "year"`, each remaining eligible year becomes its own single-year fold. With `unit = "n_years"`, remaining eligible years are grouped into consecutive, non-overlapping blocks of `n_years_per_fold` years; a short trailing block (fewer years than the block size) is dropped rather than yielded as an under-sized fold, with a warning.
- **Minimum-data filter.** A candidate fold is discarded if it contains fewer than `min_valid_obs` genuine (non-missing) `T_water` observations, since a fold with too little held-out data would produce an unreliable or undefined goodness-of-fit score.

### 4.2 Per-Fold Procedure

For each eligible fold, `run_leave_one_year_out_cv()` performs the following sequence:

1. **Mask.** The fold's `T_water` observations are set to the missing-data sentinel (`-999.0`) via `_mask_fold()`. In gap-tolerant mode, the corresponding `T_air` and discharge forcing are also masked for that window, so that gap-tolerant mode's own segmentation logic treats the held-out period as a genuine data gap and restarts/re-splices the integration around it correctly, avoiding uncontrolled state drift over a long masked window. In the default (non-gap-tolerant) whole-series mode, forcing is left intact and the ODE free-integrates through the held-out window using real `T_air`/discharge data with no restart; this is an intentional and documented asymmetry, considered acceptable because the model is structurally mean-reverting, but it means CV behaviour under `gap_tolerant: false` and `gap_tolerant: true` is not perfectly symmetric.
2. **Recompute derived statistics without leakage.** Under non-gap-tolerant operation, discharge is also masked before `compute_qmedia()` is called, specifically to prevent the median-discharge statistic used for parameterizing thermal inertia in Versions 4/7/8 from being computed using the held-out fold's discharge values, then restored immediately afterward. In gap-tolerant mode, the day-of-year climatology used for gap-filling is likewise recomputed for the fold. `aggregation()` and `statis()` are then re-run so the calibration objective's internal index mapping (`I_inf`/`I_pos`) is rebuilt to exclude the held-out rows, and, in gap-tolerant mode, `detect_segments()` is re-run against the freshly masked forcing.
3. **Calibrate.** The configured optimizer (`PSO`, `DE`, or `LATHYP`) is run on the masked data via the same dispatch path (`main.run_optimizer`) used for a normal single calibration, optionally using a cheaper `optimizer_overrides` configuration (e.g., fewer generations/particles) to reduce the N-fold cost of cross-validation relative to a single production run.
4. **Restore forcing, then simulate.** Original forcing (`T_air`, discharge) is restored (in gap-tolerant mode, segments are re-detected against the restored forcing), and the full time series is simulated once with the fold's calibrated parameters.
5. **Score strictly on the held-out block.** NSE, KGE, and RMSE are computed via `_compute_fold_metrics()` using only the rows inside the fold's index range, comparing the model's simulation there against the original (unmasked) `T_water` values saved before step 1 — genuinely missing observations inside the held-out window are excluded from scoring as well, so sentinel values cannot contaminate the metric.
6. **Restore state.** `_restore_fold()` reinstates the original `T_water` (and, in gap-tolerant mode, `T_air`/discharge) values for the fold's rows before the next fold begins, or before returning control to the caller if this was the last fold or an exception occurred.

After all folds complete, the calibration/statistics state (`data.par`, `data.par_best`, `compute_qmedia`, aggregation, and summary statistics) is restored/recomputed against the full, unmasked record, so that the shared `CommonData` object is left in a consistent state for any subsequent operation.

### 4.3 Metrics

For a fold with held-out observations \(o_i\) and simulated values \(s_i\) (restricted to indices where both are valid):

- **NSE**: \(1 - \dfrac{\sum_i (o_i - s_i)^2}{\sum_i (o_i - \bar{o})^2}\)
- **KGE**: \(1 - \sqrt{(r-1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}\), where \(r\) is the Pearson correlation between \(o\) and \(s\), \(\alpha = \sigma_s/\sigma_o\), and \(\beta = \bar{s}/\bar{o}\)
- **RMSE**: \(\sqrt{\dfrac{1}{n}\sum_i (o_i - s_i)^2}\)

If a fold's held-out block has fewer than 10 valid observations, a warning is raised, since the resulting metrics are statistically unreliable at that sample size; the CV run still proceeds and reports the metric regardless (subject to the `min_valid_obs` eligibility filter applied at fold-construction time).

## 5. Configuration

Cross-validation is enabled by adding a `cross_validation` block to the YAML configuration file:

```yaml
cross_validation:
  enabled: true
  unit: year                  # "year" or "n_years"
  n_years_per_fold: 1          # number of years held out per fold (only used if unit: n_years)
  water_year_start_month: 1    # 1 = calendar year; e.g. 10 = Oct-Sep water year
  min_train_years: 1            # number of leading eligible years reserved purely for spin-up/training
  skip_first_year: true         # the very first calendar/water year is never a candidate fold
  min_valid_obs: 10              # minimum held-out observations for a fold to be scored
  optimizer_overrides:
    n_run: 20                   # optional cheaper per-fold optimizer settings
    n_particles: 20
```

| Field | Default | Meaning |
|---|---|---|
| `unit` | `"year"` | `"year"` yields one fold per eligible year; `"n_years"` groups eligible years into fixed-size consecutive blocks |
| `n_years_per_fold` | `1` | Block size in years, used only when `unit = "n_years"` |
| `water_year_start_month` | `1` | Month at which a labelled "year" begins; use a non-January value to align fold boundaries with a hydrologically quiet point in the seasonal cycle |
| `min_train_years` | `1` | Number of additional leading eligible years (beyond the mandatory spin-up year) reserved only for training, never held out |
| `skip_first_year` | `true` | Whether the very first eligible year is automatically excluded as a candidate fold; if set to `false`, `min_train_years` must be greater than zero |
| `min_valid_obs` | `10` | Minimum number of genuine `T_water` observations a candidate fold must contain to be scored; folds below this threshold are dropped before calibration |
| `optimizer_overrides` | `None` | Optional dict of optimizer settings (e.g. `n_run`, `n_particles`) applied only for the duration of CV folds, to reduce the N-fold cost relative to running the production calibration configuration N times |

When `cross_validation.enabled: true` and `run_mode` is one of `PSO`, `DE`, or `LATHYP`, the CLI entry point (`main.py`) diverts execution to the cross-validation driver instead of a single production calibration: it runs `run_leave_one_year_out_cv()`, writes the fold-by-fold summary to `cv_results.csv`, prints it, and returns — the normal single-calibration `forward()` validation run and post-processing/plotting steps are skipped for a CV run.

Cross-validation can also be invoked programmatically:

```python
from pyair2stream.cross_validation import run_leave_one_year_out_cv, summarize, CVConfig

cv_config = CVConfig(unit="year", min_train_years=1, skip_first_year=True,
                      optimizer_overrides={"n_run": 20, "n_particles": 20})

results = run_leave_one_year_out_cv(data, cv_config, run_mode="DE")
df = summarize(results)
```

## 6. Outputs

`run_leave_one_year_out_cv()` returns a list of `FoldResult` objects, one per fold, each recording:

- `label` — the held-out period identifier (e.g. `"2014"` or `"2014-2016"`)
- `held_out_start` / `held_out_end` — the first and last calendar dates of the held-out block
- `n_obs_held_out` — the count of genuine (non-missing) `T_water` observations inside the block
- `par_best` — the parameter vector calibrated with that fold withheld
- `nse`, `kge`, `rmse` — goodness-of-fit computed strictly on the held-out block
- `obs_held_out`, `sim_held_out` — the raw observed/simulated arrays for that block, retained to support pooled (micro-averaged) metric computation

`summarize()` converts this list into a `pandas.DataFrame` with one row per fold (metrics plus calibrated parameter columns `p1..pN`), followed by three summary rows:

- **`mean`** and **`std`** — the macro-average and sample standard deviation of NSE/KGE/RMSE and of each parameter, computed across per-fold values.
- **`pooled`** — NSE/KGE/RMSE computed on the concatenation of every fold's held-out observations and simulations, i.e. a micro-average giving every out-of-sample day equal weight regardless of which fold it belonged to. `pooled` is not meaningful for the parameter columns (there is no single "pooled" parameter set), so those cells are left as `NaN` in that row.

When run from the CLI, this DataFrame is written to `cv_results.csv` in the run's output folder.

## 7. Empirical Application

Leave-one-year-out CV was applied to a six-year daily record from the Dischmabach station (DAV-2327, Switzerland), using `version: 8`, `gap_tolerant: true`, and the `DE` optimizer, with `skip_first_year: true` and `min_train_years: 1` (so the first two eligible years serve only as spin-up/training, never as a held-out fold).

### 7.1 Per-Fold and Pooled Predictive Skill

| Fold | NSE | RMSE |
|---|---|---|
| 2005 | 0.9616 | 0.5828 |
| 2006 | 0.9605 | 0.6029 |
| 2007 | 0.9476 | 0.6653 |
| 2008 | 0.8690 | 0.9836 |
| 2009 | 0.9540 | 0.6267 |
| **mean** | 0.9386 | 0.6922 |
| **std** | 0.0393 | 0.1657 |
| **pooled** | 0.9410 | 0.7081 |

The 2008 fold shows a visibly weaker out-of-sample fit than the other four years, illustrating exactly the kind of year-to-year generalization gap that a single fixed calibration/validation split would not reveal.

### 7.2 Parameter Stability as an Equifinality Diagnostic

Because each fold recalibrates all eight parameters independently, the spread of the calibrated parameter vector across folds is itself a diagnostic of parameter identifiability:

| Fold | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |
|---|---|---|---|---|---|---|---|---|
| 2005 | 4.7445 | 0.6388 | 1.4213 | 0.2582 | 0.000 | 4.7540 | 0.5843 | 0.6211 |
| 2006 | 4.9223 | 0.6374 | 1.4317 | 0.2705 | 0.000 | 4.9123 | 0.5814 | 0.6406 |
| 2007 | 4.7376 | 0.5896 | 1.3583 | 0.2835 | 0.000 | 5.1776 | 0.5835 | 0.6728 |
| 2008 | 4.9732 | 0.6403 | 1.4651 | 0.2621 | 0.000 | 5.1907 | 0.5788 | 0.6475 |
| 2009 | 4.7659 | 0.6281 | 1.3981 | 0.2663 | 0.000 | 4.8862 | 0.5820 | 0.6354 |
| **mean** | 4.8287 | 0.6268 | 1.4149 | 0.2681 | 0.000 | 4.9842 | 0.5820 | 0.6435 |
| **std** | 0.1106 | 0.0214 | 0.0398 | 0.0097 | 0.000 | 0.1922 | 0.0021 | 0.0190 |

Parameters `p1`–`p4` and `p6`–`p8` vary by only 2–5% of their mean across folds, indicating they are well-identified by this record. Parameter `p5` sits at its lower bound (0.0) in four of the five folds and is only marginally non-zero in one — a pattern consistent with mild, localized equifinality in that single parameter (a discharge-scaled constant offset), rather than broad overparameterization of the full 8-parameter model. This kind of per-parameter stability analysis is only possible because cross-validation performs an independent recalibration per fold; it is not available from the Fortran reference implementation's single calibration/validation split.

## 8. Practical Considerations and Limitations

- **Cost scales linearly with fold count.** Since a full calibration is performed independently for each fold, an N-fold CV run costs approximately N times a single production calibration. `optimizer_overrides` is provided specifically to allow a cheaper per-fold configuration (fewer generations/particles) than the production calibration, trading some per-fold optimizer precision for overall CV runtime.
- **Asymmetric state-drift protection.** As noted in Section 4.2, only `gap_tolerant: true` mode gives the held-out window genuine restart protection against ODE state drift; in the default whole-series mode the integrator free-runs through the held-out window on real forcing data with no restart safeguard. This is judged acceptable given the model's mean-reverting structure but is a documented, intentional asymmetry that users should be aware of when interpreting CV results obtained without `gap_tolerant: true`.
- **Choice of `water_year_start_month`.** Because the model includes a phase-locked annual cosine term, a fold boundary that splits a seasonal cycle mid-winter or mid-summer can bias per-fold scoring. Setting `water_year_start_month` to a month where the river's seasonal cycle is quietest is recommended over the default calendar-year boundary for rivers with a strong, sharply-peaked seasonal signal.
- **Minimum sample size per fold.** Folds are only retained if they contain at least `min_valid_obs` genuine observations; per-fold metrics computed from very small samples (fewer than 10 valid observations) are flagged with a warning as being of limited statistical reliability, even where the eligibility filter allows them through.
- **Compatible run modes.** Cross-validation is currently wired to dispatch to `PSO`, `DE`, or `LATHYP` as the per-fold calibration algorithm; it is not applicable to `FORWARD` mode (no calibration is performed) or to the MCMC-based modes (`DE-MCMC`, `DE-CV-MCMC`) as an outer loop, since those already have their own internal use of a preliminary DE fit and/or (in `DE-CV-MCMC`) their own cross-validation-informed initialization.

## 9. References

- Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water temperature as a function of air temperature and discharge. *Environmental Research Letters*, 10(11), 114011. https://doi.org/10.1088/1748-9326/10/11/114011
- Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. (General rationale for grouped/blocked cross-validation over point-wise LOOCV under autocorrelated data.)
- Gupta, H. V., Kling, H., Yilmaz, K. K., and Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology*, 377(1–2), 80–91. (KGE definition used for fold scoring.)
- Nash, J. E. and Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I — A discussion of principles. *Journal of Hydrology*, 10(3), 282–290. (NSE definition.)

## Appendix: Source Reference

Implementation: `pyair2stream/cross_validation.py` (`CVConfig`, `FoldResult`, `assign_year_groups`, `build_folds`, `run_leave_one_year_out_cv`, `summarize`), with configuration parsing in `pyair2stream/io.py` and CLI dispatch in `pyair2stream/main.py::main`. Worked example with full results and diagnostic plot: `examples/cross_validation/README.md`. Repository: https://github.com/LukeAFullard/pyair2stream.