# pyair2stream — Gap-Tolerant Mode
## Technical Design Document

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problems Identified](#2-problems-identified)
3. [Issues Summary](#3-issues-summary)
4. [Solutions](#4-solutions)
5. [Implementation Plan](#5-implementation-plan)
6. [Checks Required at Each Stage](#6-checks-required-at-each-stage)
7. [Residual Limitations](#7-residual-limitations)
8. [Backwards Compatibility Guarantee](#8-backwards-compatibility-guarantee)

---

## 1  Introduction

The original air2stream model — and its initial Python port (pyair2stream) — requires that both the air temperature (`T_air`) and discharge (`Q`) time series are completely gap-free. This is a hard constraint enforced at data loading time: any missing value in either series causes an immediate error before calibration begins.

This constraint exists for good reason. The air2stream ODE integrates forward one day at a time, so a missing forcing value at timestep *j* would propagate directly into the state at *j+1* and every subsequent timestep. Missing discharge also corrupts the mean discharge `Qmedia`, which appears as a normalising scalar in every integration step for model versions 4, 7, and 8. The original authors therefore made completeness a precondition rather than something to silently paper over.

In practice, however, completely gap-free multi-year records are rare. Discharge gauges go offline for maintenance, freeze during winter, or are destroyed by floods. Air temperature loggers fail. Data are withheld during quality control. For many real-world catchments — especially in data-sparse regions — the completeness requirement either forces users to discard otherwise usable periods, apply gap-filling methods whose uncertainty is difficult to quantify, or abandon the model entirely.

The gap-tolerant mode addresses this by rethinking two foundational assumptions.

**The first assumption** is that `Qmedia` must be computed from the same series being modelled. In reality, `Qmedia` is a catchment characteristic — the long-term mean flow that defines what "normal" discharge looks like for that reach. It can legitimately come from longer historical records, regional hydrological analysis, or published catchment statistics. Allowing users to supply `Qmedia` externally decouples the normalisation scalar from the gaps in the modelling period and removes one of the two hard failure modes.

**The second assumption** is that the model must be integrated as a single continuous run from start to finish. This is a numerical convenience, not a physical necessity. The air2stream ODE is a first-order initial value problem (IVP): given a starting water temperature and a sequence of valid forcing values, it produces a valid trajectory forward in time. There is no mathematical reason why it cannot be restarted at any point with a new initial condition. If the forcing data contains gaps, the natural solution is to treat each contiguous block of valid `T_air` and `Q` as an independent integration segment, run the ODE separately over each one, and evaluate the objective function over all segments jointly. The calibration then finds the single parameter set that best explains the observed water temperature across all segments simultaneously — exactly as it would for a continuous record, but without requiring that the forcing be fabricated over the gaps.

Critically, this approach makes an important distinction: a gap in the *forcing data* (`T_air` or `Q`) is not the same as a gap in the *observation data* (`T_water`). `T_water` gaps are already handled gracefully in both the original Fortran and the Python port via the `-999` sentinel. The new mode extends the same philosophy to the forcing variables — gaps in what drives the model simply reduce the number of timesteps that can be simulated and evaluated, rather than preventing the model from running at all.

Together, these two changes form a coherent and physically principled extension rather than a workaround. The key design constraint throughout is **full backwards compatibility**: the gap-tolerant mode is opt-in via a single `gap_tolerant: true` flag in the configuration file. All existing behaviour is preserved by default, and users who do not set the flag will observe no change whatsoever.

The remainder of this document details the problems that motivated this design, the solutions chosen, the implementation plan, the checks required at each stage, and the limitations that users must understand before applying this mode to their data.

---

## 2  Problems Identified

### 2.1  Hard Requirement for Gap-Free Forcing Data

`read_Tseries()` in `io.py` raises a `ValueError` if any `T_air` or `Q` value is null or missing. This is the primary blocker: real-world datasets routinely contain gaps due to sensor failure, data transmission loss, or deliberate removal during quality control.

> **Severity: High** — prevents the model from running at all on any gapped dataset.

---

### 2.2  Qmedia Computed Solely from the Modelling Period

The mean discharge `Qmedia` is currently computed as the arithmetic mean of all `Q` values in the input file. If `Q` gaps coincide with extreme flood events — which is common, as gauges are most likely to fail at high flows — `Qmedia` is systematically underestimated. Since every integration step normalises discharge as `θ = Q / Qmedia`, this bias propagates into parameter estimation for the depth exponent (`par[4]`) across the entire calibration.

> **Severity: High** — silent systematic bias in parameter calibration for versions 4, 7, and 8.

---

### 2.3  Single-Pass ODE Integration Breaks on Gaps

`call_model()` integrates from index `0` to `n_tot-1` in a single loop with no mechanism to skip or restart around gaps. Introducing a `-999` sentinel into the `Tair` or `Q` array mid-run would cause the ODE to evaluate expressions such as `K = par[2] * (-999.0)`, driving `Twat_mod` to physically impossible temperatures and corrupting all subsequent timesteps.

> **Severity: Critical** — produces `NaN` or floating-point overflow; no recovery possible without a full restart.

---

### 2.4  Warm-Up Block Inherits Gaps from Year 1

`read_Tseries()` constructs a 365-day warm-up period by copying the first 365 rows of the raw data into indices `0–364`. If the raw data contains `T_air` or `Q` gaps in its first calendar year, those `-999` sentinels are propagated into the warm-up block. Any integration strategy that steps through indices `0–364` will encounter these sentinels immediately, causing the same overflow described in 2.3.

> **Severity: Critical** — the warm-up block cannot be assumed clean in gap-tolerant mode.

---

### 2.5  DOY Climatology Overwritten on Validation Pass

The gap-tolerant mode builds a day-of-year (DOY) climatology of observed water temperature to provide fallback initial conditions when restarting integration segments. However, `read_Tseries()` is called twice: once for calibration (`p == 'c'`) and again for validation (`p == 'v'`). If the climatology is recomputed on the validation pass, it is overwritten with validation-period observations — allowing the model to peek at validation `T_water` values when setting restart initial conditions. This constitutes **data leakage**.

> **Severity: Medium** — invalidates validation metrics; easy to fix by gating computation to the calibration pass only.

---

### 2.6  Objective Function Evaluates IC Transient

When a segment restarts with a climatological initial condition that differs significantly from the true river temperature, the ODE produces an artificial transient in the first 10–30 days as the model trajectory converges to the correct thermal state. If the objective function evaluates observations that fall within this transient period, the optimiser (PSO/LHS) will penalise the parameter set for a model error caused by the initial condition rather than the parameters themselves. This warps the calibration search toward spurious compensation.

> **Severity: Medium** — magnitude depends on how far the climatological IC is from truth; worst in anomalous years or for large thermally sluggish rivers.

---

### 2.7  Hard Requirement to Start on January 1st

`read_Tseries()` hard-errors if the input series does not start on January 1st. The restriction is a legacy of how the `tt` (fractional time-of-year) array was originally constructed by counting positions from the array start rather than reading actual calendar dates. There is no thermodynamic or mathematical reason for this restriction; it excludes valid datasets that begin mid-year.

> **Severity: Medium** — excludes datasets beginning after a gauge installation, a quality-controlled start date, or a water year boundary.

---

### 2.8  KGE Undefined for Small n_dat or Zero Variance

The KGE objective function computes a ratio involving the standard deviations of both observed and modelled series. If `n_dat < 2` — which becomes more likely when gaps fragment the calibration period — division by `n_dat - 1` produces a zero-division error. Similarly, if all observations within a segment happen to be identical, `std_obs = 0` causes division by zero in the KGE formula.

> **Severity: Low** — a pre-existing issue made more likely by segmented evaluation.

---

### 2.9  MNAR Gaps Inflate Performance Metrics

In hydrology, data gaps are almost never missing at random (MAR). They are Missing Not At Random (MNAR): gauges most commonly fail during extreme flood peaks or sub-zero ice events — the very conditions most important to model correctly. By restricting calibration and evaluation to valid segments, the extreme tails of the observed distribution are implicitly removed. This suppresses observed variance and artificially inflates NSE and KGE scores. A model scoring NSE = 0.88 on gapped data may perform substantially worse on a complete dataset containing the missing peaks.

> **Severity: Low (documentation issue)** — cannot be fixed in code; must be clearly communicated to users so that gap-tolerant metrics are not compared directly against continuous-record benchmarks.

---

## 3  Issues Summary

| # | Issue | Location | Severity | Resolution |
|---|-------|----------|----------|------------|
| 2.1 | Hard requirement for gap-free T_air and Q | `io.py` — `read_Tseries` | **High** | Gap-tolerant mode with sentinel (-999) support |
| 2.2 | Qmedia computed only from modelling period | `io.py` — `read_Tseries` | **High** | Allow user-supplied Qmedia from external sources |
| 2.3 | Single-pass ODE integration breaks on gaps | `model.py` — `call_model` | **Critical** | Piecewise IVP: segment detection + per-segment restart |
| 2.4 | Warm-up block copies gapped Year 1 data | `model.py` — `call_model_segmented` | **Critical** | Skip warm-up entirely in gap-tolerant mode; use IC instead |
| 2.5 | DOY climatology overwritten on validation pass | `io.py` — `read_Tseries` | Medium | Restrict climatology computation to calibration pass only |
| 2.6 | Objective function evaluates IC transient | `model.py` — `funcobj` | Medium | `warmup_drop_days` mask excludes first N days of each segment |
| 2.7 | January 1st start date hard requirement | `io.py` — `read_Tseries` | Medium | Option B: calendar-date `tt` calculation; remove date restriction |
| 2.8 | KGE undefined for n_dat < 2 or zero std | `model.py` — `funcobj` / `statis` | Low | Guard clauses; warn and return -999 sentinel |
| 2.9 | MNAR gaps inflate NSE/KGE scores | Documentation | Low | Prominent warning in USER_GUIDE; recommend NSE/RMS over KGE |

---

## 4  Solutions

### 4.1  Opt-In Flag and New Config Fields

A single boolean field `gap_tolerant` is added to `CommonData` (default: `False`) and read from the YAML configuration file. Every change described below is gated on this flag. The following additional fields are introduced:

- **`Qmedia_user`** (`Optional[float]`): externally supplied mean discharge, overrides the computed value when present.
- **`segments`** (`Optional[list]`): populated at runtime with `(start_idx, end_idx)` tuples for each valid contiguous block.
- **`doy_climatology`** (`np.ndarray`, shape 366): mean observed `T_water` by day-of-year, computed from the calibration series only, used as fallback initial conditions for segment restarts.
- **`warmup_drop_days`** (`int`, default `15`): number of days at the start of each segment excluded from objective function evaluation to avoid penalising the IC transient. User-configurable; should be tuned to the thermal memory of the specific river.
- **`eval_mask`** (`np.ndarray`, bool): precomputed boolean array of shape `(n_tot,)` marking which timesteps are eligible for objective function evaluation. Computed once after segment detection and reused on every PSO/LHS iteration.
- **`min_segment_days`** (`int`, default `30`): segments shorter than this threshold are dropped with a warning. Prevents calibrating against trajectories still dominated by initial condition uncertainty.

---

### 4.2  Calendar-Date tt Calculation (Option B) — Unconditional Fix

The fractional time-of-year array `tt` is rewritten to derive from actual calendar dates rather than array position. For each index, the day-of-year is computed from the date columns and divided by the number of days in that calendar year (accounting for leap years):

```python
doy = (pd.Timestamp(year, month, day) - pd.Timestamp(year, 1, 1)).days + 1
data.tt[idx] = doy / (366 if is_leap else 365)
```

This produces **identical output** to the existing code for series that start on January 1st, and correct output for any other start date. The January 1st restriction is removed in gap-tolerant mode. In legacy mode the check is retained to preserve exact Fortran-equivalent behaviour.

---

### 4.3  Relaxed Data Validation

In gap-tolerant mode, the hard errors on `T_air` and `Q` completeness are replaced with sentinel conversion: `NaN` values are filled with `-999.0` and the data is loaded. The **date spine check** (no missing rows) remains a hard error unconditionally — the user must supply a row for every calendar day, with `-999` or `NaN` for missing values. Missing rows would corrupt the `tt` array and date-based month detection in `aggregation()`.

A `Qmedia` safety check is added: if the computed `Qmedia` is zero or negative, a hard error is raised with a suggestion to supply `Qmedia` externally. A warning is also emitted if more than 50% of `Q` values are missing and no external `Qmedia` is provided.

---

### 4.4  External Qmedia

When `Qmedia` is supplied in the YAML (key: `Qmedia`), it overrides the value computed from the data. Both values are logged at runtime so the user can verify the override is reasonable. For model versions 3 and 5, which do not use `Q` in the ODE at all, `Q` gaps are entirely harmless and a message is emitted to that effect.

---

### 4.5  DOY Climatology (Calibration Pass Only)

The `T_water` DOY climatology is computed from calibration-period observations only (indices 365 to `n_tot`, skipping the warm-up copy) and stored on the data object. It is **never recomputed during the validation pass**. DOY bins with zero observations are filled by linear interpolation across the annual cycle. If no `T_water` observations exist at all, a hard error is raised — calibration is impossible without any observations.

---

### 4.6  Segment Detection

A new function `detect_segments()` scans the data array from index 365 (start of actual data, never touching the warm-up block). A timestep is considered valid if both `T_air` and `Q` are non-sentinel — or if the model version does not use `Q` (versions 3 and 5). Contiguous valid blocks are identified and returned as `(start_idx, end_idx)` pairs. Segments shorter than `min_segment_days` are dropped with a printed warning. After detection, an `eval_mask` is built that excludes the first `warmup_drop_days` days of each segment from objective function evaluation.

---

### 4.7  Segmented ODE Integration

A new function `call_model_segmented()` replaces `call_model()` in gap-tolerant mode. The warm-up block (indices `0–364`) is **completely ignored** — it is never integrated in this mode, eliminating the sentinel contamination risk described in Problem 2.4. Each segment is treated identically:

1. Set the initial condition: use `Twat_obs[seg_start]` if observed; otherwise use `doy_climatology[doy_of(seg_start)]`.
2. Integrate from `seg_start` to `seg_end` using the existing `_step()` helper (the refactored RK2/RK4/EUL/CRN logic).
3. Apply the `Tice_cover` floor at each step.
4. Set `Twat_mod` to `-999` for all indices outside valid segments.

The existing integrator logic is extracted into a private `_step(data, j)` helper function with no change to the arithmetic. This refactor is safe and enables reuse without code duplication.

---

### 4.8  Objective Function Guard and eval_mask

`funcobj()` is updated to use the precomputed `eval_mask`. When aggregating `Twat_mod` into `Twat_mod_agg`, any `I_pos` index not in `eval_mask` is excluded. This means observations falling within the `warmup_drop_days` buffer at the start of each segment are not included in the misfit calculation. In legacy mode, `eval_mask` covers all indices from 365 onwards — identical to current behaviour.

Guard clauses are added for KGE: if `n_dat < 2` or either standard deviation is zero, the function returns `-999.0` and prints a warning. `statis()` also guards against `n_dat == 0` with a hard error.

---

### 4.9  Output Flagging

The calibration and validation output CSVs gain three additional columns:

- **`Tair_gap`**: `1` where `T_air` was a sentinel, `0` otherwise.
- **`Q_gap`**: `1` where `Q` was a sentinel, `0` otherwise.
- **`segment_id`**: integer segment index; `-999` for warm-up or gap rows.

A plain-text `gaps_summary.txt` is written to the output folder reporting: `Qmedia` source and value, gap fractions for `T_air` and `Q`, segment count and date ranges, total valid days, and number of `T_water` observations used in calibration.

---

## 5  Implementation Plan

| Step | Title | File(s) | Backwards Compatible? | Notes |
|------|-------|---------|----------------------|-------|
| 0 | Add flag fields to `CommonData` | `config.py` | Yes — all default to `False`/`None` | `gap_tolerant`, `Qmedia_user`, `segments`, `doy_climatology`, `warmup_drop_days`, `eval_mask`, `min_segment_days` |
| 1 | Read new YAML keys | `io.py` | Yes — all keys optional | `gap_tolerant` defaults `False`; legacy configs unchanged |
| 2 | Relax data validation checks | `io.py` | Yes — gated on flag | Date spine always hard-errors; `T_air`/`Q` checks conditional |
| 3 | Rewrite `tt` calculation (Option B) | `io.py` | Yes — identical output for Jan 1 starts | Calendar-date DOY; removes Jan 1 restriction unconditionally |
| 4 | Compute `Qmedia` with user override | `io.py` | Yes — gated on flag | Log both computed and user-supplied values; warn if >50% `Q` missing |
| 5 | Build DOY climatology (cal pass only) | `io.py` | Yes — gated on flag | Restrict to `p == 'c'`; interpolate missing DOYs; error if no `T_water` |
| 6 | `detect_segments()` | `model.py` (new fn) | Yes — not called in legacy | Scan from idx 365; respect version vs `Q` dependency; drop short segments |
| 7 | Store `eval_mask` on data | `model.py` (new fn) | Yes — not called in legacy | Boolean array; `warmup_drop_days` excluded from segment starts |
| 8 | Extract `_step()` helper | `model.py` (refactor) | Yes — logic unchanged | Pulls RK2/RK4/EUL/CRN integrator body into reusable private function |
| 9 | `call_model_segmented()` | `model.py` (new fn) | Yes — not called in legacy | No warm-up block; IC from obs or DOY clim; integrates strictly within segments |
| 10 | Guard `statis()` and `funcobj()` | `model.py` | Yes — pre-existing edge cases fixed | `n_dat == 0` hard error; KGE guards for `n_dat < 2` and zero std |
| 11 | Update `funcobj()` with `eval_mask` | `model.py` | Yes — legacy uses full mask | Skip days outside valid segments and `warmup_drop_days` buffer |
| 12 | Fork `sub_1()` on `gap_tolerant` | `optimization.py` | Yes — gated on flag | Single line change; all optimisers inherit automatically |
| 13 | Update `forward()` and validation | `main.py` | Yes — gated on flag | Re-run `detect_segments()` on validation data; handle zero-segment case |
| 14 | Flag outputs and write gap summary | `main.py` | Yes — additive columns only | `Tair_gap`, `Q_gap`, `segment_id` columns; `gaps_summary.txt` report |
| 15 | Documentation | `USER_GUIDE.md` | n/a | MNAR warning; `Qmedia` guidance; `warmup_drop_days` explanation; KGE caveat |

Steps are ordered by dependency. Steps 0–5 (`config`, `io`) must be completed before `model.py` work begins. Steps 6–11 (`model`) must be completed before `optimization.py` and `main.py` are updated. Step 15 (documentation) should be written in parallel with Step 14 and completed before any release.

---

## 6  Checks Required at Each Stage

### 6.1  Data Loading (`io.py`)

- **HARD ERROR**: date column contains missing rows (gaps in the date spine). Message must explain that rows are required for all dates; values should be `-999` or `NaN`.
- **HARD ERROR**: `T_air` or `Q` column absent entirely.
- **HARD ERROR** *(gap_tolerant)*: `Qmedia <= 0` after computation or user supply.
- **HARD ERROR** *(gap_tolerant)*: zero `T_water` observations — DOY climatology cannot be built; calibration is impossible.
- **WARNING** *(gap_tolerant)*: more than 50% of `Q` values are `-999` and no `Qmedia_user` supplied.
- **WARNING** *(gap_tolerant)*: computed `Qmedia` differs from `Qmedia_user` by more than 30% — user-supplied value may be inconsistent with the data.
- **INFO** *(gap_tolerant, version 3 or 5)*: `Q` gaps are ignored because this version does not use `Q` in the ODE.
- **HARD ERROR** *(legacy mode only)*: series does not start on January 1st.

---

### 6.2  Segment Detection (`model.py`)

- **HARD ERROR**: no valid segments found after gap detection and `min_segment_days` filtering.
- **HARD ERROR**: total valid forcing days across all segments is zero.
- **WARNING**: a segment was dropped for being shorter than `min_segment_days` — print segment indices and length.
- **WARNING**: total valid forcing days < 365 — fewer than one full year of data; calibration results may be unreliable.
- **WARNING**: fewer than 3 segments survive when gap-tolerant mode is active and the data is heavily fragmented.
- **ASSERTION** *(debug)*: no `-999` sentinel is ever passed to `_step()`. Should be verified during development; can be disabled in production.

---

### 6.3  Integration (`model.py`)

- Verify `Twat_mod` is initialised to `-999` before integration begins, so gaps in output are explicit rather than zero.
- Verify `_step()` is never called with `j` or `j+1` outside the current segment bounds.
- Verify `Tice_cover` floor is applied after every step.

---

### 6.4  Objective Function (`model.py`)

- **HARD ERROR**: `n_dat == 0` after aggregation — no `T_water` observations survived.
- **WARNING + return -999**: `n_dat < 2` when `fun_obj == 'KGE'`.
- **WARNING + return -999**: `std_obs == 0` or `std_mod == 0` when `fun_obj == 'KGE'`.
- Verify `eval_mask` excludes all indices before index 365 (warm-up period).
- Verify `eval_mask` excludes `warmup_drop_days` days at the start of each segment.
- Verify no `Twat_mod` value of `-999` (outside any segment) is included in any aggregation window.

---

### 6.5  Validation Pass (`main.py`)

- After `read_Tseries(data, 'v')`, re-run `detect_segments()` to rebuild `segments` and `eval_mask` for the validation data.
- If `len(segments) == 0`, skip validation and set `data.n_tot = 0` (existing skip logic applies).
- Verify `doy_climatology` is **not** recomputed during the validation pass.
- Verify `Qmedia_user` (if set) persists unchanged from calibration to validation.

---

### 6.6  Output

- Verify `gaps_summary.txt` is written even when no gaps exist — it then reports 0 gap days, serving as a positive confirmation of clean data.
- Verify `segment_id` column in output CSV matches the indices returned by `detect_segments()`.
- Verify `Tair_gap` and `Q_gap` columns correctly reflect sentinel positions in the original (pre-conversion) input data, not the processed arrays.

---

## 7  Residual Limitations

The following limitations cannot be resolved by this implementation and must be documented prominently in the `USER_GUIDE`.

### 7.1  T_water Observations Within Forcing Gaps Are Excluded

Any `T_water` observation that falls on a day where `T_air` or `Q` is a sentinel has no valid `Twat_mod` to compare against and is automatically excluded from calibration. This is correct and intentional behaviour — not a bug. Users should be aware that some observations will not contribute to calibration even if `T_water` was measured on those days.

### 7.2  Initial Condition Uncertainty at Segment Restarts

When `T_water` is not observed at the start of a segment, the DOY climatological IC introduces uncertainty proportional to the inter-annual variability of `T_water` at that time of year. The `warmup_drop_days` buffer mitigates this, but its optimal value is river-specific. Users of large, thermally sluggish rivers should consider increasing it beyond the default of 15 days.

### 7.3  MNAR Gaps and Metric Comparability

Data gaps in hydrology are Missing Not At Random (MNAR): gauges most commonly fail during floods and ice events. Removing these periods suppresses the tails of the observed distribution, reduces observed variance, and artificially inflates NSE and KGE. **Performance metrics obtained in gap-tolerant mode are not directly comparable to those obtained on continuous datasets from the same or other catchments.** NSE and RMS are more robust under MNAR conditions than KGE.

### 7.4  Qmedia Bias from Preferentially Missing High Flows

Even with an external `Qmedia_user`, if the modelling-period `Q` gaps coincide with high-flow events, the `θ = Q / Qmedia` ratios during valid periods will be systematically lower than they would be in a complete record. This is a fundamental data limitation that no implementation choice can fully resolve; it should be reported alongside any published calibration results.

### 7.5  Short Segments

Segments shorter than `warmup_drop_days` plus a handful of observation days contribute nothing to the objective function — the entire segment is masked. The `min_segment_days` parameter is designed to catch this before wasting integration time, but the appropriate threshold is problem-dependent and users with short-memory rivers may reasonably lower it below the default of 30 days.

---

## 8  Backwards Compatibility Guarantee

When `gap_tolerant: false` (the default), the following conditions are guaranteed:

- `read_Tseries()` raises `ValueError` on any `T_air` or `Q` null value, exactly as before.
- `read_Tseries()` raises `ValueError` if the series does not start on January 1st, exactly as before.
- `Qmedia` is computed identically to the existing implementation.
- `call_model()` is called unchanged; `call_model_segmented()` is never invoked.
- `funcobj()` uses the full evaluation range from index 365 to `n_tot`; `eval_mask` covers all those indices.
- All output files have the same names and formats as before — gap-tolerant mode adds columns to the CSV outputs but does not alter or rename existing ones.
- The golden test suite (`test_golden.py`) will produce identical results, bit-for-bit, regardless of whether gap-tolerant mode code is present in the codebase.

---
