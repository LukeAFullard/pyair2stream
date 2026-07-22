# Gap-Tolerant Calibration Mode in pyair2stream

## 1. Overview

`pyair2stream` is a Python reimplementation of the Fortran `air2stream` model (Toffolon and Piccolroaz, 2015), which simulates daily river water temperature from air temperature and, optionally, discharge, by numerically integrating a lumped ordinary differential equation (ODE) heat-budget model forward in time. The original Fortran implementation requires the forcing record (`T_air`, and `Discharge` for versions that use it) to be complete and continuous for the entire simulation period; any missing forcing day makes the record unusable as supplied.

`pyair2stream` adds an opt-in gap-tolerant calibration mode, enabled with `gap_tolerant: true`, that does not exist in the Fortran reference implementation. It allows the model to be calibrated on records containing missing `T_air`/`Discharge` days by automatically splitting the record into contiguous valid segments, integrating each segment independently with its own restart condition, and excluding gap-affected days from the calibration objective. This document describes the design rationale, the segment-detection and restart algorithm, its interaction with the rest of the calibration pipeline (including cross-validation), its configuration, its outputs, and empirical results characterizing its behaviour under different missing-data patterns. It is intended to serve as a citable technical description of this feature for use in derivative scientific work.

## 2. Motivation

Long daily river temperature and discharge records are rarely perfectly continuous in practice: sensor outages, ice-affected gauging, or administrative gaps in monitoring programmes routinely leave missing days scattered through an otherwise usable record. Requiring a fully continuous record, as the Fortran reference implementation does, forces a user to either discard an entire dataset because of a handful of missing days, or to fabricate/interpolate forcing values through the gap — the latter risking the introduction of artificial smoothing or bias into precisely the periods (floods, freezes) that are hardest, and most scientifically interesting, to model correctly.

Gap-tolerant mode instead treats a forcing gap as a structural discontinuity: the model integrates independently on either side of it, restarting from a defensible condition rather than either fabricating forcing data or corrupting the ODE state by integrating through fictitious values. This preserves the physically-based, stateful character of the ODE integration on the data that does exist, while allowing a fragmented record to still be used for calibration.

## 3. Algorithm

### 3.1 Segment Detection

Segment detection (`detect_segments()` in `model.py`) scans the forcing arrays from the first real data day onward (index 365 in the internal, zero-padded array — see Section 3.4) and classifies each day as valid or invalid:

- A day is invalid if `T_air` is missing (`-999.0`).
- For model versions that use discharge in their governing equation (all versions except 3 and 5), a day is also invalid if `Discharge` is missing or non-positive (`<= 0.0`). Versions 3 and 5 do not use discharge in their ODE, so gaps in `Discharge` are ignored entirely for those versions and do not fragment the record.

Maximal contiguous runs of valid days form candidate segments. A candidate segment is retained only if its length is at least `min_segment_days` (default 30); shorter runs are dropped, with a console warning naming the dropped segment's index range and length. If no segment survives this filter, gap-tolerant mode raises `ValueError("No valid segments found after gap detection and filtering.")`, since there is then no usable data to calibrate against.

Two additional diagnostics are logged (once per run, not repeated on every optimizer evaluation) if triggered: a warning if the total valid data across all retained segments is less than 365 days (results may not be reliable with less than a full annual cycle of data), and a warning if the record has fragmented into more than two segments (a signal of a highly discontinuous dataset).

### 3.2 Evaluation Mask and Warm-Up Exclusion Within Segments

Because each segment restarts from an approximate initial condition rather than a true continuous physical state (Section 3.3), the first `warmup_drop_days` (default 15) of every segment are excluded from the calibration objective, even though the ODE is still integrated through them. This is implemented by constructing a boolean `eval_mask` covering the whole record: `True` only for days that are (a) inside a retained segment, and (b) at least `warmup_drop_days` past that segment's start. This mask is consumed identically by `funcobj()` (the calibration objective) regardless of whether gap-tolerant mode is active — in the non-gap-tolerant case, `detect_segments()` still builds an `eval_mask` that is simply `True` everywhere after the model's own single, one-year warm-up (Section 3.4), so both modes share exactly the same downstream statistics/objective code path.

### 3.3 Per-Segment Restart Condition

Within each retained segment, the ODE state is not carried over from the previous segment (this would require integrating across the gap using fabricated forcing data). Instead, `call_model_segmented()` re-initializes the water temperature state at the start of every segment using, in priority order:

1. **The genuine observed water temperature** at that day, if one exists (`Twat_obs[start] != -999.0`).
2. **A day-of-year climatology value**, `doy_climatology[doy]`, otherwise — a smooth, per-calendar-day mean water temperature estimated from all valid observations in the record (see Section 3.5). This is the more common case in practice, since a segment typically begins right where forcing data resumes, which need not coincide with an observed water temperature reading.

Each segment is integrated independently by the same underlying numerical integrator (`RK4`, `RK2`, `EUL`, or `CRN`) used elsewhere in `pyair2stream`; days outside any retained segment (including the initial padding described in Section 3.4 and any dropped/too-short segments) are left at the missing-data sentinel `-999.0` in the simulated output, so there is no possibility of the model silently propagating a spurious value across a gap boundary.

### 3.4 Interaction with the Model's Own One-Year Warm-Up

Independent of gap-tolerant mode, every `pyair2stream` run prepends a duplicate of the record's first simulated year to the internal working arrays as a numerical warm-up, to reduce sensitivity to the (arbitrary) initial condition supplied for day one of the record; real data begins at internal index 365. `detect_segments()` always ignores this prepended block (scanning only from index 365 onward), so gap detection, segment restart logic, and the `warmup_drop_days` exclusion operate purely on genuine calendar data, never on the synthetic warm-up prefix. In non-gap-tolerant mode, this global one-year warm-up is the record's *only* spin-up mechanism, and the input file is required to start on 1 January so that the warm-up year and the true first year share the same annual phase; in gap-tolerant mode this restriction is lifted (Section 3.6), because each segment obtains its own initial condition locally rather than depending on that global warm-up year.

### 3.5 Day-of-Year Climatology and Discharge Normalization Without Leakage

Two auxiliary statistics feed into gap-tolerant simulation, and both are computed strictly from data available at calibration time to avoid look-ahead leakage:

- **Day-of-year climatology** (`compute_doy_climatology()`): for each of the 366 possible calendar days, the mean of all valid `T_water` observations falling on that day-of-year across the record is computed. Any calendar day with zero observations is filled by linear interpolation across a year-triplicated copy of the climatology (so the interpolation wraps correctly across the December–January boundary). If the record has zero valid `T_water` observations in total, this raises `ValueError`, since climatology-based segment restarts would otherwise be undefined.
- **Mean discharge** (`Qmedia`, via `compute_qmedia()`): used to normalize discharge-dependent terms in versions 4, 7, and 8. In gap-tolerant mode, if the resulting `Qmedia` is zero or negative (i.e., discharge is missing or invalid essentially everywhere) for a version that requires it, a `ValueError` is raised directing the user to supply `Qmedia` explicitly. If more than half of the discharge record is missing, a warning is issued recommending the same. If a user-supplied `Qmedia` differs by more than 30% from the value computed from the available data, a warning is also issued, as a sanity check against a mistyped or stale override.

Both routines are re-run from the currently valid (unmasked) arrays whenever calibration re-enters a new context — in particular, during cross-validation folds — specifically so that a held-out fold's discharge or water-temperature values cannot leak into either statistic (see Section 4).

### 3.6 Relaxed Input Validation

Outside gap-tolerant mode, `pyair2stream` enforces that the input CSV starts on 1 January and that `T_air`/`Discharge` contain no missing values, raising a `ValueError` otherwise (Section 3.4 explains why the January-1st constraint exists for the non-gap-tolerant path). Gap-tolerant mode relaxes both constraints: the record may start on any calendar date, and `T_air`/`Discharge` may contain missing values, which are the very case this mode exists to handle. The requirement that the file be a continuous daily series with no *skipped calendar dates* (as opposed to missing *values* on rows that exist) still applies in both modes — a gap must be represented as a row with a missing-value sentinel or `NaN`, not as an absent row.

## 4. Interaction with Cross-Validation

When gap-tolerant mode is combined with the leave-one-year-out cross-validation feature, masking a held-out fold's `T_water` observations is not by itself sufficient to prevent state leakage: if the forcing (`T_air`/discharge) for the held-out window were left intact, the segmented integrator would simply treat it as ordinary valid data and integrate straight through it using the very information the fold is meant to withhold from calibration. Gap-tolerant cross-validation therefore also masks the fold's `T_air` and discharge to the missing-data sentinel for the duration of that fold, causing `detect_segments()` to treat the held-out window as a genuine data gap: the segmented integrator restarts around it exactly as it would for any other missing-data period, rather than free-integrating through it as happens in the non-gap-tolerant cross-validation path. `compute_qmedia()` and `compute_doy_climatology()` are also recomputed per fold, from the masked arrays, so that the held-out fold cannot influence either auxiliary statistic used for that fold's segment restarts. Full details of the cross-validation procedure itself are given in the accompanying cross-validation documentation.

## 5. Configuration

```yaml
gap_tolerant: true

Qmedia: null              # optional explicit override for mean discharge
warmup_drop_days: 15      # days excluded from the objective at the start of each segment
min_segment_days: 30      # minimum contiguous valid-data length to keep a segment
```

| Field | Default | Meaning |
|---|---|---|
| `gap_tolerant` | `false` | Enables segmented, gap-tolerant calibration; if `false`, `T_air`/`Discharge` must be complete and the record must start on 1 January |
| `Qmedia` | Computed from data | Explicit override for the mean-discharge normalization constant; recommended whenever high-flow periods are disproportionately missing, since the computed value would otherwise be biased low |
| `warmup_drop_days` | `15` | Number of days at the start of every segment excluded from the calibration objective (still integrated, just not scored) |
| `min_segment_days` | `30` | Minimum length, in days, for a contiguous valid-data run to be kept as a usable segment |

These fields sit at the top level of the YAML configuration (not nested under `optimization:`), matching `pyair2stream/io.py`'s parsing. Gap-tolerant mode composes with every calibration mode (`PSO`, `DE`, `LATHYP`, `DE-MCMC`, `DE-CV-MCMC`) and with cross-validation (Section 4); it is not itself an optimizer, but an input-handling and simulation mode that all optimizers run on top of.

## 6. Pre-Analysis Diagnostics

Before committing to a full calibration run, `pyair2stream` provides `analyze_timeseries()` (`pre_analysis.py`) to characterize a candidate dataset's suitability for gap-tolerant calibration without running the model. Given a dataframe of `Date`, `T_air`, `T_water`, and (optionally) `Discharge`, it:

- Computes missing-data percentages for each forcing/observation column.
- Applies the same forcing-based segmentation logic as `detect_segments()` (contiguous valid-forcing runs, filtered by `min_segment_days`) to classify the record into valid segments, too-short segments, and gap periods.
- Counts how many genuine `T_water` observations fall inside the valid segments, and reports this both in absolute terms and as points-per-parameter for each of the five model versions (3, 4, 5, 7, 8) — a quick, version-specific check of whether the usable data volume is adequate for the number of parameters being fit.
- Optionally writes a text summary report and a three-panel diagnostic plot (`T_air`, `T_water`, `Discharge` over time, shaded by segment classification: valid/green, too-short/yellow, gap/red).

This is a diagnostic-only utility; it does not itself run a calibration, and its segment classification is advisory (based purely on forcing-column completeness) rather than the authoritative segmentation `detect_segments()` performs internally during an actual run.

## 7. Outputs

In addition to the calibration outputs common to all modes, gap-tolerant runs produce:

| Output | Contents |
|---|---|
| `2_<run_mode>_<objective>_<station>_<series>c_<time_res>.csv` | The usual calibration-period output, with three additional columns in gap-tolerant mode: `Tair_gap` and `Q_gap` (binary flags marking sentinel-valued forcing days) and `segment_id` (the index of the segment each row belongs to, or `-999` for days outside any retained segment) |
| `gaps_summary.txt` | Gap-tolerant-mode-only diagnostic report: the source of `Qmedia` (user-supplied or computed) and its value; the missing fraction of `T_air` and `Discharge` over the genuine (non-warm-up) record; the number of segments found, and each segment's start/end date and length |

## 8. Empirical Characterization

### 8.1 Parameter and Goodness-of-Fit Stability Under Systematic vs. Random Gaps

A controlled experiment calibrated Version 8 of the model (Differential Evolution, 3000 generations, 100-member population) on the complete Dischmabach (DAV) dataset from Switzerland, then repeated the calibration after injecting `NaN` gaps into `T_air` under six scenarios: a gap-free baseline, four systematic patterns varying the number and length of gaps (`few_short`, `many_short`, `few_long`, `many_long`), and one pattern with gaps scattered at random.

| Scenario | Missing `T_air` | NSE | R² | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.00% | 0.9558 | 0.9558 | 4.794 | 0.629 | 1.410 | 0.270 | 0.000 | 4.912 | 0.582 | 0.637 |
| few_short | 0.59% | 0.9572 | 0.9560 | 4.792 | 0.621 | 1.401 | 0.271 | 0.000 | 4.945 | 0.582 | 0.641 |
| many_short | 2.35% | 0.9618 | 0.9573 | 4.910 | 0.625 | 1.430 | 0.255 | 0.000 | 5.183 | 0.579 | 0.651 |
| few_long | 2.35% | 0.9568 | 0.9542 | 4.815 | 0.620 | 1.405 | 0.271 | 0.000 | 4.963 | 0.580 | 0.641 |
| many_long | 5.87% | 0.9588 | 0.9521 | 4.791 | 0.604 | 1.381 | 0.279 | 0.000 | 5.129 | 0.579 | 0.659 |
| random | 4.97% | 0.9842 | 0.9610 | 4.719 | 0.702 | 1.515 | 0.257 | 0.000 | 4.495 | 0.581 | 0.575 |

Across the systematic gap scenarios, both calibrated parameters and NSE remain close to the gap-free baseline — NSE shifts by no more than +0.006 even at 5.87% missing `T_air`. The random-gap scenario is the clear exception: despite a comparable proportion of missing data (4.97%), NSE rises sharply to 0.9842, a far larger deviation than any systematic scenario produced. This is consistent with the documented caveat that gaps bias performance metrics upward: gaps scattered at random tend to remove individually difficult-to-model days piecemeal without necessarily excluding an entire hard hydrological event (e.g., a freeze or flood) in one block, whereas systematic (block) gaps are more likely to remove or preserve whole events intact, giving a fairer like-for-like comparison to the continuous record. This is a direct empirical demonstration of why the documentation for this feature explicitly cautions that gap-tolerant NSE/KGE values are not directly comparable to continuous-record values.

### 8.2 Practical Implication

This result means the pattern of missingness, not merely its overall proportion, determines how much a gap-tolerant calibration's reported goodness-of-fit can be expected to diverge from what a fully continuous record would have produced. Users comparing gap-tolerant results either across datasets or against literature values calibrated on continuous records should characterize whether their missingness is closer to a systematic (block) or random (scattered) pattern using the pre-analysis diagnostics in Section 6, and treat metrics from randomly-gapped records with additional caution.

## 9. Practical Considerations and Limitations

- **Metrics are not directly comparable to a continuous-record calibration.** As demonstrated in Section 8.1, removing data — especially in a scattered, random pattern — can materially inflate NSE/KGE relative to what the same model would achieve on the full record, because the removed days are not a representative sample of the record's difficulty. NSE and RMS degrade more gracefully under gaps than KGE.
- **`T_water` inside a forcing gap is dropped, not interpolated.** Observations of the target variable that happen to fall inside a masked/gapped forcing window are excluded from both calibration and evaluation; they are never filled in or estimated.
- **Segment restart is approximate, not a true continuous state.** Because a segment's initial condition comes from either an observed value or a climatological estimate rather than a carried-over ODE state, the first `warmup_drop_days` of every segment are excluded from scoring specifically because the model has not yet had time to converge away from that approximate restart value.
- **Discharge-dependent versions need enough valid discharge to normalize correctly.** For versions 4, 7, and 8, a `Qmedia` computed from a discharge record with substantial high-flow gaps will be biased low; supplying a known historical `Qmedia` explicitly is recommended whenever this is a concern, and the implementation will raise an error rather than silently proceed if the computed value is non-positive.
- **Highly fragmented records degrade reliability.** More than two retained segments, or fewer than 365 total valid days across all segments, each trigger an explicit warning, since both situations reduce the model's ability to characterize a full annual cycle within any single segment.

## 10. References

- Toffolon, M. and Piccolroaz, S. (2015). A hybrid model for river water temperature as a function of air temperature and discharge. *Environmental Research Letters*, 10(11), 114011. https://doi.org/10.1088/1748-9326/10/11/114011
- Piotrowski, A. P. and Napiorkowski, J. J. (2018). Performance of the air2stream model that relates air and stream water temperatures depends on the calibration method. *Journal of Hydrology*, 561, 395–412.
- Beck, H. E., van Dijk, A. I. J. M., de Roo, A., et al. (2017). G