# 05 — CLI and I/O correctness

**Priority: P1.** The documented entry point cannot run a climate projection at all, and
two silent-wrong-output paths exist in normal use.

## Defect A — pure projection crashes before reaching FORWARD mode

`main.main()` calls `aggregation(data)` then `statis(data)` unconditionally
(`main.py:223-224`) before dispatching to any run mode. `statis` raises when there are no
observations (`model.py:293`).

Reproduced with a future forcing series containing `Date`, `T_air`, `Discharge` and no
`T_water`:

```
ValueError: n_dat is 0 after aggregation. No T_water observations survived.
  File "pyair2stream/main.py", line 224, in main
    statis(data)
```

`forward_mode` has a correct `has_obs` guard (`optimization.py:47-61`) that handles exactly
this case — but it is unreachable from `main()`. Every projection example in the repo
bypasses `main()` entirely, importing `read_calibration` / `read_Tseries` / `forward_mode`
directly (see `examples/forward_prediction_intervals/run_example.py:56-62`).

So `pyair2stream --config ...` cannot perform the package's headline forward-projection
use case.

### Required change 5.1

Move `aggregation` / `statis` out of the unconditional path in `main()`. Either gate them
on the presence of observations, or push them into the run modes that need them
(`PSO`, `LATHYP`, `DE`, `DE-MCMC`, `DE-CV-MCMC`), leaving `FORWARD` to decide via its
existing `has_obs` logic. The second is cleaner: `main()` should not be computing
calibration statistics for a run mode that does not calibrate.

`main.forward()` must also tolerate `finalfit == -999.0`; verify the consistency check at
`main.py:56-63` does not trip.

## Defect B — validation shorter than one year silently re-runs calibration

```python
# io.py:313-315
if p == 'v' and n_tot_raw < 365:
    print('Validation period < 1 year --> validation is skipped')
    return
```

This returns **before** `data.n_tot = n_tot` at line 320, so `n_tot` retains its
calibration value. The guard in `main.forward()` (`main.py:136`) then passes:

```
  n_tot before read_Tseries(v): 1461   years (2015, 2017)
  n_tot after  read_Tseries(v): 1461   years (2015, 2017)
  main.forward() guard is `if data.n_tot < 365: return` -> guard passes? True
  -> validation block proceeds using the CALIBRATION arrays
```

Result: a `3_*_v_*.csv` "validation" file that is actually the calibration data
re-aggregated, and a bogus validation efficiency appended to `1_*.out`. Entirely silent.

### Required change 5.2

Set an explicit skip flag. Do not overload `n_tot`, which is load-bearing for array
allocation everywhere.

```python
# io.py — shape only
data.validation_available = False
if p == 'v' and n_tot_raw < 365:
    print('Validation period < 1 year --> validation is skipped')
    return
...
if p == 'v':
    data.validation_available = True
```

Then gate the entire validation block in `main.forward()` on
`data.validation_available` rather than on `data.n_tot`. Apply the same treatment to the
missing-file branch at `io.py:262-270`, which currently signals by setting `n_tot = 0`.

## Defect C — warm-up rows leak into every output CSV

`read_Tseries` prepends a 365-day copy of year one and sets `date[0:365, :] = -999`
(`io.py:341-347`). Every output CSV is written straight from those arrays:

- `2_*.csv`, `3_*.csv` (`main.py:88-99`, `173-184`)
- `MCMC_envelopes_*.csv` (`optimization.py:747-756`)
- `Forward_Prediction_Envelopes_*.csv` (`optimization.py:172-180`)

Confirmed on a forward run: `warmup rows present in output: 365 (Year=-999)`.

`post_processing` strips them positionally (`df.iloc[365:]`, line 247), so the plots are
fine. Anyone reading these files in R or pandas gets 365 junk rows, and
`pd.to_datetime` on `Year = -999` throws.

### Required change 5.3

Drop the warm-up block on write, or emit real dates plus an `is_warmup` boolean column.
Prefer dropping — the warm-up year is an implementation detail with no interpretation.

If dropped, `post_processing`'s positional `iloc[365:]` slices must be removed in the same
change, along with the parallel slices at `post_processing.py:355-360` and `487-490`. Grep
for `365` across `post_processing.py` before declaring this done.

## Defect D — calendar assumptions block common climate model output

`io.read_Tseries` enforces:

- Series must start 1 January (`io.py:284-286`), unless `gap_tolerant`.
- Series must be a gap-free daily `date_range` (`io.py:289-291`).

GCM output on a 360-day calendar, or with leap days stripped, fails validation. Worse, if a
user pads such a series to pass validation, `tt` is computed from real calendar dates
(`io.py:359-369`) and the seasonal cosine term silently misaligns against the forcing.

### Required change 5.4

Add explicit calendar handling: accept a `calendar:` config key
(`standard` | `noleap` | `360_day`), and compute `tt` consistently with the declared
calendar rather than from `pd.Timestamp` arithmetic. At minimum, detect a non-standard
calendar and raise with an actionable message rather than accepting a padded series.

Also relax the 1-January start requirement for `FORWARD` mode — a projection period rarely
starts on 1 January, and the constraint exists only so the warm-up block's `tt` values
(`io.py:355-357`, hardcoded `(j+1)/365.0`) line up.

## Defect E — examples reference columns the code no longer writes

```
examples/forward_prediction_intervals/run_example.py:88,96   env['Twat_mod_p5'], ['Twat_mod_p95']
examples/optimizer_comparison/compare_optimizers.py:110      env_df['Twat_mod_p5'], ['Twat_mod_p95']
```

The code writes `Twat_mod_lower` / `Twat_mod_p50` / `Twat_mod_upper`
(`optimization.py:176-178`, `751-753`). Both scripts raise `KeyError`.
`post_processing.py` carries a fallback for both names (lines 366-376) — the fingerprint of
an incomplete rename. The committed PNGs predate it.

### Required change 5.5

Update both example scripts. Remove the dual-name fallback in `post_processing.py` once
done — a compatibility shim for a name that was never released is pure confusion. Add a CI
job that executes the example scripts, or at minimum imports and smoke-tests them; these
would have been caught immediately.

## Acceptance criteria

- A CLI-level test: a forcing CSV with no `T_water` column runs to completion through
  `main()` in `FORWARD` mode and writes a simulated series.
- A test with a 200-day validation file asserting no `3_*.csv` is written and no extra
  efficiency line is appended to `1_*.out`.
- A test asserting every output CSV parses with `pd.to_datetime(df[['Year','Month','Day']])`
  without error, and that row count equals input row count.
- A test asserting a 360-day-calendar series raises a clear error rather than silently
  misaligning `tt`.
- Example scripts execute in CI.
