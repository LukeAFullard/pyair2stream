# 06 — Diagnostics, plots and sensitivity analysis

**Priority: P2.** Nothing here corrupts the model, but several published figures show the
wrong quantity, including one committed to the repository as an example.

**Depends on report 03** for `eval_mask`.

## Defect A — dotty plots use the wrong column

`post_processing.py:102-105`:

```python
n_par  = len(df_0.columns) - 1
parset = df_0.iloc[:, :-1].values
eff    = df_0.iloc[:, -1].values
```

The optimizers write **12** columns (`optimization.py:329`, `433` and equivalents):

```
['par_1'...'par_8', 'eff_index', 'NSE', 'R2', 'MAE']
```

so `n_par` is inferred as 11, `parset` swallows `eff_index`, `NSE` and `R2`, and `eff` is
**MAE**. Downstream:

- the y-axis is labelled with `data.fun_obj` (lines 155-158), so an MAE axis reads "NSE";
- the `toll` filter (`eff >= 0.5` for NSE) is applied to MAE, retaining the *worst* fits;
- `i_best = np.argmax(eff)` (line 130) marks the **highest-error** parameter set as optimal.

Visible in a committed example figure,
`examples/Hopelands/output/dottyplots_DE-MCMC_NSE_Hopelands.png`: y-axis labelled "NSE"
with values from 1 to 4.3, which is impossible for NSE (bounded above by 1). Those are MAE
values in °C, and the orange "best" marker sits at the maximum.

### Required change 6.1

Select columns by name, not position.

```python
par_cols = [c for c in df_0.columns if c.startswith('par_')]
parset   = df_0[par_cols].values
eff      = df_0['eff_index'].values
n_par    = len(par_cols)
```

Regenerate the committed example figures afterwards.

## Defect B — stray `for ... else` blanks the par_8 panel

`post_processing.py:145-159`. The `else:` at line 158 is at the same indentation as the
`for` at line 145, so it is a `for...else` clause: it executes on normal loop completion
with `i == 7`, calling `axes[7].axis('off')`. The par_8 subplot loses its frame, ticks and
labels on every run.

### Required change 6.2

Remove the `else:` clause. If the intent was to hide unused panels, do it explicitly inside
the loop with `if i >= n_par: axes[i].axis('off'); continue`.

## Defect C — PSO never records NSE, R² or MAE

`PSO_mode` evaluates particles through `eval_particle_worker` in a
`concurrent.futures.ProcessPoolExecutor` (`optimization.py:229`, `275`). `sub_1` sets
`data.current_nse` / `current_r2` / `current_mae` inside the **child** process; only the
scalar return value crosses the boundary. The parent then writes its own untouched
defaults (`config.py:88-90`, `-999.0`) into the history rows at lines 236-239 and 283-286.

Confirmed on a quickstart run:

```
      eff_index    NSE     R2    MAE
min    0.057756 -999.0 -999.0 -999.0
max    0.973014 -999.0 -999.0 -999.0
n rows 45
```

Every metric column is `-999.0`. The convergence plot (`post_processing.py:58-93`) then
draws three flat lines at `-999`, and combined with Defect A the dotty plot is drawn with
y-limits around `[-999, -1099]`.

`DE_mode` and `LH_mode` run single-threaded and are unaffected.

### Required change 6.3

Return the metrics from the worker rather than relying on shared state.

```python
# optimization.py — shape only
def eval_particle_worker(args):
    data, p_vals, n_par = args
    data.par[:n_par] = p_vals
    eff = sub_1(data)
    return eff, data.current_nse, data.current_r2, data.current_mae
```

Update both call sites to unpack the tuple and use the returned values when appending to
`history`. Do not "fix" this by forcing PSO single-threaded.

While here: only 45 of ~620 evaluations were recorded, because particles that hit an
absorbing wall are never evaluated (`optimization.py:265-269`). That is Fortran-equivalent
(`AIR2STREAM_RUNMODE.f90`, `IF (status.eq.0)`) and should be left alone, but the sparse
history is worth a note in the docs so users do not think evaluations were lost.

## Defect D — sensitivity analysis runs on stale validation segments

`sensitivity.py:31-36`:

```python
read_Tseries(data, 'c')                              # reload calibration data
if data.gap_tolerant and data.segments is None:      # segments is NOT None
    detect_segments(data)                            # so this never runs
```

By the time `sensitivity_analysis` is called (`main.py:254`), `main.forward()` has already
overwritten `data.segments` with **validation** segments at `main.py:142`.

Verified:

```
calibration segments : [(365, 964), (1066, 1564)]   n_tot 1565
validation  segments : [(365, 464), (516, 964)]     n_tot  965
segments used by sensitivity_analysis on CALIBRATION data: [(365, 464), (516, 964)]
```

The analysis integrates 549 of 1200 calibration days and never touches the second half of
the record. `data.eval_mask` is stale in the same way.

### Required change 6.4

Covered by change 3.2 (drop the `data.segments is None` short-circuit and invalidate on
data load). Verify explicitly here — this is the confirmed instance.

## Defect E — sensitivity index documented as one thing, computed as another

`sensitivity.py:24` prints "perturbations = ...% of parameter range". `sensitivity.py:79-83`
computes `delta = (delta_pct/100) * abs(par_best[j])` — a percentage of the parameter
**value**. `param_range` is computed at line 68 and used only as a skip test.
`USER_GUIDE.md` §11 repeats the "% of its own value" phrasing, so the docs are internally
inconsistent with the console message.

Two consequences:

- The index is a *relative* sensitivity. A parameter with a small fitted value receives a
  small perturbation and appears insensitive regardless of its true influence. Note that
  fitted `a5 = 0.000` occurs in the bundled Switzerland results, which triggers the
  `1e-4` floor at line 81.
- The plot y-axis is labelled `Sensitivity Index [°C]` (`sensitivity.py:180`) as though it
  were absolute.

Additionally, when the optimum sits on a bound the clipping at lines 87-90 silently makes
the central difference one-sided. The `actual_delta` normalisation partially compensates
but the estimate is then first-order, not second-order, and no warning is emitted.

### Required change 6.5

- Pick one definition. Range-relative (`delta = pct/100 * (parmax-parmin)`) is the better
  choice: it is comparable across parameters and immune to the near-zero problem. Make it
  the default, expose the value-relative variant behind a config key if wanted.
- Fix the console string, the docstring, `USER_GUIDE.md` §11 and
  `docs/sensitivity_analysis.md` to match whatever is chosen.
- Relabel the plot axis to state the normalisation.
- Emit a warning row (`Status: "Bounded"`) when clipping made the difference one-sided.

## Defect F — residual ACF plot computed on a spliced series

`post_processing.py:412-414` calls `pd.plotting.autocorrelation_plot(res_clean)` where
`res_clean = residuals.dropna()`. Dropping NaNs concatenates non-adjacent days, so lag-k in
the plot is not lag-k in time. `estimate_ar1_rho` handles adjacency correctly; the plot does
not, and the two will disagree.

### Required change 6.6

Compute the ACF on a gap-aware basis — reindex to the full daily calendar with NaNs
retained and use a pairwise-complete estimator, or compute per-segment and pool. Annotate
the plot with the number of valid pairs at each lag.

## Acceptance criteria

- A test asserting the dotty-plot data extraction returns exactly 8 parameter columns and
  that `eff` equals the `eff_index` column.
- A test asserting a PSO history CSV contains no `-999.0` in the `NSE`, `R2` or `MAE`
  columns for rows whose `eff_index` is finite.
- A test asserting `sensitivity_analysis` on calibration data uses segments derived from
  calibration data (compare against a fresh `detect_segments` call).
- A test asserting the sensitivity index is invariant to a rescaling of a parameter's
  bounds that leaves the fitted value unchanged (under the range-relative definition, it
  will not be — assert the documented behaviour, whichever is chosen).
- Regenerate all committed example figures and confirm the dotty-plot y-axes are bounded by
  1 for NSE and KGE.
