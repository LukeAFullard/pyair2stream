# 03 — Objective function computed on mismatched samples; `eval_mask` never set

**Priority: P1.** Every reported NSE, KGE, R² and AIC in gap-tolerant mode is wrong; the
MCMC likelihood double-counts a year of data in every non-gap-tolerant run.

**Land this before reports 04 and 06.** Both consume `data.eval_mask`.

## Background

Two masks control which days count:

- `data.segments` — contiguous blocks of valid forcing, over which the ODE is integrated.
- `data.eval_mask` — boolean over `n_tot`, days eligible for scoring. Built by
  `detect_segments` (`model.py:14`), which excludes the warm-up block and the first
  `warmup_drop_days` of each segment.

`aggregation()` (`model.py:171`) builds `I_inf`/`I_pos`, mapping observations onto scoring
windows. `statis()` (`model.py:291`) computes `mean_obs`, `TSS_obs`, `std_obs` from those
windows. `fast_funcobj` (`model_numba.py:17`) computes the objective from them.

## Defect A — `statis()` and `funcobj()` score different samples

`statis()` loops over **all** `n_dat` windows (`model.py:301-308`). `fast_funcobj` skips
any window where no day satisfies `Twat_mod[pos] != -999.0 and eval_mask[pos]`
(`model_numba.py:80-96`). Observations falling in a `warmup_drop_days` window are counted
in the denominator but not the numerator.

Verified on synthetic gap-tolerant data:

```
statis used 800 obs; funcobj scored 750  -> 50 observation mismatch
reported NSE:                 -8.094
honest NSE on matched subset: -8.649
```

Consequences, and they differ by metric:

- **NSE** — `1 - TSS_valid/TSS_obs_all`. The excluded set is fixed for fixed segments, so
  this is a monotone affine transform of the honest NSE. **Calibration finds the same
  optimum**, but every reported NSE value is inflated.
- **KGE** — `mean_obs` and `std_obs` come from the full set while `mean_mod`/`std_mod` come
  from the subset. This is *not* monotone. **The optimum itself moves.**
- **R²** — same mixing (`model_numba.py:118-126`). Wrong.
- **AIC/BIC** — computed downstream in `post_processing.py:294-296` from the aggregated
  series; inherits the problem.

In non-gap-tolerant mode `eval_mask` is `None` and every window gets model output, so
`valid_n_dat == n_dat` and the bug does not bite. It is specific to gap-tolerant mode.

## Defect B — `eval_mask` is never set outside gap-tolerant mode

Every call site of `detect_segments` is guarded by `if data.gap_tolerant`:

```
main.py:51, 142    optimization.py:28, 60, 144, 568, 638, 699, 767, 855, 925, 986, 1046
cross_validation.py:352, 367    sensitivity.py:35
```

So in the default workflow `data.eval_mask` stays `None` for the whole run. `funcobj`
tolerates this (`model.py:337`, falls back to `np.ones`) because `I_pos` only ever contains
indices ≥ 365. The MCMC likelihood does not:

```python
# optimization.py:575-586 (and the DE-CV-MCMC copy at 862-873)
valid_mask = (data.Twat_obs != -999.0)
if data.eval_mask is not None:
    valid_mask &= data.eval_mask          # never taken
...
log_L = -0.5 * N * np.log(SSE / N)
```

Verified:

```
  rows in CSV                        : 1096
  data.n_tot (incl. warm-up copy)    : 1461
  data.eval_mask after read_Tseries  : None
  N used by log_probability()        : 1461
  N that SHOULD be used (idx>=365)   : 1096
  -> 365 first-year observations counted a SECOND time
```

The warm-up block is a verbatim copy of year one (`io.py:344-347`), so year one is weighted
twice, and its residuals are the initial-condition spin-up transient. `N` is inflated by
33% in this example, which sharpens the posterior directly (`log_L` scales with `N`). The
same mask feeds `best_sigma` (`optimization.py:641-650`) and `estimate_ar1_rho`
(`optimization.py:653-655`), so the sidecar metadata is biased too.

## Required changes

### 3.1 Make `detect_segments` unconditional

Remove the `data.gap_tolerant` guard from every call site listed above. `detect_segments`
already handles the non-gap case correctly at `model.py:22-27` — it sets
`eval_mask[365:] = True` and returns without building segments. The guard is redundant and
is the sole cause of Defect B.

Prefer a single call site over fourteen. Introduce a small helper invoked once after data
load and once after any masking operation:

```python
# model.py — shape only
def prepare_evaluation(data):
    """Rebuild segments and eval_mask for the currently loaded data. Idempotent."""
    detect_segments(data)
```

Then replace the fourteen guarded call sites with a single call in `read_Tseries` (or in
`main` immediately after it), plus explicit re-calls wherever `Twat_obs`/`Tair`/`Q` are
mutated (cross-validation folds).

### 3.2 Remove the `data.segments is None` short-circuit

Several call sites read `if data.gap_tolerant and data.segments is None: detect_segments(data)`.
The `is None` test means segments are computed once and never refreshed when the
underlying data changes. This causes a confirmed bug in `sensitivity.py:35` (see report 06)
and is a latent hazard everywhere else. Replace with explicit invalidation:
set `data.segments = None` and `data.eval_mask = None` at the top of `read_Tseries`, and
have `prepare_evaluation` always recompute.

### 3.3 Restrict `statis()` to the scored sample

`statis()` must iterate over the same windows `fast_funcobj` will accept. The cleanest fix
is to have `aggregation()` refuse to emit a window whose days are all outside `eval_mask`,
so `n_dat` is correct by construction and `statis` needs no change. This requires
`detect_segments` to run *before* `aggregation` — currently `main.py:222-224` calls
`aggregation` then `statis` with no `detect_segments` in between. Reorder.

If that reordering proves invasive, the fallback is to pass `eval_mask` into `statis` and
skip windows with no eligible day. Prefer the first approach; it removes the possibility of
the two drifting apart again.

### 3.4 Align the MCMC likelihood with the objective

Two sub-issues in `optimization.py:575-586` and `862-873`:

- After 3.1, `eval_mask` will be non-`None` and the double-counting disappears. Verify.
- The likelihood is computed on **daily** `Twat_obs` vs `Twat_mod`, while the objective is
  computed on the **aggregated** series (`Twat_obs_agg` vs `Twat_mod_agg`). For
  `time_resolution: 1d` these coincide. For weekly or monthly they do not, and the
  posterior is then inconsistent with the fit. Use the aggregated arrays.
- `if SSE == 0: return np.inf` (line 585) will poison the emcee chain. Return a large
  finite value instead.

### 3.5 Note the residual-independence caveat on AIC/BIC

`post_processing.py:294-296` computes AIC/BIC assuming independent residuals. Daily stream
temperature residuals are strongly autocorrelated, which systematically favours more
complex models. This matters if versions 3/5/7/8 are compared, and if results are compared
against the Piccolroaz AIC tables bundled in `examples/validation/Switzerland/`. Either
apply an effective-sample-size correction (`N_eff = N * (1-rho)/(1+rho)`) or add an
explicit caveat to the metrics CSV and the docs. Also note `k` excludes the variance
parameter; this shifts AIC by a constant and does not affect ranking, so it is cosmetic.

## Acceptance criteria

- A gap-tolerant test asserting `n_dat` after `aggregation` equals the number of windows
  scored by `funcobj`, and that reported NSE matches an independently computed NSE on the
  matched subset to 1e-10.
- A test asserting KGE is invariant to `warmup_drop_days` changes that do not change which
  days are scored (currently it is not).
- A non-gap-tolerant test asserting `data.eval_mask is not None` after data load and that
  `eval_mask[:365].sum() == 0`.
- A test asserting the MCMC likelihood's `N` equals the number of observations at index
  ≥ 365, not `n_tot`.

## Out of scope

Do not change the warm-up-block design (duplicating year one into indices 0-364). It is
Fortran-equivalent and the golden tests depend on it. Fix the masking, not the layout.
