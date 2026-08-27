# 01 — Qmedia is recomputed per dataset, which cancels the scenario signal

**Priority: P0.** This silently produces a null result for the abstraction study.

## Background

`air2stream` versions 4, 7 and 8 see discharge only through the dimensionless ratio

```
theta = Q / Qmedia
```

Every discharge term in the ODE — the `theta * (a5 + a6·cos(...) - a8·Tw)` advective
group and the `theta^a4` thermal-capacity divisor — is a function of θ alone. The model
is therefore **exactly invariant to any rescaling of Q that is matched by a rescaling of
Qmedia**. `Qmedia` is not a nuisance normaliser; it is part of the calibrated model. A
parameter vector is only meaningful paired with the `Qmedia` it was fitted under.

## The defect

`Qmedia` is treated as a quantity derived from whatever data is currently loaded, not as
a calibrated constant.

- `pyair2stream/io.py:372` — `compute_qmedia(data, verbose=True)` is the final statement
  of `read_Tseries()` and runs unconditionally on every load, for both `'c'` and `'v'`.
- `pyair2stream/io.py:184` — `compute_qmedia()` recomputes from `data.Q[365:n_tot]`
  unless `data.Qmedia_user` is set.
- `pyair2stream/main.py:103-108` — the only place `Qmedia` is ever written to an output
  file is `gaps_summary.txt`, which is inside `if data.gap_tolerant:`. **In the default
  non-gap-tolerant workflow the calibration Qmedia is never persisted**, so a user cannot
  recover it from calibration outputs.
- `pyair2stream/main.py:134` — `forward()` calls `read_Tseries(data, 'v')`, so within a
  single run the validation period is scored under a *different* Qmedia than calibration.

The upstream Fortran behaves identically (`AIR2STREAM_READ.f90`, `read_Tseries`, Qmedia
accumulated inside the read loop; `read_validation` calls it again). This is an inherited
design flaw, not a porting error — but it is fatal for scenario work, which the Fortran
was never used for.

## Evidence

Traced call sites in the user's two-run workflow:

```
STEP 1 - calibration run on OBSERVED flow (run_mode: DE)
    [compute_qmedia] Qmedia -> 8.01175   (io.py:372, in read_Tseries)
STEP 2 - scenario run on NATURALISED flow (run_mode: FORWARD, same params)
    [compute_qmedia] Qmedia -> 12.01763  (io.py:372, in read_Tseries)
    ratio scenario/calibration = 1.5000  (flow was scaled by 1.5)
```

Consequence, same fitted parameters, naturalised flow = observed x 1.5:

```
mean dT (naturalised - observed), Qmedia auto-recomputed : -0.0000 degC
mean dT (naturalised - observed), Qmedia pinned          : -0.9600 degC
max |dT| auto: 0.0000    max |dT| pinned: 1.2204
```

A uniform flow change produces **exactly zero** temperature response. There is no warning.
For non-uniform abstraction the cancellation is partial but always biased toward "no
effect", because the scenario mean absorbs the mean of the change.

## Required changes

### 1.1 Persist Qmedia with the calibration result (all run modes)

`Qmedia` must be written to disk by every calibration run, not only gap-tolerant ones.

- Add it to `parameters.txt` (written in `io.read_calibration`, around line 160) or, better,
  emit a new `calibration_metadata.json` next to it containing at minimum:

```python
# shape only, not the implementation
{
  "qmedia": float,            # the value the parameters were fitted under
  "qmedia_source": "computed" | "user",
  "n_q_valid": int,
  "theta_min": float, "theta_max": float,   # calibrated theta range, needed by report 02
  "version": int, "integrator": str,
  "par_best": [ ... 8 floats ... ],
  "pyair2stream_version": str
}
```

  `theta_min`/`theta_max` are consumed by the extrapolation guard in report 02, so add
  them now.

- Write this file from `main.forward()`, after `data.par_best` is final, alongside the
  existing `1_*.out` write (`main.py:66-70`).

### 1.2 Do not recompute Qmedia in FORWARD mode

In `io.read_Tseries`, the call at line 372 should not recompute when the model is being
run with externally supplied parameters.

```python
# io.py, end of read_Tseries — shape of the required guard
if data.runmode == 'FORWARD' and data.Qmedia_user is None:
    raise ValueError(
        "FORWARD mode requires an explicit `Qmedia:` in the config (or a "
        "`calibration_metadata.json` via `paths.calibration_metadata`). "
        "Recomputing Qmedia from scenario discharge rescales theta and cancels "
        "the discharge signal. See audit report 01."
    )
compute_qmedia(data, verbose=True)
```

Hard failure is correct here. A default that silently recomputes is exactly the trap being
fixed, and FORWARD mode is precisely where scenarios are run.

Add an optional config key `paths.calibration_metadata` which, when present, loads
`qmedia`, `version` and `integrator` from the JSON written in 1.1 and cross-checks them
against the current config, erroring on mismatch.

### 1.3 Freeze Qmedia across calibration and validation within a run

In `main.forward()`, the validation load at line 134 must not change `data.Qmedia`.
Capture the calibration value before the call and restore it after, or add a
`recompute_qmedia: bool = True` parameter to `read_Tseries` and pass `False` for `'v'`.

This changes validation numbers for existing non-gap-tolerant runs. That is intended — the
current behaviour scores the validation period under a normalisation the parameters were
not fitted with. Note the deviation from the Fortran in `README.md` under
"Known deviations from the Fortran reference".

### 1.4 Warn when scenario theta leaves the calibrated range

After loading scenario forcing in FORWARD mode, compare `Q / Qmedia` against the
`theta_min` / `theta_max` recorded at calibration and report the fraction of days outside.
Warn above a small threshold; this feeds directly into the stability guard in report 02.

### 1.5 Documentation

`USER_GUIDE.md` currently presents `Qmedia:` as a convenience for gappy records
(sections around lines 194 and 335). Add a dedicated subsection stating that `Qmedia` is a
**calibrated model constant** and that any scenario comparison — abstraction, naturalised
flow, climate projection — must reuse the calibration value. Include the zero-signal
demonstration above.

## Acceptance criteria

- A test that calibrates on a series, runs FORWARD on the same series with `Discharge`
  multiplied by a constant `k` and `Qmedia` pinned, and asserts the mean `|dT|` is
  non-zero and increases monotonically in `|k-1|` over `k in {1.1, 1.25, 1.5}`.
- A test asserting `FORWARD` mode without an explicit `Qmedia` or metadata file raises.
- A test asserting `data.Qmedia` is unchanged after `read_Tseries(data, 'v')`.
- A round-trip test: calibrate, read `calibration_metadata.json`, run FORWARD from it,
  and assert the reproduced objective matches `finalfit` to 1e-6.

## Out of scope

Do not attempt to make the model invariant to Qmedia or to reparameterise around absolute
discharge. The θ formulation is the published model and must be preserved.
