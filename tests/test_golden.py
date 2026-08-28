"""
Golden-master regression tests: pyair2stream's own `model.call_model` /
`model_numba.fast_run_integration` compared against the upstream Fortran
reference implementation.

docs/audit/08_testing_gaps.md, 8.1/8.2:

- Full `version x integrator` cross-product (5 versions x 4 Fortran-backed
  integrators = 20 cases), not just the 3 combinations previously covered.
  `EXP` has no Fortran equivalent (docs/audit/02_numerical_integration.md), so
  it is instead compared against `CRN` (both unconditionally stable, and
  validated in report 02 to agree to well under 0.1 degC).
- Tight tolerance over a 3-year horizon, instead of the previous
  `rtol=atol=1e-2` over 10 days -- loose enough to hide a slow systematic
  divergence, which is exactly what extending the horizon here found (see
  below).

Extending the horizon to 3 years surfaced a real divergence of up to 0.086 degC
that the old 10-day/1e-2 window could never have caught -- but the bug was in
`tests/fortran_runner.py`, not in `pyair2stream`. The Fortran reference's
`year_ini=date(366,1)` (`AIR2STREAM_READ.f90`) reads the *year* field of the
first real row to seed its per-year leap-year block lengths -- but this
harness was writing the date columns as `day month year` instead of the real
air2stream input format, `year month day` (confirmed against
`fortran/upstream/Switzerland/*_cc.txt` and `fortran/upstream/readme.txt`).
Fortran therefore received the day-of-month as `year_ini` (e.g. `1` for a
1 January start), so `leap_year(year_ini + i - 1)` evaluated the leap-ness of
essentially arbitrary small integers instead of real calendar years --
silently shortening or lengthening year blocks in `tt` for every year beyond
the first. Invisible in every previous golden test (all <=100 days, so never
crossing a year boundary at all), it would have corrupted any future attempt
to extend the horizon. Confirmed by instrumenting a debug build of the
Fortran binary to dump its own `tt`/`year_ini` values directly, and by
independently reproducing Fortran's documented block algorithm in Python
against `data.tt`, which matched exactly -- i.e. `pyair2stream`'s own
calendar-aware `tt` construction (`io.py`) was correct throughout. Fixed by
writing the date columns in the correct order.

With that harness bug fixed, the remaining floor is the Fortran reference's
own output format: `AIR2STREAM_SUBROUTINES.f90`'s `2_*.out` writes `Twat_mod`
as `f10.5` (5 decimal places, `FORMAT` 1005), so no text-based comparison
against it can be tighter than that regardless of correctness on either side.
`atol=6e-6` below is set just above that `5e-6` rounding ceiling -- roughly
1700x tighter than the previous `1e-2`, not the `1e-9` originally suggested
(unreachable through this file format, confirmed above; the actual computed
values matched far tighter than `1e-5` once the harness bug was fixed).

Each case goes through the real `read_calibration`/`read_Tseries` pipeline
(not a hand-constructed `CommonData`) so the golden comparison exercises the
same `tt` (calendar-aware day-of-year) construction production `FORWARD` runs
use, rather than a test-local reimplementation that could silently drift from
it. Discharge is held constant so `Qmedia` is trivially exact on both sides
(the Fortran driver recomputes its own `Qmedia` from the discharge file it is
given -- see `tests/fortran_runner.py`) and every version/integrator stays
comfortably inside its stability limit (`docs/audit/02_numerical_integration.md`),
isolating the comparison to the seasonal/calendar terms and each integrator's
own arithmetic.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pyair2stream.config import CommonData, PI
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import call_model, funcobj, aggregation, statis
from tests.fortran_runner import run_fortran_model

# a1..a8, matching the existing golden tests' values.
_BASE_PAR = [1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1]

# Which parameter indices (0-indexed) io.py forces to zero for each version --
# see `read_calibration`'s version-based `parmin`/`parmax`/`flag_par` zeroing
# in io.py. `model_numba`'s per-version ODE branches assume the caller has
# already zeroed the corresponding `par` entries the same way (docs/audit/08,
# Gap A), so the golden fixtures must reproduce that convention exactly.
_ZERO_INDICES_BY_VERSION = {
    3: (3, 4, 5, 6, 7),
    4: (4, 5, 6, 7),
    5: (3, 4, 7),
    7: (3,),
    8: (),
}

VERSIONS = (3, 4, 5, 7, 8)
FORTRAN_INTEGRATORS = ('EUL', 'RK2', 'RK4', 'CRN')

# 3 years -- long enough that a slow systematic divergence (e.g. a tt/ttt
# mismatch, or accumulated floating-point error) would be visible, unlike the
# previous 10-day window (docs/audit/08_testing_gaps.md, 8.2).
N_TOT_RAW = 3 * 365

QMEDIA = 10.0


def _par_for_version(version: int) -> np.ndarray:
    par = np.array(_BASE_PAR, dtype=np.float64)
    for idx in _ZERO_INDICES_BY_VERSION[version]:
        par[idx] = 0.0
    return par


def _build_golden_data(tmp_path, version: int, mod_num: str, n_tot_raw: int = N_TOT_RAW):
    """
    Build a `CommonData` for a `FORWARD` run over `n_tot_raw` days starting
    2000-01-01 (a leap year, so the comparison exercises leap-year `tt`
    handling), via the real `read_calibration`/`read_Tseries` pipeline.

    Returns `(data, par, Tair, Q)` -- the latter three for the matching
    `run_fortran_model` call.
    """
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range('2000-01-01', periods=n_tot_raw, freq='D')
    day_idx = np.arange(n_tot_raw)
    Tair = 15.0 + 10.0 * np.sin(2.0 * PI * day_idx / 365.25)
    Q = np.full(n_tot_raw, QMEDIA, dtype=np.float64)

    input_csv = tmp_path / 'input.csv'
    pd.DataFrame({'Date': dates, 'T_air': Tair, 'Discharge': Q}).to_csv(input_csv, index=False)

    par = _par_for_version(version)
    output_dir = tmp_path / 'output'

    config = {
        'project_name': 'golden_test',
        'station_name': 'GoldenStation',
        'run_mode': 'FORWARD',
        'version': version,
        'integrator': mod_num,
        'objective_function': 'NSE',
        'Qmedia': QMEDIA,
        'parameters_forward': [float(x) for x in par],
        'paths': {
            'input_data': str(input_csv),
            'output_dir': str(output_dir),
        },
    }
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

    data = read_calibration(config_file=str(config_path))
    read_Tseries(data, 'c')

    return data, par, Tair, Q


@pytest.mark.parametrize('version', VERSIONS)
@pytest.mark.parametrize('mod_num', FORTRAN_INTEGRATORS)
def test_golden_version_integrator_matrix(tmp_path, version, mod_num):
    """20-case version x integrator matrix vs the real Fortran reference (docs/audit/08, 8.1)."""
    data, par, Tair, Q = _build_golden_data(tmp_path, version, mod_num)

    call_model(data)

    golden_twat_mod = run_fortran_model(
        version=version,
        mod_num=mod_num,
        n_tot_raw=N_TOT_RAW,
        Tair=Tair,
        Q=Q,
        par=par,
        Qmedia=QMEDIA,
        Twat_initial=4.0,
    )

    np.testing.assert_allclose(
        data.Twat_mod[365:], golden_twat_mod, rtol=1e-7, atol=6e-6,
        err_msg=f"Twat_mod mismatch for version={version}, integrator={mod_num}",
    )


@pytest.mark.parametrize('version', VERSIONS)
def test_exp_matches_crn(tmp_path, version):
    """
    `EXP` has no Fortran equivalent, so it is validated against `CRN` instead
    -- both unconditionally stable, and shown in docs/audit/02_numerical_integration.md
    to agree to well under 0.1 degC. Guards against a future regression in
    either integrator silently drifting apart (docs/audit/08_testing_gaps.md, 8.1).
    """
    data_crn, _, _, _ = _build_golden_data(tmp_path / 'crn', version, 'CRN')
    call_model(data_crn)

    data_exp, _, _, _ = _build_golden_data(tmp_path / 'exp', version, 'EXP')
    call_model(data_exp)

    np.testing.assert_allclose(
        data_exp.Twat_mod[365:], data_crn.Twat_mod[365:], atol=0.1,
        err_msg=f"EXP and CRN diverged by more than 0.1 degC for version={version}",
    )


def test_funcobj_matches_manual_nse():
    """
    `funcobj`'s NSE matches an independent, manually-computed NSE over the
    same valid observations. Does not need the Fortran submodule/gfortran --
    it checks internal consistency of the objective function, not integrator
    output, so it is not parametrized into the golden matrix above.
    """
    data = CommonData()

    n_tot_raw = 100
    n_tot = n_tot_raw + 365

    data.n_tot = n_tot
    data.version = 8
    data.mod_num = 'RK4'
    data.time_res = '1d'
    data.fun_obj = 'NSE'
    data.Qmedia = np.float64(10.0)
    data.Tice_cover = np.float64(0.0)
    data.par = np.array(_BASE_PAR, dtype=np.float64)

    data.Tair = np.full(n_tot, 15.0, dtype=np.float64)
    data.Q = np.full(n_tot, 10.0, dtype=np.float64)

    data.tt = np.zeros(n_tot, dtype=np.float64)
    for j in range(1, 366):
        data.tt[j - 1] = j / 365.0
    for j in range(1, 367):
        if 365 + j - 1 >= n_tot:
            break
        data.tt[365 + j - 1] = j / 366.0

    data.date = np.zeros((n_tot, 3), dtype=np.int32)
    data.Twat_mod = np.zeros(n_tot, dtype=np.float64)
    data.Twat_mod[0] = 4.0
    data.Twat_obs = np.full(n_tot, -999.0, dtype=np.float64)
    for i in range(10):
        data.Twat_obs[365 + i] = 5.0 + i * 0.5

    call_model(data)

    data.I_pos = np.zeros(n_tot, dtype=np.int32)
    data.I_inf = np.zeros((n_tot, 3), dtype=np.int32)
    data.Twat_obs_agg = np.zeros(n_tot, dtype=np.float64)
    data.eval_mask = np.zeros(n_tot, dtype=np.bool_)
    data.eval_mask[365:] = True

    n_inf = 0
    n_pos = 0
    for i in range(1, n_tot):
        data.I_pos[n_pos] = i
        data.Twat_obs_agg[i] = data.Twat_obs[i]
        if data.Twat_obs[i] != -999.0:
            data.I_inf[n_inf, 0] = n_pos
            data.I_inf[n_inf, 1] = n_pos
            data.I_inf[n_inf, 2] = i
            n_inf += 1
        n_pos += 1

    data.n_dat = n_inf
    statis(data)

    computed_nse = funcobj(data)

    valid_obs_mask = data.Twat_obs != -999.0
    valid_obs = data.Twat_obs[valid_obs_mask]
    valid_mod = data.Twat_mod[valid_obs_mask]

    mean_obs = np.mean(valid_obs)
    tss_obs = np.sum((valid_obs - mean_obs) ** 2)
    tss_mod = np.sum((valid_mod - valid_obs) ** 2)
    expected_nse = 1.0 - (tss_mod / tss_obs)

    np.testing.assert_allclose(computed_nse, expected_nse, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
