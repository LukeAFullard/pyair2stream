"""
Regression tests for defects found in the September 2026 correctness review.

Each test reproduces a defect that previously produced a wrong result with no
error:

- a `-999` missing-value marker in `T_air` was simulated as -999 degC;
- unused parameters in `parameters_forward` changed the CRN physics;
- invalid `version`/`run_mode`/`integrator`/bounds were silently accepted;
- DE-MCMC was not reproducible across processes despite `random_seed`;
- gap-tolerant cross-validation scored a different sample than `statis()` used;
- the goodness-of-fit "R2" was actually NSE;
- a calibration-period prediction band was drawn on the validation plot;
- FORWARD runs not starting on 1 January had an out-of-phase warm-up;
- daily prediction intervals used the residual SD of weekly/monthly means;
- saved ensembles held ~-999 values on gap days;
- a FORWARD run overwrote the calibration's `calibration_metadata.json`.

It also covers FORWARD runs taking their parameters from `calibration_metadata.json`,
and a missing validation file stopping the run instead of skipping validation.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import yaml

from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, statis, call_model, funcobj
from pyair2stream import cross_validation, optimization
from pyair2stream.optimization import _daily_residual_sigma, _noisy_member, DE_mode, forward_mode
from pyair2stream.post_processing import _envelope_on_dates, post_process
from pyair2stream.main import forward

BOUNDS = {'min': [-5, -5, -5, -1, 0, 0, 0, -1], 'max': [15, 1.5, 5, 1, 20, 10, 1, 5]}
PAR = [1.0, 0.25, 0.3, 0.2, 0.5, 1.0, 0.4, 0.1]


def _csv(path, start='2015-01-01', n_days=1461, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq='D')
    doy = dates.dayofyear.values
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'T_air': 10 + 8 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 2, n_days),
        'T_water': 10 + 6 * np.sin(2 * np.pi * (doy - 125) / 365) + rng.normal(0, 0.5, n_days),
        'Discharge': np.exp(1.5 + 0.5 * np.sin(2 * np.pi * (doy - 60) / 365) + rng.normal(0, 0.3, n_days)),
    })
    df.to_csv(path, index=False)
    return df


def _config(tmp_path, name='c.yaml', **kw):
    cfg = {'project_name': 'p', 'station_name': 'S', 'series': 'c', 'time_resolution': '1d',
           'version': 8, 'objective_function': 'NSE', 'integrator': 'CRN', 'run_mode': 'DE',
           'optimization': {'n_run': 3, 'n_particles': 3}, 'parameter_bounds': BOUNDS,
           'paths': {'input_data': str(tmp_path / 'cal.csv'), 'output_dir': str(tmp_path / 'out')}}
    cfg.update(kw)
    path = tmp_path / name
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return str(path)


def _load(tmp_path, **kw):
    data = read_calibration(_config(tmp_path, **kw))
    read_Tseries(data, 'c')
    return data


# --- Input handling -----------------------------------------------------------

def test_minus999_in_tair_is_treated_as_missing(tmp_path):
    df = _csv(tmp_path / 'cal.csv')
    df.loc[500, 'T_air'] = -999.0
    df.to_csv(tmp_path / 'cal.csv', index=False)
    with pytest.raises(ValueError, match='air temperature .* must be complete'):
        _load(tmp_path)


def test_minus999_in_discharge_rejected_even_with_theta_floor(tmp_path):
    df = _csv(tmp_path / 'cal.csv')
    df.loc[500, 'Discharge'] = -999.0
    df.to_csv(tmp_path / 'cal.csv', index=False)
    with pytest.raises(ValueError, match='discharge .* must be complete'):
        _load(tmp_path, min_theta_floor=1e-6)


@pytest.mark.parametrize('key, value', [('version', 6), ('run_mode', 'DEMCMC'),
                                        ('integrator', 'RK3'), ('objective_function', 'MSE')])
def test_invalid_choices_rejected(tmp_path, key, value):
    _csv(tmp_path / 'cal.csv')
    with pytest.raises(ValueError, match=key):
        read_calibration(_config(tmp_path, **{key: value}))


def test_bounds_must_list_8_values_and_be_ordered(tmp_path):
    _csv(tmp_path / 'cal.csv')
    with pytest.raises(ValueError, match='exactly 8'):
        read_calibration(_config(tmp_path, parameter_bounds={'min': [0, 0, 0], 'max': [1, 1, 1]}))
    bad = {'min': list(BOUNDS['min']), 'max': list(BOUNDS['max'])}
    bad['min'][1] = 5.0  # a2 min > max
    with pytest.raises(ValueError, match='min > max'):
        read_calibration(_config(tmp_path, parameter_bounds=bad))


def test_optimizer_refuses_to_run_without_free_parameters(tmp_path):
    _csv(tmp_path / 'cal.csv')
    cfg = yaml.safe_load(open(_config(tmp_path)))
    del cfg['parameter_bounds']
    with open(tmp_path / 'c.yaml', 'w') as f:
        yaml.safe_dump(cfg, f)
    data = read_calibration(str(tmp_path / 'c.yaml'))
    read_Tseries(data, 'c')
    aggregation(data)
    statis(data)
    with pytest.raises(ValueError, match='No parameter is free'):
        DE_mode(data, seed=1)


# --- Model versions: unused parameters ---------------------------------------

@pytest.mark.parametrize('version, par', [
    (3, [1.0, 0.3, 0.3, 0.0, 0.0, 3.0, 0.5, 0.0]),   # a6 unused by v3
    (4, [1.0, 0.3, 0.3, 0.2, 2.0, 3.0, 0.5, 0.2]),   # a5-a8 unused by v4
    (7, [1.0, 0.3, 0.3, 0.8, 0.5, 1.0, 0.5, 0.1]),   # a4 unused by v7
])
def test_forward_unused_parameters_ignored_by_every_integrator(tmp_path, version, par):
    _csv(tmp_path / 'cal.csv')
    sims = {}
    for integ in ('CRN', 'RK4', 'EXP'):
        data = _load(tmp_path, version=version, integrator=integ, run_mode='FORWARD',
                     Qmedia=4.5, parameters_forward=par)
        call_model(data)
        sims[integ] = data.Twat_mod[365:].copy()
    # Different integrators solve the same equation: agreement to well under 0.5 degC
    # (previously CRN differed from RK4 by 1.3-10 degC here).
    assert np.max(np.abs(sims['CRN'] - sims['RK4'])) < 0.5
    assert np.max(np.abs(sims['CRN'] - sims['EXP'])) < 0.5


def test_integration_ignores_unused_parameters_set_programmatically(tmp_path):
    _csv(tmp_path / 'cal.csv')
    data = _load(tmp_path, version=3, run_mode='FORWARD', Qmedia=4.5,
                 parameters_forward=[1.0, 0.3, 0.3, 0, 0, 0, 0, 0])
    call_model(data)
    clean = data.Twat_mod.copy()
    data.par[5] = 3.0  # a6 is unused by version 3
    call_model(data)
    np.testing.assert_array_equal(data.Twat_mod, clean)


# --- FORWARD runs -------------------------------------------------------------

def test_forward_warmup_phase_follows_copied_rows(tmp_path):
    _csv(tmp_path / 'cal.csv', start='2015-07-01', n_days=800)
    data = _load(tmp_path, run_mode='FORWARD', Qmedia=4.5, parameters_forward=PAR)
    np.testing.assert_array_equal(data.tt[:365], data.tt[365:730])


def test_jan1_standard_calendar_keeps_fortran_warmup_phase(tmp_path):
    _csv(tmp_path / 'cal.csv', start='2016-01-01')  # leap year: real-record phase is j/366
    data = _load(tmp_path)
    np.testing.assert_allclose(data.tt[:365], np.arange(1, 366) / 365.0)


def test_forward_run_does_not_overwrite_calibration_metadata(tmp_path):
    _csv(tmp_path / 'cal.csv')
    out = tmp_path / 'out'
    out.mkdir()
    sentinel = {'qmedia': 4.5, 'written_by': 'calibration'}
    (out / 'calibration_metadata.json').write_text(json.dumps(sentinel))
    data = _load(tmp_path, run_mode='FORWARD', Qmedia=4.5, parameters_forward=PAR)
    forward_mode(data)
    forward(data)
    assert json.loads((out / 'calibration_metadata.json').read_text()) == sentinel


def test_forward_takes_parameters_from_calibration_metadata(tmp_path):
    # Without parameters_forward, a FORWARD run uses the calibrated parameters
    # recorded in calibration_metadata.json, so nobody has to copy them by hand.
    _csv(tmp_path / 'cal.csv')
    meta = {'qmedia': 4.5, 'version': 8, 'integrator': 'CRN', 'par_best': PAR}
    (tmp_path / 'meta.json').write_text(json.dumps(meta))
    paths = {'input_data': str(tmp_path / 'cal.csv'), 'output_dir': str(tmp_path / 'out'),
             'calibration_metadata': str(tmp_path / 'meta.json')}
    data = _load(tmp_path, run_mode='FORWARD', paths=paths)
    np.testing.assert_array_equal(data.par, PAR)
    assert data.Qmedia == 4.5
    with pytest.raises(ValueError, match='FORWARD mode needs parameters'):
        _load(tmp_path, run_mode='FORWARD', Qmedia=4.5)


def test_missing_validation_file_is_an_error(tmp_path):
    # A validation_data path that does not exist (e.g. a typo) must not silently
    # skip validation; leaving validation_data out still skips it.
    _csv(tmp_path / 'cal.csv')
    paths = {'input_data': str(tmp_path / 'cal.csv'), 'output_dir': str(tmp_path / 'out'),
             'validation_data': str(tmp_path / 'typo.csv')}
    data = _load(tmp_path, paths=paths)
    with pytest.raises(FileNotFoundError, match='typo.csv'):
        read_Tseries(data, 'v')
    data = _load(tmp_path)
    read_Tseries(data, 'v')
    assert data.n_tot == 0


# --- Calibration --------------------------------------------------------------

def test_de_keeps_de_solution_if_polish_is_worse(tmp_path, monkeypatch):
    _csv(tmp_path / 'cal.csv')
    data = _load(tmp_path, version=3)
    aggregation(data)
    statis(data)

    class Worse:
        def __init__(self, x):
            self.x = np.asarray(x) * 0 + 0.001
            self.fun = 1e29

    monkeypatch.setattr(optimization, 'minimize', lambda f, x0, **kw: Worse(x0))
    DE_mode(data, seed=3)
    # The DE optimum is kept, not the worse "polished" point.
    assert not np.allclose(data.par_best[:3], 0.001)


def test_mcmc_reproducible_across_processes(tmp_path):
    _csv(tmp_path / 'cal.csv', n_days=730)
    script = textwrap.dedent(f"""
        import contextlib, io, sys
        from pyair2stream.io import read_calibration, read_Tseries
        from pyair2stream.model import aggregation, statis
        from pyair2stream.optimization import DE_MCMC_mode
        data = read_calibration(sys.argv[1]); read_Tseries(data, 'c'); aggregation(data); statis(data)
        with contextlib.redirect_stdout(io.StringIO()):
            DE_MCMC_mode(data, seed=data.random_seed)
    """)
    chains = []
    for run in range(2):
        cfg = _config(tmp_path, name=f'c{run}.yaml', version=3, run_mode='DE-MCMC', random_seed=7,
                      optimization={'n_run': 3, 'n_particles': 3, 'mcmc_walkers': 8, 'mcmc_steps': 40},
                      uncertainty_options={'strict_convergence': False},
                      paths={'input_data': str(tmp_path / 'cal.csv'), 'output_dir': str(tmp_path / f'o{run}')})
        subprocess.run([sys.executable, '-c', script, cfg], check=True, capture_output=True)
        chains.append(pd.read_csv(tmp_path / f'o{run}' / 'MCMC_chain_S_c_1d.csv').values)
    np.testing.assert_array_equal(chains[0], chains[1])


def test_gap_tolerant_cv_statis_uses_the_scored_sample(tmp_path, monkeypatch):
    _csv(tmp_path / 'cal.csv')
    data = _load(tmp_path, gap_tolerant=True)
    aggregation(data)
    statis(data)
    checked = []

    def fake_optimizer(d, run_mode, overrides):
        # Every window statis() counted must be scoreable by funcobj() under the
        # current eval_mask, or NSE mixes two different samples.
        assert all(d.eval_mask[d.I_inf[i, 2]] for i in range(d.n_dat))
        d.par_best = np.array(PAR)
        checked.append(True)

    monkeypatch.setattr(cross_validation, '_run_optimizer', fake_optimizer)
    cross_validation.run_leave_one_year_out_cv(data, cross_validation.CVConfig(), 'DE')
    assert checked


# --- Uncertainty --------------------------------------------------------------

def test_prediction_interval_sigma_is_daily(tmp_path):
    _csv(tmp_path / 'cal.csv')
    for res in ('1d', '1w'):
        data = _load(tmp_path, time_resolution=res)
        aggregation(data)
        statis(data)
        data.par[:] = PAR
        call_model(data)
        funcobj(data)
        m = data.eval_mask & (data.Twat_obs != -999.0)
        daily = np.sqrt(np.mean((data.Twat_mod[m] - data.Twat_obs[m]) ** 2))
        assert _daily_residual_sigma(data, data.eval_mask) == pytest.approx(daily)
        if res == '1d':  # at daily resolution this is also the RMSE of the scored series
            ma = (data.Twat_obs_agg != -999.0) & data.eval_mask
            agg = np.sqrt(np.mean((data.Twat_mod_agg[ma] - data.Twat_obs_agg[ma]) ** 2))
            assert daily == pytest.approx(agg)


def test_ensemble_members_are_nan_on_gap_days():
    tw = np.array([10.0, -999.0, 12.0])
    member = _noisy_member(tw, np.array([0.5, 0.5, 0.5]))
    assert member[0] == 10.5 and np.isnan(member[1]) and member[2] == 12.5


# --- Reporting ----------------------------------------------------------------

def _envelope_csv(folder, name, start, n):
    dates = pd.date_range(start, periods=n, freq='D')
    pd.DataFrame({'Year': dates.year, 'Month': dates.month, 'Day': dates.day,
                  'Twat_mod_lower': 1.0, 'Twat_mod_p50': 2.0, 'Twat_mod_upper': 3.0}
                 ).to_csv(os.path.join(folder, name), index=False)


def test_envelope_matched_by_date_not_row_position(tmp_path):
    _csv(tmp_path / 'cal.csv')
    data = read_calibration(_config(tmp_path, run_mode='DE-MCMC'))
    _envelope_csv(data.folder, 'MCMC_envelopes_S_c_1d.csv', '2015-01-01', 1095)
    assert _envelope_on_dates(data, pd.date_range('2018-01-01', periods=365)) is None
    band = _envelope_on_dates(data, pd.date_range('2015-03-01', periods=10))
    assert band is not None and len(band) == 10
    # A FORWARD run uses its own envelope, not a calibration envelope in the same folder.
    data.runmode = 'FORWARD'
    assert _envelope_on_dates(data, pd.date_range('2015-03-01', periods=10)) is None


def test_goodness_of_fit_reports_nse_and_r2_separately(tmp_path):
    _csv(tmp_path / 'cal.csv')
    data = read_calibration(_config(tmp_path))
    dates = pd.date_range('2015-01-01', periods=365)
    obs = 10 + 5 * np.sin(np.arange(365) / 58.0)
    mod = 1.2 * obs + 1.0  # perfectly correlated but biased: R2 = 1, NSE < 1
    pd.DataFrame({'Year': dates.year, 'Month': dates.month, 'Day': dates.day, 'Tair': obs,
                  'Twat_obs': obs, 'Twat_mod': mod, 'Twat_obs_agg': obs, 'Twat_mod_agg': mod,
                  'Q': 5.0}).to_csv(os.path.join(data.folder, '2_DE_NSE_S_cc_1d.csv'), index=False)
    post_process(data)
    gof = pd.read_csv(os.path.join(data.folder, 'goodness_of_fit_calibration_DE_NSE_S.csv')).set_index('Metric')['Value']
    assert gof['R2'] == pytest.approx(1.0)
    assert gof['NSE'] == pytest.approx(1 - np.sum((mod - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))
    assert gof['NSE'] < 0.9


# --- Added checks -------------------------------------------------------------

def test_interval_coverage_reports_share_of_observations_inside(tmp_path):
    _csv(tmp_path / 'cal.csv')
    data = _load(tmp_path)
    obs = data.Twat_obs
    inside = np.zeros(data.n_tot, dtype=bool)
    inside[365::2] = True  # every other real day inside the band
    env = pd.DataFrame({'Twat_mod_lower': np.where(inside, obs - 1, obs + 1),
                        'Twat_mod_upper': np.where(inside, obs + 1, obs + 2)})
    result = optimization._interval_coverage(data, env, 90.0)
    n = data.n_tot - 365
    assert result['interval_coverage_n_days'] == n
    assert result['interval_coverage'] == pytest.approx(np.sum(inside[365:]) / n)


def test_segment_warmup_warning_when_relaxation_is_slow(tmp_path, capsys):
    from pyair2stream.model import check_segment_warmup
    _csv(tmp_path / 'cal.csv')
    data = _load(tmp_path, version=3, gap_tolerant=True, warmup_drop_days=15)
    data.par[:] = [0.5, 0.05, 0.05, 0, 0, 0, 0, 0]  # B = a3 = 0.05/day -> 60 days needed
    check_segment_warmup(data)
    assert 'warmup_drop_days: 60' in capsys.readouterr().out
    data.par[2] = 1.0  # B = 1/day -> 3 days needed: no warning
    check_segment_warmup(data)
    assert 'warmup_drop_days' not in capsys.readouterr().out
