"""
pyair2stream.run(), the run summary (summary.md) and the gap-filled water temperature files.
"""

import os

import numpy as np
import pandas as pd
import pytest

import pyair2stream
from pyair2stream.io import read_calibration
from pyair2stream.results import messages_from

BOUNDS = {'min': [-5, -5, -5, -1, 0, 0, 0, -1], 'max': [15, 1.5, 5, 1, 20, 10, 1, 5]}


def _csv(path, start='2015-01-01', n_days=1096, seed=0, drop=None):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq='D')
    doy = dates.dayofyear.values
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'T_air': 10 + 8 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 2, n_days),
        'T_water': 10 + 6 * np.sin(2 * np.pi * (doy - 125) / 365) + rng.normal(0, 0.5, n_days),
        'Discharge': np.exp(1.5 + 0.5 * np.sin(2 * np.pi * (doy - 60) / 365) + rng.normal(0, 0.3, n_days)),
    })
    if drop is not None:
        df.loc[drop, 'T_water'] = np.nan
    df.to_csv(path, index=False)
    return df


def _config(tmp_path, **kw):
    cfg = {'project_name': 'p', 'station_name': 'S', 'series': 'c', 'time_resolution': '1d',
           'version': 5, 'objective_function': 'NSE', 'integrator': 'CRN', 'run_mode': 'DE',
           'random_seed': 1, 'optimization': {'n_run': 3, 'n_particles': 3}, 'parameter_bounds': BOUNDS,
           'paths': {'input_data': str(tmp_path / 'cal.csv'), 'output_dir': str(tmp_path / 'out')}}
    cfg.update(kw)
    return cfg


def test_read_calibration_accepts_a_dict(tmp_path):
    _csv(tmp_path / 'cal.csv')
    cfg = _config(tmp_path)
    data = read_calibration(cfg)
    assert data.version == 5 and data.runmode == 'DE'
    assert cfg['paths']['output_dir'] == str(tmp_path / 'out')      # the caller's dict is not changed


def test_run_returns_the_results_and_writes_summary_and_filled_series(tmp_path, capsys):
    missing = list(range(200, 230))                                    # a month without water temperature
    _csv(tmp_path / 'cal.csv', drop=missing)
    _csv(tmp_path / 'val.csv', start='2018-01-01', seed=1)
    cfg = _config(tmp_path, paths={'input_data': str(tmp_path / 'cal.csv'),
                                   'validation_data': str(tmp_path / 'val.csv'),
                                   'output_dir': str(tmp_path / 'out')})
    result = pyair2stream.run(cfg, verbose=False)
    assert capsys.readouterr().out == ''                               # verbose=False prints nothing

    assert result.output_dir == str(tmp_path / 'out')
    assert result.run_mode == 'DE' and result.version == 5
    assert set(result.parameters) == {'a1', 'a2', 'a3', 'a6', 'a7'}  # version 5's parameters only
    assert np.allclose(list(result.parameters.values()),
                       [result.data.par_best[j] for j in (0, 1, 2, 5, 6)])
    assert {'calibration', 'validation'} <= set(result.scores)
    gof = pd.read_csv(os.path.join(result.output_dir, 'goodness_of_fit_validation_DE_NSE_S.csv'))
    assert result.scores['validation']['RMSE'] == pytest.approx(float(gof.set_index('Metric').Value['RMSE']))

    summary = open(result.summary, encoding='utf-8').read()
    assert result.summary == os.path.join(result.output_dir, 'summary.md') and 'summary.md' in result.files
    for heading in ('## Settings', '## Data', '## How well the model fits', '## Parameters',
                    '## Warnings and notes', '## Output files'):
        assert heading in summary
    assert f"{result.scores['validation']['RMSE']:.2f}" in summary

    # The same page as HTML, with every figure embedded, and summary.md showing and linking them.
    page = open(os.path.join(result.output_dir, 'summary.html'), encoding='utf-8').read()
    assert 'summary.html' in result.files and page.startswith('<!doctype html>')
    assert '<h2>How well the model fits</h2>' in page and f"{result.scores['validation']['RMSE']:.2f}" in page
    figures = [f for f in result.files if f.endswith('.png')]
    assert figures and page.count('<img src="data:image/png;base64,') == len(figures)
    assert all(f"]({f})" in summary for f in figures)

    filled = pd.read_csv(os.path.join(result.output_dir, 'filled_water_temperature_calibration.csv'),
                         parse_dates=['Date'])
    assert len(filled) == 1096
    gap = filled.iloc[missing]
    assert (gap.source == 'model').all() and gap.T_water_measured.isna().all()
    assert np.allclose(gap.T_water_filled, gap.T_water_model)
    kept = filled.drop(index=missing)
    assert (kept.source == 'measured').all()
    assert np.allclose(kept.T_water_filled, kept.T_water_measured)
    assert os.path.exists(os.path.join(result.output_dir, 'filled_water_temperature_validation.csv'))


def test_run_records_warnings_and_notes(tmp_path):
    _csv(tmp_path / 'cal.csv')
    _csv(tmp_path / 'val.csv', start='2016-01-01', seed=1)     # overlaps the calibration years
    cfg = _config(tmp_path, paths={'input_data': str(tmp_path / 'cal.csv'),
                                   'validation_data': str(tmp_path / 'val.csv'),
                                   'output_dir': str(tmp_path / 'out')})
    result = pyair2stream.run(cfg, verbose=False)
    overlap = [m for m in result.messages if 'calibration file' in m and m.startswith('Warning')]
    assert overlap, result.messages
    assert overlap[0] in open(result.summary, encoding='utf-8').read()


def test_run_raises_instead_of_exiting(tmp_path):
    with pytest.raises(FileNotFoundError):
        pyair2stream.run(str(tmp_path / 'missing.yaml'), verbose=False)


def test_dropped_segment_warnings_are_counted():
    lines = ["Warning: Dropped segment (400, 410) of length 11 days (min_segment_days=30)",
             "Note: something", "Warning: Dropped segment (500, 505) of length 6 days (min_segment_days=30)",
             "Note: something", "an ordinary line"]
    messages = messages_from(lines)
    assert messages[0].startswith("Warning: 2 stretch(es)")
    assert messages[1:] == ["Note: something"]
