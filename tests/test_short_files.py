"""
Validation and scenario (FORWARD) files shorter than a year, down to 30 days.

A file shorter than a year has no first year to copy as the model's warm-up, so the
warm-up holds the first day's conditions: the model settles at the water temperature
that matches them, its first warmup_drop_days are not scored, and the outputs mark them
(`warm_up`). A calibration still needs a whole year (tests/test_data_checks.py).
"""

import numpy as np
import pandas as pd
import pytest

import pyair2stream
from pyair2stream.model import call_model, check_segment_warmup
from tests.test_correctness_fixes import PAR, _config, _csv, _load


def _short(tmp_path, start='2016-06-01', n=60, **kw):
    df = _csv(tmp_path / 'long.csv')
    df[df.Date >= start].head(n).to_csv(tmp_path / 'cal.csv', index=False)
    return _load(tmp_path, run_mode='FORWARD', Qmedia=4.5, parameters_forward=PAR, **kw)


def test_warm_up_holds_the_first_days_conditions(tmp_path):
    data = _short(tmp_path)
    assert data.warmup_from_first_day and data.n_tot == 365 + 60
    for series in (data.Tair, data.Q, data.tt):
        assert np.all(series[:365] == series[365])
    assert not data.eval_mask[365:380].any() and data.eval_mask[380:].all()


def test_the_start_is_forgotten_within_the_unscored_days(tmp_path):
    # The same 60 days, simulated as a short file and inside the 4-year record.
    data = _short(tmp_path)
    call_model(data)
    long = _load(tmp_path, run_mode='FORWARD', Qmedia=4.5, parameters_forward=PAR,
                 paths={'input_data': str(tmp_path / 'long.csv'), 'output_dir': str(tmp_path / 'out')})
    call_model(long)
    i = 365 + int(np.flatnonzero(pd.read_csv(tmp_path / 'long.csv').Date == '2016-06-01')[0])
    difference = np.abs(data.Twat_mod[365:] - long.Twat_mod[i:i + 60])
    assert difference[15:].max() < 0.05


def test_files_of_a_year_or_more_and_gap_tolerant_files_are_unchanged(tmp_path):
    _csv(tmp_path / 'cal.csv')
    data = _load(tmp_path, run_mode='FORWARD', Qmedia=4.5, parameters_forward=PAR)
    assert not data.warmup_from_first_day and data.eval_mask[365:].all()
    data = _short(tmp_path, gap_tolerant=True)
    assert not data.warmup_from_first_day


def test_warning_when_the_model_needs_longer_to_forget_its_start(tmp_path, capsys):
    data = _short(tmp_path, warmup_drop_days=1)
    capsys.readouterr()
    check_segment_warmup(data)
    assert "the first days of the file may still reflect" in capsys.readouterr().out
    data = _short(tmp_path, warmup_drop_days=60)
    capsys.readouterr()
    check_segment_warmup(data)
    assert capsys.readouterr().out == ""


def test_short_scenario_run_marks_its_warm_up_days(tmp_path):
    _short(tmp_path, n=45)
    result = pyair2stream.run(_config(tmp_path, run_mode='FORWARD', Qmedia=4.5, parameters_forward=PAR),
                              verbose=False)
    assert any("has 45 days, less than a year" in m for m in result.messages)
    out = [f for f in result.files if f.startswith('2_')][0]
    df = pd.read_csv(f"{result.output_dir}/{out}")
    assert len(df) == 45 and df.warm_up.tolist() == [1] * 15 + [0] * 30


def test_short_validation_file_is_used(tmp_path):
    _csv(tmp_path / 'cal.csv', n_days=400)
    _csv(tmp_path / 'val.csv', start='2017-06-01', n_days=60, seed=1)
    result = pyair2stream.run(_config(tmp_path, optimization={'n_run': 1, 'n_particles': 2},
                                      paths={'input_data': str(tmp_path / 'cal.csv'),
                                             'validation_data': str(tmp_path / 'val.csv'),
                                             'output_dir': str(tmp_path / 'out')}), verbose=False)
    out = [f for f in result.files if f.startswith('3_')][0]
    df = pd.read_csv(f"{result.output_dir}/{out}")
    assert len(df) == 60 and df.warm_up.sum() == 15
    assert "validation" in result.scores
    with pytest.raises(KeyError):
        df['segment_id']
