"""
The settings a calibration was fitted with go with its parameters, and a later run must
use the same ones; Qmedia must be positive.

- calibration_metadata.json and the MCMC chain's _meta.json record Tice_cover and
  min_theta_floor (which change what the model computes) and calendar, gap_tolerant and
  time_resolution (which describe the fit).
- A FORWARD run that loads either file stops if its Tice_cover or min_theta_floor differ
  (they were accepted without a word); files without the record give a note.
- Qmedia <= 0 is refused for the versions that use discharge, in every mode (outside
  gap-tolerant mode every simulated temperature was NaN and the run stopped with a
  divergence error that blamed the integrator).
"""

import json

import numpy as np
import pytest

from pyair2stream.main import _write_calibration_metadata
from tests.test_correctness_fixes import PAR, _csv, _load

SETTINGS = {'Tice_cover': 0.0, 'min_theta_floor': None}


def _forward_from_metadata(tmp_path, meta, **kw):
    _csv(tmp_path / 'cal.csv')
    (tmp_path / 'meta.json').write_text(json.dumps(
        {'qmedia': 4.5, 'version': kw.get('version', 8), 'integrator': 'CRN', 'par_best': PAR, **meta}))
    paths = {'input_data': str(tmp_path / 'cal.csv'), 'output_dir': str(tmp_path / 'out'),
             'calibration_metadata': str(tmp_path / 'meta.json')}
    return _load(tmp_path, run_mode='FORWARD', paths=paths, **kw)


def test_calibration_metadata_records_the_settings(tmp_path):
    _csv(tmp_path / 'cal.csv')
    data = _load(tmp_path, Tice_cover=0.5, min_theta_floor=1e-6, time_resolution='1w')
    data.par_best = np.array(PAR)
    (tmp_path / 'out').mkdir(exist_ok=True)
    data.folder = str(tmp_path / 'out')
    _write_calibration_metadata(data)
    meta = json.loads((tmp_path / 'out' / 'calibration_metadata.json').read_text())
    assert meta['Tice_cover'] == 0.5 and meta['min_theta_floor'] == 1e-6
    assert meta['calendar'] == 'standard' and meta['gap_tolerant'] is False
    assert meta['time_resolution'] == '1w'


def test_matching_settings_are_accepted(tmp_path, capsys):
    data = _forward_from_metadata(tmp_path, SETTINGS)
    assert data.Qmedia == 4.5
    assert 'cannot be checked' not in capsys.readouterr().out


@pytest.mark.parametrize('meta,kw,word', [
    ({'Tice_cover': 0.0, 'min_theta_floor': None}, {'Tice_cover': 4.0}, 'Tice_cover 0.0'),
    ({'Tice_cover': 0.0, 'min_theta_floor': 1e-6}, {}, 'min_theta_floor 1e-06'),
    ({'Tice_cover': 0.0, 'min_theta_floor': None}, {'min_theta_floor': 1e-6}, 'min_theta_floor None'),
    ({'Tice_cover': 0.0, 'min_theta_floor': 1e-6}, {'min_theta_floor': 1e-3}, 'min_theta_floor'),
])
def test_different_settings_are_refused(tmp_path, meta, kw, word):
    with pytest.raises(ValueError, match=word):
        _forward_from_metadata(tmp_path, meta, **kw)


def test_floor_is_not_compared_for_versions_without_discharge(tmp_path):
    data = _forward_from_metadata(tmp_path, {'Tice_cover': 0.0, 'min_theta_floor': 1e-6}, version=5)
    assert data.min_theta_floor is None


def test_metadata_without_the_record_gives_a_note(tmp_path, capsys):
    _forward_from_metadata(tmp_path, {})
    assert 'cannot be checked' in capsys.readouterr().out


@pytest.mark.parametrize('qmedia', [0, -5])
def test_qmedia_not_positive_is_refused(tmp_path, qmedia):
    _csv(tmp_path / 'cal.csv')
    with pytest.raises(ValueError, match='Qmedia must be positive'):
        _load(tmp_path, run_mode='FORWARD', Qmedia=qmedia, parameters_forward=PAR)
    with pytest.raises(ValueError, match='Qmedia must be positive'):
        _load(tmp_path, Qmedia=qmedia)
    # Version 5 does not use discharge, so its Qmedia does not matter.
    _load(tmp_path, run_mode='FORWARD', Qmedia=qmedia, parameters_forward=PAR, version=5)
