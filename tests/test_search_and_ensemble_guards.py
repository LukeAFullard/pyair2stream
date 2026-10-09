"""
Guards added by the October 2026 audit:

1. LATHYP keeps its best sample even when every sample scores below -999 (it used to
   start from a floor of -999 and could return the all-zero start parameters).
2. An ensemble draw run with an explicit integrator (RK4/RK2/EUL) is excluded when a
   difference can grow more than `stability_max_growth` times (`largest_growth`), even if
   its simulation stays below `max_plausible_twat`.
"""

import shutil

import numpy as np
import pytest

import pyair2stream.optimization as optimization
from pyair2stream.model import is_numerically_divergent, largest_growth

from test_numerical_stability import FLOOD_PAR, _build_theta_data
from test_optimization import TestOptimization


def _lathyp_data(tmp_path):
    case = TestOptimization()
    case.setUp()
    data = case.data
    shutil.rmtree(data.folder, ignore_errors=True)   # setUp's folder: write to tmp_path instead
    data.folder = str(tmp_path)
    data.runmode = 'LATHYP'
    data.n_run = 6
    return data


def test_lathyp_keeps_its_best_sample_when_every_score_is_below_minus_999(tmp_path, monkeypatch):
    data = _lathyp_data(tmp_path)
    scores = iter([-5000.0, -2000.0, np.nan, -3000.0, -4000.0, -6000.0])
    tried = []

    def fake_sub_1(d):
        tried.append(d.par.copy())
        return next(scores)

    monkeypatch.setattr(optimization, 'sub_1', fake_sub_1)
    optimization.LH_mode(data, seed=1)
    assert data.finalfit == -2000.0
    np.testing.assert_array_equal(data.par_best, tried[1])


def test_lathyp_refuses_when_no_sample_has_a_finite_score(tmp_path, monkeypatch):
    data = _lathyp_data(tmp_path)
    monkeypatch.setattr(optimization, 'sub_1', lambda d: np.nan)
    with pytest.raises(RuntimeError, match="finite score"):
        optimization.LH_mode(data, seed=1)


def test_an_unstable_explicit_draw_counts_as_divergent_even_if_its_values_look_plausible():
    # A 20-day flood takes B to 3.5, above RK4's limit of 2.785.
    theta = np.ones(800)
    theta[500:520] = 6.0
    for mod_num, expected in (('RK4', True), ('CRN', False)):
        data = _build_theta_data(mod_num, FLOOD_PAR, theta)
        data.Twat_mod[:] = 10.0               # plausible values: only the stability check can object
        assert is_numerically_divergent(data, 60.0) is expected
    assert largest_growth(_build_theta_data('RK4', FLOOD_PAR, theta))['growth'] > 100.0
