import numpy as np
import pandas as pd
import pytest
from datetime import date
from pyair2stream.config import CommonData
from pyair2stream.cross_validation import (
    CVConfig, assign_year_groups, build_folds, run_leave_one_year_out_cv
)

@pytest.fixture
def dummy_data():
    data = CommonData()
    n_tot = 365 * 4
    data.n_tot = n_tot
    data.date = np.zeros((n_tot, 3), dtype=np.int32)

    start_date = pd.Timestamp("2010-01-01")
    dates = pd.date_range(start_date, periods=n_tot, freq="D")
    data.date[:, 0] = dates.year
    data.date[:, 1] = dates.month
    data.date[:, 2] = dates.day

    return data

def test_assign_year_groups_calendar(dummy_data):
    wy = assign_year_groups(dummy_data, water_year_start_month=1)
    assert np.all(wy == dummy_data.date[:, 0])

def test_assign_year_groups_water_year(dummy_data):
    wy = assign_year_groups(dummy_data, water_year_start_month=10)
    # 2010-09-30 should be 2010
    idx_sept = np.where((dummy_data.date[:, 0] == 2010) & (dummy_data.date[:, 1] == 9))[0][-1]
    assert wy[idx_sept] == 2010
    # 2010-10-01 should be 2011
    idx_oct = np.where((dummy_data.date[:, 0] == 2010) & (dummy_data.date[:, 1] == 10))[0][0]
    assert wy[idx_oct] == 2011

def test_build_folds_single_year(dummy_data):
    # min_train_years=1, skip_first_year=True -> first 2 years skipped
    config = CVConfig(unit="year", n_years_per_fold=1, water_year_start_month=1, min_train_years=1, skip_first_year=True)
    folds = build_folds(dummy_data, config)
    assert len(folds) == 2 # 2012, 2013 (2010 and 2011 skipped)
    assert folds[0][0] == "2012"
    assert folds[1][0] == "2013"

def test_build_folds_n_years(dummy_data):
    config = CVConfig(unit="n_years", n_years_per_fold=2, water_year_start_month=1, min_train_years=0, skip_first_year=True)
    folds = build_folds(dummy_data, config)
    # 2010 skipped. 2011, 2012, 2013 remain. Blocks of 2 -> [2011, 2012]
    # 2013 is dropped because it's a partial block.
    assert len(folds) == 1
    assert folds[0][0] == "2011-2012"

def test_run_leave_one_year_out_cv(dummy_data):
    dummy_data.gap_tolerant = False
    dummy_data.n_run = 10
    dummy_data.mineff_index = -999.0
    dummy_data.version = 5
    n_tot = dummy_data.n_tot
    dummy_data.Tair = np.sin(np.linspace(0, 4 * np.pi, n_tot)) + 10
    dummy_data.Twat_obs = np.sin(np.linspace(0, 4 * np.pi, n_tot)) * 0.8 + 10
    dummy_data.Q = np.ones(n_tot) * 10
    dummy_data.tt = np.linspace(0, 4, n_tot)

    dummy_data.parmin = np.zeros(8)
    dummy_data.parmax = np.ones(8) * 10
    dummy_data.par = np.ones(8)
    dummy_data.par_best = np.ones(8)
    dummy_data.flag_par = np.ones(8, dtype=bool)

    # Initialize required arrays
    dummy_data.Twat_mod = np.zeros(n_tot)
    dummy_data.Twat_mod_agg = np.zeros(n_tot)
    dummy_data.Twat_obs_agg = np.zeros(n_tot)
    dummy_data.I_pos = np.zeros(n_tot, dtype=np.int32)
    dummy_data.I_inf = np.zeros(n_tot, dtype=np.int32)

    # Needs a runmode for the fallback optimizer override
    dummy_data.runmode = 'PSO'
    dummy_data.time_res = "1d"
    dummy_data.n_particles = 5
    dummy_data.mod_num = "RK4"
    dummy_data.fun_obj = "NSE"

    config = CVConfig(
        unit="year",
        n_years_per_fold=1,
        water_year_start_month=1,
        min_train_years=1,
        skip_first_year=True,
        optimizer_overrides={"n_run": 2}
    )

    # Back up original obs to ensure they aren't mutated permanently
    orig_obs = dummy_data.Twat_obs.copy()

    results = run_leave_one_year_out_cv(dummy_data, config, 'PSO')

    # 2010 skipped (skip_first_year), 2011 skipped (min_train_years) -> 2012, 2013 tested
    assert len(results) == 2
    assert results[0].label == "2012"
    assert results[1].label == "2013"

    # Verify no permanent mutation
    np.testing.assert_array_equal(dummy_data.Twat_obs, orig_obs)

    # Verify we got metrics
    for r in results:
        assert not np.isnan(r.rmse)
