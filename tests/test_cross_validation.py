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
    dummy_data.Twat_obs = np.ones(dummy_data.n_tot)
    # min_train_years=1, skip_first_year=True -> first 2 years skipped
    config = CVConfig(unit="year", n_years_per_fold=1, water_year_start_month=1, min_train_years=1, skip_first_year=True)
    folds = build_folds(dummy_data, config)
    assert len(folds) == 2 # 2012, 2013 (2010 and 2011 skipped)
    assert folds[0][0] == "2012"
    assert folds[1][0] == "2013"

def test_build_folds_n_years(dummy_data):
    dummy_data.Twat_obs = np.ones(dummy_data.n_tot)
    config = CVConfig(unit="n_years", n_years_per_fold=2, water_year_start_month=1, min_train_years=0, skip_first_year=True)
    folds = build_folds(dummy_data, config)
    # 2010 skipped. 2011, 2012, 2013 remain. Blocks of 2 -> [2011, 2012]
    # 2013 is dropped because it's a partial block.
    assert len(folds) == 1
    assert folds[0][0] == "2011-2012"

def test_cross_validation_leak_prevention(dummy_data):
    """
    Regression test ensuring that held-out observations are masked using the hardcoded
    MISSING_DATA_SENTINEL, regardless of the user's `mineff_index` setting.
    """
    from pyair2stream.cross_validation import _mask_fold
    from pyair2stream.model import aggregation, statis

    # Set mineff_index to a plausible but non-sentinel value (e.g. 0.5)
    dummy_data.mineff_index = 0.5
    dummy_data.gap_tolerant = False

    n_tot = dummy_data.n_tot
    dummy_data.Tair = np.ones(n_tot) * 10
    dummy_data.Twat_obs = np.ones(n_tot) * 10  # Genuinely 10 everywhere
    dummy_data.Q = np.ones(n_tot) * 10
    dummy_data.tt = np.linspace(0, 4, n_tot)

    dummy_data.parmin = np.array([-5, -5, -5, -1, 0, 0, 0, -1], dtype=np.float64)
    dummy_data.parmax = np.array([15, 1.5, 5, 1, 20, 10, 1, 5], dtype=np.float64)
    dummy_data.par = np.zeros(8, dtype=np.float64)
    dummy_data.par_best = np.zeros(8, dtype=np.float64)
    dummy_data.flag_par = np.ones(8, dtype=int)
    dummy_data.Twat_mod = np.zeros(n_tot)
    dummy_data.Twat_mod_agg = np.zeros(n_tot)
    dummy_data.Twat_obs_agg = np.zeros(n_tot)
    dummy_data.I_pos = np.zeros(n_tot, dtype=np.int32)
    dummy_data.I_inf = np.zeros(n_tot, dtype=np.int32)
    dummy_data.time_res = "1d"
    dummy_data.parmin = np.array([-5, -5, -5, -1, 0, 0, 0, -1], dtype=np.float64)
    dummy_data.parmax = np.array([15, 1.5, 5, 1, 20, 10, 1, 5], dtype=np.float64)
    dummy_data.par = np.zeros(8, dtype=np.float64)
    dummy_data.par_best = np.zeros(8, dtype=np.float64)
    dummy_data.flag_par = np.ones(8, dtype=int)

    # Define a mock fold targeting the second half of the data
    fold_idx = np.arange(n_tot // 2, n_tot)

    # 1. Base case: no masking, everything is valid
    # Since time_res is "1d", statis will set eval_mask to true and populates I_inf
    dummy_data.eval_mask = np.ones(n_tot, dtype=bool)
    aggregation(dummy_data)
    statis(dummy_data)
    n_dat_unmasked = dummy_data.n_dat

    # aggregation skips the first 365 days of warm-up padding when counting n_dat
    assert n_dat_unmasked == n_tot - 365

    # 2. Mask the fold
    orig_twat, orig_tair, orig_q = _mask_fold(dummy_data, fold_idx)

    # Verify the masking used the codebase sentinel, NOT mineff_index
    assert np.all(dummy_data.Twat_obs[fold_idx] == -999.0)

    # 3. Re-aggregate to rebuild I_inf and I_pos mapping
    aggregation(dummy_data)
    statis(dummy_data)
    n_dat_masked = dummy_data.n_dat

    # The masked rows should be fully excluded from the calibration objective size (n_dat)
    assert n_dat_masked == n_dat_unmasked - len(fold_idx)

    # The valid objective indices (I_inf) must not contain any indices from fold_idx
    for valid_agg_idx in range(n_dat_masked):
        # I_inf[:, 2] stores the original row index matching the target array
        original_row_idx = dummy_data.I_inf[valid_agg_idx, 2]
        assert original_row_idx not in fold_idx, "Data leak: held-out row entered calibration objective"

def test_run_leave_one_year_out_cv(dummy_data):
    # Seed numpy random to de-flake the stochastic optimizers with tiny particle/run counts
    np.random.seed(42)

    dummy_data.gap_tolerant = False
    dummy_data.n_run = 10
    dummy_data.version = 5
    dummy_data.mod_num = "RK4"
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
    dummy_data.objective_function = "NSE"
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

def test_run_leave_one_year_out_cv_gap_tolerant(dummy_data):
    from pyair2stream.cross_validation import run_leave_one_year_out_cv, summarize
    np.random.seed(42)

    dummy_data.gap_tolerant = True
    dummy_data.n_run = 10
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

    # Required for compute_qmedia
    dummy_data._n_tot_raw = n_tot - 365
    dummy_data.doy_climatology = np.zeros(366)

    dummy_data.segments = [(0, n_tot - 1)]

    # Needs a runmode for the fallback optimizer override
    dummy_data.runmode = 'PSO'
    dummy_data.time_res = "1d"
    dummy_data.n_particles = 5
    dummy_data.mod_num = "RK4"
    dummy_data.fun_obj = "NSE"
    dummy_data.objective_function = "NSE"

    config = CVConfig(
        unit="year",
        n_years_per_fold=1,
        water_year_start_month=1,
        min_train_years=1,
        skip_first_year=True,
        min_valid_obs=1,
        optimizer_overrides={"n_run": 2, "n_particles": 2} # speed up test
    )

    results = run_leave_one_year_out_cv(dummy_data, config, 'PSO')

    assert len(results) > 0, "CV did not return any results"
    for r in results:
        assert r.nse >= -1e6  # Just verify it computed something

    df = summarize(results)
    assert not df.empty

    # Assert data was restored
    assert (dummy_data.Tair != -999.0).all()
    assert (dummy_data.Q != -999.0).all()

def test_run_leave_one_year_out_cv_rejects_first_year(dummy_data):
    from pyair2stream.cross_validation import run_leave_one_year_out_cv
    np.random.seed(42)

    dummy_data.gap_tolerant = False
    dummy_data.n_run = 10
    dummy_data.version = 5
    n_tot = dummy_data.n_tot
    dummy_data.Tair = np.sin(np.linspace(0, 4 * np.pi, n_tot)) + 10
    dummy_data.Twat_obs = np.sin(np.linspace(0, 4 * np.pi, n_tot)) * 0.8 + 10
    dummy_data.Q = np.ones(n_tot) * 10
    dummy_data.tt = np.linspace(0, 4, n_tot)

    dummy_data.Twat_mod_agg = np.zeros(n_tot)
    dummy_data.Twat_obs_agg = np.zeros(n_tot)
    dummy_data.I_pos = np.zeros(n_tot, dtype=np.int32)
    dummy_data.I_inf = np.zeros(n_tot, dtype=np.int32)
    dummy_data.time_res = "1d"
    dummy_data.parmin = np.array([-5, -5, -5, -1, 0, 0, 0, -1], dtype=np.float64)
    dummy_data.parmax = np.array([15, 1.5, 5, 1, 20, 10, 1, 5], dtype=np.float64)
    dummy_data.par = np.zeros(8, dtype=np.float64)
    dummy_data.par_best = np.zeros(8, dtype=np.float64)
    dummy_data.flag_par = np.ones(8, dtype=int)
    dummy_data.Twat_mod = np.zeros(n_tot)
    dummy_data.mod_num = "RK4"
    dummy_data.fun_obj = "NSE"
    dummy_data.folder = ""
    dummy_data.station = "test"
    dummy_data.series = "test"

    # We want to test unit="n_years", water_year_start_month=10, skip_first_year=False, min_train_years=0
    # Our dummy data is 4 years (1460 days) starting 2010-01-01
    # 2010 water year (Oct 2009-Sep 2010): first 273 days (Jan 1 to Sep 30 2010)
    # 2011 water year (Oct 2010-Sep 2011): next 365 days
    # 2012 water year (Oct 2011-Sep 2012): next 366 days (leap)
    # 2013 water year (Oct 2012-Sep 2013): next 365 days
    # 2014 water year (Oct 2013-Dec 2013): remaining 91 days
    config = CVConfig(
        unit="n_years",
        n_years_per_fold=2,
        water_year_start_month=10,
        min_train_years=0,
        skip_first_year=False,
        optimizer_overrides={"n_run": 2, "n_particles": 2}
    )

    import pytest
    with pytest.raises(ValueError, match="The first year cannot be a candidate fold"):
        run_leave_one_year_out_cv(dummy_data, config, 'PSO')
