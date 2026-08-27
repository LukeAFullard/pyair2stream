"""
Acceptance tests for docs/audit/06_diagnostics_and_plots.md.

Covers: dotty-plot column selection by name (Defect A / 6.1), PSO history NSE/R2/MAE
surviving the multiprocessing boundary (Defect C / 6.3), sensitivity_analysis using
calibration-derived (not stale validation) segments (Defect D / 6.4), the
sensitivity-index normalization mode (Defect E / 6.5), and the gap-aware residual ACF
(Defect F / 6.6).
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from pyair2stream.config import CommonData, PI
from pyair2stream.post_processing import select_dotty_data, gap_aware_acf
from pyair2stream.optimization import PSO_mode
from pyair2stream.model import detect_segments, statis
from pyair2stream.io import read_Tseries
from pyair2stream.sensitivity import sensitivity_analysis


# ---------------------------------------------------------------------------
# 6.1 -- dotty-plot data extraction by name
# ---------------------------------------------------------------------------

def test_select_dotty_data_uses_names_not_position():
    df_0 = pd.DataFrame({
        'par_1': [1.0, 2.0, 3.0],
        'par_2': [0.1, 0.2, 0.3],
        'par_3': [0.1, 0.2, 0.3],
        'par_4': [0.5, 0.5, 0.5],
        'par_5': [1.0, 1.0, 1.0],
        'par_6': [1.0, 1.0, 1.0],
        'par_7': [0.5, 0.5, 0.5],
        'par_8': [0.1, 0.1, 0.1],
        'eff_index': [0.5, 0.9, 0.2],
        'NSE': [0.5, 0.9, 0.2],
        'R2': [0.6, 0.95, 0.3],
        'MAE': [4.3, 1.0, 9.9],  # deliberately NOT correlated with eff_index/NSE ranking
    })

    parset, eff, n_par = select_dotty_data(df_0)

    assert n_par == 8
    assert parset.shape == (3, 8)
    np.testing.assert_array_equal(eff, df_0['eff_index'].to_numpy())
    # The best (highest NSE/eff_index) row is index 1; under the old positional bug,
    # eff would have been MAE and the "best" row would have been index 2 instead.
    assert int(np.argmax(eff)) == 1


# ---------------------------------------------------------------------------
# 6.3 -- PSO history records real NSE/R2/MAE, not parent-process -999.0 defaults
# ---------------------------------------------------------------------------

def _build_pso_data(folder):
    data = CommonData()
    data.n_tot = 375
    data.date = np.zeros((data.n_tot, 3), dtype=np.int32)
    data.tt = np.zeros(data.n_tot, dtype=np.float64)
    data.Tair = np.zeros(data.n_tot, dtype=np.float64)
    data.Q = np.zeros(data.n_tot, dtype=np.float64)
    data.Twat_obs = np.full(data.n_tot, -999.0, dtype=np.float64)
    data.Twat_mod = np.zeros(data.n_tot, dtype=np.float64)
    data.Twat_obs_agg = np.full(data.n_tot, -999.0, dtype=np.float64)
    data.Twat_mod_agg = np.full(data.n_tot, -999.0, dtype=np.float64)

    data.version = 8
    data.mod_num = 'RK4'
    data.time_res = '1d'
    data.fun_obj = 'RMS'
    data.runmode = 'PSO'
    data.station = 'test_station'
    data.series = 'test_series'
    data.Qmedia = np.float64(10.0)
    data.Tice_cover = np.float64(0.0)

    for i in range(data.n_tot):
        data.tt[i] = np.float64(i / 365.0)
        data.Tair[i] = 15.0 + 10.0 * np.sin(2.0 * PI * data.tt[i])
        data.Q[i] = 10.0 + 5.0 * np.cos(2.0 * PI * data.tt[i])
        if i >= 365:
            data.Twat_obs[i] = 12.0 + 8.0 * np.sin(2.0 * PI * data.tt[i])

    n_dat = data.n_tot - 365
    data.n_dat = n_dat
    data.I_inf = np.zeros((n_dat, 3), dtype=np.int32)
    data.I_pos = np.zeros(n_dat, dtype=np.int32)
    for k, i in enumerate(range(365, data.n_tot)):
        data.I_inf[k, 0] = k
        data.I_inf[k, 1] = k
        data.I_inf[k, 2] = i
        data.I_pos[k] = i
        data.Twat_obs_agg[i] = data.Twat_obs[i]

    data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)
    data.parmin = np.zeros(8, dtype=np.float64)
    data.parmax = np.array([2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float64)
    data.flag_par = np.ones(8, dtype=np.bool_)
    data.c1 = 2.0
    data.c2 = 2.0
    data.wmin = 0.4
    data.wmax = 0.9
    data.mineff_index = -1e30
    data.folder = folder
    os.makedirs(folder, exist_ok=True)

    # Populate mean_obs/TSS_obs/std_obs so NSE/R2 are computed against real
    # observation variance, rather than the CommonData defaults of 0.0 (which
    # would make every particle hit the "TSS_obs == 0.0" degenerate branch in
    # model_numba.fast_funcobj regardless of the multiprocessing fix under test).
    statis(data)
    return data


def test_pso_history_records_real_nse_r2_mae():
    folder = 'test_report06_pso_output'
    try:
        data = _build_pso_data(folder)
        data.n_particles = 8
        data.n_run = 3

        PSO_mode(data, seed=1)

        csv_path = os.path.join(folder, "0_PSO_RMS_test_station_test_series_1d.csv")
        df = pd.read_csv(csv_path)

        assert len(df) > 0
        finite_rows = df[np.isfinite(df['eff_index'])]
        assert len(finite_rows) > 0

        # NSE and MAE have no legitimate -999.0 sentinel path here (TSS_obs is fixed
        # and non-zero for this dataset's real, non-degenerate observations -- see
        # model_numba.fast_funcobj); if the multiprocessing return value is dropped
        # and the parent's own untouched defaults get written instead, EVERY row
        # reads exactly -999.0 for both.
        for col in ('NSE', 'MAE'):
            assert not (finite_rows[col] == -999.0).any(), f"{col} contains parent-process -999.0 default"

        # R2 does have a legitimate -999.0 ("undefined, zero simulated variance")
        # sentinel per-particle, so it may be -999.0 for *some* rows; the bug made
        # it -999.0 for *every* row regardless of that particle's actual fit.
        assert not (finite_rows['R2'] == -999.0).all(), "R2 is -999.0 for every row (parent-process default never overwritten)"
    finally:
        if os.path.exists(folder):
            shutil.rmtree(folder)


# ---------------------------------------------------------------------------
# 6.4 -- sensitivity_analysis uses calibration-derived segments
# ---------------------------------------------------------------------------

def _build_sensitivity_data(tmpdir, n_days=400):
    data = CommonData()
    data.folder = tmpdir
    data.runmode = "FORWARD"
    data.fun_obj = "NSE"
    data.station = "test_station"
    data.series = "c"

    csv_path = os.path.join(tmpdir, "mock_data.csv")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'Date': pd.date_range('2000-01-01', periods=n_days, freq='D').strftime('%Y-%m-%d'),
        'T_air': rng.random(n_days) * 10 + 10,
        'T_water': rng.random(n_days) * 5 + 5,
        'Discharge': rng.random(n_days) * 50 + 10,
    })
    df.to_csv(csv_path, index=False)
    data._input_data_path_cal = csv_path
    data.name = tmpdir

    data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1])
    data.par_best = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1])
    data.parmin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    data.parmax = np.array([5.0, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0])
    data.flag_par = np.array([True] * 8)

    data.version = 8
    data.mod_num = 'RK4'
    data.Qmedia_user = 20.0
    data.gap_tolerant = True
    data.sensitivity_perturbations = [1.0]
    return data


def test_sensitivity_analysis_uses_calibration_segments_not_stale_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _build_sensitivity_data(tmpdir)

        # Establish what a fresh read of the calibration data actually produces.
        read_Tseries(data, 'c')
        if data.segments is None:
            detect_segments(data)
        expected_segments = list(data.segments)

        # Now poison data.segments/eval_mask the way a prior main.forward() run
        # left them: stale segments/mask derived from a *different* (here,
        # deliberately wrong) window, simulating report 06's confirmed Defect D
        # scenario where validation-derived segments survived into the
        # sensitivity analysis.
        data.segments = [(365, 380)]
        data.eval_mask = np.zeros(data.n_tot, dtype=bool)

        sensitivity_analysis(data)

        assert data.segments == expected_segments, (
            "sensitivity_analysis used stale segments instead of re-deriving them "
            "from the calibration data (docs/audit/06_diagnostics_and_plots.md, Defect D)"
        )


# ---------------------------------------------------------------------------
# 6.5 -- sensitivity index normalization mode
# ---------------------------------------------------------------------------

def test_sensitivity_index_value_mode_invariant_to_bound_rescaling():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _build_sensitivity_data(tmpdir)
        data.sensitivity_perturbation_mode = 'value'
        df_value_narrow = sensitivity_analysis(data)

    with tempfile.TemporaryDirectory() as tmpdir2:
        data2 = _build_sensitivity_data(tmpdir2)
        data2.sensitivity_perturbation_mode = 'value'
        # Rescale bounds only (par_best/par unchanged, and still far from either
        # bound so nothing gets clipped in either configuration).
        data2.parmax = data2.parmax * 10.0
        df_value_wide = sensitivity_analysis(data2)

    idx_narrow = df_value_narrow.set_index('Parameter')['Sensitivity_Index']
    idx_wide = df_value_wide.set_index('Parameter')['Sensitivity_Index']
    np.testing.assert_allclose(idx_narrow.to_numpy(), idx_wide.to_numpy(), rtol=1e-8, atol=1e-10)


def test_sensitivity_index_range_mode_not_invariant_to_bound_rescaling():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _build_sensitivity_data(tmpdir)
        data.sensitivity_perturbation_mode = 'range'
        df_range_narrow = sensitivity_analysis(data)

    with tempfile.TemporaryDirectory() as tmpdir2:
        data2 = _build_sensitivity_data(tmpdir2)
        data2.sensitivity_perturbation_mode = 'range'
        data2.parmax = data2.parmax * 10.0
        df_range_wide = sensitivity_analysis(data2)

    idx_narrow = df_range_narrow.set_index('Parameter')['Sensitivity_Index']
    idx_wide = df_range_wide.set_index('Parameter')['Sensitivity_Index']
    # Under range-relative normalization, widening the bounds widens the absolute
    # perturbation applied, so the reported index is NOT expected to match --
    # this is the documented behaviour (docs/audit/06_diagnostics_and_plots.md, 6.5).
    assert not np.allclose(idx_narrow.to_numpy(), idx_wide.to_numpy(), equal_nan=True)


def test_sensitivity_bounded_status_on_one_sided_clip():
    with tempfile.TemporaryDirectory() as tmpdir:
        data = _build_sensitivity_data(tmpdir)
        # par_3 (index 2) calibrated at 0.999, close to its max bound of 1.0 --
        # a 10% perturbation clips only the plus side.
        data.par[2] = 0.999
        data.par_best[2] = 0.999
        data.parmax[2] = 1.0
        data.parmin[2] = 0.0
        data.sensitivity_perturbations = [10.0]

        df_sens = sensitivity_analysis(data)
        row = df_sens[(df_sens['Parameter'] == 'par_3') & (df_sens['Perturbation_%'] == 10.0)].iloc[0]
        assert row['Status'] == 'Bounded'


# ---------------------------------------------------------------------------
# 6.6 -- gap-aware residual ACF
# ---------------------------------------------------------------------------

def test_gap_aware_acf_skips_pairs_across_a_gap():
    # Construct a series where a naive dropna()-then-lag-1 approach would pair
    # index 4 (day 4) with index 5 (day 7, since days 5-6 are missing) -- a
    # 3-day gap masquerading as a 1-day lag.
    values = np.array([1.0, 2.0, 1.0, 2.0, 1.0, np.nan, np.nan, 2.0, 1.0, 2.0, 1.0, 2.0])
    residuals = pd.Series(values)

    lags, acf, n_pairs = gap_aware_acf(residuals, max_lag=3)

    # Lag 1 must only use pairs of temporally adjacent valid days: (0,1),(1,2),
    # (2,3),(3,4),(7,8),(8,9),(9,10),(10,11) -- 8 pairs, none crossing the gap.
    assert n_pairs[0] == 8
    # This alternating +1/-1-around-1.5 series has a strong negative lag-1
    # autocorrelation.
    assert acf[0] < -0.5

    # Lag 2 pairs that would cross the gap (index 4 with index 6, and index 5
    # with index 7) must be excluded; only genuinely 2-apart valid pairs count.
    valid = ~np.isnan(values)
    expected_lag2_pairs = int((valid[:-2] & valid[2:]).sum())
    assert n_pairs[1] == expected_lag2_pairs


def test_gap_aware_acf_matches_full_series_pearson_when_no_gaps():
    rng = np.random.default_rng(2)
    n = 200
    e = rng.standard_normal(n)
    # Build a mildly autocorrelated series (AR(1)-like) with no gaps.
    series = np.cumsum(e) * 0.0  # placeholder, overwritten below
    series = np.empty(n)
    series[0] = e[0]
    for t in range(1, n):
        series[t] = 0.6 * series[t - 1] + e[t]
    residuals = pd.Series(series)

    lags, acf, n_pairs = gap_aware_acf(residuals, max_lag=1)

    expected = np.corrcoef(series[:-1], series[1:])[0, 1]
    assert n_pairs[0] == n - 1
    assert np.isclose(acf[0], expected, atol=1e-10)
