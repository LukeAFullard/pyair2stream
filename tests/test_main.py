import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import pandas as pd
import numpy as np

from pyair2stream.main import main, forward, run_optimizer
from pyair2stream.config import CommonData

class TestMain(unittest.TestCase):
    @patch('pyair2stream.main.read_calibration')
    @patch('pyair2stream.main.read_Tseries')
    @patch('pyair2stream.main.aggregation')
    @patch('pyair2stream.main.statis')
    @patch('pyair2stream.main.forward_mode')
    @patch('pyair2stream.main.post_process')
    @patch('pyair2stream.main.sensitivity_analysis')
    @patch('sys.argv', ['main.py', '--config', 'dummy.yaml'])
    def test_main_orchestration(self, mock_sens, mock_post, mock_fwd_mode, mock_statis, mock_agg, mock_read_ts, mock_read_cal):
        # Setup mock data
        data = CommonData()
        data.runmode = 'FORWARD'
        data.sensitivity_analysis = True
        # to avoid forward crashing
        data.n_tot = 100
        data.par_best = np.array([1.0]*8)
        data.par = np.array([1.0]*8)
        data.Tair = np.ones(100)
        data.Q = np.ones(100)
        data.Twat_obs = np.ones(100)
        data.Twat_mod = np.ones(100)
        data.Twat_obs_agg = np.ones(100)
        data.Twat_mod_agg = np.ones(100)
        data.date = np.ones((100,3), dtype=np.int32)
        data.gap_tolerant = False
        data.finalfit = 1.0
        data.folder = tempfile.mkdtemp()
        data.fun_obj = 'NSE'
        data.station = 'test'
        data.series = 'c'
        data.time_res = '1d'

        mock_read_cal.return_value = data

        # Patch forward so it doesn't crash trying to do real math
        with patch('pyair2stream.main.forward') as mock_fwd:
            main()

            # Assert correct orchestration sequence
            mock_read_cal.assert_called_once_with(config_file='dummy.yaml')
            mock_read_ts.assert_called_once_with(data, 'c')
            # FORWARD mode does not calibrate and may have no observations at all
            # (a pure projection); main() must not call aggregation()/statis()
            # unconditionally -- forward_mode() (mocked here) handles both cases
            # itself via its own has_obs check (report 05, Defect A).
            mock_agg.assert_not_called()
            mock_statis.assert_not_called()
            mock_fwd_mode.assert_called_once_with(data)
            mock_fwd.assert_called_once_with(data)
            mock_post.assert_called_once_with(data)
            mock_sens.assert_called_once_with(data)

    @patch('pyair2stream.main.forward_mode')
    @patch('pyair2stream.main.PSO_mode')
    @patch('pyair2stream.main.LH_mode')
    @patch('pyair2stream.main.DE_mode')
    @patch('pyair2stream.main.DE_MCMC_mode')
    def test_run_optimizer_dispatch(self, mock_de_mcmc, mock_de, mock_lh, mock_pso, mock_fwd):
        data = CommonData()

        # `run_optimizer` threads `data.random_seed` through to every optimizer
        # entry point (docs/audit/07_reproducibility_and_provenance.md, 7.1).
        data.random_seed = 42

        data.runmode = 'FORWARD'
        run_optimizer(data)
        mock_fwd.assert_called_once_with(data)

        data.runmode = 'PSO'
        run_optimizer(data)
        mock_pso.assert_called_once_with(data, seed=42)

        data.runmode = 'LATHYP'
        run_optimizer(data)
        mock_lh.assert_called_once_with(data, seed=42)

        data.runmode = 'DE'
        run_optimizer(data)
        mock_de.assert_called_once_with(data, seed=42)

        data.runmode = 'DE-MCMC'
        run_optimizer(data)
        mock_de_mcmc.assert_called_once_with(data, seed=42)

    @patch('pyair2stream.main.call_model')
    @patch('pyair2stream.main.funcobj')
    @patch('pyair2stream.main.read_Tseries')
    @patch('pyair2stream.main.aggregation')
    @patch('pyair2stream.main.statis')
    def test_forward_gap_tolerant(self, mock_statis, mock_agg, mock_read_ts, mock_funcobj, mock_call_model):
        data = CommonData()
        data.folder = tempfile.mkdtemp()
        data.gap_tolerant = True
        data.segments = [(0, 49), (60, 99)]
        data.runmode = 'PSO'
        data.fun_obj = 'NSE'
        data.station = 'test'
        data.series = 'c'
        data.time_res = '1d'
        data.n_tot = 100
        data.par = np.array([1.0]*8)
        data.par_best = np.array([1.0]*8)
        data.finalfit = 0.95
        data.Qmedia = 10.0
        data.Qmedia_user = None
        data.n_dat = 50
        data.date = np.ones((100, 3), dtype=np.int32)
        data.Tair = np.ones(100)
        data.Tair[50:60] = -999.0
        data.Q = np.ones(100)
        data.Twat_obs = np.ones(100)
        data.Twat_mod = np.ones(100)
        data.Twat_obs_agg = np.ones(100)
        data.Twat_mod_agg = np.ones(100)

        mock_funcobj.return_value = 0.95

        forward(data)

        mock_call_model.assert_called()
        self.assertTrue(os.path.exists(os.path.join(data.folder, "gaps_summary.txt")))
        self.assertTrue(os.path.exists(os.path.join(data.folder, "2_PSO_NSE_test_cc_1d.csv")))

    @patch('pyair2stream.main.call_model')
    @patch('pyair2stream.main.funcobj')
    @patch('pyair2stream.main.read_Tseries')
    @patch('pyair2stream.main.aggregation')
    @patch('pyair2stream.main.statis')
    def test_forward_gap_intolerant(self, mock_statis, mock_agg, mock_read_ts, mock_funcobj, mock_call_model):
        data = CommonData()
        data.folder = tempfile.mkdtemp()
        data.gap_tolerant = False
        data.runmode = 'PSO'
        data.fun_obj = 'NSE'
        data.station = 'test'
        data.series = 'c'
        data.time_res = '1d'
        data.n_tot = 100
        data.par = np.array([1.0]*8)
        data.par_best = np.array([1.0]*8)
        data.finalfit = 0.95
        data.Qmedia = 10.0
        data.Qmedia_user = None
        data.n_dat = 50
        data.date = np.ones((100, 3), dtype=np.int32)
        data.Tair = np.ones(100)
        data.Q = np.ones(100)
        data.Twat_obs = np.ones(100)
        data.Twat_mod = np.ones(100)
        data.Twat_obs_agg = np.ones(100)
        data.Twat_mod_agg = np.ones(100)

        mock_funcobj.return_value = 0.95

        forward(data)

        mock_call_model.assert_called()
        self.assertFalse(os.path.exists(os.path.join(data.folder, "gaps_summary.txt")))
        self.assertTrue(os.path.exists(os.path.join(data.folder, "2_PSO_NSE_test_cc_1d.csv")))

    @patch('pyair2stream.main.read_calibration')
    @patch('pyair2stream.main.read_Tseries')
    @patch('pyair2stream.main.aggregation')
    @patch('pyair2stream.main.statis')
    @patch('pyair2stream.cross_validation.cross_validate')
    @patch('sys.argv', ['main.py', '--config', 'dummy.yaml'])
    def test_main_cross_validation(self, mock_cross_validate, mock_statis, mock_agg, mock_read_ts, mock_read_cal):
        data = CommonData()
        data.runmode = "DE"
        data.cross_validation = "loyo"
        data.folder = tempfile.mkdtemp()
        data.mean_obs = 10.0
        data.TSS_obs = 100.0
        data.std_obs = 2.0

        mock_read_cal.return_value = data
        mock_df = pd.DataFrame({'fold': [1], 'NSE': [0.9]})
        # One held-out year with a known error: +0.5 °C in July, -0.2 °C otherwise.
        from pyair2stream.cross_validation import FoldResult
        days = pd.date_range("2011-01-01", "2011-12-31")
        obs = np.full(len(days), 10.0)
        sim = obs + np.where(days.month == 7, 0.5, -0.2)
        fold = FoldResult(fold_id=0, label="2011", held_out_start=days[0], held_out_end=days[-1],
                          n_obs_held_out=len(days), par_best=np.zeros(8), nse=0.9, kge=0.9, rmse=0.3,
                          obs_held_out=obs, sim_held_out=sim, dates_held_out=days)
        mock_cross_validate.return_value = (mock_df, [fold])

        main()

        # Non-FORWARD run modes calibrate and must still get aggregation()/statis()
        # from main() before dispatching (report 05, Defect A only concerns FORWARD).
        mock_agg.assert_called_once_with(data)
        mock_statis.assert_called_once_with(data)
        mock_cross_validate.assert_called_once_with(data, data.runmode, return_folds=True)
        self.assertTrue(os.path.exists(os.path.join(data.folder, "cv_results.csv")))
        bias = pd.read_csv(os.path.join(data.folder, "cv_bias_by_month.csv")).set_index("period")
        self.assertAlmostEqual(bias.loc["Jul", "bias"], 0.5)
        self.assertAlmostEqual(bias.loc["Jan", "bias"], -0.2)
        self.assertTrue(os.path.exists(os.path.join(data.folder, "cv_bias_by_month.png")))

if __name__ == '__main__':
    unittest.main()
