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
            mock_agg.assert_called_once_with(data)
            mock_statis.assert_called_once_with(data)
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

        data.runmode = 'FORWARD'
        run_optimizer(data)
        mock_fwd.assert_called_once_with(data)

        data.runmode = 'PSO'
        run_optimizer(data)
        mock_pso.assert_called_once_with(data)

        data.runmode = 'LATHYP'
        run_optimizer(data)
        mock_lh.assert_called_once_with(data)

        data.runmode = 'DE'
        run_optimizer(data)
        mock_de.assert_called_once_with(data)

        data.runmode = 'DE-MCMC'
        run_optimizer(data)
        mock_de_mcmc.assert_called_once_with(data)

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
    @patch('pyair2stream.cross_validation.run_leave_one_year_out_cv')
    @patch('pyair2stream.cross_validation.summarize')
    @patch('sys.argv', ['main.py', '--config', 'dummy.yaml'])
    def test_main_cross_validation(self, mock_summarize, mock_run_cv, mock_statis, mock_agg, mock_read_ts, mock_read_cal):
        data = CommonData()
        data.runmode = "DE"
        data.cross_validation = "loyo"
        data.folder = tempfile.mkdtemp()
        data.mean_obs = 10.0
        data.TSS_obs = 100.0
        data.std_obs = 2.0

        mock_read_cal.return_value = data
        mock_df = pd.DataFrame({'fold': [1], 'NSE': [0.9]})
        mock_summarize.return_value = mock_df

        main()

        mock_run_cv.assert_called_once_with(data, "loyo", data.runmode)
        mock_summarize.assert_called_once()
        self.assertTrue(os.path.exists(os.path.join(data.folder, "cv_results.csv")))

if __name__ == '__main__':
    unittest.main()
