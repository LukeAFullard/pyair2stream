import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import pandas as pd
import numpy as np

from pyair2stream.main import main, forward
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

if __name__ == '__main__':
    unittest.main()
