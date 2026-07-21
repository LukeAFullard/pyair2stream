import pytest
import numpy as np
import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from pyair2stream.optimization import DE_MCMC_mode, DE_CV_MCMC_mode

def test_mcmc_autocorr_invalid_json():
    # Setup mock data for DE_MCMC_mode
    data = MagicMock()
    data.n_tot = 3
    data.segments = [(0, 2)]
    data.mcmc_walkers = 20
    data.mcmc_steps = 10
    data.flag_par = [True] * 8
    data.parmin = np.zeros(8)
    data.parmax = np.ones(8)
    data.par = np.zeros(8)
    data.par_best = np.zeros(8)
    data.Twat_obs = np.array([1, 2, 3])
    data.Twat_mod = np.array([1, 2, 3])
    data.eval_mask = None
    data.gap_tolerant = False
    data.uncertainty_options = {}
    data.date = np.array([[2000, 1, 1], [2000, 1, 2], [2000, 1, 3]])
    data.station = "test"
    data.series = "test"
    data.time_res = "daily"
    data.mod_num = "RK4"
    data.model = "version_7"
    data.fun_obj = "NSE"
    data.Tair = np.zeros(3)
    data.Q = np.zeros(3)
    data.theta_a = 0
    data.a1 = 0
    data.Tice_cover = 0
    data.Tair_mean = 0
    data.Qmedia = 0
    data.dt = 1.0
    data.missing_data_sentinel = -999.0

    with tempfile.TemporaryDirectory() as temp_dir:
        data.folder = temp_dir

        class MockSampler:
            def __init__(self, *args, **kwargs):
                self.acceptance_fraction = np.array([0.5])
            def run_mcmc(self, *args, **kwargs):
                pass
            def get_autocorr_time(self, quiet=False):
                return np.array([np.nan] * 8)
            def get_chain(self, discard=0, flat=False):
                return np.zeros((10, 8))

        with patch('emcee.EnsembleSampler', MockSampler), \
             patch('pyair2stream.optimization.DE_mode'), \
             patch('pyair2stream.optimization.call_model'), \
             patch('pyair2stream.optimization.funcobj', return_value=1.0), \
             patch('pyair2stream.uncertainty.generate_ar1_noise', return_value=np.zeros(3)):

            DE_MCMC_mode(data)

            # Check json
            json_file = os.path.join(temp_dir, f"MCMC_chain_test_test_daily_meta.json")
            with open(json_file, 'r') as f:
                content = json.load(f)

            assert content['mean_autocorr_time'] is None

def test_mcmc_ndim_zero_de_runs():
    data = MagicMock()
    data.n_tot = 3
    data.segments = [(0, 2)]
    data.mcmc_walkers = 10
    data.mcmc_steps = 10
    # Set no active params
    data.flag_par = [True] * 8
    data.parmin = np.zeros(8)
    data.parmax = np.zeros(8)

    with patch('pyair2stream.optimization.DE_mode') as mock_de:
        DE_MCMC_mode(data)
        mock_de.assert_called_once()

        mock_de.reset_mock()
        DE_CV_MCMC_mode(data)
        mock_de.assert_called_once()
