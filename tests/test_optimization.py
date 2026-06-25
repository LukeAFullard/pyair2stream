import unittest
import numpy as np
import os
import shutil
import pandas as pd

from pyair2stream.config import CommonData, PI
from pyair2stream.optimization import forward_mode, PSO_mode, LH_mode
from pyair2stream.main import forward

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.data = CommonData()
        self.data.n_tot = 365 + 10 # 375 total points
        self.data.date = np.zeros((self.data.n_tot, 3), dtype=np.int32)
        self.data.tt = np.zeros(self.data.n_tot, dtype=np.float64)
        self.data.Tair = np.zeros(self.data.n_tot, dtype=np.float64)
        self.data.Q = np.zeros(self.data.n_tot, dtype=np.float64)
        self.data.Twat_obs = np.full(self.data.n_tot, -999.0, dtype=np.float64)
        self.data.Twat_mod = np.zeros(self.data.n_tot, dtype=np.float64)
        self.data.Twat_obs_agg = np.full(self.data.n_tot, -999.0, dtype=np.float64)
        self.data.Twat_mod_agg = np.full(self.data.n_tot, -999.0, dtype=np.float64)

        # Add basic config
        self.data.version = 8
        self.data.mod_num = 'RK4'
        self.data.time_res = '1d'
        self.data.fun_obj = 'RMS'
        self.data.runmode = 'PSO'
        self.data.station = 'test_station'
        self.data.series = 'test_series'

        self.data.Qmedia = np.float64(10.0)
        self.data.Tice_cover = np.float64(0.0)

        # Populate basic inputs for model calculation
        for i in range(self.data.n_tot):
            self.data.tt[i] = np.float64(i / 365.0)
            self.data.Tair[i] = 15.0 + 10.0 * np.sin(2.0 * PI * self.data.tt[i])
            self.data.Q[i] = 10.0 + 5.0 * np.cos(2.0 * PI * self.data.tt[i])
            if i >= 365:
                self.data.Twat_obs[i] = 12.0 + 8.0 * np.sin(2.0 * PI * self.data.tt[i])

        # Required for aggregation
        self.data.n_dat = 10
        self.data.I_inf = np.zeros((10, 3), dtype=np.int32)
        self.data.I_pos = np.zeros(10, dtype=np.int32)
        n_inf = 0
        n_pos = 0
        for i in range(365, self.data.n_tot):
            self.data.I_inf[n_inf, 0] = n_pos
            self.data.I_inf[n_inf, 1] = n_pos
            self.data.I_inf[n_inf, 2] = i
            self.data.I_pos[n_pos] = i
            self.data.Twat_obs_agg[i] = self.data.Twat_obs[i]
            n_inf += 1
            n_pos += 1

        # Par ranges and config
        self.data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)
        self.data.parmin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.parmax = np.array([2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float64)
        self.data.flag_par = np.ones(8, dtype=np.bool_)

        self.data.c1 = 2.0
        self.data.c2 = 2.0
        self.data.wmin = 0.4
        self.data.wmax = 0.9

        self.data.mineff_index = -1e30 # Accept any fit

        self.data.folder = 'test_opt_output'
        os.makedirs(self.data.folder, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.data.folder):
            shutil.rmtree(self.data.folder)

    def test_forward_mode(self):
        forward_mode(self.data)
        self.assertIsNotNone(self.data.finalfit)
        np.testing.assert_array_equal(self.data.par_best, self.data.par)

    def test_PSO_mode(self):
        self.data.n_particles = 10
        self.data.n_run = 5
        self.data.runmode = 'PSO'

        PSO_mode(self.data, seed=42)

        self.assertIsNotNone(self.data.finalfit)
        self.assertEqual(len(self.data.par_best), 8)

        expected_csv = os.path.join(self.data.folder, f"0_PSO_RMS_test_station_test_series_1d.csv")
        self.assertTrue(os.path.exists(expected_csv))

        df = pd.read_csv(expected_csv)
        self.assertTrue(len(df) > 0)
        self.assertTrue("eff_index" in df.columns)

    def test_LH_mode(self):
        self.data.n_run = 5
        self.data.runmode = 'LATHYP'

        LH_mode(self.data, seed=42)

        self.assertIsNotNone(self.data.finalfit)
        self.assertEqual(len(self.data.par_best), 8)

        expected_csv = os.path.join(self.data.folder, f"0_LATHYP_RMS_test_station_test_series_1d.csv")
        self.assertTrue(os.path.exists(expected_csv))

    def test_DE_mode(self):
        from pyair2stream.optimization import DE_mode
        self.data.n_particles = 3
        self.data.n_run = 2
        self.data.runmode = 'DE'

        DE_mode(self.data, seed=42)

        self.assertIsNotNone(self.data.finalfit)
        self.assertEqual(len(self.data.par_best), 8)

        expected_csv = os.path.join(self.data.folder, f"0_DE_RMS_test_station_test_series_1d.csv")
        self.assertTrue(os.path.exists(expected_csv))

        df = pd.read_csv(expected_csv)
        self.assertTrue(len(df) > 0)
        self.assertTrue("eff_index" in df.columns)

    def test_forward_routine(self):
        self.data.n_particles = 2
        self.data.n_run = 1
        self.data.runmode = 'PSO'
        PSO_mode(self.data, seed=42)
        forward(self.data)

        expected_csv = os.path.join(self.data.folder, f"2_PSO_RMS_test_station_test_seriesc_1d.csv")
        self.assertTrue(os.path.exists(expected_csv))
        df = pd.read_csv(expected_csv)
        # We initialized n_tot = 375 originally in setUp.
        # But forward(self.data) resets n_tot=0 because validation file is skipped.
        # We check the length of the CSV written BEFORE validation was attempted.
        self.assertEqual(len(df), 375)

if __name__ == '__main__':
    unittest.main()
