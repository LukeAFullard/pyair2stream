import unittest
import os
import shutil
import numpy as np
import pandas as pd

from pyair2stream.config import CommonData, PI
from pyair2stream.optimization import PSO_mode
from pyair2stream.main import forward
from pyair2stream.post_processing import post_process

class TestPostProcessing(unittest.TestCase):
    def setUp(self):
        self.data = CommonData()
        self.data.n_tot = 380  # Just past 1 year to test subsetting
        self.data.date = np.zeros((self.data.n_tot, 3), dtype=np.int32)
        self.data.tt = np.zeros(self.data.n_tot, dtype=np.float64)
        self.data.Tair = np.zeros(self.data.n_tot, dtype=np.float64)
        self.data.Q = np.zeros(self.data.n_tot, dtype=np.float64)
        self.data.Twat_obs = np.full(self.data.n_tot, -999.0, dtype=np.float64)
        self.data.Twat_mod = np.zeros(self.data.n_tot, dtype=np.float64)
        self.data.Twat_obs_agg = np.full(self.data.n_tot, -999.0, dtype=np.float64)
        self.data.Twat_mod_agg = np.full(self.data.n_tot, -999.0, dtype=np.float64)

        self.data.version = 8
        self.data.mod_num = 'RK4'
        self.data.time_res = '1d'
        self.data.fun_obj = 'RMS'
        self.data.runmode = 'PSO'
        self.data.station = 'test_station'
        self.data.series = 'test_series'
        self.data.folder = 'test_pp_output'
        os.makedirs(self.data.folder, exist_ok=True)

        self.data.Qmedia = np.float64(10.0)
        self.data.Tice_cover = np.float64(0.0)

        dates = pd.date_range(start="2020-01-01", periods=self.data.n_tot, freq='D')
        for i in range(self.data.n_tot):
            self.data.date[i, 0] = dates.year[i]
            self.data.date[i, 1] = dates.month[i]
            self.data.date[i, 2] = dates.day[i]
            self.data.tt[i] = np.float64(i / 365.0)
            self.data.Tair[i] = 15.0 + 10.0 * np.sin(2.0 * PI * self.data.tt[i])
            self.data.Q[i] = 10.0 + 5.0 * np.cos(2.0 * PI * self.data.tt[i])
            if i >= 365:
                self.data.Twat_obs[i] = 12.0 + 8.0 * np.sin(2.0 * PI * self.data.tt[i])

        self.data.n_dat = self.data.n_tot - 365
        self.data.I_inf = np.zeros((self.data.n_dat, 3), dtype=np.int32)
        self.data.I_pos = np.zeros(self.data.n_dat, dtype=np.int32)
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

        self.data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)
        self.data.parmin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.parmax = np.array([2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float64)
        self.data.flag_par = np.ones(8, dtype=np.bool_)
        self.data.c1 = 2.0
        self.data.c2 = 2.0
        self.data.wmin = 0.4
        self.data.wmax = 0.9
        self.data.mineff_index = -1e30

    def tearDown(self):
        if os.path.exists(self.data.folder):
            shutil.rmtree(self.data.folder)

    def test_post_processing_output(self):
        # 1. Run minimal model to generate output files
        self.data.n_particles = 2
        self.data.n_run = 1
        PSO_mode(self.data, seed=42)
        forward(self.data)

        # We manually reset n_tot for test purposes because forward skipped validation and reset n_tot
        self.data.n_tot = 380

        # 2. Call the post_process function
        post_process(self.data, toll=2.0)

        # 3. Assert plot files were created
        expected_files = [
            f"dottyplots_{self.data.runmode}_{self.data.fun_obj}_{self.data.station}.pdf",
            f"dottyplots_{self.data.runmode}_{self.data.fun_obj}_{self.data.station}.png",
            f"calibration_{self.data.runmode}_{self.data.fun_obj}_{self.data.station}.pdf",
            f"calibration_{self.data.runmode}_{self.data.fun_obj}_{self.data.station}.png"
        ]

        for file in expected_files:
            file_path = os.path.join(self.data.folder, file)
            self.assertTrue(os.path.exists(file_path), f"Expected file {file} was not generated.")

if __name__ == '__main__':
    unittest.main()
