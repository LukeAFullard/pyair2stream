import unittest
import numpy as np
from pyair2stream.config import CommonData, PI, TTT
from pyair2stream.model import call_model, aggregation, statis, funcobj

class TestModel(unittest.TestCase):
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
        # Adjust par so that DD^p[4] doesn't underflow/overflow K
        self.data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)
        self.data.Tice_cover = np.float64(0.0)
        self.data.Qmedia = np.float64(10.0)
        self.data.time_res = '1d'

        # Populate basic inputs
        for i in range(self.data.n_tot):
            self.data.tt[i] = np.float64(i / 365.0)
            self.data.Tair[i] = 15.0 + 10.0 * np.sin(2.0 * PI * self.data.tt[i])
            self.data.Q[i] = 10.0 + 5.0 * np.cos(2.0 * PI * self.data.tt[i])
            if i >= 365:
                self.data.Twat_obs[i] = 12.0 + 8.0 * np.sin(2.0 * PI * self.data.tt[i])

    def test_integrators(self):
        # We test RK4 integrator specifically
        self.data.version = 8
        self.data.mod_num = 'RK4'
        call_model(self.data)

        # Ensure it advanced and updated Twat_mod
        self.assertNotEqual(self.data.Twat_mod[1], 0.0)
        # Check initial condition since obs[0] is -999
        self.assertEqual(self.data.Twat_mod[0], 4.0)

        # Basic sanity check on RK2
        data_rk2 = CommonData()
        data_rk2.n_tot = self.data.n_tot
        data_rk2.Tair = self.data.Tair.copy()
        data_rk2.Q = self.data.Q.copy()
        data_rk2.tt = self.data.tt.copy()
        data_rk2.Twat_obs = self.data.Twat_obs.copy()
        data_rk2.Twat_mod = np.zeros(self.data.n_tot, dtype=np.float64)
        data_rk2.par = self.data.par.copy()
        data_rk2.Qmedia = self.data.Qmedia
        data_rk2.Tice_cover = self.data.Tice_cover
        data_rk2.version = 8
        data_rk2.mod_num = 'RK2'

        call_model(data_rk2)
        self.assertNotEqual(data_rk2.Twat_mod[-1], 0.0)
        # RK4 and RK2 should produce slightly different results
        self.assertNotAlmostEqual(self.data.Twat_mod[-1], data_rk2.Twat_mod[-1], places=3)

    def test_aggregation_statis_funcobj(self):
        # We need to set some Twat_mod first to have meaningful data
        self.data.version = 8
        self.data.mod_num = 'RK4'
        call_model(self.data)

        aggregation(self.data)

        # With n_tot=375 and time_res='1d', we should have 10 data points
        self.assertEqual(self.data.n_dat, 10)
        self.assertEqual(len(self.data.I_inf), 10)
        self.assertEqual(len(self.data.I_pos), 10)

        statis(self.data)
        self.assertNotEqual(self.data.mean_obs, 0.0)
        self.assertNotEqual(self.data.TSS_obs, 0.0)
        self.assertNotEqual(self.data.std_obs, 0.0)

        self.data.fun_obj = 'NSE'
        nse = funcobj(self.data)
        self.assertIsInstance(nse, np.float64)

        self.data.fun_obj = 'KGE'
        kge = funcobj(self.data)
        self.assertIsInstance(kge, np.float64)

        self.data.fun_obj = 'RMS'
        rms = funcobj(self.data)
        self.assertIsInstance(rms, np.float64)
        self.assertTrue(rms <= 0.0) # RMS in this code is strictly negative

if __name__ == '__main__':
    unittest.main()
