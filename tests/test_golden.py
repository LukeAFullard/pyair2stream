import unittest
import numpy as np

from pyair2stream.config import CommonData, PI
from pyair2stream.model import call_model, funcobj, aggregation, statis

class TestGoldenOutputs(unittest.TestCase):
    def test_rk4_version_8_golden(self):
        """
        Validates the RK4 numerical integration and objective functions
        against a set of manually verified golden outputs ensuring strict reproducibility.
        """
        data = CommonData()

        # We test with exactly 10 days
        n_tot = 10
        data.n_tot = n_tot
        data.n_dat = n_tot - 1

        data.version = 8
        data.mod_num = 'RK4'
        data.time_res = '1d'
        data.fun_obj = 'NSE'
        data.Qmedia = np.float64(10.0)
        data.Tice_cover = np.float64(0.0)
        data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)

        data.Tair = np.full(n_tot, 15.0, dtype=np.float64)
        data.Q = np.full(n_tot, 10.0, dtype=np.float64)
        data.tt = np.array([i/365.0 for i in range(n_tot)], dtype=np.float64)
        data.date = np.zeros((n_tot, 3), dtype=np.int32)

        data.Twat_mod = np.zeros(n_tot, dtype=np.float64)
        data.Twat_mod[0] = 4.0 # Init condition matching Fortran

        # Predict Twat_obs to calculate meaningful objective values later
        # Making up an observed set slightly offset from expected golden outputs
        data.Twat_obs = np.array([
            -999.0, # Day 1 is usually part of warm-up in aggregation logic
            5.6, 6.9, 7.9, 8.8, 9.4, 10.0, 10.5, 10.9, 11.2
        ], dtype=np.float64)

        # We call the model directly
        call_model(data)

        # 1. Golden Reference for `Twat_mod` after RK4 Version 8 Execution
        golden_twat_mod = np.array([
            4.0,
            5.540813708131667,
            6.802602297834198,
            7.836212193519514,
            8.683272897271095,
            9.377867630125097,
            9.94790114184635,
            10.416219582480279,
            10.801527378639673,
            11.119137910824824
        ], dtype=np.float64)

        np.testing.assert_allclose(data.Twat_mod, golden_twat_mod, rtol=1e-12, atol=1e-12,
            err_msg="Twat_mod output did not match golden output exactly")

        # 2. Golden Reference for `funcobj` after aggregation and statistics

        # We manually setup I_inf and I_pos just like aggregation would
        data.I_pos = np.zeros(n_tot, dtype=np.int32)
        data.I_inf = np.zeros((n_tot - 1, 3), dtype=np.int32)
        data.Twat_obs_agg = np.zeros(n_tot, dtype=np.float64)

        n_inf = 0
        n_pos = 0
        for i in range(1, n_tot):
            data.I_inf[n_inf, 0] = n_pos
            data.I_inf[n_inf, 1] = n_pos
            data.I_inf[n_inf, 2] = i
            data.I_pos[n_pos] = i
            data.Twat_obs_agg[i] = data.Twat_obs[i]
            n_inf += 1
            n_pos += 1

        data.n_dat = n_inf
        statis(data)

        golden_nse = funcobj(data)

        # We calculate offline manually the NSE value:
        # TSS_obs = sum((obs - mean_obs)^2)
        mean_obs = np.mean(data.Twat_obs[1:])
        tss_obs = np.sum((data.Twat_obs[1:] - mean_obs)**2)

        # TSS_mod = sum((mod - obs)^2)
        tss_mod = np.sum((data.Twat_mod[1:] - data.Twat_obs[1:])**2)

        expected_nse = 1.0 - (tss_mod / tss_obs)

        np.testing.assert_allclose(golden_nse, expected_nse, rtol=1e-12, atol=1e-12,
            err_msg="NSE objective function calculation did not match golden output")

if __name__ == '__main__':
    unittest.main()
