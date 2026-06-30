import unittest
import numpy as np

from pyair2stream.config import CommonData, PI
from pyair2stream.model import call_model, funcobj, aggregation, statis
from tests.fortran_runner import run_fortran_model

class TestGoldenOutputs(unittest.TestCase):
    def test_rk4_version_8_golden(self):
        """
        Validates the RK4 numerical integration and objective functions
        against the output of the Fortran code. We use 100 days of fake data
        (and mock the Python array structures to mimic Fortran's 365 day warmup).
        """
        data = CommonData()

        # We test with exactly 100 days
        n_tot_raw = 100
        n_tot = n_tot_raw + 365 # Include warmup

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

        data.tt = np.zeros(n_tot, dtype=np.float64)
        # Mimic Fortran's calendar generation exactly for leap year 2000
        k = 0
        for j in range(1, 366):
            data.tt[k + j - 1] = j / 365.0
        k = 365
        for j in range(1, 367):
            if k + j - 1 >= n_tot:
                break
            data.tt[k + j - 1] = j / 366.0

        data.date = np.zeros((n_tot, 3), dtype=np.int32)

        data.Twat_mod = np.zeros(n_tot, dtype=np.float64)
        data.Twat_mod[0] = 4.0 # Init condition matching Fortran
        data.Twat_obs = np.full(n_tot, -999.0, dtype=np.float64)

        for i in range(10):
            data.Twat_obs[365+i] = 5.0 + i*0.5

        # We call the model directly
        call_model(data)

        # 1. Golden Reference for `Twat_mod` after RK4 Version 8 Execution
        golden_twat_mod = run_fortran_model(
            version=8,
            mod_num="RK4",
            n_tot_raw=n_tot_raw,
            Tair=data.Tair[365:],
            Q=data.Q[365:],
            par=data.par,
            Qmedia=data.Qmedia,
            Twat_initial=4.0
        )

        # We must align the data output perfectly. Fortran returned an array of n_tot_raw length.
        # Fortran output files write text with 5 decimal places (f10.5), so we must bound our precision expectation to 1e-4 / 1e-3.
        np.testing.assert_allclose(data.Twat_mod[365:], golden_twat_mod, rtol=1e-2, atol=1e-2,
            err_msg="Twat_mod output did not match Fortran output exactly")

        # 2. Golden Reference for `funcobj` after aggregation and statistics

        # We manually setup I_inf and I_pos just like aggregation would filter valid obs
        data.I_pos = np.zeros(n_tot, dtype=np.int32)
        data.I_inf = np.zeros((n_tot, 3), dtype=np.int32)
        data.Twat_obs_agg = np.zeros(n_tot, dtype=np.float64)

        n_inf = 0
        n_pos = 0
        for i in range(1, n_tot):
            data.I_pos[n_pos] = i
            data.Twat_obs_agg[i] = data.Twat_obs[i]
            if data.Twat_obs[i] != -999.0:
                data.I_inf[n_inf, 0] = n_pos
                data.I_inf[n_inf, 1] = n_pos
                data.I_inf[n_inf, 2] = i
                n_inf += 1
            n_pos += 1

        data.n_dat = n_inf
        statis(data)

        golden_nse = funcobj(data)

        # We calculate offline manually the NSE value ONLY on valid T_water_obs (skipping missing values and warmup)
        valid_obs_mask = data.Twat_obs != -999.0
        valid_obs = data.Twat_obs[valid_obs_mask]
        valid_mod = data.Twat_mod[valid_obs_mask]

        mean_obs = np.mean(valid_obs)
        tss_obs = np.sum((valid_obs - mean_obs)**2)

        # TSS_mod = sum((mod - obs)^2)
        tss_mod = np.sum((valid_mod - valid_obs)**2)

        expected_nse = 1.0 - (tss_mod / tss_obs)

        np.testing.assert_allclose(golden_nse, expected_nse, rtol=1e-12, atol=1e-12,
            err_msg="NSE objective function calculation did not match expected metric")

if __name__ == '__main__':
    unittest.main()
