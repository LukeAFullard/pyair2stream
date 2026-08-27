"""
Regression tests for docs/audit/02_numerical_integration.md.

The air2stream ODE is linear in Tw with a discharge-dependent decay rate B.
Explicit integrators (RK4/RK2/EUL) are only conditionally stable in B*dt (dt=1
day); a parameter set stable at calibration discharge can diverge -- silently,
with no NaN or warning -- at a different (e.g. scenario) discharge. These tests
check that:

1. The divergence guard (`check_numerical_divergence`) catches an RK4 blow-up
   under a scenario flow, while CRN (now the default integrator) does not.
2. CRN and the new EXP (exponential/integrating-factor) integrator agree to
   within 0.1 degC, both on a real dataset and on a deliberately stiff
   parameter set where RK4 diverges outright.
3. `stability_report` flags max(B) exceeding the RK4 stability limit (2.785)
   as a pre-flight screening heuristic.
4. The default integrator is now CRN (unconditionally stable), not RK4.
"""

import os
import unittest

import numpy as np

from pyair2stream.config import CommonData
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import (
    call_model, compute_B_series, stability_report, warn_on_stability,
    check_numerical_divergence, NumericalDivergenceError, STABILITY_LIMITS,
)

QUICKSTART_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "quickstart", "data", "calibration_data.csv",
)

# A version-8 parameter set with a negative a4 (rating-curve exponent), reproducing the
# audit's real-world failure mode: higher discharge shrinks theta**a4, which amplifies
# B = (a3 + a8*theta) / theta**a4 super-linearly in theta. Stable at flow x1.0, diverges
# under RK4 at flow x1.5 (see docs/audit/02_numerical_integration.md, "Real failure on a
# real fit").
RUN1_LIKE_PAR = [0.3, 0.8, 0.1, -0.317, 11.16, 8.09, 0.5, 1.15]

# A deliberately stiff parameter set (a3=3.0, a8=3.0, a4=0) from the audit report's "Note
# on CRN in the stiff regime", giving B roughly in [3.85, 8.45] -- comfortably past RK4's
# stability limit for the whole record, not just a scenario tail.
STIFF_PAR = [0.5, 0.7, 3.0, 0.0, 1.0, 1.0, 0.4, 3.0]


def _build_synthetic_data(mod_num, par, q_scale=1.0, n_days=400, Qmedia=8.0, version=8):
    n_tot = n_days + 365
    data = CommonData()
    data.n_tot = n_tot
    data.version = version
    data.mod_num = mod_num
    data.Qmedia = Qmedia
    data.Tice_cover = 0.0
    data.par = np.array(par, dtype=np.float64)
    data.par_best = data.par.copy()

    t = np.arange(n_tot) - 365
    data.Tair = 10.0 + 12.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
    data.Q = q_scale * (8.0 + 4.0 * np.sin(2 * np.pi * t / 365.0))
    data.tt = np.array([((i % 365) + 1) / 365.0 for i in range(n_tot)])
    data.date = np.zeros((n_tot, 3), dtype=np.int32)
    data.Twat_obs = np.full(n_tot, -999.0)
    data.Twat_mod = np.zeros(n_tot)
    return data


def _build_stiff_data(mod_num, par, n_days=400, Qmedia=10.0, theta_lo=0.283, theta_hi=1.483):
    n_tot = n_days + 365
    data = CommonData()
    data.n_tot = n_tot
    data.version = 8
    data.mod_num = mod_num
    data.Qmedia = Qmedia
    data.Tice_cover = 0.0
    data.par = np.array(par, dtype=np.float64)
    data.par_best = data.par.copy()

    t = np.arange(n_tot)
    data.Tair = 10.0 + 12.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
    theta = theta_lo + (theta_hi - theta_lo) * 0.5 * (1 + np.sin(2 * np.pi * t / 365.0))
    data.Q = theta * Qmedia
    data.tt = np.array([((i % 365) + 1) / 365.0 for i in range(n_tot)])
    data.date = np.zeros((n_tot, 3), dtype=np.int32)
    data.Twat_obs = np.full(n_tot, -999.0)
    data.Twat_mod = np.zeros(n_tot)
    return data


class TestNumericalStability(unittest.TestCase):
    def test_rk4_diverges_but_crn_does_not_on_scenario_flow(self):
        data_rk4 = _build_synthetic_data('RK4', RUN1_LIKE_PAR, q_scale=1.5)
        call_model(data_rk4)
        with self.assertRaises(NumericalDivergenceError):
            check_numerical_divergence(data_rk4, max_plausible_twat=60.0)

        data_crn = _build_synthetic_data('CRN', RUN1_LIKE_PAR, q_scale=1.5)
        call_model(data_crn)
        check_numerical_divergence(data_crn, max_plausible_twat=60.0)  # must not raise
        self.assertTrue(np.all(np.isfinite(data_crn.Twat_mod)))
        self.assertLess(np.max(data_crn.Twat_mod), 60.0)

    def test_rk4_is_stable_at_calibration_flow_for_the_same_parameters(self):
        # Same parameters, un-rescaled discharge: RK4 must NOT raise. This is the crux of
        # the audit finding -- stability at calibration flow says nothing about stability
        # at scenario flow.
        data = _build_synthetic_data('RK4', RUN1_LIKE_PAR, q_scale=1.0)
        call_model(data)
        check_numerical_divergence(data, max_plausible_twat=60.0)  # must not raise

    def test_stability_report_flags_rk4_exceeding_limit_on_scenario_flow(self):
        data = _build_synthetic_data('RK4', RUN1_LIKE_PAR, q_scale=1.5)
        report = stability_report(data)
        self.assertEqual(report['limit'], STABILITY_LIMITS['RK4'])
        self.assertGreater(report['max_B'], 2.785)
        self.assertGreater(report['n_exceeding'], 0)

        with self.assertRaises(NumericalDivergenceError):
            warn_on_stability(data, error_fraction=0.10)

    def test_stability_report_never_flags_crn_or_exp(self):
        for mod_num in ('CRN', 'EXP'):
            data = _build_synthetic_data(mod_num, RUN1_LIKE_PAR, q_scale=2.0)
            report = stability_report(data)
            self.assertEqual(report['limit'], np.inf)
            # Must not raise regardless of how large max_B is.
            warn_on_stability(data)

    def test_crn_and_exp_agree_on_stiff_parameter_set(self):
        data_rk4 = _build_stiff_data('RK4', STIFF_PAR)
        call_model(data_rk4)
        self.assertFalse(np.all(np.isfinite(data_rk4.Twat_mod)))  # RK4 must diverge here

        data_crn = _build_stiff_data('CRN', STIFF_PAR)
        call_model(data_crn)
        data_exp = _build_stiff_data('EXP', STIFF_PAR)
        call_model(data_exp)

        self.assertTrue(np.all(np.isfinite(data_crn.Twat_mod)))
        self.assertTrue(np.all(np.isfinite(data_exp.Twat_mod)))

        diff = np.abs(data_crn.Twat_mod[365:] - data_exp.Twat_mod[365:])
        self.assertLess(np.max(diff), 0.1)

    def test_crn_and_exp_agree_on_quickstart_dataset(self):
        cal_data = CommonData()
        cal_data.runmode = 'PSO'
        cal_data.version = 8
        cal_data._input_data_path_cal = QUICKSTART_CSV
        read_Tseries(cal_data, 'c')

        par = [1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1]

        def run(mod_num):
            d = CommonData()
            d.n_tot = cal_data.n_tot
            d.version = 8
            d.mod_num = mod_num
            d.Qmedia = cal_data.Qmedia
            d.Tice_cover = 0.0
            d.par = np.array(par, dtype=np.float64)
            d.par_best = d.par.copy()
            d.Tair = cal_data.Tair
            d.Q = cal_data.Q
            d.tt = cal_data.tt
            d.date = cal_data.date
            d.Twat_obs = cal_data.Twat_obs
            d.Twat_mod = np.zeros(d.n_tot)
            call_model(d)
            return d.Twat_mod

        crn = run('CRN')
        exp = run('EXP')
        self.assertTrue(np.all(np.isfinite(crn)))
        self.assertTrue(np.all(np.isfinite(exp)))
        self.assertLess(np.max(np.abs(crn - exp)), 0.1)

    def test_default_integrator_is_crn(self):
        import tempfile
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump({
                    'project_name': os.path.join(tmp, 'proj'),
                    'version': 8,
                    'run_mode': 'PSO',
                    'paths': {'output_dir': os.path.join(tmp, 'out')},
                }, f)
            data = read_calibration(config_path)
            self.assertEqual(data.mod_num, 'CRN')


if __name__ == '__main__':
    unittest.main()
