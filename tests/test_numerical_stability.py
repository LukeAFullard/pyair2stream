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
5. `step_amplification` is exactly what one step of each integrator does to a
   difference between two simulations, and `largest_growth` (the most a
   difference can grow over a stretch of days) stops an explicit-integrator run
   above `stability_max_growth`, even when B exceeds the limit on few days.
"""

import os
import unittest

import numpy as np

from pyair2stream.config import CommonData
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import (
    call_model, compute_B_series, stability_report, warn_on_stability,
    check_numerical_divergence, NumericalDivergenceError, STABILITY_LIMITS,
    step_amplification, largest_growth, STABILITY_MAX_GROWTH,
)

MENTUE_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "switzerland", "MAH_2369_calibration.csv",
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


def _build_theta_data(mod_num, par, theta, version=7, Qmedia=10.0):
    """`n_tot` days with the scaled discharge `theta` (warm-up included), no measurements."""
    n_tot = len(theta)
    data = CommonData()
    data.n_tot = n_tot
    data.version = version
    data.mod_num = mod_num
    data.Qmedia = Qmedia
    data.Tice_cover = 0.0
    data.par = np.array(par, dtype=np.float64)
    data.par_best = data.par.copy()
    t = np.arange(n_tot)
    data.Tair = 10.0 + 12.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
    data.Q = np.asarray(theta, dtype=np.float64) * Qmedia
    data.tt = np.array([((i % 365) + 1) / 365.0 for i in range(n_tot)])
    data.date = np.zeros((n_tot, 3), dtype=np.int32)
    data.Twat_obs = np.full(n_tot, -999.0)
    data.Twat_mod = np.zeros(n_tot)
    return data


# Version 7 with B = a3 + a8*theta: B = 1.0 at theta = 1, and B = 3.5 (above every explicit
# integrator's limit) at theta = 6.
FLOOD_PAR = [0.5, 0.7, 0.5, 0.0, 1.0, 1.0, 0.4, 0.5]


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

    def test_crn_and_exp_agree_on_real_forcing(self):
        cal_data = CommonData()
        cal_data.runmode = 'PSO'
        cal_data.version = 8
        cal_data._input_data_path_cal = MENTUE_CSV
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

    def test_step_amplification_is_what_each_integrator_does(self):
        # With no forcing (a1 = a2 = a5 = a6 = 0) and no ice floor, every step multiplies the
        # water temperature by exactly R_j, so the integrators themselves check the formulas,
        # including which day's B each one uses and RK4's midpoint discharge.
        par = list(RUN1_LIKE_PAR)
        par[0] = par[1] = par[4] = par[5] = 0.0
        for version in (5, 8):
            for mod_num in ('CRN', 'EXP', 'RK4', 'RK2', 'EUL'):
                data = _build_synthetic_data(mod_num, par if version == 8 else [0, 0, 2.2, 0, 0, 0, 0, 0],
                                             q_scale=1.5, version=version)
                data.Tice_cover = -1e300
                call_model(data)
                tw = data.Twat_mod
                ok = (np.abs(tw[:-1]) > 1e-200) & (np.abs(tw[:-1]) < 1e200) & np.isfinite(tw[1:])
                self.assertGreater(int(ok.sum()), 50, (version, mod_num))
                R = step_amplification(data)
                np.testing.assert_allclose(tw[1:][ok] / tw[:-1][ok], R[ok], rtol=1e-12, atol=1e-14,
                                           err_msg=f"version {version}, {mod_num}")

    def test_largest_growth_with_constant_B(self):
        n = 400 + 365
        theta = np.ones(n)
        for mod_num, B, factor in (('RK4', 3.0, 1.375), ('EUL', 3.0, 2.0), ('RK2', 3.0, 2.5),
                                   ('RK4', 1.0, 1.0), ('CRN', 3.0, 1.0), ('EXP', 3.0, 1.0)):
            data = _build_theta_data(mod_num, [0.5, 0.7, B, 0, 0, 0, 0, 0], theta, version=3)
            growth = largest_growth(data)
            self.assertAlmostEqual(growth['log10_growth'], (n - 1) * np.log10(factor), places=6,
                                   msg=f"{mod_num}, B={B}")
            if factor > 1:
                self.assertEqual((growth['start'], growth['end']), (0, n - 1))

    def test_growth_stops_a_short_flood_that_the_share_of_days_lets_through(self):
        theta = np.ones(400 + 365)
        theta[500:515] = 6.0                    # 15 days with B = 3.5: 2% of the days
        data = _build_theta_data('RK4', FLOOD_PAR, theta)
        report = stability_report(data)
        self.assertLess(report['frac_exceeding'], 0.10)
        self.assertGreater(report['max_growth'], STABILITY_MAX_GROWTH)
        self.assertEqual(report['growth_stretch'], (500, 514))    # the 14 steps between flood days
        with self.assertRaises(NumericalDivergenceError):
            warn_on_stability(data, error_fraction=0.10)
        warn_on_stability(data, error_fraction=0.10, max_growth=1e300)    # warns only

        # And without the check RK4 indeed goes wrong there, while CRN does not.
        call_model(data)
        crn = _build_theta_data('CRN', FLOOD_PAR, theta)
        call_model(crn)
        self.assertGreater(np.nanmax(np.abs(data.Twat_mod[500:530] - crn.Twat_mod[500:530])), 1.0)

    def test_short_spikes_above_the_limit_only_warn(self):
        # A single day with B above the limit: each step mixes it with a day below the limit,
        # and no difference grows at all, although the day counts in the share.
        theta = np.ones(400 + 365)
        theta[400::30] = 6.0
        data = _build_theta_data('RK4', FLOOD_PAR, theta)
        report = warn_on_stability(data, error_fraction=0.10)               # must not raise
        self.assertGreater(report['n_exceeding'], 0)
        self.assertEqual(report['max_growth'], 1.0)
        # Three days in a row: two steps with R(-3.5) = 2.73, so a difference grows 7.5 times.
        for k in (1, 2):
            theta[400 + k::30] = 6.0
        data = _build_theta_data('RK4', FLOOD_PAR, theta)
        report = warn_on_stability(data, error_fraction=0.10)               # must not raise
        self.assertAlmostEqual(report['max_growth'], (1 - 3.5 + 3.5**2 / 2 - 3.5**3 / 6 + 3.5**4 / 24) ** 2,
                               places=9)

    def test_negative_B_is_named_as_the_reason(self):
        # B < 0 (impossible physics) makes the equation itself unstable: every step grows.
        data = _build_theta_data('RK4', [0.5, 0.7, -0.05, 0, 0, 0, 0, 0], np.ones(400 + 365), version=3)
        self.assertEqual(stability_report(data)['n_exceeding'], 0)
        with self.assertRaisesRegex(NumericalDivergenceError, "B is negative"):
            warn_on_stability(data, error_fraction=0.10)

    def test_growth_does_not_chain_across_gap_tolerant_segments(self):
        n = 400 + 365
        data = _build_theta_data('RK4', [0.5, 0.7, 3.0, 0, 0, 0, 0, 0], np.ones(n), version=3)
        data.gap_tolerant = True
        data.segments = [(365, 500), (510, 600)]
        growth = largest_growth(data)
        self.assertAlmostEqual(growth['log10_growth'], 135 * np.log10(1.375), places=6)
        self.assertEqual((growth['start'], growth['end']), (365, 500))

    def test_crn_growth_is_bounded_and_crn_and_exp_are_never_stopped(self):
        # With CRN a difference can grow for a step when B falls sharply, but over any stretch
        # by no more than max(1, max(B)/2 - 1): the factors telescope.
        rng = np.random.default_rng(3)
        theta = np.exp(rng.normal(0.0, 1.5, 400 + 365))
        for mod_num in ('CRN', 'EXP'):
            data = _build_theta_data(mod_num, [0.5, 0.7, 1.0, 0.3, 1.0, 1.0, 0.4, 2.0], theta, version=8)
            B = compute_B_series(data)
            growth = largest_growth(data)['growth']
            if mod_num == 'EXP':
                self.assertEqual(growth, 1.0)
            else:
                self.assertGreater(growth, 1.0)
                self.assertLessEqual(growth, max(1.0, B.max() / 2 - 1) * (1 + 1e-9))
            warn_on_stability(data, error_fraction=0.0, max_growth=1.0)     # must not raise

    def test_stability_max_growth_is_read_from_the_config(self):
        import tempfile
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, 'config.yaml')
            for value, ok in ((1000, True), (0.5, False)):
                with open(config_path, 'w') as f:
                    yaml.dump({'project_name': os.path.join(tmp, 'proj'), 'version': 8, 'run_mode': 'PSO',
                               'stability_max_growth': value, 'paths': {'output_dir': os.path.join(tmp, 'out')}}, f)
                if ok:
                    self.assertEqual(read_calibration(config_path).stability_max_growth, value)
                else:
                    with self.assertRaises(ValueError):
                        read_calibration(config_path)
            self.assertEqual(CommonData().stability_max_growth, STABILITY_MAX_GROWTH)

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
