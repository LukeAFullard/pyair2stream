"""
Regression tests for docs/audit/10_zero_discharge_handling.md.

For model versions 4, 7, and 8, discharge only enters the ODE through
`theta = Q / Qmedia`, and the thermal-capacity divisor is `theta ** a4`. At
`Q == 0` this either raises a bare `ZeroDivisionError` (a4 > 0) or silently
evaluates to `inf`, collapsing that day's update towards zero with no error, no
NaN, and no warning (a4 < 0) -- undetectable by `check_numerical_divergence`.
These tests check that:

1. `check_nonpositive_discharge` (wired into `read_Tseries`, so it applies
   identically to the calibration record and a FORWARD-mode scenario record)
   raises a clear `ValueError` naming the offending index/date and count,
   before any parameter vector or integrator is ever involved.
2. Bypassing that guard and running the model directly reproduces the two raw
   failure modes described above (documenting *why* the guard is needed).
3. The opt-in `min_theta_floor` escape hatch lets the same run complete with
   finite output, behaving as a floor/limit (identical output for any Q at or
   below the floor), across every integrator and both signs of a4.
4. Versions 3/5 (which never evaluate theta) are completely unaffected.
5. The golden tests (tests/test_golden.py, tests/test_physics_golden.py) are
   unchanged by all of the above (verified separately -- they exercise no zero
   discharge and no min_theta_floor).
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from pyair2stream.config import CommonData
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import (
    call_model, check_nonpositive_discharge, check_numerical_divergence,
)

ALL_INTEGRATORS = ('CRN', 'RK2', 'RK4', 'EUL', 'EXP')

# Version-8 parameter sets differing only in the sign of a4 (index 3), otherwise
# a stable, plausible-fit-like set (reused shape from test_numerical_stability.py).
PAR_A4_POS = [0.3, 0.8, 0.1, 0.317, 11.16, 8.09, 0.5, 1.15]
PAR_A4_NEG = [0.3, 0.8, 0.1, -0.317, 11.16, 8.09, 0.5, 1.15]

# A parameter set with a well-behaved (bounded) equilibrium for versions 3/5, which
# have no theta term at all: Tw* = (a1 + a2*Ta)/a3, comfortably under the 60 degC
# sanity bound for the whole Ta range used by `_build_synthetic_data`.
PAR_STABLE_NO_THETA = [1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]


def _write_series_csv(path, n_days, q_zero_index=None, start='2001-01-01'):
    """A daily CSV starting 1 January, with T_water present (so this can be used
    for both calibration and FORWARD flows) and one optional zero-discharge day."""
    t = np.arange(n_days)
    dates = pd.date_range(start=start, periods=n_days, freq='D')
    T_air = 10.0 + 12.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
    Q = 8.0 + 4.0 * np.sin(2 * np.pi * t / 365.0)
    if q_zero_index is not None:
        Q[q_zero_index] = 0.0
    T_water = 5.0 + 8.0 * np.sin(2 * np.pi * (t - 60) / 365.0)
    df = pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'), 'T_air': T_air,
        'Discharge': Q, 'T_water': T_water,
    })
    df.to_csv(path, index=False)


def _build_synthetic_data(mod_num, par, n_days=400, Qmedia=8.0, version=8,
                           q_zero_index=200, min_theta_floor=None):
    """In-memory CommonData with a sinusoidal, otherwise-normal Q series and a
    single Q=0 day at `q_zero_index` (post-warm-up), bypassing read_Tseries/the
    guard entirely -- used to exercise the integrators directly."""
    n_tot = n_days + 365
    data = CommonData()
    data.n_tot = n_tot
    data.version = version
    data.mod_num = mod_num
    data.Qmedia = Qmedia
    data.Tice_cover = 0.0
    data.min_theta_floor = min_theta_floor
    data.par = np.array(par, dtype=np.float64)
    data.par_best = data.par.copy()

    t = np.arange(n_tot) - 365
    data.Tair = 10.0 + 12.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
    data.Q = 8.0 + 4.0 * np.sin(2 * np.pi * t / 365.0)
    if q_zero_index is not None:
        data.Q[365 + q_zero_index] = 0.0
    data.tt = np.array([((i % 365) + 1) / 365.0 for i in range(n_tot)])
    data.date = np.zeros((n_tot, 3), dtype=np.int32)
    data.Twat_obs = np.full(n_tot, -999.0)
    data.Twat_mod = np.zeros(n_tot)
    return data


class TestCheckNonpositiveDischargeUnit(unittest.TestCase):
    """Direct tests of the guard function against in-memory CommonData."""

    def test_raises_for_version_8_zero_day(self):
        data = _build_synthetic_data('CRN', PAR_A4_POS, q_zero_index=200)
        with self.assertRaisesRegex(ValueError, r'Non-positive discharge'):
            check_nonpositive_discharge(data)

    def test_error_names_index_date_and_count(self):
        data = _build_synthetic_data('CRN', PAR_A4_POS, q_zero_index=200)
        data.date[:] = -999
        data.date[365:, 0] = 2001
        data.date[365:, 1] = 1
        data.date[365:, 2] = np.arange(1, data.n_tot - 365 + 1)
        try:
            check_nonpositive_discharge(data)
            self.fail("expected ValueError")
        except ValueError as e:
            msg = str(e)
            self.assertIn(str(365 + 200), msg)
            self.assertIn('1 day(s)', msg)

    def test_counts_multiple_offending_days(self):
        data = _build_synthetic_data('CRN', PAR_A4_POS, q_zero_index=None)
        data.Q[400] = 0.0
        data.Q[401] = -1.0  # negative is also non-positive
        data.Q[450] = 0.0
        try:
            check_nonpositive_discharge(data)
            self.fail("expected ValueError")
        except ValueError as e:
            self.assertIn('3 day(s)', str(e))
            # first offending index reported
            self.assertIn(str(400), str(e))

    def test_negative_a4_does_not_change_whether_guard_fires(self):
        # The guard is purely a Q-data check: it must fire identically regardless
        # of which a4 sign is currently loaded on `data.par`.
        data_pos = _build_synthetic_data('CRN', PAR_A4_POS, q_zero_index=200)
        data_neg = _build_synthetic_data('CRN', PAR_A4_NEG, q_zero_index=200)
        with self.assertRaises(ValueError):
            check_nonpositive_discharge(data_pos)
        with self.assertRaises(ValueError):
            check_nonpositive_discharge(data_neg)

    def test_no_zero_days_does_not_raise(self):
        data = _build_synthetic_data('CRN', PAR_A4_POS, q_zero_index=None)
        check_nonpositive_discharge(data)  # must not raise

    def test_gap_tolerant_mode_is_left_alone(self):
        # Gap-tolerant mode already excludes Q<=0 days from every segment via a
        # heavier mechanism (segment restart); the guard must not duplicate/
        # interfere with that.
        data = _build_synthetic_data('CRN', PAR_A4_POS, q_zero_index=200)
        data.gap_tolerant = True
        check_nonpositive_discharge(data)  # must not raise

    def test_min_theta_floor_set_skips_the_guard(self):
        data = _build_synthetic_data('CRN', PAR_A4_POS, q_zero_index=200,
                                      min_theta_floor=1e-6)
        check_nonpositive_discharge(data)  # must not raise


class TestVersions3And5Unaffected(unittest.TestCase):
    """Versions 3/5 never evaluate theta, so Q=0 is not a numerical problem
    for them -- no new restriction should apply."""

    def test_check_does_not_raise_for_version_3_or_5(self):
        for version in (3, 5):
            data = _build_synthetic_data('CRN', PAR_A4_POS, version=version, q_zero_index=200)
            check_nonpositive_discharge(data)  # must not raise

    def test_call_model_runs_fine_for_version_3_or_5_with_zero_discharge(self):
        for version in (3, 5):
            for mod_num in ALL_INTEGRATORS:
                data = _build_synthetic_data(mod_num, PAR_STABLE_NO_THETA, version=version, q_zero_index=200)
                call_model(data)
                self.assertTrue(np.all(np.isfinite(data.Twat_mod)))
                check_numerical_divergence(data, max_plausible_twat=60.0)  # must not raise


class TestRawFailureModesWithoutTheGuard(unittest.TestCase):
    """Documents the two failure modes the guard exists to prevent, by calling
    call_model() directly (i.e. bypassing check_nonpositive_discharge/read_Tseries)
    on a version-8 zero-discharge day."""

    def test_positive_a4_raises_bare_zerodivisionerror(self):
        for mod_num in ALL_INTEGRATORS:
            data = _build_synthetic_data(mod_num, PAR_A4_POS, q_zero_index=200)
            with self.assertRaises(ZeroDivisionError):
                call_model(data)

    def test_negative_a4_silently_produces_finite_but_wrong_value(self):
        for mod_num in ALL_INTEGRATORS:
            data_zero = _build_synthetic_data(mod_num, PAR_A4_NEG, q_zero_index=200)
            call_model(data_zero)
            # No crash, no NaN/inf -- and check_numerical_divergence does not catch it,
            # since the value stays inside the plausible range. This is the "silent
            # mis-simulation" failure mode.
            self.assertTrue(np.all(np.isfinite(data_zero.Twat_mod)))
            check_numerical_divergence(data_zero, max_plausible_twat=60.0)  # must not raise

            data_normal = _build_synthetic_data(mod_num, PAR_A4_NEG, q_zero_index=None)
            call_model(data_normal)

            # The zero day (and everything downstream in the same integration
            # segment) differs from the same run without the zero day.
            diff = np.abs(data_zero.Twat_mod[365:] - data_normal.Twat_mod[365:])
            self.assertGreater(np.max(diff), 1e-6)


class TestMinThetaFloorEscapeHatch(unittest.TestCase):
    """The opt-in floor lets the run complete, and behaves as a limit (not a
    special case): any Q at or below floor*Qmedia clamps to the exact same
    theta, and therefore produces an identical simulated series."""

    def test_floor_produces_finite_output_every_integrator_both_a4_signs(self):
        # A very small floor amplifies theta**a4 sharply for that single day (an
        # expected, physically-driven consequence of clamping close to zero -- not
        # a numerical crash), so this only asserts finiteness/completion, not that
        # the single floored day stays under the (unrelated) sanity bound.
        for mod_num in ALL_INTEGRATORS:
            for par in (PAR_A4_POS, PAR_A4_NEG):
                data = _build_synthetic_data(mod_num, par, q_zero_index=200,
                                              min_theta_floor=1e-6)
                call_model(data)
                self.assertTrue(np.all(np.isfinite(data.Twat_mod)),
                                 msg=f"{mod_num}, a4={par[3]}")

    def test_floor_behaves_as_a_limit_not_a_special_case(self):
        # A day with Q=0 and a day with a very small positive Q that is still
        # below the floor threshold both clamp to the identical theta at that
        # day, and must therefore produce essentially identical output -- proving
        # the floor is a genuine clamp/limit, not a Q==0-only special case.
        #
        # CRN/EXP/EUL/RK2 evaluate theta (and apply the floor) directly from each
        # day's own Q, so those match to machine precision. RK4 additionally
        # evaluates a midpoint using Q_mid = 0.5*(Q[j] + Q[j+1]) *before* flooring
        # (exactly like it does for Tair) -- since the neighbouring day's Q is a
        # normal, un-floored value, Q_mid itself is not near zero and is not
        # floored, so the (tiny) difference between Q=0 and Q=tiny_q at the
        # zeroed day survives, undamped, into that one midpoint evaluation. This
        # is expected RK4 behaviour (the same averaging it applies to Tair), not a
        # discontinuity introduced by flooring, and stays tiny (of order tiny_q).
        floor = 1e-6
        Qmedia = 8.0
        tiny_q = 0.01 * floor * Qmedia  # well below floor*Qmedia

        for mod_num in ALL_INTEGRATORS:
            data_zero = _build_synthetic_data(mod_num, PAR_A4_NEG, q_zero_index=200,
                                                min_theta_floor=floor, Qmedia=Qmedia)
            call_model(data_zero)

            data_tiny = _build_synthetic_data(mod_num, PAR_A4_NEG, q_zero_index=None,
                                               min_theta_floor=floor, Qmedia=Qmedia)
            data_tiny.Q[365 + 200] = tiny_q
            call_model(data_tiny)

            atol = 1e-6 if mod_num == 'RK4' else 1e-10
            np.testing.assert_allclose(
                data_zero.Twat_mod[365:], data_tiny.Twat_mod[365:], rtol=0, atol=atol,
                err_msg=f"floor did not behave as a limit for {mod_num}"
            )

    def test_floor_matches_boundary_run_closely(self):
        # A day whose real Q lands exactly on the floor (unclamped, since the
        # comparison `theta < theta_floor` is strict) must match a Q=0 day with
        # the same floor (clamped) closely, since both evaluate theta at
        # precisely the floor value -- see the RK4 midpoint-averaging note above
        # for why this is a close match rather than an exact one for RK4.
        floor = 1e-6
        Qmedia = 8.0

        for mod_num in ALL_INTEGRATORS:
            data_zero = _build_synthetic_data(mod_num, PAR_A4_POS, q_zero_index=200,
                                                min_theta_floor=floor, Qmedia=Qmedia)
            call_model(data_zero)

            data_boundary = _build_synthetic_data(mod_num, PAR_A4_POS, q_zero_index=None,
                                                   min_theta_floor=floor, Qmedia=Qmedia)
            data_boundary.Q[365 + 200] = floor * Qmedia
            call_model(data_boundary)

            atol = 1e-4 if mod_num == 'RK4' else 1e-10
            np.testing.assert_allclose(
                data_zero.Twat_mod[365:], data_boundary.Twat_mod[365:], rtol=0, atol=atol,
            )


class TestReadTseriesIntegration(unittest.TestCase):
    """End-to-end: the guard is wired into read_Tseries, so it fires identically
    for a calibration record and a FORWARD-mode scenario record, before any
    calibration search or forward run ever begins."""

    def test_calibration_load_raises_clear_error_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'cal.csv')
            _write_series_csv(csv_path, n_days=400, q_zero_index=200)

            for mod_num in ALL_INTEGRATORS:
                data = CommonData()
                data.runmode = 'DE'
                data.version = 8
                data.mod_num = mod_num
                data._input_data_path_cal = csv_path
                with self.assertRaisesRegex(ValueError, r'Non-positive discharge'):
                    read_Tseries(data, 'c')

    def test_forward_scenario_load_raises_clear_error_by_default(self):
        # Same guard fires identically for a FORWARD-mode scenario discharge
        # record (naturalised flow / climate projection), not just calibration.
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'scenario.csv')
            _write_series_csv(csv_path, n_days=400, q_zero_index=200)

            data = CommonData()
            data.runmode = 'FORWARD'
            data.version = 8
            data.mod_num = 'CRN'
            data.Qmedia_user = 8.0  # FORWARD mode requires an explicit Qmedia
            data._input_data_path_cal = csv_path
            with self.assertRaisesRegex(ValueError, r'Non-positive discharge'):
                read_Tseries(data, 'c')

    def test_calibration_load_does_not_raise_without_zero_discharge(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'cal.csv')
            _write_series_csv(csv_path, n_days=400, q_zero_index=None)

            data = CommonData()
            data.runmode = 'DE'
            data.version = 8
            data.mod_num = 'CRN'
            data._input_data_path_cal = csv_path
            read_Tseries(data, 'c')  # must not raise

    def test_min_theta_floor_via_config_allows_load_to_proceed(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'cal.csv')
            _write_series_csv(csv_path, n_days=400, q_zero_index=200)

            data = CommonData()
            data.runmode = 'DE'
            data.version = 8
            data.mod_num = 'CRN'
            data.min_theta_floor = 1e-6
            data._input_data_path_cal = csv_path
            read_Tseries(data, 'c')  # must not raise
            self.assertEqual(data.Q[365 + 200], 0.0)  # raw Q is untouched; only theta is floored

    def test_min_theta_floor_parsed_from_config_yaml(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'cal.csv')
            _write_series_csv(csv_path, n_days=400, q_zero_index=200)
            config_path = os.path.join(tmp, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump({
                    'project_name': os.path.join(tmp, 'proj'),
                    'version': 8,
                    'run_mode': 'DE',
                    'min_theta_floor': 1e-6,
                    'paths': {'input_data': csv_path, 'output_dir': os.path.join(tmp, 'out')},
                }, f)
            data = read_calibration(config_path)
            self.assertEqual(data.min_theta_floor, 1e-6)
            read_Tseries(data, 'c')  # must not raise

    def test_invalid_min_theta_floor_rejected(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'cal.csv')
            _write_series_csv(csv_path, n_days=400, q_zero_index=None)
            config_path = os.path.join(tmp, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump({
                    'project_name': os.path.join(tmp, 'proj'),
                    'version': 8,
                    'run_mode': 'DE',
                    'min_theta_floor': -1.0,
                    'paths': {'input_data': csv_path, 'output_dir': os.path.join(tmp, 'out')},
                }, f)
            with self.assertRaisesRegex(ValueError, r'min_theta_floor must be a positive float'):
                read_calibration(config_path)


if __name__ == '__main__':
    unittest.main()
