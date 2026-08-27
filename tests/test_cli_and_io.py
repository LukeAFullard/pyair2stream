"""
Regression tests for docs/audit/05_cli_and_io_correctness.md.

1. Defect A: main() called aggregation()/statis() unconditionally before
   dispatching to any run mode, so a pure FORWARD-mode projection (no T_water
   at all -- the package's headline climate-projection use case) crashed with
   "n_dat is 0 after aggregation" before ever reaching forward_mode()'s own
   (correct) has_obs handling.
2. Defect B: a validation period shorter than one year returned from
   read_Tseries before data.n_tot was overwritten, so it silently retained the
   calibration value and passed main.forward()'s `if data.n_tot < 365` guard,
   re-running "validation" on the calibration arrays.
3. Defect C: every output CSV was written with the 365-day warm-up block
   (Year=-999) still in it, breaking pd.to_datetime and row-count expectations
   for anyone reading the file directly.
4. Defect D: a non-standard-calendar (e.g. 360-day GCM) series either failed
   validation outright or, if padded to fake Gregorian dates, silently
   misaligned the seasonal term. `calendar: noleap`/`360_day` computes tt from
   row position against the declared calendar instead.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from pyair2stream.config import CommonData
from pyair2stream.io import read_Tseries
from pyair2stream.main import main

PAR0 = [1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1]


def _write_config(path, **kwargs):
    with open(path, 'w') as f:
        yaml.dump(kwargs, f)


def _write_series_csv(path, n_days, start='2001-01-01', with_twat=True, q_scale=1.0):
    t = np.arange(n_days)
    dates = pd.date_range(start=start, periods=n_days, freq='D')
    df_data = {
        'Date': dates.strftime('%Y-%m-%d'),
        'T_air': 10.0 + 12.0 * np.sin(2 * np.pi * (t - 80) / 365.0),
        'Discharge': q_scale * (20.0 + 10.0 * np.sin(2 * np.pi * t / 365.0)),
    }
    if with_twat:
        df_data['T_water'] = 8.0 + 6.0 * np.sin(2 * np.pi * (t - 100) / 365.0)
    pd.DataFrame(df_data).to_csv(path, index=False)


class TestCliAndIoCorrectness(unittest.TestCase):
    def test_pure_projection_runs_to_completion_through_main(self):
        # Defect A: a forcing CSV with NO T_water column, run through the actual
        # CLI entry point (main()) in FORWARD mode, must not crash.
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'future.csv')
            n_days = 400
            dates = pd.date_range('2050-01-01', periods=n_days, freq='D')
            pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'T_air': 10.0 + 12.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.0),
                'Discharge': np.full(n_days, 10.0),
            }).to_csv(csv_path, index=False)

            out_dir = os.path.join(tmp, 'out')
            config_path = os.path.join(tmp, 'config.yaml')
            _write_config(
                config_path, project_name=os.path.join(tmp, 'proj'), station_name='X',
                water_station='X', version=8, integrator='CRN', run_mode='FORWARD',
                objective_function='NSE', Qmedia=10.0, parameters_forward=PAR0,
                paths={'input_data': csv_path, 'output_dir': out_dir},
            )

            old_argv = sys.argv
            try:
                sys.argv = ['pyair2stream', '--config', config_path]
                main()
            finally:
                sys.argv = old_argv

            out_csv = os.path.join(out_dir, '2_FORWARD_NSE_X_seriesc_1d.csv')
            self.assertTrue(os.path.exists(out_csv))
            df = pd.read_csv(out_csv)
            self.assertEqual(len(df), n_days)
            self.assertTrue((df['Twat_mod'] != -999.0).any())

    def test_short_validation_period_skips_cleanly(self):
        # Defect B: a <1-year validation file must not silently re-run
        # "validation" on the calibration arrays.
        with tempfile.TemporaryDirectory() as tmp:
            cal_csv = os.path.join(tmp, 'cal.csv')
            val_csv = os.path.join(tmp, 'val.csv')
            _write_series_csv(cal_csv, 400)
            _write_series_csv(val_csv, 200)  # < 365 days

            out_dir = os.path.join(tmp, 'out')
            config_path = os.path.join(tmp, 'config.yaml')
            _write_config(
                config_path, project_name=os.path.join(tmp, 'proj'), station_name='X',
                water_station='X', version=8, integrator='CRN', run_mode='DE',
                objective_function='NSE',
                optimization={'n_run': 1, 'n_particles': 2},
                parameter_bounds={'min': [0.0] * 8, 'max': [2.0] * 8},
                paths={'input_data': cal_csv, 'validation_data': val_csv, 'output_dir': out_dir},
            )

            old_argv = sys.argv
            try:
                sys.argv = ['pyair2stream', '--config', config_path]
                main()
            finally:
                sys.argv = old_argv

            val_csv_out = os.path.join(out_dir, '3_DE_NSE_X_seriesv_1d.csv')
            self.assertFalse(os.path.exists(val_csv_out))

            out_file = os.path.join(out_dir, '1_DE_NSE_X_series_1d.out')
            with open(out_file) as f:
                lines = [l for l in f.readlines() if l.strip()]
            # Header/params line + calibration efficiency line only, no validation line appended.
            self.assertEqual(len(lines), 2)

    def test_output_csv_has_no_warmup_rows_and_parses_cleanly(self):
        # Defect C: output CSVs must not contain the 365-day warm-up block, and
        # every row must parse as a real date.
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'cal.csv')
            n_days = 400
            _write_series_csv(csv_path, n_days)

            out_dir = os.path.join(tmp, 'out')
            config_path = os.path.join(tmp, 'config.yaml')
            _write_config(
                config_path, project_name=os.path.join(tmp, 'proj'), station_name='X',
                water_station='X', version=8, integrator='CRN', run_mode='DE',
                objective_function='NSE',
                optimization={'n_run': 1, 'n_particles': 2},
                parameter_bounds={'min': [0.0] * 8, 'max': [2.0] * 8},
                paths={'input_data': csv_path, 'output_dir': out_dir},
            )

            old_argv = sys.argv
            try:
                sys.argv = ['pyair2stream', '--config', config_path]
                main()
            finally:
                sys.argv = old_argv

            out_csv = os.path.join(out_dir, '2_DE_NSE_X_seriesc_1d.csv')
            df = pd.read_csv(out_csv)

            self.assertEqual(len(df), n_days)  # no warm-up rows, exactly the input row count
            self.assertFalse((df['Year'] == -999).any())
            pd.to_datetime(df[['Year', 'Month', 'Day']])  # must not raise

    def test_360_day_calendar_series_raises_under_default_standard_calendar(self):
        # Defect D: a genuine non-standard-calendar series (here, a noleap year
        # with Feb 29 stripped) must raise a clear error under the default
        # 'standard' calendar rather than being silently accepted.
        with tempfile.TemporaryDirectory() as tmp:
            dates = pd.date_range('2000-01-01', '2000-12-31', freq='D')
            dates = dates[~((dates.month == 2) & (dates.day == 29))]  # strip Feb 29
            csv_path = os.path.join(tmp, 'noleap.csv')
            pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'T_air': 10.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, len(dates))),
                'Discharge': np.full(len(dates), 10.0),
            }).to_csv(csv_path, index=False)

            data = CommonData()
            data.runmode = 'DE'
            data.calendar = 'standard'
            data._input_data_path_cal = csv_path
            with self.assertRaises(ValueError):
                read_Tseries(data, 'c')

    def test_noleap_calendar_computes_tt_from_row_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates = pd.date_range('2000-01-01', '2000-12-31', freq='D')
            dates = dates[~((dates.month == 2) & (dates.day == 29))]  # 365 real days
            csv_path = os.path.join(tmp, 'noleap.csv')
            pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'T_air': 10.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, len(dates))),
                'Discharge': np.full(len(dates), 10.0),
            }).to_csv(csv_path, index=False)

            data = CommonData()
            data.runmode = 'DE'
            data.calendar = 'noleap'
            data._input_data_path_cal = csv_path
            read_Tseries(data, 'c')  # must not raise

            # Day 1 of the real record -> 1/365; the last real day -> exactly 1.0.
            self.assertAlmostEqual(data.tt[365], 1.0 / 365.0)
            self.assertAlmostEqual(data.tt[365 + 364], 1.0)

    def test_360_day_calendar_computes_tt_from_row_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 720  # two 360-day years, padded as consecutive real Gregorian days
            dates = pd.date_range('2000-01-01', periods=n, freq='D')
            csv_path = os.path.join(tmp, 'threesixty.csv')
            pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'T_air': 10.0 + 5.0 * np.sin(np.linspace(0, 4 * np.pi, n)),
                'Discharge': np.full(n, 10.0),
            }).to_csv(csv_path, index=False)

            data = CommonData()
            data.runmode = 'DE'
            data.calendar = '360_day'
            data._input_data_path_cal = csv_path
            read_Tseries(data, 'c')  # must not raise

            self.assertAlmostEqual(data.tt[365], 1.0 / 360.0)          # day 1 of year 1
            self.assertAlmostEqual(data.tt[365 + 359], 1.0)            # day 360, end of year 1
            self.assertAlmostEqual(data.tt[365 + 360], 1.0 / 360.0)    # day 1 of year 2, restarts

    def test_invalid_calendar_value_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, 'config.yaml')
            _write_config(
                config_path, project_name=os.path.join(tmp, 'proj'), version=8,
                run_mode='DE', calendar='gregorian_v2',
                paths={'output_dir': os.path.join(tmp, 'out')},
            )
            from pyair2stream.io import read_calibration
            with self.assertRaises(ValueError):
                read_calibration(config_path)


if __name__ == '__main__':
    unittest.main()
