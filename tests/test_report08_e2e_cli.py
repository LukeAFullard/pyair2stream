"""
End-to-end CLI tests for docs/audit/08_testing_gaps.md, Gap E.

Nothing in the suite previously invoked `main()` on a realistic config for a
full calibration-with-validation run or a gap-tolerant-plus-sensitivity run,
so a `main()`-level defect in either path (argument parsing, output-file
writing, the two dispatched independently by `run_mode`) could only be caught
by a real user. The other two scenarios in report 08's minimum set --
FORWARD with no `T_water` (report 05, Defect A) and a sub-365-day validation
file (report 05, Defect B) -- are already covered end-to-end through
`main()` in `tests/test_cli_and_io.py`; not duplicated here.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from pyair2stream.main import main

PAR0 = [1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1]


def _write_config(path, **kwargs):
    with open(path, 'w') as f:
        yaml.dump(kwargs, f)


def _write_series_csv(path, n_days, start='2001-01-01', tair_gap_days=None, q_gap_days=None):
    t = np.arange(n_days)
    dates = pd.date_range(start=start, periods=n_days, freq='D')
    tair = 10.0 + 12.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
    q = 20.0 + 10.0 * np.sin(2 * np.pi * t / 365.0)
    twat = 8.0 + 6.0 * np.sin(2 * np.pi * (t - 100) / 365.0)

    if tair_gap_days:
        tair = tair.astype(object)
        for d in tair_gap_days:
            tair[d] = np.nan
    if q_gap_days:
        q = q.astype(object)
        for d in q_gap_days:
            q[d] = np.nan

    pd.DataFrame({
        'Date': dates.strftime('%Y-%m-%d'),
        'T_air': tair,
        'Discharge': q,
        'T_water': twat,
    }).to_csv(path, index=False)


def _run_main(config_path):
    old_argv = sys.argv
    try:
        sys.argv = ['pyair2stream', '--config', config_path]
        main()
    finally:
        sys.argv = old_argv


class TestDECalibrationWithValidation(unittest.TestCase):
    """DE calibration with a real (>=1 year) validation period: 1_/2_/3_ files all exist and parse."""

    def test_de_calibration_with_validation_e2e(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal_csv = os.path.join(tmp, 'cal.csv')
            val_csv = os.path.join(tmp, 'val.csv')
            n_cal_days = 400
            n_val_days = 400
            _write_series_csv(cal_csv, n_cal_days, start='2001-01-01')
            _write_series_csv(val_csv, n_val_days, start='2003-01-01')

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

            _run_main(config_path)

            out_file = os.path.join(out_dir, '1_DE_NSE_X_series_1d.out')
            self.assertTrue(os.path.exists(out_file))
            with open(out_file) as f:
                lines = [l for l in f.readlines() if l.strip()]
            # params+eff line, calibration efficiency, validation efficiency.
            self.assertEqual(len(lines), 3)
            par_line = [float(x) for x in lines[0].split()]
            self.assertEqual(len(par_line), 8)
            float(lines[1])  # calibration efficiency, parses as a number
            float(lines[2])  # validation efficiency, parses as a number

            cal_out = os.path.join(out_dir, '2_DE_NSE_X_seriesc_1d.csv')
            df_cal = pd.read_csv(cal_out)
            self.assertEqual(len(df_cal), n_cal_days)
            pd.to_datetime(df_cal[['Year', 'Month', 'Day']])  # must not raise

            val_out = os.path.join(out_dir, '3_DE_NSE_X_seriesv_1d.csv')
            df_val = pd.read_csv(val_out)
            self.assertEqual(len(df_val), n_val_days)
            pd.to_datetime(df_val[['Year', 'Month', 'Day']])  # must not raise

            metadata_path = os.path.join(out_dir, 'calibration_metadata.json')
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.assertIn('qmedia', metadata)
            self.assertIn('par_best', metadata)
            self.assertEqual(len(metadata['par_best']), 8)


class TestGapTolerantCalibrationWithSensitivity(unittest.TestCase):
    """Gap-tolerant calibration followed by sensitivity analysis (report 06, Defect D)."""

    def test_gap_tolerant_calibration_and_sensitivity_e2e(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal_csv = os.path.join(tmp, 'cal.csv')
            n_days = 500
            # A handful of T_air gaps well inside the record, isolated enough that
            # min_segment_days still leaves usable segments on both sides.
            _write_series_csv(cal_csv, n_days, tair_gap_days=[200, 201, 202])

            out_dir = os.path.join(tmp, 'out')
            config_path = os.path.join(tmp, 'config.yaml')
            _write_config(
                config_path, project_name=os.path.join(tmp, 'proj'), station_name='X',
                water_station='X', version=8, integrator='CRN', run_mode='DE',
                objective_function='NSE', gap_tolerant=True, min_segment_days=30,
                warmup_drop_days=5,
                optimization={'n_run': 1, 'n_particles': 2},
                parameter_bounds={'min': [0.0] * 8, 'max': [2.0] * 8},
                paths={'input_data': cal_csv, 'output_dir': out_dir},
                sensitivity_analysis=True, sensitivity_perturbations=[5.0],
            )

            _run_main(config_path)

            gaps_summary = os.path.join(out_dir, 'gaps_summary.txt')
            self.assertTrue(os.path.exists(gaps_summary))

            sens_csv = os.path.join(out_dir, 'sensitivity_DE_NSE_X.csv')
            self.assertTrue(os.path.exists(sens_csv))
            df_sens = pd.read_csv(sens_csv)
            self.assertEqual(
                list(df_sens.columns),
                ['Parameter', 'Perturbation_%', 'Sensitivity_Index', 'Status'],
            )
            self.assertEqual(len(df_sens), 8)  # one row per parameter


if __name__ == '__main__':
    unittest.main()
