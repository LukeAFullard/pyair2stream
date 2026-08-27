"""
Regression tests for docs/audit/01_qmedia_scenario_invariance.md.

Qmedia is a calibrated model constant (theta = Q / Qmedia feeds every discharge
term in the ODE). These tests check that:

1. Pinning Qmedia across a discharge rescaling produces a real, non-zero,
   monotonically growing temperature response (the bug made this exactly zero).
2. FORWARD mode refuses to run without an explicit Qmedia or calibration metadata.
3. Qmedia is frozen (not recomputed) when the validation period is loaded.
4. `calibration_metadata.json` round-trips: a FORWARD run built from it
   reproduces the calibration objective value.
"""

import os
import json
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from pyair2stream.config import CommonData
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import call_model, aggregation, statis, funcobj
from pyair2stream.main import forward

# A stable version-8 parameter set (all 8 parameters active) reused from the
# golden-test fixtures, known not to blow up under RK4.
PAR0 = [1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1]


def _write_series_csv(path, n_days, q_scale=1.0, start='2001-01-01', with_twat=False):
    t = np.arange(n_days)
    dates = pd.date_range(start=start, periods=n_days, freq='D')
    T_air = 10.0 + 15.0 * np.sin(2 * np.pi * (t - 80) / 365.0)
    Q = q_scale * (20.0 + 10.0 * np.sin(2 * np.pi * t / 365.0))
    df_data = {'Date': dates.strftime('%Y-%m-%d'), 'T_air': T_air, 'Discharge': Q}
    if with_twat:
        df_data['T_water'] = 8.0 + 6.0 * np.sin(2 * np.pi * (t - 100) / 365.0)
    pd.DataFrame(df_data).to_csv(path, index=False)


def _write_config(path, **kwargs):
    with open(path, 'w') as f:
        yaml.dump(kwargs, f)


class TestQmediaScenarioInvariance(unittest.TestCase):
    def test_scenario_signal_not_cancelled_when_qmedia_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline_csv = os.path.join(tmp, 'baseline.csv')
            _write_series_csv(baseline_csv, 400, q_scale=1.0)

            # "Calibration": load the baseline data and capture the Qmedia the
            # (fixed, externally-supplied) parameters are implicitly fitted under.
            cal_cfg = os.path.join(tmp, 'config_cal.yaml')
            _write_config(
                cal_cfg, project_name=os.path.join(tmp, 'cal'), version=8, integrator='RK4',
                run_mode='DE', objective_function='NSE',
                paths={'input_data': baseline_csv, 'output_dir': os.path.join(tmp, 'cal_out')},
            )
            cal_data = read_calibration(cal_cfg)
            read_Tseries(cal_data, 'c')
            qmedia_pinned = float(cal_data.Qmedia)

            def run_forward_scenario(k):
                scen_csv = os.path.join(tmp, f'scenario_{k}.csv')
                _write_series_csv(scen_csv, 400, q_scale=k)
                fwd_cfg = os.path.join(tmp, f'config_fwd_{k}.yaml')
                _write_config(
                    fwd_cfg, project_name=os.path.join(tmp, f'fwd_{k}'), version=8, integrator='RK4',
                    run_mode='FORWARD', objective_function='NSE',
                    Qmedia=qmedia_pinned, parameters_forward=PAR0,
                    paths={'input_data': scen_csv, 'output_dir': os.path.join(tmp, f'fwd_out_{k}')},
                )
                data = read_calibration(fwd_cfg)
                read_Tseries(data, 'c')
                call_model(data)
                return data.Twat_mod.copy(), data.n_tot

            base_twat, n_tot = run_forward_scenario(1.0)

            mean_abs_dT = {}
            for k in (1.1, 1.25, 1.5):
                scen_twat, _ = run_forward_scenario(k)
                dT = scen_twat[365:n_tot] - base_twat[365:n_tot]
                mean_abs_dT[k] = float(np.mean(np.abs(dT)))

            # A uniform discharge rescaling must produce a real, non-zero signal...
            for k in (1.1, 1.25, 1.5):
                self.assertGreater(mean_abs_dT[k], 1e-6)

            # ...that grows monotonically with |k - 1|.
            ks = sorted(mean_abs_dT.keys())
            for a, b in zip(ks, ks[1:]):
                self.assertGreater(mean_abs_dT[b], mean_abs_dT[a])

    def test_forward_without_qmedia_or_metadata_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'data.csv')
            _write_series_csv(csv_path, 400)
            cfg = os.path.join(tmp, 'config.yaml')
            _write_config(
                cfg, project_name=os.path.join(tmp, 'proj'), version=8, integrator='RK4',
                run_mode='FORWARD', objective_function='NSE', parameters_forward=PAR0,
                paths={'input_data': csv_path, 'output_dir': os.path.join(tmp, 'out')},
            )
            data = read_calibration(cfg)
            self.assertIsNone(data.Qmedia_user)
            with self.assertRaises(ValueError):
                read_Tseries(data, 'c')

    def test_forward_versions_without_discharge_do_not_require_qmedia(self):
        # Versions 3 and 5 zero out every discharge-related parameter and never
        # evaluate theta, so the guard should not apply to them.
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'data.csv')
            _write_series_csv(csv_path, 400)
            for version in (3, 5):
                cfg = os.path.join(tmp, f'config_v{version}.yaml')
                _write_config(
                    cfg, project_name=os.path.join(tmp, f'proj_v{version}'), version=version,
                    integrator='RK4', run_mode='FORWARD', objective_function='NSE',
                    parameters_forward=PAR0,
                    paths={'input_data': csv_path, 'output_dir': os.path.join(tmp, f'out_v{version}')},
                )
                data = read_calibration(cfg)
                read_Tseries(data, 'c')  # must not raise

    def test_qmedia_frozen_across_validation_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal_csv = os.path.join(tmp, 'cal.csv')
            val_csv = os.path.join(tmp, 'val.csv')
            _write_series_csv(cal_csv, 400, q_scale=1.0)
            _write_series_csv(val_csv, 400, q_scale=5.0)  # deliberately different discharge scale

            data = CommonData()
            data.runmode = 'DE'
            data._input_data_path_cal = cal_csv
            data._input_data_path_val = val_csv

            read_Tseries(data, 'c')
            qmedia_after_cal = float(data.Qmedia)

            read_Tseries(data, 'v', recompute_qmedia=False)

            self.assertEqual(float(data.Qmedia), qmedia_after_cal)

    def test_calibration_metadata_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'cal.csv')
            _write_series_csv(csv_path, 400, with_twat=True)

            cal_cfg = os.path.join(tmp, 'config_cal.yaml')
            out_dir = os.path.join(tmp, 'cal_out')
            _write_config(
                cal_cfg, project_name=os.path.join(tmp, 'cal'), version=8, integrator='RK4',
                run_mode='DE', objective_function='NSE',
                paths={'input_data': csv_path, 'output_dir': out_dir},
            )

            data = read_calibration(cal_cfg)
            read_Tseries(data, 'c')
            aggregation(data)
            statis(data)

            # Stand in for a completed optimizer run.
            data.par[:] = PAR0
            call_model(data)
            data.finalfit = funcobj(data)
            data.par_best[:] = PAR0

            forward(data)

            metadata_path = os.path.join(data.folder, 'calibration_metadata.json')
            self.assertTrue(os.path.exists(metadata_path))
            with open(metadata_path) as f:
                metadata = json.load(f)

            self.assertAlmostEqual(metadata['qmedia'], float(data.Qmedia), places=5)
            self.assertEqual(metadata['version'], 8)
            self.assertEqual(metadata['integrator'], 'RK4')
            self.assertIsNotNone(metadata['theta_min'])
            self.assertIsNotNone(metadata['theta_max'])

            # Round trip: run FORWARD from the saved metadata on the SAME data and
            # confirm the reproduced objective matches the calibration's finalfit.
            fwd_cfg = os.path.join(tmp, 'config_fwd.yaml')
            fwd_out_dir = os.path.join(tmp, 'fwd_out')
            _write_config(
                fwd_cfg, project_name=os.path.join(tmp, 'fwd'), version=8, integrator='RK4',
                run_mode='FORWARD', objective_function='NSE', parameters_forward=PAR0,
                paths={
                    'input_data': csv_path, 'output_dir': fwd_out_dir,
                    'calibration_metadata': metadata_path,
                },
            )

            fwd_data = read_calibration(fwd_cfg)
            self.assertAlmostEqual(fwd_data.Qmedia_user, metadata['qmedia'], places=9)

            read_Tseries(fwd_data, 'c')
            aggregation(fwd_data)
            statis(fwd_data)
            call_model(fwd_data)
            reproduced_ei = funcobj(fwd_data)

            self.assertAlmostEqual(reproduced_ei, data.finalfit, places=6)

    def test_calibration_metadata_version_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = os.path.join(tmp, 'calibration_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    "qmedia": 10.0, "version": 8, "integrator": "RK4",
                }, f)

            csv_path = os.path.join(tmp, 'data.csv')
            _write_series_csv(csv_path, 400)
            cfg = os.path.join(tmp, 'config.yaml')
            _write_config(
                cfg, project_name=os.path.join(tmp, 'proj'), version=4, integrator='RK4',
                run_mode='FORWARD', objective_function='NSE', parameters_forward=PAR0,
                paths={
                    'input_data': csv_path, 'output_dir': os.path.join(tmp, 'out'),
                    'calibration_metadata': metadata_path,
                },
            )
            with self.assertRaises(ValueError):
                read_calibration(cfg)


if __name__ == '__main__':
    unittest.main()
