import unittest
import os
import tempfile
import numpy as np
import pandas as pd
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.config import CommonData

class TestIO(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_file = os.path.join(self.test_dir.name, 'config.yaml')

        self.proj_dir = os.path.join(self.test_dir.name, 'TestProject')
        os.makedirs(self.proj_dir, exist_ok=True)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_read_calibration(self):
        # Create mock config.yaml
        yaml_content = f"""
project_name: "{self.proj_dir}"
station_name: "AirStation"
water_station: "WaterStation"
series: "c"
time_resolution: "1d"
version: 8
Tice_cover: 0.0
objective_function: "NSE"
integrator: "RK4"
run_mode: "PSO"
prc: 1.0

optimization:
  n_runs: 100
  mineff_index: 0.0
  n_particles: 50
  c1: 2.0
  c2: 2.0
  wmax: 0.9
  wmin: 0.4

parameter_bounds:
  min: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  max: [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
"""
        with open(self.config_file, 'w') as f:
            f.write(yaml_content)

        # Call function
        data = read_calibration(config_file=self.config_file)

        # Assertions
        self.assertEqual(data.name, self.proj_dir)
        self.assertEqual(data.air_station, "AirStation")
        self.assertEqual(data.water_station, "WaterStation")
        self.assertEqual(data.station, "AirStation_WaterStation")
        self.assertEqual(data.version, 8)
        self.assertEqual(data.runmode, "PSO")
        self.assertEqual(data.n_particles, 50)
        self.assertEqual(data.c1, 2.0)
        self.assertEqual(data.wmax, 0.9)

        # Test bug fix for version 8
        np.testing.assert_array_equal(data.parmin, [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(data.parmax, [11.0, 12.0, 13.0, 14.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(data.flag_par, [True, True, True, True, False, False, False, False])

    def test_read_Tseries(self):
        # Create a mock CommonData
        data = CommonData()
        data.name = self.proj_dir
        data.station = "AirStation"
        data.series = "c"

        # Create mock Tseries file
        ts_file = os.path.join(self.proj_dir, f"input_timeseries.csv")
        data._input_data_path_cal = ts_file

        # Generate 400 days of mock data spanning a leap year (e.g. 2020)
        dates = pd.date_range(start="2020-01-01", periods=400, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'T_air': np.random.rand(400) * 20,
            'T_water': np.random.rand(400) * 15,
            'Discharge': np.random.rand(400) * 50
        })

        # Add a sentinel value to Q to test Qmedia calculation
        df.loc[10, 'Discharge'] = -999.0

        df.to_csv(ts_file, index=False)

        # Call function
        read_Tseries(data, 'c')

        # Expected total elements: 400 raw + 365 warm-up = 765
        self.assertEqual(data.n_tot, 765)

        # Test Qmedia without sentinel value
        valid_q = df.loc[df['Discharge'] != -999.0, 'Discharge']
        self.assertAlmostEqual(data.Qmedia, valid_q.mean(), places=5)
        self.assertEqual(data.n_Q, 399)

        # Test 1st year replication
        self.assertTrue(np.all(data.date[0:365, :] == -999))
        np.testing.assert_allclose(data.Tair[0:365], df['T_air'].values[:365], rtol=1e-7, atol=1e-7)

        # Test tt array: first 365 are warm-up (i.e. divided by 365)
        self.assertAlmostEqual(data.tt[0], 1.0/365.0)
        self.assertAlmostEqual(data.tt[364], 365.0/365.0)

        # The 366th day in the array (index 365) is 2020-01-01, a leap year, so it divides by 366
        self.assertAlmostEqual(data.tt[365], 1.0/366.0)
        self.assertAlmostEqual(data.tt[365 + 365], 366.0/366.0) # 2020-12-31

if __name__ == '__main__':
    unittest.main()
