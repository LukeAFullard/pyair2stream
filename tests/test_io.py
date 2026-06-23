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
        self.input_file = os.path.join(self.test_dir.name, 'input.txt')
        self.pso_file = os.path.join(self.test_dir.name, 'PSO.txt')
        self.param_file = os.path.join(self.test_dir.name, 'parameters.txt')

        # We need to mock the directory structure the code expects
        # data.name = 'TestProject'
        self.proj_dir = os.path.join(self.test_dir.name, 'TestProject')
        os.makedirs(self.proj_dir, exist_ok=True)
        self.proj_param_file = os.path.join(self.proj_dir, 'parameters.txt')

    def tearDown(self):
        self.test_dir.cleanup()

    def test_read_calibration(self):
        # Create mock input.txt
        with open(self.input_file, 'w') as f:
            f.write("header\n")
            f.write(f"{self.proj_dir}\n") # name
            f.write("AirStation\n") # air_station
            f.write("WaterStation\n") # water_station
            f.write("c\n") # series
            f.write("1d\n") # time_res
            f.write("8\n") # version
            f.write("0.0\n") # Tice_cover
            f.write("NSE\n") # fun_obj
            f.write("RK4\n") # mod_num
            f.write("PSO\n") # runmode
            f.write("1.0\n") # prc
            f.write("100\n") # n_run
            f.write("0.0\n") # mineff_index

        # Create mock PSO.txt
        with open(self.pso_file, 'w') as f:
            f.write("header\n")
            f.write("50\n")
            f.write("2.0 2.0\n")
            f.write("0.9 0.4\n")

        # Create mock parameters.txt inside proj_dir
        with open(self.proj_param_file, 'w') as f:
            f.write("1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0\n")
            f.write("11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0\n")

        # Call function
        data = read_calibration(input_file=self.input_file, pso_file=self.pso_file, parameters=self.proj_param_file)

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
        data.station = "AirStation_WaterStation"
        data.series = "c"

        # Create mock Tseries file
        ts_file = os.path.join(self.proj_dir, f"{data.station}_cc.txt")

        # Generate 400 days of mock data spanning a leap year (e.g. 2020)
        dates = pd.date_range(start="2020-01-01", periods=400, freq='D')
        df = pd.DataFrame({
            'Year': dates.year,
            'Month': dates.month,
            'Day': dates.day,
            'Tair': np.random.rand(400) * 20,
            'Twat_obs': np.random.rand(400) * 15,
            'Q': np.random.rand(400) * 50
        })

        # Add a sentinel value to Q to test Qmedia calculation
        df.loc[10, 'Q'] = -999.0

        df.to_csv(ts_file, sep='\t', header=False, index=False)

        # Call function
        read_Tseries(data, 'c')

        # Expected total elements: 400 raw + 365 warm-up = 765
        self.assertEqual(data.n_tot, 765)

        # Test Qmedia without sentinel value
        valid_q = df.loc[df['Q'] != -999.0, 'Q']
        self.assertAlmostEqual(data.Qmedia, valid_q.mean(), places=5)
        self.assertEqual(data.n_Q, 399)

        # Test 1st year replication
        self.assertTrue(np.all(data.date[0:365, :] == -999))
        np.testing.assert_allclose(data.Tair[0:365], df['Tair'].values[:365], rtol=1e-7, atol=1e-7)

        # Test tt array: first 365 are warm-up (i.e. divided by 365)
        self.assertAlmostEqual(data.tt[0], 1.0/365.0)
        self.assertAlmostEqual(data.tt[364], 365.0/365.0)

        # The 366th day in the array (index 365) is 2020-01-01, a leap year, so it divides by 366
        self.assertAlmostEqual(data.tt[365], 1.0/366.0)
        self.assertAlmostEqual(data.tt[365 + 365], 366.0/366.0) # 2020-12-31

if __name__ == '__main__':
    unittest.main()
