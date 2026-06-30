import unittest
import numpy as np
import pandas as pd
import tempfile
import os

from pyair2stream.config import CommonData
from pyair2stream.sensitivity import sensitivity_analysis
from pyair2stream.model import call_model, aggregation

class TestSensitivity(unittest.TestCase):
    def test_sensitivity_analysis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = CommonData()
            data.folder = tmpdir
            data.runmode = "FORWARD"
            data.fun_obj = "NSE"
            data.station = "test_station"
            data.series = "c"

            csv_path = os.path.join(tmpdir, "mock_data.csv")
            df = pd.DataFrame({
                'Date': pd.date_range('2000-01-01', periods=400, freq='D').strftime('%Y-%m-%d'),
                'T_air': np.random.rand(400) * 10 + 10,
                'T_water': np.random.rand(400) * 5 + 5,
                'Discharge': np.random.rand(400) * 50 + 10
            })
            df.to_csv(csv_path, index=False)
            data._input_data_path_cal = csv_path
            data.name = tmpdir

            data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1])
            data.par_best = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1])
            data.parmin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            data.parmax = np.array([5.0, 1.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0])
            data.flag_par = np.array([True]*8)

            data.version = 8
            data.mod_num = 'RK4'

            # `eval_mask` is initialized in `detect_segments` or defaults.
            # We can enable gap tolerant to make it init properly, or we can mock it
            data.gap_tolerant = True

            data.sensitivity_perturbations = [1.0] # 1% perturbation

            df_sens = sensitivity_analysis(data)

            # Check dataframe has 8 rows
            self.assertEqual(len(df_sens), 8)

            # Check outputs were created
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "sensitivity_FORWARD_NSE_test_station.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "sensitivity_FORWARD_NSE_test_station.png")))

if __name__ == '__main__':
    unittest.main()
