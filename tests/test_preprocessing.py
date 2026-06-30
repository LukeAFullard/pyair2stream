import unittest
import pandas as pd
import numpy as np
import io
import tempfile
import os

from pyair2stream.preprocessing import read_and_resample, merge_timeseries

class TestPreprocessing(unittest.TestCase):
    def test_read_and_resample(self):
        # CSV with specific datetime formats
        csv_content = """date,temp
01/01/2000,10.5
02/01/2000,11.0
03/01/2000,11.5
03/01/2000,12.5
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write(csv_content)
            tmp_path = tmp.name

        try:
            df = read_and_resample(tmp_path, date_col='date', value_col='temp',
                                   standard_col_name='T_air', date_format='%d/%m/%Y')
            self.assertEqual(len(df), 3)
            self.assertEqual(df.iloc[0]['T_air'], 10.5)
            self.assertEqual(df.iloc[2]['T_air'], 12.0) # Average of 11.5 and 12.5
            self.assertEqual(df.iloc[0]['Date'], '2000-01-01')
        finally:
            os.remove(tmp_path)

    def test_merge_timeseries(self):
        csv_air = """date,temp
01/01/2000,10
02/01/2000,11
04/01/2000,13
05/01/2000,14
"""
        csv_wat = """Date,Twat
2000-01-01,5
2000-01-03,7
2000-01-04,8
2000-01-05,9
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_air:
            tmp_air.write(csv_air)
            path_air = tmp_air.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_wat:
            tmp_wat.write(csv_wat)
            path_wat = tmp_wat.name

        try:
            file_configs = [
                {'file_path': path_air, 'date_col': 'date', 'value_col': 'temp', 'standard_col_name': 'T_air', 'date_format': '%d/%m/%Y'},
                {'file_path': path_wat, 'date_col': 'Date', 'value_col': 'Twat', 'standard_col_name': 'T_water', 'date_format': '%Y-%m-%d'}
            ]

            merged = merge_timeseries(file_configs)

            # Length should be 5 because it pads out the missing dates from 1st to 5th
            self.assertEqual(len(merged), 5)

            # Test Date column conversion
            self.assertEqual(merged['Date'].iloc[0], '2000-01-01')
            self.assertEqual(merged['Date'].iloc[4], '2000-01-05')

            # Test padding correctly placed NaNs where expected
            self.assertTrue(pd.isna(merged['T_air'].iloc[2])) # Jan 3rd missing in air
            self.assertTrue(pd.isna(merged['T_water'].iloc[1])) # Jan 2nd missing in water
            self.assertEqual(merged['T_air'].iloc[3], 13.0) # Jan 4th
            self.assertEqual(merged['T_water'].iloc[2], 7.0) # Jan 3rd
        finally:
            os.remove(path_air)
            os.remove(path_wat)

if __name__ == '__main__':
    unittest.main()
