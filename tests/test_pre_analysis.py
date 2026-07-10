import unittest
import pandas as pd
import numpy as np
import tempfile
import os

from pyair2stream.pre_analysis import analyze_timeseries

class TestPreAnalysis(unittest.TestCase):
    def test_analyze_timeseries_segments(self):
        # Create 100 days
        idx = pd.date_range('2000-01-01', periods=100, freq='D')

        # All valid initially
        df = pd.DataFrame({
            'Date': idx,
            'T_air': np.random.rand(100) * 10 + 10,
            'T_water': np.random.rand(100) * 5 + 5,
            'Discharge': np.random.rand(100) * 50 + 10
        })

        # Introduce a gap in T_air from day 40 to 49 (10 days missing)
        df.loc[40:49, 'T_air'] = -999.0

        # Introduce a 0 discharge value at day 80 (should be treated as gap)
        df.loc[80, 'Discharge'] = 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            out_plot = os.path.join(tmpdir, "plot.png")
            out_report = os.path.join(tmpdir, "report.txt")

            summary, text = analyze_timeseries(
                df,
                output_plot_path=out_plot,
                output_summary_path=out_report,
                gap_tolerant=True,
                min_segment_days=10
            )

            # Assert segments
            # Segment 1: days 0-39 (40 days) -> Valid
            # Gap: days 40-49 (10 days) -> Gap
            # Segment 2: days 50-79 (30 days) -> Valid
            # Gap: day 80 (1 day) -> Gap
            # Segment 3: days 81-99 (19 days) -> Valid

            self.assertEqual(summary['valid_segments_count'], 3)
            self.assertEqual(summary['too_short_segments_count'], 0)
            self.assertEqual(summary['gap_segments_count'], 2)

            # total valid days: 40 + 30 + 19 = 89
            self.assertEqual(summary['total_valid_days'], 89)

            # Check files were created
            self.assertTrue(os.path.exists(out_plot))
            self.assertTrue(os.path.exists(out_report))

    def test_analyze_timeseries_missing_columns(self):
        df = pd.DataFrame({
            'Date': pd.date_range('2000-01-01', periods=10, freq='D'),
            'T_air': np.random.rand(10) * 10 + 10,
            # Missing 'T_water' and 'Discharge'
        })

        # It doesn't actually raise an error, just skips creating keys for missing stats
        summary, _ = analyze_timeseries(df)
        self.assertNotIn('T_water', summary['missing_stats'])

    def test_analyze_timeseries_gap_intolerant(self):
        idx = pd.date_range('2000-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Date': idx,
            'T_air': np.random.rand(10) * 10 + 10,
            'T_water': np.random.rand(10) * 5 + 5,
            'Discharge': np.random.rand(10) * 50 + 10
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            out_plot = os.path.join(tmpdir, "plot.png")
            out_report = os.path.join(tmpdir, "report.txt")

            summary, _ = analyze_timeseries(
                df,
                output_plot_path=out_plot,
                output_summary_path=out_report,
                gap_tolerant=False
            )

            # Since we did not modify core logic to behave differently for gap_tolerant=False
            # (as requested by the reviewer), we just check that it runs without errors
            # and returns a valid summary dictionary.
            self.assertIn('total_days', summary)

    def test_analyze_timeseries_all_segments_too_short(self):
        idx = pd.date_range('2000-01-01', periods=20, freq='D')
        df = pd.DataFrame({
            'Date': idx,
            'T_air': np.random.rand(20) * 10 + 10,
            'T_water': np.random.rand(20) * 5 + 5,
            'Discharge': np.random.rand(20) * 50 + 10
        })

        # Introduce a gap every 5 days, so segments are length 4
        df.loc[4, 'T_air'] = -999.0
        df.loc[9, 'T_air'] = -999.0
        df.loc[14, 'T_air'] = -999.0
        df.loc[19, 'T_air'] = -999.0

        with tempfile.TemporaryDirectory() as tmpdir:
            out_plot = os.path.join(tmpdir, "plot.png")
            out_report = os.path.join(tmpdir, "report.txt")

            summary, _ = analyze_timeseries(
                df,
                output_plot_path=out_plot,
                output_summary_path=out_report,
                gap_tolerant=True,
                min_segment_days=10 # All segments are 4 days, so they should be rejected
            )

            self.assertEqual(summary['valid_segments_count'], 0)
            self.assertEqual(summary['too_short_segments_count'], 4)
            self.assertEqual(summary['total_valid_days'], 0)

if __name__ == '__main__':
    unittest.main()
