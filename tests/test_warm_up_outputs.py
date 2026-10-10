"""
The unscored warm-up days (`warm_up` = 1: the first warmup_drop_days of each gap-tolerant
segment, or of a file shorter than a year) are left out of the mean error by month, as from
the scores, and are marked in the gap-filled file.
"""

import glob
import os

import pandas as pd

import pyair2stream
from tests.test_zero_flow_gap_tolerant import _config


def test_bias_by_month_leaves_out_the_warm_up_days_and_the_filled_file_marks_them(tmp_path):
    # Two segments: 2001-01-01 to 2001-07-19 and from 2001-08-14 (25 dry days in between).
    result = pyair2stream.run(_config(tmp_path), verbose=False)
    out = result.output_dir
    sim = pd.read_csv(glob.glob(os.path.join(out, "2_*.csv"))[0])
    assert sim.warm_up.sum() == 30
    scored = sim[(sim.warm_up == 0) & (sim.Twat_obs != -999) & (sim.Twat_mod != -999)]
    bias = pd.read_csv(glob.glob(os.path.join(out, "bias_by_month_calibration_*.csv"))[0]).set_index("period")
    # January 2001 loses its first 15 days (and still counts: the record covers it); 2002 and 2003 are whole.
    assert bias.loc["Jan", "n_days"] == (scored.Month == 1).sum() == 16 + 31 + 31
    filled = pd.read_csv(os.path.join(out, "filled_water_temperature_calibration.csv"))
    assert filled.warm_up.tolist() == sim.warm_up.tolist()
