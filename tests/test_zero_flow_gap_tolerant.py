"""
Zero or negative discharge in gap-tolerant mode (versions 4, 7 and 8 cannot simulate a day
without flow).

- A calibration (or validation) file: the days are treated as gaps, and the run says so,
  marks them in Q_gap and counts them in gaps_summary.txt.
- A FORWARD scenario file: the run stops, because the scenario's answer is about those days.
- With min_theta_floor set, the days are simulated (at the floor) instead.
- Counting functions warn about days without a simulated value, and a paired difference
  refuses two runs that simulated different days.
"""

import os

import numpy as np
import pandas as pd
import pytest
import yaml

import pyair2stream
from pyair2stream import scenario
from pyair2stream.config import CommonData
from pyair2stream.data_checks import check_table
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import find_segments

BOUNDS = {"min": [-5, -5, -5, -1, 0, 0, 0, -1], "max": [15, 1.5, 5, 1, 20, 10, 1, 5]}
DRY = slice(200, 225)                     # 25 days without flow


def _table(n=800, start="2001-01-01"):
    t = np.arange(n)
    q = 2.0 + np.cos(2 * np.pi * t / 365.0)
    q[DRY] = 0.0
    return pd.DataFrame({
        "Date": pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d"),
        "T_air": 8.0 + 10.0 * np.sin(2 * np.pi * (t - 100) / 365.0),
        "T_water": 8.0 + 6.0 * np.sin(2 * np.pi * (t - 110) / 365.0),
        "Discharge": q,
    })


def _config(tmp_path, **kw):
    _table().to_csv(tmp_path / "in.csv", index=False)
    cfg = {"version": 8, "run_mode": "DE", "random_seed": 1, "gap_tolerant": True,
           "paths": {"input_data": str(tmp_path / "in.csv"), "output_dir": str(tmp_path / "out")},
           "parameter_bounds": BOUNDS, "optimization": {"n_run": 2, "n_particles": 4}, **kw}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_the_data_check_explains_what_happens_to_zero_flow_days():
    df = _table()
    cal = check_table(df, "f.csv", gap_tolerant=True)
    assert cal.errors == []
    assert "25 day(s) (first: 2001-07-20, line 202)" in cal.warnings[0]
    assert "treated as gaps: they are not simulated or scored" in cal.warnings[0]
    fwd = check_table(df, "f.csv", period="scenario", gap_tolerant=True)
    assert "would have no water temperature on exactly these days" in fwd.errors[0]
    for kw in ({"min_theta_floor": 1e-6}, {"version": 5}):
        assert check_table(df, "f.csv", gap_tolerant=True, **kw).problems == []
        assert check_table(df, "f.csv", period="scenario", gap_tolerant=True, **kw).errors == []


def test_with_min_theta_floor_the_days_are_simulated_not_gaps():
    data = CommonData()
    data.version, data.n_tot = 8, 365 + 800
    data.Tair = np.full(data.n_tot, 10.0)
    data.Q = np.concatenate([np.full(365, 2.0), _table().Discharge.to_numpy()])
    data.min_theta_floor = None
    assert find_segments(data, 30)[0] == [(365, 365 + 199), (365 + 225, data.n_tot - 1)]
    data.min_theta_floor = 1e-6
    assert find_segments(data, 30)[0] == [(365, data.n_tot - 1)]


def test_a_forward_run_stops_on_zero_flow_days(tmp_path):
    data = read_calibration(_config(tmp_path, run_mode="FORWARD", Qmedia=2.0,
                                    parameters_forward=[1, 0.5, 0.5, 0.2, 1, 1, 0.5, 0.5]))
    with pytest.raises(ValueError, match="Set min_theta_floor to simulate them"):
        read_Tseries(data, "c")


def test_a_calibration_says_so_and_marks_and_counts_the_days(tmp_path):
    result = pyair2stream.run(_config(tmp_path), verbose=False)
    assert any("25 day(s)" in m and "treated as gaps" in m for m in result.messages)
    assert "treated as gaps" in open(result.summary, encoding="utf-8").read()
    sim = pd.read_csv([os.path.join(result.output_dir, f) for f in result.files if f.startswith("2_")][0])
    assert (sim.Q_gap.iloc[DRY] == 1).all() and (sim.Twat_mod.iloc[DRY] == -999).all()
    gaps = open(os.path.join(result.output_dir, "gaps_summary.txt")).read()
    assert "of which zero or negative discharge (treated as gaps; set min_theta_floor to simulate them): 25" in gaps


def test_counting_functions_warn_about_days_without_a_value(capsys):
    ens = np.full((3, 10), 20.0)
    ens[:, 4:6] = np.nan
    assert list(scenario.exceedance(ens, 18.0)) == [8, 8, 8]
    assert "2 day(s) have no simulated value" in capsys.readouterr().out
    scenario.aggregate(ens, pd.date_range("2001-01-01", periods=10), freq="5D")
    assert "uses only the days that have one" in capsys.readouterr().out
    scenario.exceedance(np.full((3, 10), 20.0), 18.0)
    assert capsys.readouterr().out == ""


def test_a_paired_difference_refuses_runs_that_simulated_different_days():
    base = np.full((3, 10), 20.0)
    dry = base.copy()
    dry[:, 4:6] = np.nan
    with pytest.raises(ValueError, match="2 day\\(s\\) are simulated in one and not the other"):
        scenario.paired_difference(dry, base)
    assert np.isnan(scenario.paired_difference(dry, dry)[:, 4:6]).all()
