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
    for period in ("calibration", "scenario"):
        floored = check_table(df, "f.csv", period=period, gap_tolerant=True, min_theta_floor=1e-6)
        assert floored.errors == []
        assert "Zero discharge in f.csv: 25 day(s)" in floored.warnings[0] and "min_theta_floor" in floored.warnings[0]
        assert check_table(df, "f.csv", period=period, gap_tolerant=True, version=5).problems == []


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


def test_negative_discharge_is_an_error_whatever_the_settings():
    df = _table()
    df.loc[300:309, "Discharge"] = -9999.0                    # a missing-value code other than -999
    for kw in ({}, {"gap_tolerant": True}, {"min_theta_floor": 1e-6}, {"gap_tolerant": True, "min_theta_floor": 1e-6}):
        msg = check_table(df, "f.csv", **kw).errors[0]
        assert "Negative discharge in f.csv: 10 day(s) (first: -9999 on 2001-10-28, line 302)" in msg
        assert "write a missing value as -999 or leave the cell blank" in msg
    assert check_table(df, "f.csv", version=5).errors == []    # versions 3 and 5 do not use discharge


def test_a_forward_run_reports_every_day_outside_the_calibrated_flows(tmp_path, capsys):
    # The calibration saw theta from 0.5 to 1.5; the scenario has 25 zero-flow days run at the
    # floor (0.6% of 800 days would once have been under the 1% threshold, and floored days
    # were not counted at all) and one day of high flow.
    df = _table()
    df.loc[400, "Discharge"] = 3.4
    df.to_csv(tmp_path / "scenario.csv", index=False)
    meta = tmp_path / "calibration_metadata.json"
    meta.write_text('{"qmedia": 2.0, "theta_min": 0.5, "theta_max": 1.5, "version": 8, "integrator": "CRN", '
                    '"par_best": [1, 0.5, 0.5, 0.2, 1, 1, 0.5, 0.5]}')
    data = read_calibration(_config(tmp_path, run_mode="FORWARD", gap_tolerant=False, min_theta_floor=1e-6,
                                    paths={"input_data": str(tmp_path / "scenario.csv"), "output_dir": str(tmp_path / "o"),
                                           "calibration_metadata": str(meta)}))
    read_Tseries(data, "c")
    out = capsys.readouterr().out
    assert "26 day(s) (" in out and "of the days with discharge; first: 2001-07-20) have theta = Q/Qmedia outside" in out
    assert "(lowest 1e-06, highest 1.7), 25 of them zero-flow days run at min_theta_floor" in out


def test_theta_of_days_is_what_the_model_runs_with():
    from pyair2stream.io import theta_of_days
    q = np.array([2.0, 0.0, -999.0, 4.0])
    theta, rows = theta_of_days(q, 2.0, None)            # zero flow is not simulated: left out
    assert list(rows) == [0, 3] and list(theta) == [1.0, 2.0]
    theta, rows = theta_of_days(q, 2.0, 1e-6)            # simulated at the floor: included
    assert list(rows) == [0, 1, 3] and list(theta) == [1.0, 1e-6, 2.0]
