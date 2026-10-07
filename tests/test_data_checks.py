"""
Checks of input data (pyair2stream.data_checks), and that every way of loading data
uses them:

- a run checks every file it reads (calibration, validation, FORWARD scenario) with
  `check_table`, and names the file, the column and the first line of a problem;
- the validation file is checked before any calibration, not after it;
- `analyze_timeseries` reports exactly the problems a run would stop on;
- `merge_timeseries` reports the rows and readings it cannot use.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import pyair2stream
from pyair2stream import main as main_module
from pyair2stream.data_checks import check_table
from pyair2stream.io import read_calibration, read_Tseries, precheck_validation
from pyair2stream.preprocessing import merge_timeseries, read_and_resample

BOUNDS = {"min": [-5, -5, -5, -1, 0, 0, 0, -1], "max": [15, 1.5, 5, 1, 20, 10, 1, 5]}


def _table(n=400, start="2001-01-01"):
    t = np.arange(n)
    return pd.DataFrame({
        "Date": pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d"),
        "T_air": 8.0 + 10.0 * np.sin(2 * np.pi * (t - 100) / 365.0),
        "T_water": 8.0 + 6.0 * np.sin(2 * np.pi * (t - 110) / 365.0),
        "Discharge": 2.0 + np.cos(2 * np.pi * t / 365.0),
    })


def _errors(df, **kw):
    return check_table(df, "f.csv", **kw).errors


def _config(tmp_path, cal, val=None, **kw):
    cal_path = tmp_path / "cal.csv"
    cal.to_csv(cal_path, index=False)
    paths = {"input_data": str(cal_path), "output_dir": str(tmp_path / "out")}
    if val is not None:
        val.to_csv(tmp_path / "val.csv", index=False)
        paths["validation_data"] = str(tmp_path / "val.csv")
    cfg = {"version": 8, "run_mode": "DE", "random_seed": 1, "paths": paths, "parameter_bounds": BOUNDS,
           "optimization": {"n_run": 2, "n_particles": 2}, **kw}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


# --- check_table: one message per kind of problem, naming the file and the line ---------

def test_a_complete_table_passes():
    assert _errors(_table()) == []


def test_missing_date_names_the_first_missing_day():
    msg = _errors(_table().drop(index=[99, 100]).reset_index(drop=True))[0]
    assert "must be continuous at a daily time scale" in msg
    assert "2 date(s) have no row (first: 2001-04-10, after line 100)" in msg


def test_repeated_date_names_its_lines():
    df = _table()
    df = pd.concat([df.iloc[:11], df.iloc[10:]]).reset_index(drop=True)
    assert "2001-01-11 appears on lines 12, 13" in _errors(df)[0]


def test_dates_out_of_order_named():
    df = _table()
    df.iloc[[20, 21]] = df.iloc[[21, 20]].to_numpy()
    assert "2001-01-21 on line 23 comes after 2001-01-22 on line 22" in _errors(df)[0]


def test_blank_and_unreadable_dates_named_by_line():
    df = _table()
    df.loc[5, "Date"] = ""
    assert "Date is blank on line 7" in _errors(df)[0]
    df = _table()
    df.loc[58, "Date"] = "2001-02-30"
    assert "Date '2001-02-30' on line 60 of f.csv cannot be read" in _errors(df)[0]


def test_text_value_named_with_column_and_line():
    df = _table()
    df["T_air"] = df["T_air"].astype(object)
    df.loc[30, "T_air"] = "n.a."
    assert "Column T_air in f.csv has 1 value(s) that are not numbers (first: 'n.a.' on line 32)" in _errors(df)[0]
    df.loc[30, "T_air"] = "12,5"
    assert "decimal separator" in _errors(df)[0]


def test_column_name_with_a_space_explained():
    msg = _errors(_table().rename(columns={"T_air": "T_air "}))[0]
    assert "Missing 'T_air' column" in msg and "found 'T_air '" in msg


def test_water_temperature_required_for_calibration_and_validation_only():
    df = _table().drop(columns="T_water")
    assert "Missing 'T_water' column" in _errors(df)[0]
    assert "Missing 'T_water' column" in _errors(df, period="validation")[0]
    assert _errors(df, period="scenario") == []
    empty = _table().assign(T_water=np.nan)
    assert "has no water temperature measurements" in _errors(empty)[0]
    assert "remove paths.validation_data" in _errors(empty, period="validation")[0]
    assert _errors(empty, period="scenario") == []


def test_gaps_named_with_first_date_and_line():
    df = _table()
    df.loc[[40, 41], "T_air"] = np.nan
    df.loc[50, "Discharge"] = -999.0
    errors = _errors(df)
    assert "air temperature in f.csv must be complete: 2 day(s) have no value (first: 2001-02-10, line 42)" in errors[0]
    assert "discharge in f.csv must be complete: 1 day(s) have no value (first: 2001-02-20, line 52)" in errors[1]
    assert _errors(df, gap_tolerant=True) == []
    assert not any("discharge" in e for e in _errors(df, version=5))


def test_nonpositive_discharge_as_a_run_treats_it():
    df = _table()
    df.loc[60, "Discharge"] = 0.0
    assert "Non-positive discharge (Q <= 0) in f.csv: 1 day(s) (first: 2001-03-02, line 62)" in _errors(df)[0]
    for kw in ({"gap_tolerant": True}, {"version": 5}, {"min_theta_floor": 1e-6}):
        assert _errors(df, **kw) == []


def test_start_date_and_length():
    late = _table(start="2001-01-02")
    assert "must start on January 1st (it starts on 2001-01-02)" in _errors(late)[0]
    assert _errors(late, period="scenario") == []
    assert _errors(late, gap_tolerant=True) == []
    short = _table(n=300)
    assert "The calibration time series in f.csv has only 300 day(s)" in _errors(short)[0]
    assert "The scenario (FORWARD) time series in f.csv has only 300 day(s)" in _errors(short, period="scenario")[0]
    checked = check_table(short, "f.csv", period="validation")
    assert checked.errors == [] and "validation will be skipped" in checked.warnings[0]


def test_implausible_values_warn_with_date_and_line():
    df = _table()
    df.loc[10, "T_water"] = -99.0
    checked = check_table(df, "f.csv")
    assert checked.errors == []
    assert "1 value(s) of T_water in f.csv are outside -2 to 50 degC (first: -99 on 2001-01-11, line 12)" \
        in checked.warnings[0]


def test_nonstandard_calendar_checks_only_the_order():
    df = _table().drop(index=[99]).reset_index(drop=True)
    assert _errors(df, calendar="noleap") == []
    df.iloc[[20, 21]] = df.iloc[[21, 20]].to_numpy()
    assert "must have non-decreasing dates" in _errors(df, calendar="noleap")[0]


# --- A run uses the same checks for every file --------------------------------------------

@pytest.mark.parametrize("period, p, run_mode", [("calibration", "c", "DE"), ("scenario", "c", "FORWARD")])
def test_read_Tseries_raises_the_checkers_first_error(tmp_path, period, p, run_mode):
    bad = _table().drop(index=[99]).reset_index(drop=True)
    extra = {"parameters_forward": [1, 0.5, 0.5, 0, 0, 0, 0, 0], "Qmedia": 2.0} if run_mode == "FORWARD" else {}
    data = read_calibration(_config(tmp_path, bad, run_mode=run_mode, **extra))
    with pytest.raises(ValueError) as info:
        read_Tseries(data, p)
    assert str(info.value) == _errors(bad, period=period).pop(0).replace("f.csv", str(tmp_path / "cal.csv"))


def test_a_bad_validation_file_stops_the_run_before_any_calibration(tmp_path, monkeypatch):
    bad_val = _table(start="2002-01-01").drop(index=[99]).reset_index(drop=True)
    monkeypatch.setattr(main_module, "run_optimizer",
                        lambda data: pytest.fail("the calibration ran before the validation file was checked"))
    monkeypatch.setattr(sys, "argv", ["pyair2stream", "--config", _config(tmp_path, _table(), bad_val)])
    with pytest.raises(ValueError, match="val.csv must be continuous at a daily time scale"):
        main_module.main()


def test_a_missing_validation_file_stops_the_run_before_any_calibration(tmp_path, monkeypatch):
    cfg = yaml.safe_load(open(_config(tmp_path, _table())))
    cfg["paths"]["validation_data"] = str(tmp_path / "typo.csv")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(main_module, "run_optimizer", lambda data: pytest.fail("calibrated first"))
    monkeypatch.setattr(sys, "argv", ["pyair2stream", "--config", str(tmp_path / "config.yaml")])
    with pytest.raises(FileNotFoundError, match="typo.csv"):
        main_module.main()


def test_validation_warnings_are_printed_once(tmp_path, monkeypatch, capsys):
    val = _table(start="2002-01-01")
    val.loc[10, "T_water"] = -99.0
    monkeypatch.setattr(sys, "argv", ["pyair2stream", "--config", _config(tmp_path, _table(), val)])
    main_module.main()
    assert capsys.readouterr().out.count("value(s) of T_water") == 1


def test_overlap_of_validation_and_calibration_is_reported(tmp_path, capsys):
    cal = _table()
    data = read_calibration(_config(tmp_path, cal, cal.iloc[:370]))
    read_Tseries(data, "c")
    precheck_validation(data)
    out = capsys.readouterr().out
    assert "370 measured day(s) of the validation file" in out and "also measured days of the calibration" in out

    separate = read_calibration(_config(tmp_path, cal, _table(start="2003-01-01")))
    read_Tseries(separate, "c")
    precheck_validation(separate)
    assert "also measured days" not in capsys.readouterr().out


def test_cross_validation_says_the_validation_file_is_not_used(tmp_path, capsys):
    data = read_calibration(_config(tmp_path, _table(), cross_validation={"enabled": True}))
    data._input_data_path_val = str(tmp_path / "never_read.csv")
    precheck_validation(data)
    assert "not used by a cross-validation run" in capsys.readouterr().out


# --- analyze_timeseries reports what a run would stop on -----------------------------------

def test_analyze_timeseries_reports_the_runs_errors():
    bad = _table().drop(index=range(100, 110)).reset_index(drop=True)
    summary, report = pyair2stream.analyze_timeseries(bad, source="f.csv")
    assert summary["run_would_stop"]
    assert summary["errors"] == _errors(bad)
    assert summary["missing_dates"] == 10
    assert summary["missing_stats"]["T_air"]["missing_count"] == 10
    assert "A run would STOP on this data" in report

    summary, report = pyair2stream.analyze_timeseries(_table(), source="f.csv")
    assert not summary["run_would_stop"] and "A run would accept this data." in report


def test_analyze_timeseries_uses_the_runs_settings():
    df = _table()
    df.loc[[40, 41], "T_air"] = np.nan
    assert pyair2stream.analyze_timeseries(df)[0]["run_would_stop"]
    assert not pyair2stream.analyze_timeseries(df, gap_tolerant=True)[0]["run_would_stop"]


# --- merge_timeseries reports what it cannot use ---------------------------------------------

def _hourly(tmp_path, name, days=10, drop_hours_on=None, extra=None):
    t = pd.date_range("2001-01-01", periods=24 * days, freq="h")
    df = pd.DataFrame({"time": t.strftime("%Y-%m-%d %H:%M"), "value": 10.0 + np.sin(np.arange(len(t)) / 24 * 2 * np.pi)})
    if drop_hours_on is not None:
        day = pd.Timestamp(drop_hours_on)
        df = df[(pd.to_datetime(df.time).dt.normalize() != day) | (pd.to_datetime(df.time).dt.hour == 12)]
    if extra:
        extra(df)
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return {"file_path": str(path), "date_col": "time", "value_col": "value", "standard_col_name": "T_air"}


def test_merge_reports_unreadable_times(tmp_path, capsys):
    cfg = _hourly(tmp_path, "a", extra=lambda d: d.__setitem__("time", d["time"].where(d.index != 5, "not a date")))
    out = merge_timeseries([cfg])
    assert "1 row(s) of" in capsys.readouterr().out and len(out) == 10


def test_merge_treats_minus999_as_missing(tmp_path, capsys):
    def code(d):
        d.loc[d.index[:24], "value"] = -999.0
    out = merge_timeseries([_hourly(tmp_path, "b", extra=code)])
    assert "24 value(s) of -999" in capsys.readouterr().out
    assert np.isnan(out.T_air.iloc[0]) and out.T_air.iloc[1] > 0


def test_merge_reports_days_with_few_readings_and_can_blank_them(tmp_path, capsys):
    cfg = _hourly(tmp_path, "c", drop_hours_on="2001-01-04")
    out = merge_timeseries([cfg])
    assert "fewer than half the usual number of readings (usual: 24 a day; first: 2001-01-04 with 1)" \
        in capsys.readouterr().out
    assert np.isfinite(out.T_air.iloc[3])
    out = merge_timeseries([cfg], min_readings_per_day=20)
    assert np.isnan(out.T_air.iloc[3]) and "left blank" in capsys.readouterr().out


def test_merge_stops_on_text_values_and_accepts_named_codes(tmp_path):
    def text(d):
        d["value"] = d["value"].astype(object)
        d.loc[d.index[7], "value"] = "ERR"
    cfg = _hourly(tmp_path, "d", extra=text)
    with pytest.raises(ValueError, match=r"not numbers \(first: 'ERR' on line 9\).*na_values"):
        merge_timeseries([cfg])
    out = merge_timeseries([{**cfg, "na_values": ["ERR"]}])
    assert out.T_air.notna().all()


def test_read_and_resample_names_a_missing_column(tmp_path):
    cfg = _hourly(tmp_path, "e")
    with pytest.raises(ValueError, match="Column 'temp' not found"):
        read_and_resample(cfg["file_path"], "time", "temp", "T_air")
