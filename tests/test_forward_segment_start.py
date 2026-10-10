"""
Gap-tolerant FORWARD runs start each segment from the water temperature that matches its
first day's conditions (the equation at rest: A/B), not from measurements: a scenario file
needs no water temperature, its start does not come from measurements made under other
conditions, and paired runs start alike. Calibration and validation are unchanged.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

import pyair2stream
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import call_model, equilibrium_temperature
from pyair2stream.model_numba import fast_rk_version

PAR = [1, 0.5, 0.5, 0.2, 1, 1, 0.5, 0.5]
GAP = slice(300, 311)


def _scenario(tmp_path, water=False, warmer=0.0, name="scen"):
    t = np.arange(730)
    df = pd.DataFrame({"Date": pd.date_range("2001-01-01", periods=730).strftime("%Y-%m-%d"),
                       "T_air": 10 + 8 * np.sin(2 * np.pi * (t - 110) / 365) + warmer,
                       "Discharge": 2 + np.cos(2 * np.pi * t / 365)})
    if water:
        df["T_water"] = 25.0                                  # far from the model, on purpose
    df.loc[df.index[GAP], "T_air"] = np.nan                  # rows 300-310
    df.to_csv(tmp_path / f"{name}.csv", index=False)
    cfg = {"version": 8, "run_mode": "FORWARD", "gap_tolerant": True, "Qmedia": 2.0, "parameters_forward": PAR,
           "paths": {"input_data": str(tmp_path / f"{name}.csv"), "output_dir": str(tmp_path / name)}}
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    return str(tmp_path / f"{name}.yaml")


def _simulate(config):
    data = read_calibration(config)
    read_Tseries(data, "c")
    call_model(data)
    return data


@pytest.mark.parametrize("water", [False, True])
def test_each_segment_starts_at_rest_under_its_first_days_conditions(tmp_path, water):
    data = _simulate(_scenario(tmp_path, water=water))
    p = np.concatenate([[0.0], data.par])
    assert len(data.segments) == 2
    for start, _ in data.segments:
        start_value = data.Twat_mod[start]
        assert start_value == pytest.approx(equilibrium_temperature(data, start, p))
        assert start_value != 25.0                               # not the measurement
        rate = fast_rk_version(8, *p[1:], data.Tair[start], data.Q[start], start_value, data.tt[start], 2.0)
        assert rate == pytest.approx(0.0, abs=1e-9)              # the equation is at rest there


def test_a_scenario_without_water_temperature_runs_and_marks_its_warm_up_days(tmp_path):
    result = pyair2stream.run(_scenario(tmp_path), verbose=False)
    out = [f for f in result.files if f.startswith("2_")][0]
    df = pd.read_csv(f"{result.output_dir}/{out}")
    assert df.warm_up.sum() == 2 * 15
    assert df.warm_up.iloc[:15].all() and df.warm_up.iloc[311:326].all() and not df.warm_up.iloc[326:].any()


def test_paired_runs_start_alike_whatever_was_measured(tmp_path):
    # The same scenario with and without a water temperature column gives the same simulation.
    a = _simulate(_scenario(tmp_path, water=False, name="a"))
    b = _simulate(_scenario(tmp_path, water=True, name="b"))
    np.testing.assert_array_equal(a.Twat_mod, b.Twat_mod)
