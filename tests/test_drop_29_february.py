"""
calendar: "noleap" refuses a file with 29 February unless drop_29_february: true, which removes
those rows with a warning that counts them and the water-temperature measurements they held.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from pyair2stream.data_checks import check_table
from pyair2stream.io import read_calibration, read_Tseries


def _table(start="2003-01-01", n=800):
    d = pd.date_range(start, periods=n)
    t = np.arange(n)
    return pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "T_air": 8 + 10 * np.sin(2 * np.pi * (t - 100) / 365),
                         "T_water": 8 + 6 * np.sin(2 * np.pi * (t - 110) / 365), "Discharge": 2.0})


def test_noleap_refuses_29_february_by_default_and_names_the_option():
    checked = check_table(_table(), "f.csv", calendar="noleap")
    assert "29 February" in checked.errors[0] and "drop_29_february: true" in checked.errors[0]


def test_drop_29_february_removes_the_rows_with_a_warning():
    df = _table()
    checked = check_table(df, "f.csv", calendar="noleap", drop_29_february=True)
    assert checked.errors == []
    assert "Removed 1 row(s) dated 29 February from f.csv (drop_29_february), with 1 water-temperature" \
        in checked.warnings[0]
    assert len(checked.df) == len(df) - 1 and not (checked.dates.dt.strftime("%m-%d") == "02-29").any()
    # The option does nothing with the standard calendar.
    assert len(check_table(df, "f.csv", drop_29_february=True).df) == len(df)


def test_later_messages_still_name_the_lines_of_the_file():
    df = _table()
    df["T_air"] = df["T_air"].astype(object)
    df.loc[500, "T_air"] = "n.a."              # after 29 February 2004 (row 424)
    checked = check_table(df, "f.csv", calendar="noleap", drop_29_february=True)
    assert "on line 502)" in checked.errors[0]


def test_a_run_with_the_option(tmp_path):
    _table().to_csv(tmp_path / "in.csv", index=False)
    cfg = {"version": 8, "run_mode": "FORWARD", "calendar": "noleap", "drop_29_february": True, "Qmedia": 2.0,
           "parameters_forward": [0.5, 0.6, 0.6, 0.0, 0.0, 0.0, 0.5, 0.0],
           "paths": {"input_data": str(tmp_path / "in.csv"), "output_dir": str(tmp_path / "out")}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    data = read_calibration(str(tmp_path / "c.yaml"))
    read_Tseries(data, "c")
    assert data.n_tot == 365 + 799
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({**cfg, "drop_29_february": "yes"}))
    with pytest.raises(ValueError, match="drop_29_february must be true or false"):
        read_calibration(str(tmp_path / "c.yaml"))
