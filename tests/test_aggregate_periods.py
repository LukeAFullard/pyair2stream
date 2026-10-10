"""
scenario.aggregate computes a period only when enough of its days have a simulated value:
by default all of them. A partial week or month (at either end of the file, or across a
gap) has no value, with a warning, since a 3-day mean is not a 7-day mean.
"""

import numpy as np
import pandas as pd
import pytest

from pyair2stream import scenario


def test_a_partial_last_block_has_no_value(capsys):
    dates = pd.date_range("2003-07-01", periods=10)
    ens = np.tile(np.arange(10.0), (2, 1))
    weekly, periods = scenario.aggregate(ens, dates, freq="7D", return_periods=True)
    assert weekly[:, 0] == pytest.approx(3.0) and np.isnan(weekly[:, 1]).all()
    assert list(periods) == list(pd.to_datetime(["2003-07-01", "2003-07-08"]))
    assert "1 of 2 period(s) (7D) have fewer than all their days" in capsys.readouterr().out
    # min_days accepts the 3-day block, from the days it has.
    assert scenario.aggregate(ens, dates, freq="7D", min_days=3)[0] == pytest.approx([3.0, 8.0])
    assert np.isnan(scenario.aggregate(ens, dates, freq="7D", min_days=4)[0, 1])


def test_calendar_months_count_their_own_lengths(capsys):
    dates = pd.date_range("2003-07-20", "2003-09-10")
    monthly, periods = scenario.aggregate(np.ones((1, len(dates))), dates, how="sum", freq="MS",
                                          return_periods=True)
    assert np.isnan(monthly[0, 0]) and monthly[0, 1] == 31.0 and np.isnan(monthly[0, 2])
    assert list(periods.month) == [7, 8, 9]
    # February of a leap year needs its 29 days.
    dates = pd.date_range("2004-01-01", "2004-12-31")
    monthly = scenario.aggregate(np.ones((1, len(dates))), dates, how="sum", freq="MS")
    assert monthly[0].tolist() == list(dates.days_in_month[dates.is_month_start])
    assert capsys.readouterr().out.count("Warning") == 1


def test_a_period_with_a_gap_has_no_value_in_that_series_only():
    dates = pd.date_range("2001-01-01", periods=14)
    ens = np.ones((2, 14))
    ens[1, 3] = np.nan
    weekly = scenario.aggregate(ens, dates)
    assert weekly[0].tolist() == [1.0, 1.0]
    assert np.isnan(weekly[1, 0]) and weekly[1, 1] == 1.0


def test_complete_periods_are_unchanged(capsys):
    dates = pd.date_range("2020-01-01", periods=14)
    ens = np.random.default_rng(0).normal(size=(3, 14))
    expected = ens.reshape(3, 2, 7).mean(axis=2)
    assert scenario.aggregate(ens, dates) == pytest.approx(expected)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bad", [dict(freq="12h"), dict(min_days=0), dict(min_days=2.5)])
def test_invalid_settings_are_refused(bad):
    with pytest.raises(ValueError):
        scenario.aggregate(np.ones((1, 14)), pd.date_range("2020-01-01", periods=14), **bad)
