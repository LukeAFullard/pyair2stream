"""Mean error by month and season (post_processing.bias_by_month)."""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from pyair2stream.post_processing import bias_by_month, write_bias_by_month


def _series(year_offsets, july=0.0):
    """Daily obs/sim over whole years: error = the year's offset, plus `july` in July."""
    days = pd.date_range(f"{2001}-01-01", f"{2000 + len(year_offsets)}-12-31")
    obs = 10 + 5 * np.sin(np.arange(len(days)) / 58.0)
    offset = np.array([year_offsets[d.year - 2001] for d in days])
    sim = obs + offset + np.where(days.month == 7, july, 0.0)
    return days, obs, sim


def test_monthly_bias_is_the_mean_of_yearly_values_with_a_t_interval():
    offsets = [0.1, 0.3, -0.1]
    days, obs, sim = _series(offsets, july=0.5)
    table, yearly = bias_by_month(days, obs, sim)
    table = table.set_index("period")
    jul = np.array(offsets) + 0.5
    assert table.loc["Jul", "n_years"] == 3
    assert np.isclose(table.loc["Jul", "bias"], jul.mean())
    half = student_t.ppf(0.975, 2) * jul.std(ddof=1) / np.sqrt(3)
    assert np.isclose(table.loc["Jul", "ci95_lower"], jul.mean() - half)
    assert np.isclose(table.loc["Jul", "ci95_upper"], jul.mean() + half)
    assert np.isclose(table.loc["Jan", "bias"], np.mean(offsets))
    # Seasons: Jun-Aug mixes one July-shifted month with two plain ones in each year.
    assert np.isclose(table.loc["Jun-Aug", "bias"], np.mean(offsets) + 0.5 * 31 / 92)
    assert table.loc["All year", "n_years"] == 3
    assert set(yearly.year) == {2001, 2002, 2003}


def test_the_interval_reflects_year_to_year_spread_not_the_number_of_days():
    # The same offset every day of a year (a whole-season error): many days, but only
    # three independent values, so the interval must be wide.
    days, obs, sim = _series([0.6, -0.6, 0.0])
    table = bias_by_month(days, obs, sim)[0].set_index("period")
    assert np.isclose(table.loc["Aug", "bias"], 0.0)
    assert table.loc["Aug", "ci95_lower"] < -1.0 < 1.0 < table.loc["Aug", "ci95_upper"]


def test_sparse_months_and_single_years():
    days, obs, sim = _series([0.2])
    obs = obs.copy()
    obs[(days.month == 3) & (days.day > 5)] = np.nan          # March: 5 days, below the minimum
    table = bias_by_month(days, obs, sim)[0].set_index("period")
    assert table.loc["Mar", "n_years"] == 0 and np.isnan(table.loc["Mar", "bias"])
    assert table.loc["Apr", "n_years"] == 1                    # one year: a bias but no interval
    assert np.isclose(table.loc["Apr", "bias"], 0.2) and np.isnan(table.loc["Apr", "ci95_lower"])


def test_write_bias_by_month_writes_table_and_figure(tmp_path):
    days, obs, sim = _series([0.1, 0.2], july=-0.4)
    table = write_bias_by_month(days, obs, sim, str(tmp_path), "bias_by_month_test", "Test")
    assert (tmp_path / "bias_by_month_test.csv").exists()
    assert (tmp_path / "bias_by_month_test.png").exists()
    saved = pd.read_csv(tmp_path / "bias_by_month_test.csv").set_index("period")
    assert np.isclose(saved.loc["Jul", "bias"], table.set_index("period").loc["Jul", "bias"], atol=1e-4)


def test_a_period_counts_only_if_the_record_covers_all_its_days():
    # A record from 19 July 2001 to 3 March 2004: July 2001 has 13 days, 2001 and 2004 are
    # partial, and the errors are seasonal (zero over any whole year).
    days = pd.date_range("2001-07-19", "2004-03-03")
    obs = np.full(len(days), 10.0)
    sim = obs + np.sin(2 * np.pi * (days.dayofyear.to_numpy() - 1) / 365.0)
    table, yearly = bias_by_month(days, obs, sim)
    table = table.set_index("period")
    assert table.loc["All year", "n_years"] == 2                  # 2002 and 2003 only
    assert set(yearly[yearly.period == "Jul"].year) == {2002, 2003}  # not the 13 days of July 2001
    assert set(yearly[yearly.period == "Jun-Aug"].year) == {2002, 2003}
    assert set(yearly[yearly.period == "Feb"].year) == {2002, 2003, 2004}
    assert 2004 not in set(yearly[yearly.period == "Mar"].year)
    # A noleap record (no 29 February) still has whole Februaries and years.
    noleap = pd.date_range("2004-01-01", "2005-12-31")
    noleap = noleap[~((noleap.month == 2) & (noleap.day == 29))]
    t2, y2 = bias_by_month(noleap, np.full(len(noleap), 10.0), np.full(len(noleap), 10.5))
    assert t2.set_index("period").loc["All year", "n_years"] == 2
