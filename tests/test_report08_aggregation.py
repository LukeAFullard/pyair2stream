"""
Tests for docs/audit/08_testing_gaps.md, Gap C (aggregation coverage).

Every other test in this suite uses `time_res = '1d'`, leaving the entire
`unit == 'w'` and `unit == 'm'` branches of `model.aggregation()` (hand-
translated 1-based-to-0-based index arithmetic) unexercised. This file:

- Parametrizes aggregation over `{'1d', '1w', '2w', '1m'}` and checks each
  window's mean against an independent pandas computation (8.3).
- Checks `read_calibration` rejects a malformed `time_resolution` with a
  clear error instead of failing deep inside `aggregation()` (8.3).
- Documents/tests the trailing-partial-month behaviour: a short trailing
  fragment is accepted as a full period because `prc` is compared against
  the partial period's own day count, not a full period's (8.3).
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from pyair2stream.config import CommonData
from pyair2stream.io import read_calibration
from pyair2stream.model import aggregation


def _build_aggregation_data(n_tot_raw: int, time_res: str, prc: float = 0.0,
                             start: str = '2001-01-01') -> CommonData:
    """
    A `CommonData` with a complete (no missing values), real-calendar-dated
    `Twat_obs` series, wired up just enough to exercise `aggregation()`
    directly (no physics simulation needed -- `aggregation()` only reads
    `Twat_obs`/`date`/`eval_mask`/`n_tot`/`time_res`/`prc`).
    """
    n_tot = n_tot_raw + 365
    data = CommonData()
    data.n_tot = n_tot
    data.time_res = time_res
    data.prc = np.float64(prc)

    dates = pd.date_range(start, periods=n_tot_raw, freq='D')
    data.date = np.zeros((n_tot, 3), dtype=np.int32)
    data.date[365:, 0] = dates.year.values
    data.date[365:, 1] = dates.month.values
    data.date[365:, 2] = dates.day.values

    day_idx = np.arange(n_tot_raw)
    values = 10.0 + 5.0 * np.sin(2.0 * np.pi * day_idx / 37.0) + 0.01 * day_idx
    data.Twat_obs = np.full(n_tot, -999.0, dtype=np.float64)
    data.Twat_obs[365:] = values

    # Every real day is eval-eligible: no warm-up (excluded separately by
    # aggregation()'s own `i in range(365, n_tot)`) and no gap-tolerant
    # warmup_drop_days here.
    data.eval_mask = np.zeros(n_tot, dtype=np.bool_)
    data.eval_mask[365:] = True

    return data, dates, values


def test_aggregation_1d_matches_raw_series():
    n_tot_raw = 30
    data, dates, values = _build_aggregation_data(n_tot_raw, '1d')

    aggregation(data)

    assert data.n_dat == n_tot_raw
    agg_values = np.array([data.Twat_obs_agg[data.I_inf[i, 2]] for i in range(data.n_dat)])
    np.testing.assert_allclose(agg_values, values)


@pytest.mark.parametrize('time_res,n_days', [('1w', 7), ('2w', 14)])
def test_aggregation_weekly_matches_pandas_resample(time_res, n_days):
    """
    Weekly aggregation windows are fixed-length `n_days`-day chunks starting
    at the first real day (docs/audit/08, 8.3). Cross-checked against an
    independent pandas `resample(..., origin='start')` computation, which
    chunks the same way.
    """
    n_tot_raw = 100
    data, dates, values = _build_aggregation_data(n_tot_raw, time_res, prc=0.0)

    aggregation(data)

    series = pd.Series(values, index=dates)
    # No `origin=` needed: pandas' default ('start_day') already aligns to
    # midnight of the series' first timestamp for a plain day-based freq,
    # matching aggregation()'s own fixed-step chunking from the first real day.
    expected = series.resample(f'{n_days}D').mean()

    assert data.n_dat == len(expected)
    for i in range(data.n_dat):
        actual = data.Twat_obs_agg[data.I_inf[i, 2]]
        np.testing.assert_allclose(actual, expected.iloc[i], rtol=1e-12,
                                    err_msg=f"window {i} mismatch for time_res={time_res}")


def test_aggregation_monthly_matches_pandas_groupby():
    """Monthly windows align to real calendar-month boundaries (docs/audit/08, 8.3)."""
    n_tot_raw = 730  # 2 full-ish years, well beyond a handful of months
    data, dates, values = _build_aggregation_data(n_tot_raw, '1m', prc=0.0)

    aggregation(data)

    series = pd.Series(values, index=dates)
    expected = series.groupby([dates.year, dates.month]).mean()

    assert data.n_dat == len(expected)
    for i in range(data.n_dat):
        actual = data.Twat_obs_agg[data.I_inf[i, 2]]
        np.testing.assert_allclose(actual, expected.iloc[i], rtol=1e-12,
                                    err_msg=f"month window {i} mismatch")


def test_trailing_partial_month_accepted_as_full_period():
    """
    A trailing partial month is accepted as a full month because `prc` is
    compared against the *partial* period's own day count (`n_days` inside
    the monthly loop, reset per calendar month), not a full ~30-day month's
    (`model.py` aggregation, `unit == 'm'` branch). This is Fortran-equivalent
    and correct-as-ported, but easy to miss -- documented here rather than
    discovered (docs/audit/08_testing_gaps.md, 8.3).

    4 days into a new month, all present, passes even a demanding
    `prc=0.9`: 4 >= 4*0.9, even though 4 days is nowhere near 90% of a full
    calendar month.
    """
    # 2 full months (Jan, Feb 2001 non-leap) plus a 4-day fragment into March.
    n_tot_raw = 31 + 28 + 4
    data, dates, values = _build_aggregation_data(n_tot_raw, '1m', prc=0.9)

    aggregation(data)

    # Three windows: Jan, Feb, and the 4-day March fragment -- not dropped.
    assert data.n_dat == 3
    last_window_date = data.date[data.I_inf[2, 2]]
    assert last_window_date[1] == 3  # March

    march_values = values[-4:]
    actual = data.Twat_obs_agg[data.I_inf[2, 2]]
    np.testing.assert_allclose(actual, np.mean(march_values), rtol=1e-12)


class TestTimeResolutionValidation:
    """8.3: `time_resolution` is validated in `read_calibration`, not deep inside `aggregation()`."""

    def setUp_config(self, tmp_path, time_resolution):
        input_csv = tmp_path / 'input.csv'
        n_days = 366
        dates = pd.date_range('2001-01-01', periods=n_days, freq='D')
        pd.DataFrame({
            'Date': dates,
            'T_air': np.full(n_days, 15.0),
            'Discharge': np.full(n_days, 10.0),
        }).to_csv(input_csv, index=False)

        config = {
            'project_name': 'p', 'station_name': 'S', 'run_mode': 'FORWARD',
            'version': 8, 'time_resolution': time_resolution, 'Qmedia': 10.0,
            'parameters_forward': [1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1],
            'paths': {'input_data': str(input_csv), 'output_dir': str(tmp_path / 'out')},
        }
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        return str(config_path)

    @pytest.mark.parametrize('time_resolution', ['1d', '1w', '2w', '12w', '1m', '9m'])
    def test_valid_time_resolutions_accepted(self, tmp_path, time_resolution):
        config_path = self.setUp_config(tmp_path, time_resolution)
        data = read_calibration(config_file=config_path)
        assert data.time_res == time_resolution

    @pytest.mark.parametrize('time_resolution', ['daily', '2d', 'weekly', '1', 'w', '100w'])
    def test_invalid_time_resolutions_rejected_with_clear_error(self, tmp_path, time_resolution):
        config_path = self.setUp_config(tmp_path, time_resolution)
        with pytest.raises(ValueError, match='time_resolution'):
            read_calibration(config_file=config_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
