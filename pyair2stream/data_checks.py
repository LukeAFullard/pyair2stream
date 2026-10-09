"""
Checks of an input data table (Date, T_air, T_water, Discharge).

The same function checks every file a run reads (`io.read_Tseries`: the
calibration file, a FORWARD scenario file and the validation file) and the data
given to the pre-analysis report (`pre_analysis.analyze_timeseries`), so the report
can never pass a table that a run would reject.

Each problem names the file, the column and the first line concerned (line 1 is
the header row of the CSV file, so a table row i is on line i + 2).
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# Daily means outside these ranges (degC) are almost certainly not real, e.g. a
# missing-value code other than -999 (such as -99 or -9999) or a unit error.
PLAUSIBLE_RANGES = {'T_air': (-60.0, 60.0), 'T_water': (-2.0, 50.0)}

MISSING_CODE = -999.0
CALENDARS = ('standard', 'noleap')
NO_360_DAY = ("calendar '360_day' is not supported. A 360-day year has dates such as 30 February, which the "
              "package's dates, outputs and plots cannot hold, and relabelling the rows with ordinary dates "
              "would put months and years out of step with the model's seasons. Convert the file to the "
              "standard calendar first (for example with xarray's convert_calendar), and record how the added "
              "days were filled: USER_GUIDE.md §5.")
MIN_DAYS = 365          # calibration: a whole year, so the seasonal parameters can be fitted
MIN_SHORT_DAYS = 30     # validation and scenario files shorter than a year
PERIOD_NAMES = {'calibration': 'calibration', 'validation': 'validation', 'scenario': 'scenario (FORWARD)'}


@dataclass
class Problem:
    level: str        # 'error': a run stops; 'warning': a run continues
    message: str


@dataclass
class CheckedTable:
    """The table with T_air, T_water and Discharge as numbers (missing values as NaN),
    the parsed dates, and every problem found."""
    df: pd.DataFrame
    dates: Optional[pd.Series]
    problems: List[Problem] = field(default_factory=list)

    @property
    def errors(self) -> List[str]:
        return [p.message for p in self.problems if p.level == 'error']

    @property
    def warnings(self) -> List[str]:
        return [p.message for p in self.problems if p.level == 'warning']

    def raise_first_error(self) -> None:
        if self.errors:
            raise ValueError(self.errors[0])

    def print_warnings(self) -> None:
        for message in self.warnings:
            print(f"Warning: {message}")


def _line(i: int) -> int:
    """Line of the CSV file that holds table row i (line 1 is the header)."""
    return int(i) + 2


def _column_hint(columns, name: str) -> str:
    """' (found ...)' when a column differs from `name` only by case or surrounding spaces."""
    for c in columns:
        if isinstance(c, str) and c != name and c.strip().lower() == name.lower():
            return f" (found {c!r}: column names must match exactly, without spaces)"
    return ""


def _as_numbers(df: pd.DataFrame, col: str, source: str, problems: List[Problem]) -> pd.Series:
    """The column as floats with missing values as NaN; text that is not a number is an error."""
    raw = df[col]
    values = pd.to_numeric(raw, errors='coerce')
    text = raw.notna() & values.isna()
    if text.any():
        i = int(np.argmax(text.to_numpy()))
        bad = str(raw.iloc[i]).strip()
        hint = " Use a point, not a comma, as the decimal separator." if re.fullmatch(r"-?\d+,\d+", bad) else ""
        problems.append(Problem('error',
            f"Column {col} in {source} has {int(text.sum())} value(s) that are not numbers "
            f"(first: {bad!r} on line {_line(i)}). Leave a missing value blank or write -999.{hint}"))
    return values.astype(np.float64).where(values != MISSING_CODE)


def _first(mask: pd.Series, dates: pd.Series) -> str:
    i = int(np.argmax(mask.to_numpy()))
    return f"{dates.iloc[i].date()}, line {_line(i)}"


def check_table(df: pd.DataFrame, source: str, *, period: str = 'calibration', version: int = 8,
                gap_tolerant: bool = False, calendar: str = 'standard',
                min_theta_floor: Optional[float] = None) -> CheckedTable:
    """
    Check a data table as a run would, and collect every problem.

    Parameters
    ----------
    df : DataFrame with the columns Date, T_air, T_water and (versions 4, 7, 8) Discharge.
    source : the file name (or a description) used in the messages.
    period : 'calibration', 'validation' or 'scenario' (the input of a FORWARD run).
    version, gap_tolerant, calendar, min_theta_floor : as in the configuration.

    Returns
    -------
    CheckedTable. A run stops at the first error (`raise_first_error`) and prints the
    warnings; the pre-analysis report lists all of them.
    """
    if period not in PERIOD_NAMES:
        raise ValueError(f"period must be one of {', '.join(PERIOD_NAMES)}, got {period!r}")
    if calendar == '360_day':
        raise ValueError(NO_360_DAY)
    if calendar not in CALENDARS:
        raise ValueError(f"calendar must be one of {', '.join(CALENDARS)}, got {calendar!r}")
    problems: List[Problem] = []
    df = df.copy()
    columns = list(df.columns)
    uses_q = version not in (3, 5)

    # --- Columns -----------------------------------------------------------------------
    for col in ('Date', 'T_air'):
        if col not in columns:
            problems.append(Problem('error', f"Missing '{col}' column in {source}{_column_hint(columns, col)}."))
    if 'Discharge' not in columns:
        if uses_q:
            problems.append(Problem('error', f"Missing 'Discharge' column in {source}{_column_hint(columns, 'Discharge')}: "
                                             f"model version {version} uses discharge."))
        else:
            df['Discharge'] = np.nan
    if 'T_water' not in columns:
        if period != 'scenario':
            problems.append(Problem('error', f"Missing 'T_water' column in {source}{_column_hint(columns, 'T_water')}: "
                                             f"a {PERIOD_NAMES[period]} file needs measured water temperature."))
        df['T_water'] = np.nan
    if 'Date' not in columns or 'T_air' not in columns or 'Discharge' not in df.columns:
        return CheckedTable(df, None, problems)       # nothing more can be checked

    # --- Dates -------------------------------------------------------------------------
    raw_dates = df['Date']
    dates = pd.to_datetime(raw_dates, errors='coerce')
    blank = raw_dates.isna() | (raw_dates.astype(str).str.strip() == "")
    unreadable = dates.isna() & ~blank
    if blank.any():
        i = int(np.argmax(blank.to_numpy()))
        problems.append(Problem('error', f"Date is blank on line {_line(i)} of {source} "
                                         f"({int(blank.sum())} blank date(s)). Every row needs a date."))
    if unreadable.any():
        i = int(np.argmax(unreadable.to_numpy()))
        problems.append(Problem('error', f"Date {str(raw_dates.iloc[i]).strip()!r} on line {_line(i)} of {source} "
                                         f"cannot be read as a date ({int(unreadable.sum())} such date(s)). "
                                         "Write dates as YYYY-MM-DD, in the same format on every row."))
    if blank.any() or unreadable.any():
        dates = None
    else:
        leap_days = (dates.dt.month == 2) & (dates.dt.day == 29)
        duplicated = dates.duplicated(keep=False)
        backwards = dates.diff() < pd.Timedelta(0)
        if calendar == 'noleap' and leap_days.any():
            i = int(np.argmax(leap_days.to_numpy()))
            problems.append(Problem('error', f"{dates.iloc[i].date()} on line {_line(i)} of {source} is 29 February, "
                                             "which the noleap calendar does not have. Remove the 29 February rows, "
                                             "or use calendar: 'standard'."))
        elif dates.duplicated().any():
            d = dates[duplicated].iloc[0]
            lines = [_line(i) for i in np.flatnonzero((dates == d).to_numpy())]
            problems.append(Problem('error', f"The time series in {source} must be continuous at a daily time scale "
                                             f"with each date once: {d.date()} appears on lines "
                                             f"{', '.join(map(str, lines))} ({int(dates.duplicated().sum())} repeated "
                                             "date(s)). Remove the repeated rows."))
        elif backwards.any():
            i = int(np.argmax(backwards.to_numpy()))
            problems.append(Problem('error', f"The time series in {source} must be continuous at a daily time scale, "
                                             f"in date order: {dates.iloc[i].date()} on line {_line(i)} comes after "
                                             f"{dates.iloc[i - 1].date()} on line {_line(i - 1)}. Sort the rows by date."))
        elif len(dates):
            expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq='D')
            if calendar == 'noleap':
                expected = expected[~((expected.month == 2) & (expected.day == 29))]
            missing = expected.difference(pd.DatetimeIndex(dates))
            if len(missing):
                after = int(np.searchsorted(dates.to_numpy(), missing[0].to_datetime64())) - 1
                problems.append(Problem('error', f"The time series in {source} must be continuous at a daily time scale "
                                                 f"with no missing dates: {len(missing)} date(s) have no row (first: "
                                                 f"{missing[0].date()}, after line {_line(after)}). Add a row for each "
                                                 "missing date and leave its values blank."))

    n_days = len(df)
    if period == 'calibration' and n_days < MIN_DAYS:
        problems.append(Problem('error', f"The calibration time series in {source} has only {n_days} day(s); at "
                                         f"least {MIN_DAYS} are required. A calibration needs at least a whole year "
                                         "so the parameters that describe the yearly cycle can be fitted: from part "
                                         "of a year they would mean nothing for the other seasons."))
    elif n_days < MIN_SHORT_DAYS:
        if period == 'validation':
            problems.append(Problem('warning', f"The validation file {source} has only {n_days} day(s), fewer than "
                                               f"{MIN_SHORT_DAYS}, so validation will be skipped."))
        else:
            problems.append(Problem('error', f"The {PERIOD_NAMES[period]} time series in {source} has only {n_days} "
                                             f"day(s); at least {MIN_SHORT_DAYS} are required."))
    elif n_days < MIN_DAYS:
        problems.append(Problem('warning', f"The {PERIOD_NAMES[period]} file {source} has {n_days} days, less than a "
                                           "year. It is used, but its first days are not scored while the model "
                                           "settles (warmup_drop_days, default 15), and a short period says little "
                                           "about the other seasons: a year or more is recommended (USER_GUIDE.md §5)."))

    # --- Values ------------------------------------------------------------------------
    for col in ('T_air', 'T_water', 'Discharge'):
        df[col] = _as_numbers(df, col, source, problems)
    label = dates if dates is not None else pd.Series(pd.NaT, index=df.index)

    def where(mask):
        return _first(mask, label) if dates is not None else f"line {_line(int(np.argmax(mask.to_numpy())))}"

    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        bad = (df[col] < lo) | (df[col] > hi)
        if bad.any():
            i = int(np.argmax(bad.to_numpy()))
            problems.append(Problem('warning', f"{int(bad.sum())} value(s) of {col} in {source} are outside "
                                               f"{lo:g} to {hi:g} degC (first: {df[col].iloc[i]:g} on {where(bad)}). "
                                               "Check for missing-value codes other than -999 or a blank cell, and "
                                               "for unit errors; these values are used as given."))

    if uses_q:
        negative = df['Discharge'] < 0.0
        if negative.any():
            problems.append(Problem('error', f"Negative discharge in {source}: {int(negative.sum())} day(s) (first: "
                                             f"{df['Discharge'][negative].iloc[0]:g} on {where(negative)}). Flow "
                                             "cannot be negative, so this is probably a code for a missing value: "
                                             "write a missing value as -999 or leave the cell blank."))
        if min_theta_floor is not None:
            zero = df['Discharge'] == 0.0
            if zero.any():
                problems.append(Problem('warning', f"Zero discharge in {source}: {int(zero.sum())} day(s) (first: "
                                                   f"{where(zero)}). They are simulated at theta = min_theta_floor "
                                                   f"({min_theta_floor:g}), where theta^a4 is extreme: expect "
                                                   "large simulated responses on those days (USER_GUIDE.md "
                                                   "§9.2)."))

    if not gap_tolerant:
        gaps = df['T_air'].isna()
        if gaps.any():
            problems.append(Problem('error', f"The series of observed air temperature in {source} must be complete: "
                                             f"{int(gaps.sum())} day(s) have no value (first: {where(gaps)}). Fill "
                                             "them, or set gap_tolerant: true."))
        if uses_q:
            gaps = df['Discharge'].isna()
            if gaps.any():
                problems.append(Problem('error', f"The series of discharge in {source} must be complete: "
                                                 f"{int(gaps.sum())} day(s) have no value (first: {where(gaps)}). "
                                                 "Fill them, or set gap_tolerant: true."))
        if uses_q and min_theta_floor is None:
            nonpositive = df['Discharge'] <= 0.0
            if nonpositive.any():
                problems.append(Problem('error', f"Non-positive discharge (Q <= 0) in {source}: {int(nonpositive.sum())} "
                                                 f"day(s) (first: {where(nonpositive)}). Model version {version} "
                                                 "divides by a power of discharge, which is undefined at zero flow. "
                                                 "Correct the data, set gap_tolerant: true to treat such days as "
                                                 "gaps, or set min_theta_floor (USER_GUIDE.md §9.2)."))

    elif uses_q and min_theta_floor is None:
        nonpositive = df['Discharge'] <= 0.0
        if nonpositive.any():
            n = int(nonpositive.sum())
            what = (f"Zero or negative discharge in {source}: {n} day(s) (first: {where(nonpositive)}). Model "
                    f"version {version} cannot simulate a day without flow (theta = Q / Qmedia is zero, and the "
                    "equation divides by a power of theta). ")
            if period == 'scenario':
                problems.append(Problem('error', what + "In gap-tolerant mode these days would be left out as "
                                        "gaps, so the scenario would have no water temperature on exactly these "
                                        "days and they would not count in its results (for example in the number "
                                        "of warm days). Set min_theta_floor to simulate them at a very low flow "
                                        "(beyond the calibrated flows, so treat them with care), or correct the "
                                        "data (USER_GUIDE.md §9.2)."))
            else:
                problems.append(Problem('warning', what + "In gap-tolerant mode these days are treated as gaps: "
                                        "they are not simulated or scored, and the model starts again after them. "
                                        "Set min_theta_floor to simulate them instead (USER_GUIDE.md §9.2)."))

    if period != 'scenario' and 'T_water' in columns and not df['T_water'].notna().any():
        what = "so it cannot test the model. Fill it, or remove paths.validation_data" if period == 'validation' \
            else "so the model cannot be calibrated on it"
        problems.append(Problem('error', f"The {PERIOD_NAMES[period]} file {source} has no water temperature "
                                         f"measurements (column T_water is empty), {what}."))
    return CheckedTable(df, dates, problems)
