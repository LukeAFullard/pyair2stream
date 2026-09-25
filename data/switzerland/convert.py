"""
Convert the original air2stream text files in original/ into pyair2stream CSVs.

    python data/switzerland/convert.py

Each original row is `year month day T_air T_water Discharge`, with -999 for a
missing value. The CSVs have the columns Date, T_air, T_water, Discharge and
leave missing values empty. Values are copied unchanged.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STATIONS = ("MAH_2369", "SIO_2011", "DAV_2327")
PERIODS = {"cc": "calibration", "cv": "validation"}


def convert(txt_path: str, csv_path: str) -> None:
    raw = pd.read_csv(txt_path, sep=r"\s+", header=None)
    df = pd.DataFrame({
        "Date": pd.to_datetime(dict(year=raw[0], month=raw[1], day=raw[2])).dt.strftime("%Y-%m-%d"),
        "T_air": raw[3],
        "T_water": raw[4],
        "Discharge": raw[5],
    })
    df = df.replace(-999.0, np.nan)
    df.to_csv(csv_path, index=False, float_format="%.3f")


if __name__ == "__main__":
    for station in STATIONS:
        for code, period in PERIODS.items():
            src = os.path.join(HERE, "original", f"{station}_{code}.txt")
            dst = os.path.join(HERE, f"{station}_{period}.csv")
            convert(src, dst)
            print(f"{src} -> {dst}")
