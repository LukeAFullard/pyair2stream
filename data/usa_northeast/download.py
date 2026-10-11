"""
Download and select the river data of the north-eastern United States used by validation V20.

    python data/usa_northeast/download.py

The stations are chosen by a fixed rule, written here before any model was run on them, so that the
choice cannot depend on how well the model or its intervals do:

1. Every USGS stream gauge (site type ST) in hydrologic regions 01 (New England) and 02 (Mid-Atlantic)
   with daily mean water temperature (parameter 00010, statistic 00003) and daily mean discharge (00060, 00003).
2. Discharge is given, and above zero, on every day from 2011-01-01 to 2023-12-31 (versions 4 and 8
   need it on every day; no gap is filled).
3. Water temperature has at least 300 daily means in each of at least 5 of the calibration years
   2011-2020, and at least 300 daily means in all in the later years 2021-2023.
4. Daymet (version 4) gives air temperature at the gauge's coordinates.
5. If more than 30 stations qualify, 30 are drawn at random (numpy default_rng(2026)), so that the
   check's run time stays near V18's.

Air temperature is Daymet's daily (tmax + tmin) / 2 at the gauge, as many river temperature studies
use. Daymet has 365 days in every year: in leap years it leaves out 31 December, which is filled here
with the mean of 30 December and 1 January. Water temperature and discharge are USGS daily means,
approved or provisional, as published; values flagged as estimated are kept.

Writes, for each selected station, <site>_calibration.csv (2011-2020) and <site>_later.csv
(2021-2023) with the columns Date, T_air, T_water, Discharge (discharge in m³/s), stations.csv (the
selected stations with their names, coordinates, drainage area and counts) and candidates.csv (every
station examined, with the reason it was or was not selected).
"""

import io
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HUCS = ("01", "02")
START, END = "2011-01-01", "2023-12-31"
CAL_YEARS = range(2011, 2021)
LATER = ("2021-01-01", "2023-12-31")
MIN_DAYS_PER_YEAR = 300
MIN_CAL_YEARS = 5
MIN_LATER_DAYS = 300
MAX_STATIONS = 30
SEED = 2026
CFS_TO_M3S = 0.028316846592

NWIS = "https://waterservices.usgs.gov/nwis"
DAYMET = "https://daymet.ornl.gov/single-pixel/api/data"


def _get(url, retries=4):
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read().decode("utf-8")
        except Exception as exc:                     # network hiccup: wait and retry
            if k == retries - 1:
                raise
            print(f"  retry ({exc})", file=sys.stderr)
            time.sleep(2 ** (k + 1))


def candidates() -> pd.DataFrame:
    """Stream gauges of the regions with daily mean water temperature and discharge."""
    return pd.concat([_candidates(huc) for huc in HUCS]).drop_duplicates("site_no").sort_values(
        "site_no").reset_index(drop=True)


def _candidates(huc) -> pd.DataFrame:
    url = (f"{NWIS}/site/?format=rdb&huc={huc}&siteType=ST&hasDataTypeCd=dv&outputDataTypeCd=dv"
           f"&seriesCatalogOutput=true&siteStatus=all&parameterCd=00010,00060")
    df = _rdb(url)
    df = df[(df.data_type_cd == "dv") & (df.stat_cd == "00003")]
    have = df.groupby("site_no").parm_cd.agg(set)
    sites = have[have.map(lambda s: {"00010", "00060"} <= s)].index
    info = df[df.site_no.isin(sites)].drop_duplicates("site_no")
    info = info[["site_no", "station_nm", "dec_lat_va", "dec_long_va"]].sort_values("site_no")
    area = _rdb(f"{NWIS}/site/?format=rdb&huc={huc}&siteType=ST&siteStatus=all&siteOutput=expanded"
                f"&hasDataTypeCd=dv&parameterCd=00010")
    area = area.drop_duplicates("site_no").set_index("site_no").drain_area_va
    return info.assign(drain_area_sq_mi=info.site_no.map(area)).reset_index(drop=True)


def _rdb(url) -> pd.DataFrame:
    """A USGS tab-separated (rdb) table: comment lines and the row of field widths removed."""
    text = "\n".join(line for line in _get(url).splitlines() if not line.startswith("#"))
    return pd.read_csv(io.StringIO(text), sep="\t", dtype=str).iloc[1:]


def daily_values(site: str) -> pd.DataFrame:
    """USGS daily means of water temperature and discharge; for each parameter, the time series with the
    most values when a site has several."""
    url = (f"{NWIS}/dv/?format=json&sites={site}&parameterCd=00010,00060&statCd=00003"
           f"&startDT={START}&endDT={END}&siteStatus=all")
    js = json.loads(_get(url))
    best = {}
    for ts in js["value"]["timeSeries"]:
        parm = ts["variable"]["variableCode"][0]["value"]
        nodata = float(ts["variable"]["noDataValue"])
        for block in ts["values"]:
            vals = [(v["dateTime"][:10], float(v["value"])) for v in block["value"]
                    if float(v["value"]) != nodata]
            if len(vals) > len(best.get(parm, [])):
                best[parm] = vals
    out = pd.DataFrame({"Date": pd.date_range(START, END).strftime("%Y-%m-%d")})
    for parm, col in (("00010", "T_water"), ("00060", "Discharge")):
        s = pd.Series(dict(best.get(parm, [])), dtype=float)
        out[col] = out.Date.map(s)
    out["Discharge"] *= CFS_TO_M3S
    return out


def daymet_air(lat: float, lon: float) -> pd.Series:
    """Daily mean air temperature, (tmax + tmin) / 2, from Daymet at a point, on every calendar day."""
    url = f"{DAYMET}?lat={lat}&lon={lon}&vars=tmax,tmin&start={START}&end={END}&format=csv"
    text = _get(url)
    lines = text.splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith("year"))
    d = pd.read_csv(io.StringIO("\n".join(lines[head:])))
    dates = pd.to_datetime(d.year.astype(str), format="%Y") + pd.to_timedelta(d.yday - 1, unit="D")
    ta = pd.Series(((d["tmax (deg c)"] + d["tmin (deg c)"]) / 2).to_numpy(), index=dates)
    full = ta.reindex(pd.date_range(START, END))
    return full.interpolate(limit=1, limit_area="inside")     # 31 December of leap years


def check(site: str, dv: pd.DataFrame):
    """The selection rule (2 and 3 of the module docstring): (qualifies, reason, counts)."""
    q = dv.Discharge
    if q.isna().any():
        return False, f"discharge missing on {int(q.isna().sum())} days", {}
    if (q <= 0).any():
        return False, f"discharge zero or below on {int((q <= 0).sum())} days", {}
    year = pd.to_datetime(dv.Date).dt.year
    per_year = dv.T_water.notna().groupby(year).sum()
    cal_ok = int(sum(per_year.get(y, 0) >= MIN_DAYS_PER_YEAR for y in CAL_YEARS))
    later = int(dv.T_water[(dv.Date >= LATER[0]) & (dv.Date <= LATER[1])].notna().sum())
    counts = {"calibration years with at least 300 days": cal_ok, "later-year days": later}
    if cal_ok < MIN_CAL_YEARS:
        return False, f"{cal_ok} calibration years with at least {MIN_DAYS_PER_YEAR} days", counts
    if later < MIN_LATER_DAYS:
        return False, f"{later} water-temperature days in 2021-2023", counts
    return True, "qualifies", counts


def main():
    cand = candidates()
    print(f"{len(cand)} candidate gauges in regions {', '.join(HUCS)}")
    rows, kept = [], {}
    for _, c in cand.iterrows():
        dv = daily_values(c.site_no)
        ok, reason, counts = check(c.site_no, dv)
        if ok:
            try:
                ta = daymet_air(float(c.dec_lat_va), float(c.dec_long_va))
            except Exception as exc:
                ok, reason = False, f"no Daymet air temperature ({exc})"
            else:
                if ta.isna().any():
                    ok, reason = False, f"Daymet air temperature missing on {int(ta.isna().sum())} days"
                else:
                    dv = dv.assign(T_air=np.round(ta.to_numpy(), 2))
                    kept[c.site_no] = dv
        rows.append({**c.to_dict(), "qualifies": ok, "reason": reason, **counts})
        print(f"{c.site_no}: {reason}")
    table = pd.DataFrame(rows)
    chosen = sorted(kept)
    if len(chosen) > MAX_STATIONS:
        chosen = sorted(np.random.default_rng(SEED).choice(chosen, MAX_STATIONS, replace=False).tolist())
    table["selected"] = table.site_no.isin(chosen)
    table.to_csv(os.path.join(HERE, "candidates.csv"), index=False)
    for site in chosen:
        dv = kept[site][["Date", "T_air", "T_water", "Discharge"]]
        dv[dv.Date <= "2020-12-31"].to_csv(os.path.join(HERE, f"{site}_calibration.csv"), index=False,
                                           float_format="%.3f")
        dv[dv.Date >= LATER[0]].to_csv(os.path.join(HERE, f"{site}_later.csv"), index=False, float_format="%.3f")
    table[table.selected].drop(columns=["qualifies", "selected"]).to_csv(os.path.join(HERE, "stations.csv"),
                                                                         index=False)
    print(f"{int(table.qualifies.sum())} qualify; {len(chosen)} selected")


if __name__ == "__main__":
    main()
