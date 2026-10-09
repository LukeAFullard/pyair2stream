"""
What a run produced, in one place.

- `RunResult`, returned by `pyair2stream.run`: the best parameters, the scores, the
  warnings and notes the run printed, and the output files.
- `summary.md`, a one-page summary written into every output folder.
- `filled_water_temperature_<period>.csv`: the measured water temperature, with the
  model's values on the days without a measurement (and the prediction range where
  the run made one).
"""

import contextlib
import datetime
import glob
import io
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import ACTIVE_PARAMS, CommonData

PARAM_NAMES = tuple(f"a{i}" for i in range(1, 9))
MISSING = -999.0


@dataclass
class RunResult:
    """What a run produced.

    output_dir: the folder with every output file.
    run_mode, version: as in the settings.
    parameters: {"a1": ..., "a8": ...}: the best fit (calibration) or the parameters used
        (FORWARD); None for a cross-validation run, whose folds each have their own.
    scores: {"calibration": {"NSE": ..., "RMSE": ..., ...}, "validation": {...}} as written
        in goodness_of_fit_*.csv; a FORWARD run's are under "forward", a cross-validation
        run's (all held-out days together) under "cross-validation".
    messages: the warnings and notes the run printed, in order.
    summary: the path of summary.md.
    files: every output file, relative to output_dir.
    data: the run's internal state (CommonData), for advanced use.
    """
    output_dir: str
    run_mode: str
    version: int
    parameters: Optional[dict]
    scores: dict
    messages: list
    summary: Optional[str]
    files: list
    data: CommonData = field(repr=False, default=None)


# --- Capturing what a run prints --------------------------------------------------------------------

class _Tee(io.TextIOBase):
    """A text stream that passes everything on (unless quiet) and keeps the lines."""

    def __init__(self, echo_to):
        self.echo_to = echo_to
        self.lines = []
        self._partial = ""

    def write(self, s):
        if self.echo_to is not None:
            self.echo_to.write(s)
        *complete, self._partial = (self._partial + s).split("\n")
        self.lines.extend(complete)
        return len(s)

    def flush(self):
        if self.echo_to is not None:
            self.echo_to.flush()


@contextlib.contextmanager
def capture_output(verbose: bool = True):
    """Collect the lines printed inside the block; also show them unless verbose is False."""
    tee = _Tee(sys.stdout if verbose else None)
    with contextlib.redirect_stdout(tee):
        yield tee
    if tee._partial:
        tee.lines.append(tee._partial)


def messages_from(lines) -> list:
    """The warnings and notes among printed lines, each once, in order. In gap-tolerant runs,
    the one-per-stretch 'Dropped segment' warnings are counted instead of listed."""
    out, dropped = [], 0
    for line in lines:
        text = line.strip()
        if text.startswith("Warning: Dropped segment"):
            dropped += 1
        elif text.startswith(("Warning", "Note")) and text not in out:
            out.append(text)
    if dropped:
        out.insert(0, f"Warning: {dropped} stretch(es) of data between gaps were shorter than min_segment_days "
                      "and were not used (gap-tolerant mode; see gaps_summary.txt).")
    return out


# --- Reading a run's outputs --------------------------------------------------------------------------

def _simulation_files(data: CommonData) -> dict:
    """The daily simulation files of this run, by period."""
    stem = f"{data.runmode}_{data.fun_obj}_{data.station}_{data.series}"
    files = {"calibration" if data.runmode != "FORWARD" else "forward":
             os.path.join(data.folder, f"2_{stem}c_{data.time_res}.csv"),
             "validation": os.path.join(data.folder, f"3_{stem}v_{data.time_res}.csv")}
    return {k: v for k, v in files.items() if os.path.exists(v)}


def _read_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.index = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day))
    df.index.name = "Date"
    return df.replace(MISSING, np.nan)


def _range_file(data: CommonData, period: str) -> Optional[str]:
    """The daily prediction range the run wrote for this period, if any."""
    if period == "calibration" and data.runmode == "DE-MCMC":
        name = f"MCMC_envelopes_{data.station}_{data.series}_{data.time_res}.csv"
    elif period == "forward":
        name = f"Forward_Prediction_Envelopes_{data.station}_{data.series}_{data.time_res}.csv"
    else:
        return None
    path = os.path.join(data.folder, name)
    return path if os.path.exists(path) else None


def _level(data: CommonData) -> float:
    return float((data.uncertainty_options or {}).get("prediction_interval", 90.0))


def collect_scores(data: CommonData) -> dict:
    """The scores in goodness_of_fit_*.csv (or, for cross-validation, the pooled row of
    cv_results.csv)."""
    scores = {}
    for label, key in (("calibration", "calibration"), ("validation", "validation"),
                       ("forward_projection", "forward")):
        path = os.path.join(data.folder, f"goodness_of_fit_{label}_{data.runmode}_{data.fun_obj}_{data.station}.csv")
        if os.path.exists(path):
            table = pd.read_csv(path)
            scores[key] = {m: float(v) for m, v in zip(table.Metric, table.Value)}
    cv = os.path.join(data.folder, "cv_results.csv")
    if os.path.exists(cv):
        table = pd.read_csv(cv).set_index("fold")
        if "pooled" in table.index:
            row = table.loc["pooled"]
            scores["cross-validation"] = {"N": float(row.n_obs_held_out), "NSE": float(row.NSE),
                                          "KGE": float(row.KGE), "RMSE": float(row.RMSE)}
    return scores


def collect_parameters(data: CommonData) -> Optional[dict]:
    if getattr(data, "cross_validation", None) and data.runmode in ("PSO", "DE", "LATHYP"):
        return None
    par = data.par if data.runmode == "FORWARD" else getattr(data, "par_best", None)
    if par is None:
        return None
    return {PARAM_NAMES[j]: float(par[j]) for j in ACTIVE_PARAMS.get(data.version, range(8))}


def output_files(folder: str) -> list:
    out = []
    for root, _, names in os.walk(folder):
        for name in names:
            out.append(os.path.relpath(os.path.join(root, name), folder))
    return sorted(out)


# --- Gap-filled water temperature --------------------------------------------------------------------

def write_filled_series(data: CommonData) -> list:
    """For each period simulated, write filled_water_temperature_<period>.csv: the measured
    water temperature where there is one, the model's value elsewhere, which of the two it is,
    and the prediction range where the run made one. Returns the paths written."""
    written = []
    level = _level(data)
    for period, path in _simulation_files(data).items():
        sim = _read_daily(path)
        out = pd.DataFrame({"T_water_measured": sim.Twat_obs.round(3), "T_water_model": sim.Twat_mod.round(3)},
                           index=sim.index)
        out["T_water_filled"] = out.T_water_measured.where(out.T_water_measured.notna(), out.T_water_model)
        out["source"] = np.where(out.T_water_measured.notna(), "measured",
                                 np.where(out.T_water_model.notna(), "model", "none"))
        rng = _range_file(data, period)
        if rng:
            band = _read_daily(rng)
            out[f"model_lower_{level:g}"] = band.Twat_mod_lower.reindex(out.index).round(3)
            out[f"model_upper_{level:g}"] = band.Twat_mod_upper.reindex(out.index).round(3)
        target = os.path.join(data.folder, f"filled_water_temperature_{period}.csv")
        out.to_csv(target, date_format="%Y-%m-%d")
        written.append(target)
    return written


# --- summary.md --------------------------------------------------------------------------------------

FILE_DESCRIPTIONS = (
    ("summary.md", "this page"),
    ("filled_water_temperature_", "the measured water temperature, with the model's values on the days without "
                                  "a measurement (`source` says which), and the prediction range where there is one"),
    ("0_", "every parameter set tried during the calibration, with its score"),
    ("1_", "the best parameters and the calibration (and validation) score"),
    ("2_", "one row per day of the calibration (or FORWARD input) file: measured and simulated temperatures"),
    ("3_", "one row per day of the validation file: measured and simulated temperatures"),
    ("calibration_metadata.json", "Qmedia, the calibrated flow range, version, integrator, parameters and seed; "
                                  "FORWARD runs reuse it"),
    ("parameters.txt", "the parameter bounds used"),
    ("gaps_summary.txt", "the stretches of data used (gap-tolerant mode)"),
    ("goodness_of_fit_", "the scores: N, NSE, R2, RMSE, MAE, AIC, BIC"),
    ("bias_by_month_", "the mean error in each month and season, with a 95% interval"),
    ("predicted_vs_measured_", "simulated against measured temperature"),
    ("residual_diagnostics_", "the errors' histogram, normal Q-Q plot and autocorrelation"),
    ("calibration_", "measured and simulated temperature, calibration years"),
    ("validation_", "measured and simulated temperature, validation years"),
    ("full_simulation_", "the calibration period on every day"),
    ("forward_projection", "the FORWARD simulation and its range"),
    ("convergence_", "the best score against the number of parameter sets tried (it should flatten out)"),
    ("dottyplots_", "the score against each parameter"),
    ("parameter_correlation_", "how the parameters move together in the MCMC sample"),
    ("parameter_significance_", "each parameter's mean and range from the MCMC sample"),
    ("MCMC_chain_", "the MCMC sample of parameter sets (and, in _meta.json, its convergence and error model)"),
    ("MCMC_envelopes_", "the daily prediction range for the calibration file"),
    ("Forward_Prediction_Envelopes_", "the daily prediction range"),
    ("Forward_Prediction_Ensemble_", "the simulated series, one per parameter set (and its settings in _meta.json)"),
    ("cv_results.csv", "cross-validation: the scores and parameters of each held-out year, and the jackknife "
                       "parameter intervals"),
    ("cv_bias_by_month", "cross-validation: the mean error in each month over the held-out years"),
    ("cv_yearly_statistics", "cross-validation: the check of yearly peaks and counts"),
    ("cv_interval_coverage.csv", "cross-validation: how often the intervals held, at several levels"),
    ("sensitivity", "the sensitivity analysis"),
)


def _describe(name: str) -> str:
    base = os.path.basename(name)
    for prefix, text in FILE_DESCRIPTIONS:
        if base.startswith(prefix) or name.startswith(prefix):
            return text
    return ""


def _period_rows(data: CommonData) -> list:
    rows = []
    for period, path in _simulation_files(data).items():
        sim = _read_daily(path)
        no_air = int(sim.Tair.isna().sum()) if "Tair" in sim else 0
        no_q = int(sim.Q.isna().sum()) if "Q" in sim and data.version in (4, 7, 8) else 0
        rows.append((period, f"{sim.index[0]:%Y-%m-%d} to {sim.index[-1]:%Y-%m-%d}", len(sim),
                     int(sim.Twat_obs.notna().sum()), no_air, no_q, os.path.basename(path)))
    return rows


def _resolution(time_res: str) -> str:
    if time_res == "1d":
        return "daily values"
    if time_res == "1m":
        return "monthly means"
    if time_res.endswith("w") and time_res[:-1].isdigit():
        weeks = int(time_res[:-1])
        return "weekly means" if weeks == 1 else f"{weeks}-week means"
    return f"{time_res} values"


def _input_file(data: CommonData, period: str) -> str:
    path = getattr(data, "_input_data_path_val" if period == "validation" else "_input_data_path_cal", None)
    return f"`{path}`" if path else ""


def _fmt(x, digits=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def write_summary(data: CommonData, scores: dict, parameters: Optional[dict], messages: list,
                  settings: str = "", seconds: Optional[float] = None) -> str:
    """Write summary.md into the output folder and return its path."""
    from . import __version__
    lines = [f"# pyair2stream run summary: {data.station}", ""]
    when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines += [f"{data.runmode} run of model version {data.version}, {when}, pyair2stream {__version__}"
              + (f", {seconds / 60:.1f} minutes" if seconds else "") + ".", ""]

    lines += ["## Settings", "", "| | |", "|---|---|"]
    if settings:
        lines.append(f"| Settings | `{settings}` |")
    lines.append(f"| Run mode | {data.runmode} |")
    if data.runmode != "FORWARD":
        lines.append(f"| Score | {data.fun_obj}, on {_resolution(data.time_res)} |")
    lines.append(f"| Integrator | {data.mod_num} |")
    lines.append(f"| Random seed | {data.random_seed if data.random_seed is not None else 'none (not repeatable)'} |")
    lines.append(f"| Gap-tolerant mode | {'yes' if data.gap_tolerant else 'no'} |")
    if data.version in (4, 7, 8):
        lines.append(f"| Qmedia (mean discharge) | {data.Qmedia:.4g} ({'set in the settings' if data.Qmedia_user is not None else 'computed from the calibration file'}) |")
    lines.append("")

    rows = _period_rows(data)
    if rows:
        lines += ["## Data", "", "| Period | Input file | Dates | Days | Water temperature measured "
                  "| Days without air temperature" + (" | Days without discharge" if data.version in (4, 7, 8) else "")
                  + " |", "|---|---|---|---|---|---" + ("|---" if data.version in (4, 7, 8) else "") + "|"]
        for period, span, days, measured, no_air, no_q, _ in rows:
            lines.append(f"| {period} | {_input_file(data, period)} | {span} | {days:,} | {measured:,} | {no_air:,}"
                         + (f" | {no_q:,}" if data.version in (4, 7, 8) else "") + " |")
        lines.append("")

    if scores:
        lines += ["## How well the model fits", "",
                  "| Period | Values scored | NSE | RMSE (°C) | MAE (°C) |", "|---|---|---|---|---|"]
        for key, s in scores.items():
            lines.append(f"| {key} | {int(s.get('N', 0)):,} | {_fmt(s.get('NSE'))} | {_fmt(s.get('RMSE'), 2)} | "
                         f"{_fmt(s.get('MAE'), 2)} |")
        lines += ["", "NSE: 1 is perfect, 0 is no better than the average. RMSE and MAE: the typical daily error. "
                      "The validation (or cross-validation) scores show how well the model predicts years it was "
                      "not fitted to.", ""]

    if parameters:
        names = list(parameters)
        lo = {PARAM_NAMES[j]: float(data.parmin[j]) for j in range(8)}
        hi = {PARAM_NAMES[j]: float(data.parmax[j]) for j in range(8)}
        lines += ["## Parameters", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
                  "| " + ("best fit" if data.runmode != "FORWARD" else "used") + " | "
                  + " | ".join(f"{parameters[n]:.4g}" for n in names) + " |"]
        if data.runmode != "FORWARD":
            lines.append("| lower bound | " + " | ".join(f"{lo[n]:.4g}" for n in names) + " |")
            lines.append("| upper bound | " + " | ".join(f"{hi[n]:.4g}" for n in names) + " |")
            sig = glob.glob(os.path.join(data.folder, "parameter_significance_*.csv"))
            if sig:
                table = pd.read_csv(sig[0])
                lower = [c for c in table.columns if c.endswith("_CI_Lower")]
                upper = [c for c in table.columns if c.endswith("_CI_Upper")]
                if lower and upper:
                    level = lower[0].split("%")[0]
                    rng = {f"a{int(p.split('_')[1])}": (lo_, hi_) for p, lo_, hi_ in
                           zip(table.Parameter, table[lower[0]], table[upper[0]])}
                    lines.append(f"| {level}% range (MCMC) | "
                                 + " | ".join(f"{rng[n][0]:.4g} to {rng[n][1]:.4g}" if n in rng else ""
                                              for n in names) + " |")
            at_bound = [f"{n} ({'lower' if abs(parameters[n] - lo[n]) <= abs(parameters[n] - hi[n]) else 'upper'})"
                        for n in names if hi[n] > lo[n]
                        and min(abs(parameters[n] - lo[n]), abs(parameters[n] - hi[n])) <= 0.01 * (hi[n] - lo[n])]
            lines += ["", "At a bound (within 1% of the range): " + (", ".join(at_bound) if at_bound else "none") + "."
                      + (" Widen that bound and calibrate again." if at_bound else "")]
        lines.append("")

    meta = glob.glob(os.path.join(data.folder, "MCMC_chain_*_meta.json")) + \
        glob.glob(os.path.join(data.folder, "Forward_Prediction_Ensemble_*_meta.json"))
    if meta:
        m = json.load(open(meta[0]))
        level = _level(data)
        lines += ["## Uncertainty", "", "| | |", "|---|---|"]
        if "converged" in m:
            lines.append(f"| MCMC converged | {'yes' if m['converged'] else '**no**: do not use these ranges'}"
                         f" ({m.get('steps_run', '?')} steps) |")
        if "source_chain_converged" in m:
            lines.append(f"| MCMC sample converged | {'yes' if m['source_chain_converged'] else '**no**'} |")
        sigma = m.get("sigma", m.get("residual_sigma"))
        if sigma is not None:
            lines.append(f"| Daily error size σ | {sigma:.3f} °C |")
        if m.get("rho") is not None:
            lines.append(f"| Error persistence ρ | {m['rho']:.3f} |")
        if m.get("interval_coverage") is not None:
            lines.append(f"| Measured days inside the {level:g}% range | {100 * m['interval_coverage']:.1f}% of "
                         f"{m.get('interval_coverage_n_days', '?')} (should be close to {level:g}%) |")
        lines.append("")

    lines += ["## Warnings and notes", ""]
    lines += [f"- {msg}" for msg in messages] if messages else ["None."]
    lines.append("")

    files = output_files(data.folder)
    if "summary.md" not in files:
        files = sorted(files + ["summary.md"])
    # A figure saved as .png and .pdf is listed once, with both extensions.
    grouped = {}
    for name in files:
        stem, ext = os.path.splitext(name)
        key = stem if ext in (".png", ".pdf") else name
        grouped.setdefault(key, []).append(name)
    lines += ["## Output files", "", "| File | What it is |", "|---|---|"]
    for key, names in grouped.items():
        shown = f"`{names[0]}`" if len(names) == 1 else f"`{key}` (" + ", ".join(
            os.path.splitext(n)[1] for n in names) + ")"
        lines.append(f"| {shown} | {_describe(names[0])} |")
    lines += ["", "## Next", "",
              "- How to read these results: USER_GUIDE.md §8.",
              "- Before using them for a decision: the checklist in USER_GUIDE.md §14.", ""]
    path = os.path.join(data.folder, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
