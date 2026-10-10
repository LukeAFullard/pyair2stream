"""
What a run produced, in one place.

- `RunResult`, returned by `pyair2stream.run`: the best parameters, the scores, the
  warnings and notes the run printed, and the output files.
- `summary.md`, a one-page summary written into every output folder, and the same page as
  `summary.html`, with the figures in it, to open in a web browser or send as one file.
- `filled_water_temperature_<period>.csv`: the measured water temperature, with the
  model's values on the days without a measurement (and the prediction range where
  the run made one).
"""

import base64
import contextlib
import datetime
import glob
import html
import io
import json
import os
import re
import sys
import urllib.parse
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
    summary: the path of summary.md (summary.html, beside it, is the same page with the figures).
    files: the files this run wrote or replaced, relative to output_dir (the folder may also
        hold files from earlier runs).
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
    return {k: v for k, v in files.items() if os.path.exists(v) and _this_run(data, v)}


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
    return path if os.path.exists(path) and _this_run(data, path) else None


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


def _stamp(path: str):
    st = os.stat(path)
    return st.st_mtime_ns, st.st_size


def snapshot_folder(folder: str) -> dict:
    """The files in a folder now, with when they were last written: {relative path: stamp}."""
    return {name: _stamp(os.path.join(folder, name)) for name in output_files(folder)} if os.path.isdir(folder) else {}


def files_of_this_run(data: CommonData) -> list:
    """The files in the output folder that this run wrote or replaced (all of them if the
    folder was not recorded when the run started)."""
    before = getattr(data, "folder_before", None)
    files = output_files(data.folder)
    if before is None:
        return files
    return [f for f in files if before.get(f) != _stamp(os.path.join(data.folder, f))]


def _this_run(data: CommonData, path: str) -> bool:
    return os.path.relpath(path, data.folder) in set(files_of_this_run(data))


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
        if "warm_up" in sim:
            # The model is still settling on these days (not scored): its values are less reliable.
            out["warm_up"] = sim.warm_up.fillna(0).astype(int)
        rng = _range_file(data, period)
        if rng:
            band = _read_daily(rng)
            out[f"model_lower_{level:g}"] = band.Twat_mod_lower.reindex(out.index).round(3)
            out[f"model_upper_{level:g}"] = band.Twat_mod_upper.reindex(out.index).round(3)
            if "Twat_mod_lower_conformal" in band:
                out[f"model_lower_{level:g}_conformal"] = band.Twat_mod_lower_conformal.reindex(out.index).round(3)
                out[f"model_upper_{level:g}_conformal"] = band.Twat_mod_upper_conformal.reindex(out.index).round(3)
        target = os.path.join(data.folder, f"filled_water_temperature_{period}.csv")
        out.to_csv(target, date_format="%Y-%m-%d")
        written.append(target)
    return written


# --- summary.md --------------------------------------------------------------------------------------

FILE_DESCRIPTIONS = (
    ("summary.md", "this page"),
    ("summary.html", "this page with the figures in it, to open in a web browser or send as one file"),
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
    ("cv_conformal_margins.csv", "cross-validation: the margins that widen the intervals to the held-out years' "
                                 "coverage (forward_options.conformal_margins)"),
    ("sensitivity", "the sensitivity analysis"),
)


def _describe(name: str) -> str:
    base = os.path.basename(name)
    for prefix, text in FILE_DESCRIPTIONS:
        if base.startswith(prefix) or name.startswith(prefix):
            return text
    return ""


# The figures in summary.md and summary.html: the simulations first, then the checks of the
# errors, then the calibration's diagnostics; any other figure last.
FIGURE_ORDER = ("calibration_", "validation_", "full_simulation_", "forward_projection", "predicted_vs_measured_",
                "bias_by_month_", "residual_diagnostics_", "parameter_significance_", "parameter_correlation_",
                "convergence_", "dottyplots_")


def _figure_rank(name: str):
    base = os.path.basename(name)
    rank = next((i for i, prefix in enumerate(FIGURE_ORDER) if base.startswith(prefix)), len(FIGURE_ORDER))
    return rank, name


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


def seed_text(data: CommonData) -> str:
    """What made the run's random choices repeatable, for the summary. A FORWARD run's only
    random choice is the draw of parameter sets for its prediction intervals."""
    if data.runmode != "FORWARD":
        return str(data.random_seed) if data.random_seed is not None else "none (not repeatable)"
    draw = getattr(data, "forward_draw", None)
    if not draw:
        return "not needed (no random choices in this run)"
    if draw.get("reused_from"):
        return f"not needed: the parameter sets were reused from `{draw['reused_from']}`"
    if draw.get("seed") is None:
        return "none (not repeatable)"
    return f"{draw['seed']} ({draw['source']})"


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
    lines.append(f"| Random seed | {seed_text(data)} |")
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
            sig = [f for f in glob.glob(os.path.join(data.folder, "parameter_significance_*.csv"))
                   if _this_run(data, f)]
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

    # The uncertainty record of this run only: a FORWARD run's prediction intervals, or a
    # DE-MCMC run's chain (not those of another run sharing the output folder).
    pattern = {"FORWARD": "Forward_Prediction_Ensemble_*_meta.json", "DE-MCMC": "MCMC_chain_*_meta.json"}
    meta = [f for f in glob.glob(os.path.join(data.folder, pattern.get(data.runmode, "-")))
            if _this_run(data, f)]
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
            if m.get("rho_measured") is False:
                lines.append(f"| Error persistence ρ | {m['rho']:.3f}: **not measured**. There were too few pairs of "
                             "consecutive measured days (at least 30 are needed), so no persistence was assumed. "
                             "The parameter ranges and the ranges of anything longer than a day are therefore too "
                             "narrow (docs/UNCERTAINTY.md §5) |")
            else:
                lines.append(f"| Error persistence ρ | {m['rho']:.3f} |")
        if m.get("interval_coverage") is not None:
            lines.append(f"| Measured days inside the {level:g}% range | {100 * m['interval_coverage']:.1f}% of "
                         f"{m.get('interval_coverage_n_days', '?')} (should be close to {level:g}%) |")
        lines.append("")

    lines += ["## Warnings and notes", ""]
    lines += [f"- {msg}" for msg in messages] if messages else ["None."]
    lines.append("")

    files = sorted(set(files_of_this_run(data)) | {"summary.md", "summary.html"})
    figures = sorted((name for name in files if name.lower().endswith(".png")), key=_figure_rank)
    if figures:
        lines += ["## Figures", ""]
        for name in figures:
            lines += [f"![{_describe(name) or os.path.basename(name)}]({_href(name)})", ""]
    # A figure saved as .png and .pdf is listed once, with both extensions.
    grouped = {}
    for name in files:
        stem, ext = os.path.splitext(name)
        key = stem if ext in (".png", ".pdf") else name
        grouped.setdefault(key, []).append(name)
    lines += ["## Output files", "", "| File | What it is |", "|---|---|"]
    for key, names in grouped.items():
        shown = f"[`{names[0]}`]({_href(names[0])})" if len(names) == 1 else f"`{key}` (" + ", ".join(
            f"[{os.path.splitext(n)[1]}]({_href(n)})" for n in names) + ")"
        lines.append(f"| {shown} | {_describe(names[0])} |")
    before = getattr(data, "folder_before", None) or {}
    replaced = sorted(f for f in before if f in files)
    earlier = len(before) - len(replaced)
    if before:
        lines += ["", f"This folder also holds {earlier:,} file(s) from earlier runs, not described here."
                  + (" This run replaced these files of an earlier run: " + ", ".join(f"`{f}`" for f in replaced)
                     + "." if replaced else "")]
    lines += ["", "## Next", "",
              "- How to read these results: USER_GUIDE.md §8.",
              "- Before using them for a decision: the checklist in USER_GUIDE.md §14.", ""]
    path = os.path.join(data.folder, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(data.folder, "summary.html"), "w", encoding="utf-8") as f:
        f.write(summary_html(lines, data.folder))
    return path


# --- summary.html ------------------------------------------------------------------------------------

FIGURE_WIDTH = 1200     # pixels: figures are saved at 300 dpi; the page holds a smaller copy (256 colours)

_CSS = """
:root { --fg: #1d2125; --muted: #5b6570; --bg: #ffffff; --line: #d8dde3; --head: #f3f5f7; --link: #0b5cad; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e9ec; --muted: #a3acb6; --bg: #16191c; --line: #343a40; --head: #1f2327; --link: #7db7f0; }
}
body { font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg); background: var(--bg);
       max-width: 1000px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.6em; margin: 0 0 4px; } h2 { font-size: 1.2em; margin-top: 2em; border-bottom: 1px solid var(--line); }
table { border-collapse: collapse; margin: 8px 0; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 4px 10px; text-align: left; vertical-align: top; }
th { background: var(--head); } td:has(code) { min-width: 14em; }
code { font: 0.9em ui-monospace, Menlo, Consolas, monospace; word-break: break-all; }
a { color: var(--link); }
figure { margin: 16px 0 28px; } figure img { max-width: 100%; height: auto; background: #fff; }
figcaption { color: var(--muted); font-size: 0.92em; }
"""


def _href(name: str) -> str:
    return urllib.parse.quote(name.replace(os.sep, "/"))


def _inline(text: str) -> str:
    """The inline Markdown of summary.md (code, bold, links) as HTML."""
    out = html.escape(text, quote=False)
    out = re.sub(r"\[(.+?)\]\(([^)\s]+)\)", lambda m: f'<a href="{html.escape(m.group(2))}">{m.group(1)}</a>', out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)


def _image_data(path: str) -> Optional[str]:
    """The figure as a data: URI, at most FIGURE_WIDTH pixels wide and in 256 colours (about a
    quarter of the size, with no visible loss for line plots); None if it cannot be read."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > FIGURE_WIDTH:
                im = im.resize((FIGURE_WIDTH, round(im.height * FIGURE_WIDTH / im.width)), Image.LANCZOS)
            buf = io.BytesIO()
            im.quantize(256, method=Image.Quantize.MEDIANCUT).save(buf, format="PNG", optimize=True)
            raw = buf.getvalue()
    except Exception:  # noqa: BLE001  (a figure that cannot be scaled is embedded as it is)
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            return None
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _table_html(rows: list) -> str:
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    head, body = cells[0], [c for c in cells[1:] if not all(set(x) <= set("-: ") for x in c)]
    out = ["<table>"]
    if any(head):
        out.append("<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead>")
    out.append("<tbody>")
    for row in body:
        tag = "th" if not any(head) and row and row[0] else "td"
        out.append("<tr>" + "".join(f"<{tag if i == 0 else 'td'}>{_inline(c)}</{tag if i == 0 else 'td'}>"
                                    for i, c in enumerate(row)) + "</tr>")
    return "\n".join(out + ["</tbody></table>"])


def summary_html(lines: list, folder: str) -> str:
    """summary.md (its lines) as a self-contained web page, with the figures embedded."""
    title = lines[0].lstrip("# ").strip() if lines else "pyair2stream run summary"
    body, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            body.append(_table_html(lines[i:j]))
            i = j
            continue
        if line.startswith("- "):
            j = i
            while j < len(lines) and lines[j].startswith("- "):
                j += 1
            body.append("<ul>" + "".join(f"<li>{_inline(x[2:])}</li>" for x in lines[i:j]) + "</ul>")
            i = j
            continue
        figure = re.fullmatch(r"!\[(.*)\]\((.+)\)", line)
        if figure:
            name = urllib.parse.unquote(figure.group(2))
            src = _image_data(os.path.join(folder, name))
            caption = f"{_inline(figure.group(1))} (<a href=\"{figure.group(2)}\"><code>{html.escape(name)}</code></a>)"
            if src:
                body.append(f'<figure><img src="{src}" alt="{html.escape(figure.group(1))}">'
                            f"<figcaption>{caption}</figcaption></figure>")
        elif line.startswith("## "):
            body.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.strip():
            body.append(f"<p>{_inline(line)}</p>")
        i += 1
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{html.escape(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
            + "\n".join(body) + "\n</body>\n</html>\n")
