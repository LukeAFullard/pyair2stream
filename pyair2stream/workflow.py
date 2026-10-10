"""
pyair2stream.Model: the usual sequence of runs (calibrate, check by cross-validation,
predict) from one set of settings, with the files of each step passed on to the next.

    m = pyair2stream.Model("settings.yaml", output_dir="output")
    m.calibrate()                                     # DE-MCMC: parameters, their range, the error model
    m.check()                                         # cross-validation of the calibration years
    m.predict("validation.csv", name="prediction", conformal=True)
    m.predict(warmer, name="warmer", paired_with="prediction")    # a DataFrame; the same parameter sets

Each step is an ordinary `pyair2stream.run`. Its settings are written first, as
`<output_dir>/<step>.yaml`, and run from that file, so each step is recorded, and can be
run again from the command line (`pyair2stream --config output/<step>.yaml`), exactly as
if the settings had been written by hand. Nothing is kept only in memory: the files are the
record, as for any run (USER_GUIDE.md §7.3).
"""

import copy
import glob
import json
import os
from typing import Optional

import pandas as pd
import yaml

# Settings that must be the same in every step: the parameters mean something only with the
# settings they were fitted with. Set them in the Model's settings, not in a step's `changes`.
SHARED_SETTINGS = ("station_name", "water_station", "series", "time_resolution", "version", "integrator",
                   "Tice_cover", "min_theta_floor", "calendar", "drop_29_february", "Qmedia")
SHARED_UNCERTAINTY = ("noise_model", "rho_timescale", "ar1_rho", "residual_sigma", "likelihood")
CALIBRATION_MODES = ("DE", "PSO", "LATHYP")
RESERVED_NAMES = ("calibration", "check", "inputs", "calibration_data", "validation_data")


class Model:
    """
    One river, one model: calibrate, check and predict with the same settings.

    Parameters
    ----------
    settings : str or dict
        A settings file (YAML) or a dict with the same keys (USER_GUIDE.md §6): the station,
        model version, integrator, seed, parameter bounds, `uncertainty_options` (error model
        and level), `cross_validation` (for `check`) and `paths.input_data`, the calibration
        data. `paths.input_data` and `paths.validation_data` may also be DataFrames (columns
        Date, T_air, T_water, Discharge). `run_mode` is set by each step.
    output_dir : str, optional
        The folder for every step: `<output_dir>/calibration`, `<output_dir>/check` and one
        folder per prediction. Default: `paths.output_dir` of the settings, else "output".
        Paths are relative to the folder Python runs in, as for `pyair2stream.run`.
    verbose : bool
        Print each run's messages (they are also in each step's summary.md).
    """

    def __init__(self, settings, output_dir: Optional[str] = None, verbose: bool = True):
        if isinstance(settings, str):
            with open(settings) as f:
                settings = yaml.safe_load(f) or {}
        if not isinstance(settings, dict):
            raise TypeError("settings must be the path of a settings file or a dict")
        self.settings = copy.deepcopy({k: v for k, v in settings.items() if k != "paths"})
        self.paths = dict(settings.get("paths") or {})
        self.output_dir = output_dir or self.paths.pop("output_dir", None) or "output"
        self.paths.pop("output_dir", None)       # each step has a folder of its own in output_dir
        if self.paths.get("input_data") is None:
            raise ValueError("settings need paths.input_data: the calibration data")
        if self.settings.get("run_mode") == "FORWARD":
            raise ValueError("a Model's settings describe the calibration; leave run_mode out "
                             "(each step sets its own) or set DE, PSO, LATHYP or DE-MCMC")
        self.verbose = verbose
        self.calibration = None          # the RunResult of calibrate(), if run in this session
        self.calibration_dir = None      # the calibration's folder (calibrate or use_calibration)
        self.check_result = None
        self.check_dir = None
        self.predictions = {}            # name: RunResult
        self._prediction_dirs = {}       # name: folder

    # --- The steps ---------------------------------------------------------------------------------

    def calibrate(self, uncertainty: bool = True, changes: Optional[dict] = None):
        """
        Calibrate on `paths.input_data` (and score on `paths.validation_data`, if given).

        With `uncertainty` (the default), `DE-MCMC`: the best fit, then the MCMC sample of
        parameter sets and the error model that `predict` needs for prediction intervals.
        Without, a best fit only (`run_mode` of the settings if it is DE, PSO or LATHYP,
        else DE). `changes` (a dict, merged into the step's settings) changes this step only.
        Returns the run's RunResult.
        """
        mode = self.settings.get("run_mode", "DE")
        mode = "DE-MCMC" if uncertainty else (mode if mode in CALIBRATION_MODES else "DE")
        cfg = self._base(drop=("cross_validation",))
        cfg["run_mode"] = mode
        cfg["paths"].update(self._data_paths("calibration", validation=True))
        result = self._run("calibration", cfg, changes)
        self.calibration, self.calibration_dir = result, result.output_dir
        return result

    def check(self, cross_validation: Optional[dict] = None, changes: Optional[dict] = None):
        """
        Cross-validate on the calibration data: each year is hidden in turn and predicted
        from the others (USER_GUIDE.md §13). It uses the calibration's model settings,
        discharge scaling (`Qmedia`), error model and level, so its interval coverage and
        conformal margins apply to `predict`. Needs `calibrate` (or `use_calibration`) first.

        `cross_validation` (a dict) adds to the settings' `cross_validation` block, for
        example `{"threshold": 18, "season_months": [6, 7, 8, 9]}`. Returns the RunResult.
        """
        meta = self._calibration_metadata()
        mode = self.settings.get("run_mode", "DE")
        cfg = self._base()
        cfg["run_mode"] = mode if mode in CALIBRATION_MODES else "DE"
        cfg["Qmedia"] = float(meta["qmedia"])      # as the calibration, not recomputed in each fold
        cfg["cross_validation"] = {**(self.settings.get("cross_validation") or {}), **(cross_validation or {}),
                                   "enabled": True}
        cfg["paths"].update(self._data_paths("check", validation=False))
        result = self._run("check", cfg, changes)
        self.check_result, self.check_dir = result, result.output_dir
        return result

    def predict(self, data, name: str = "prediction", paired_with=None, conformal=False,
                intervals: Optional[bool] = None, n_samples: int = 1000, save_ensemble: bool = True,
                changes: Optional[dict] = None):
        """
        Simulate `data` (a file or a DataFrame: Date, T_air, Discharge, and T_water where it
        was measured) with the calibration: a FORWARD run, in `<output_dir>/<name>`.

        intervals : bool, optional
            Prediction intervals from the calibration's MCMC sample (default: when the
            calibration has one). Without, the best fit only.
        n_samples : int
            Parameter sets drawn from the sample (with intervals).
        save_ensemble : bool
            Keep every simulated series (`ensemble`, `difference`; 7-day and 30-day means).
        paired_with : str or RunResult, optional
            An earlier prediction (its name, folder or result): this one then uses the same
            parameter sets, in the same order, so the two can be compared simulation by
            simulation (`difference`). Its `n_samples` and seed are used.
        conformal : bool or str
            Widen the intervals by the conformal margins of `check` (True), or of a
            `cv_conformal_margins.csv` (its path).
        changes : dict, optional
            Merged into this step's settings.

        Returns the RunResult.
        """
        if name in RESERVED_NAMES:
            raise ValueError(f"name {name!r} is used for another step; choose another")
        if name in self._prediction_dirs:
            raise ValueError(f"a prediction named {name!r} was already made with this Model; choose another name")
        cal = self._need_calibration()
        meta_file = os.path.join(cal, "calibration_metadata.json")
        chains = _find(cal, "MCMC_chain_*.csv")
        if intervals is None:
            intervals = bool(chains)
        if intervals and not chains:
            raise ValueError(f"{cal} holds no MCMC sample: calibrate with uncertainty=True for intervals, "
                             "or predict with intervals=False")
        cfg = self._base(drop=("cross_validation", "sensitivity_analysis", "parameter_bounds",
                               "objective_function", "optimization", "run_mode"))
        cfg["run_mode"] = "FORWARD"
        cfg["paths"] = {"input_data": self._file(data, name), "calibration_metadata": meta_file}
        if intervals:
            options = cfg.setdefault("uncertainty_options", {})
            options["save_ensemble"] = bool(save_ensemble)
            forward = {"enable_prediction_intervals": True, "mcmc_chain_path": _one(chains, cal, "MCMC sample")}
            if paired_with is not None:
                forward["reuse_sample_indices_from"] = _one(
                    _find(self._prediction_dir(paired_with), "Forward_Prediction_Ensemble_*_meta.json"),
                    self._prediction_dir(paired_with), "record of the parameter sets used")
            else:
                forward["n_samples"] = int(n_samples)
                if self.settings.get("random_seed") is not None:
                    forward["random_seed"] = self.settings["random_seed"]
            if conformal:
                forward["conformal_margins"] = conformal if isinstance(conformal, str) else self._margins_file()
            cfg["forward_options"] = forward
        elif paired_with is not None or conformal:
            raise ValueError("paired_with and conformal need prediction intervals (intervals=True)")
        result = self._run(name, cfg, changes)
        self.predictions[name] = result
        self._prediction_dirs[name] = result.output_dir
        return result

    # --- Earlier results ------------------------------------------------------------------------------

    def use_calibration(self, folder: str):
        """Use a calibration made earlier (its output folder) instead of running `calibrate`."""
        if not os.path.exists(os.path.join(folder, "calibration_metadata.json")):
            raise FileNotFoundError(f"{folder} holds no calibration_metadata.json: not a calibration's output folder")
        self.calibration, self.calibration_dir = None, folder
        return self

    def use_check(self, folder: str):
        """Use a cross-validation made earlier (its output folder) instead of running `check`."""
        if not os.path.exists(os.path.join(folder, "cv_results.csv")):
            raise FileNotFoundError(f"{folder} holds no cv_results.csv: not a cross-validation's output folder")
        self.check_result, self.check_dir = None, folder
        return self

    def use_prediction(self, name: str, folder: str):
        """Use a prediction made earlier (its output folder) under `name`, for `paired_with`,
        `ensemble` and `difference`."""
        if not _find(folder, "Forward_Prediction_*"):
            raise FileNotFoundError(f"{folder} holds no FORWARD prediction")
        self._prediction_dirs[name] = folder
        return self

    # --- Reading the results --------------------------------------------------------------------------

    def ensemble(self, prediction):
        """(series, dates) of a prediction saved with `save_ensemble` (`scenario.load_ensemble`)."""
        from .scenario import load_ensemble
        return load_ensemble(self._ensemble_file(prediction))

    def difference(self, prediction, baseline):
        """`prediction` minus `baseline`, simulation by simulation, from paired predictions
        (`paired_with`; `scenario.paired_difference_from_files`, which checks the pairing)."""
        from .scenario import paired_difference_from_files
        return paired_difference_from_files(self._ensemble_file(prediction), self._ensemble_file(baseline))

    def margins(self):
        """The conformal margins of `check` (`scenario.read_conformal_margins`)."""
        from .scenario import read_conformal_margins
        return read_conformal_margins(self._margins_file())

    def settings_file(self, step: str) -> str:
        """The settings file a step was run from: `<output_dir>/<step>.yaml`."""
        return os.path.join(self.output_dir, f"{step}.yaml")

    # --- Helpers -------------------------------------------------------------------------------------

    def _base(self, drop=()):
        cfg = {k: copy.deepcopy(v) for k, v in self.settings.items() if k not in drop}
        cfg["paths"] = {}
        return cfg

    def _data_paths(self, step, validation):
        paths = {"input_data": self._file(self.paths["input_data"], "calibration_data")}
        if validation and self.paths.get("validation_data") is not None:
            paths["validation_data"] = self._file(self.paths["validation_data"], "validation_data")
        return paths

    def _file(self, data, name):
        """A data file's path; a DataFrame is first written to `<output_dir>/inputs/<name>.csv`."""
        if isinstance(data, (str, os.PathLike)):
            return os.fspath(data)
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be the path of a file or a pandas DataFrame")
        frame = data
        if "Date" not in frame.columns:
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError("a DataFrame needs a Date column, or dates as its index")
            frame = frame.rename_axis("Date").reset_index()
        folder = os.path.join(self.output_dir, "inputs")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{name}.csv")
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        return path

    def _run(self, step, cfg, changes):
        from .main import run
        if changes:
            _refuse_shared(changes)
            cfg = _merged(cfg, changes)
        cfg["paths"]["output_dir"] = os.path.join(self.output_dir, step)
        os.makedirs(self.output_dir, exist_ok=True)
        path = self.settings_file(step)
        with open(path, "w") as f:
            f.write(f"# Settings of the step '{step}', written by pyair2stream.Model and run from this file.\n"
                    f"# Paths are relative to the folder the run started in: {os.getcwd()}\n"
                    f"# To run this step again: pyair2stream --config {path}\n")
            yaml.safe_dump(_plain(cfg), f, sort_keys=False, allow_unicode=True)
        return run(path, verbose=self.verbose)

    def _need_calibration(self):
        if self.calibration_dir is None:
            raise ValueError("calibrate first (or use_calibration with the folder of an earlier calibration)")
        return self.calibration_dir

    def _calibration_metadata(self):
        with open(os.path.join(self._need_calibration(), "calibration_metadata.json")) as f:
            return json.load(f)

    def _margins_file(self):
        if self.check_dir is None:
            raise ValueError("check first (or use_check with the folder of an earlier cross-validation) "
                             "for the conformal margins")
        path = os.path.join(self.check_dir, "cv_conformal_margins.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} was not written: the cross-validation had too few held-out years "
                                    "with measurements for conformal margins")
        return path

    def _prediction_dir(self, prediction):
        if hasattr(prediction, "output_dir"):
            return prediction.output_dir
        if prediction in self._prediction_dirs:
            return self._prediction_dirs[prediction]
        if isinstance(prediction, str) and os.path.isdir(prediction):
            return prediction
        known = ", ".join(self._prediction_dirs) or "none yet"
        raise ValueError(f"no prediction named {prediction!r} (predictions of this Model: {known})")

    def _ensemble_file(self, prediction):
        folder = self._prediction_dir(prediction)
        return _one(_find(folder, "Forward_Prediction_Ensemble_*.npz"), folder,
                    "saved series (predict with save_ensemble=True)")


def _find(folder, pattern):
    return sorted(glob.glob(os.path.join(glob.escape(folder), pattern)))


def _one(found, folder, what):
    if len(found) != 1:
        raise FileNotFoundError(f"{folder}: expected one {what}, found {len(found)}"
                                + (f" ({', '.join(os.path.basename(p) for p in found)}); give each run "
                                   "a folder of its own" if found else ""))
    return found[0]


def _refuse_shared(changes):
    shared = [k for k in SHARED_SETTINGS if k in changes]
    shared += [f"uncertainty_options.{k}" for k in SHARED_UNCERTAINTY
               if k in (changes.get("uncertainty_options") or {})]
    shared += [f"paths.{k}" for k in (changes.get("paths") or {})]
    if shared:
        raise ValueError(f"{', '.join(shared)} must be the same in every step (or are set by the Model): "
                         "set them in the Model's settings instead of a step's changes")


def _plain(value):
    """Settings as plain Python values (numpy numbers and arrays included), for YAML."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):                 # numpy scalar or array
        return _plain(value.tolist())
    return value


def _merged(base, changes):
    out = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merged(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out
