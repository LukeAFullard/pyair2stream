"""
Tests for the DE-MCMC sampler set-up (optimization._make_sampler and the
adaptive, convergence-checked run).

The known-answer test matters most: convergence diagnostics alone cannot show
that a sampler is correct. emcee's DEMove + DESnookerMove combination passes
every diagnostic yet gives 90% intervals ~11% too narrow on a correlated 8-D
Gaussian (DESnookerMove alone never leaves its start point in emcee 3.1.6), so
the sampler the package uses is checked against a target with a known answer.
"""

import json
import os

import emcee
import numpy as np
import pandas as pd
import pytest
import yaml

from pyair2stream import optimization
from pyair2stream.io import read_calibration, read_Tseries
from pyair2stream.model import aggregation, statis
from pyair2stream.optimization import DE_MCMC_mode, _make_sampler, _run_until_converged

QUICKSTART = os.path.join(os.path.dirname(__file__), "..", "examples", "quickstart", "data", "calibration_data.csv")


# Mean, standard deviations and correlations of the Mentue (MAH-2369) version-8
# posterior (AR(1) likelihood, CRN). Parameters differ in scale by ~100x and
# several pairs are correlated above 0.95: the conditions under which emcee's
# DESnookerMove gives intervals ~11% too narrow.
MENTUE_V8_MU = [1.11396, 0.80859, 0.95866, -0.04789, 4.64286, 3.23026, 0.59084, 0.47755]
MENTUE_V8_SD = [0.10022, 0.022882, 0.028101, 0.028593, 0.2896, 0.19391, 0.0024962, 0.028479]
MENTUE_V8_CORR = [
    [1.0000, 0.2320, 0.4649, -0.1145, -0.3344, -0.1884, 0.1228, -0.3079],
    [0.2320, 1.0000, 0.9623, -0.5855, 0.5443, 0.5826, -0.0519, 0.5368],
    [0.4649, 0.9623, 1.0000, -0.5533, 0.4008, 0.4758, -0.0208, 0.3914],
    [-0.1145, -0.5855, -0.5533, 1.0000, -0.2930, -0.3178, 0.0423, -0.2871],
    [-0.3344, 0.5443, 0.4008, -0.2930, 1.0000, 0.9606, -0.2947, 0.9876],
    [-0.1884, 0.5826, 0.4758, -0.3178, 0.9606, 1.0000, -0.2853, 0.9521],
    [0.1228, -0.0519, -0.0208, 0.0423, -0.2947, -0.2853, 1.0000, -0.2894],
    [-0.3079, 0.5368, 0.3914, -0.2871, 0.9876, 0.9521, -0.2894, 1.0000],
]


def test_sampler_uses_de_move_only():
    s = _make_sampler(16, 3, lambda x: 0.0, seed=1)
    assert len(s._moves) == 1 and type(s._moves[0]) is emcee.moves.DEMove


def test_sampler_reproduces_known_gaussian_interval_widths():
    mu = np.array(MENTUE_V8_MU)
    sd = np.array(MENTUE_V8_SD)
    cov = np.array(MENTUE_V8_CORR) * np.outer(sd, sd)
    icov = np.linalg.inv(cov)

    def logp(x):
        d = x - mu
        return -0.5 * d @ icov @ d

    rng = np.random.default_rng(1)
    p0 = mu + 1e-3 * sd * rng.standard_normal((32, 8))   # tight start, as in the package
    s = _make_sampler(32, 8, logp, seed=1)
    s.run_mcmc(p0, 10000, progress=False)
    chain = s.get_chain(discard=3000, flat=True)
    width = np.subtract(*np.percentile(chain, [95, 5], axis=0))
    exact = 2 * 1.6448536 * sd
    np.testing.assert_allclose(width / exact, 1.0, atol=0.05)


def test_adaptive_run_stops_once_converged():
    mu, cov = np.zeros(3), np.diag([1.0, 2.0, 0.5])
    icov = np.linalg.inv(cov)
    s = _make_sampler(16, 3, lambda x: -0.5 * x @ icov @ x, seed=2)
    p0 = np.random.default_rng(2).normal(size=(16, 3)) * 1e-3
    diag = _run_until_converged(s, p0, max_steps=20000, uncertainty_options={})
    assert diag["converged"]
    assert diag["steps"] < 20000 and diag["steps"] % optimization.MCMC_CHECK_INTERVAL == 0
    assert diag["steps"] >= optimization.MCMC_TAU_FACTOR * diag["max_tau"]
    assert diag["max_rhat"] < optimization.MCMC_MAX_RHAT


def _quickstart_data(tmp_path, **uncertainty):
    cfg = {"station_name": "S", "version": 3, "run_mode": "DE-MCMC", "random_seed": 1,
           "optimization": {"n_run": 3, "n_particles": 3, "mcmc_walkers": 8, "mcmc_steps": 50},
           "parameter_bounds": {"min": [-5, -5, -5, -1, 0, 0, 0, -1], "max": [15, 1.5, 5, 1, 20, 10, 1, 5]},
           "uncertainty_options": uncertainty,
           "paths": {"input_data": QUICKSTART, "output_dir": str(tmp_path / "out")}}
    with open(tmp_path / "c.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    data = read_calibration(str(tmp_path / "c.yaml"))
    read_Tseries(data, "c")
    aggregation(data)
    statis(data)
    return data


def test_unconverged_run_stops_by_default_and_keeps_diagnostics(tmp_path):
    data = _quickstart_data(tmp_path)
    with pytest.raises(RuntimeError, match="did not converge"):
        DE_MCMC_mode(data, seed=1)
    meta = json.load(open(tmp_path / "out" / "MCMC_chain_S_series_1d_meta.json"))
    assert meta["converged"] is False and meta["steps_run"] == 50
    assert not (tmp_path / "out" / "MCMC_envelopes_S_series_1d.csv").exists()


def test_unconverged_run_can_opt_out_and_is_marked(tmp_path):
    data = _quickstart_data(tmp_path, strict_convergence=False)
    DE_MCMC_mode(data, seed=1)
    meta = json.load(open(tmp_path / "out" / "MCMC_chain_S_series_1d_meta.json"))
    assert meta["converged"] is False
    assert (tmp_path / "out" / "MCMC_envelopes_S_series_1d.csv").exists()


def test_saved_chain_is_thinned(tmp_path):
    data = _quickstart_data(tmp_path, strict_convergence=False)
    data.mcmc_steps = 3000
    DE_MCMC_mode(data, seed=1)
    meta = json.load(open(tmp_path / "out" / "MCMC_chain_S_series_1d_meta.json"))
    rows = len(pd.read_csv(tmp_path / "out" / "MCMC_chain_S_series_1d.csv"))
    assert meta["thin"] >= 1
    assert rows == (meta["steps_run"] - meta["burnin"] + meta["thin"] - 1) // meta["thin"] * 8
