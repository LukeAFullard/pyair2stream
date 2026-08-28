"""
Regression tests for docs/audit/11_ensemble_divergence_handling.md.

`check_numerical_divergence` only ever ran on the single deterministic best-fit
simulation -- never inside the posterior/prediction-interval ensemble loops in
`optimization.forward_mode()` and `optimization._run_mcmc_uncertainty()`, which each
call `call_model()` once per posterior draw (hundreds to ~1000 times) with no
divergence check at all. A single bad draw either crashed the whole batch, or (if it
stayed finite) was silently written into the percentile envelope and raw ensemble
`scenario.paired_difference` consumes. These tests check that:

1. `optimization._check_ensemble_divergence` (the helper shared by both loops)
   drops/reports a small fraction of divergent draws, raises above
   `max_divergent_fraction`, and always raises if every draw diverges.
2. `forward_mode()`'s prediction-interval loop is actually wired to this: a single
   divergent posterior draw is excluded from the saved ensemble/envelope and
   reported in the new `Forward_Prediction_Envelopes_*_meta.json` sidecar, without
   crashing the run.
3. `uncertainty_options.on_divergent_draw: "raise"` makes the same scenario raise.
4. Existing forward_mode/DE-MCMC/DE-CV-MCMC behaviour on well-behaved input is
   unchanged (covered by the full existing suite; see tests/test_optimization.py).
"""

import json
import os
import shutil
import unittest

import numpy as np
import pandas as pd

from pyair2stream.config import CommonData, PI
from pyair2stream.model import NumericalDivergenceError
from pyair2stream.optimization import forward_mode, _check_ensemble_divergence


class TestCheckEnsembleDivergenceUnit(unittest.TestCase):
    """Direct tests of the shared drop/raise/report helper."""

    def test_no_excluded_draws_returns_summary_without_raising(self):
        summary = _check_ensemble_divergence(5, [], 'drop', 0.10, 'test')
        self.assertEqual(summary['n_draws_requested'], 5)
        self.assertEqual(summary['n_divergent_draws_excluded'], 0)
        self.assertEqual(summary['divergent_draw_fraction'], 0.0)
        self.assertEqual(summary['excluded_draws'], [])

    def test_small_excluded_fraction_drops_and_reports(self):
        excluded = [{"draw_index": 2, "chain_row": 7, "params": {"par_1": 999.0}}]
        summary = _check_ensemble_divergence(10, excluded, 'drop', 0.5, 'test')
        self.assertEqual(summary['n_divergent_draws_excluded'], 1)
        self.assertAlmostEqual(summary['divergent_draw_fraction'], 0.1)
        self.assertEqual(summary['excluded_draws'], excluded)

    def test_fraction_above_threshold_raises(self):
        excluded = [{"draw_index": i, "chain_row": i, "params": {}} for i in range(4)]
        with self.assertRaises(NumericalDivergenceError):
            _check_ensemble_divergence(10, excluded, 'drop', 0.10, 'test')

    def test_all_draws_divergent_raises_even_with_lenient_threshold(self):
        # A depleted (empty) ensemble must never be silently returned as if it had
        # succeeded, regardless of how generous max_divergent_fraction is.
        excluded = [{"draw_index": i, "chain_row": i, "params": {}} for i in range(5)]
        with self.assertRaises(NumericalDivergenceError):
            _check_ensemble_divergence(5, excluded, 'drop', 1.0, 'test')

    def test_zero_requested_draws_raises_with_distinct_message(self):
        # n_total=0 (e.g. an empty/misconfigured chain) is not itself "every draw
        # diverged" -- the message must say so distinctly, not conflate the two.
        with self.assertRaisesRegex(NumericalDivergenceError, r'No test draws were requested'):
            _check_ensemble_divergence(0, [], 'drop', 0.10, 'test')

    def test_valid_draw_indices_computed_when_sample_indices_given(self):
        sample_indices = np.array([10, 20, 30, 40, 50])
        excluded = [{"draw_index": 1, "chain_row": 20, "params": {}}]
        summary = _check_ensemble_divergence(
            5, excluded, 'drop', 0.5, 'test', sample_indices=sample_indices
        )
        self.assertEqual(summary['valid_draw_indices'], [10, 30, 40, 50])


def _build_forward_data(folder):
    """A small, otherwise-normal FORWARD-mode CommonData, modelled on
    tests/test_optimization.py's setUp -- CRN (unconditionally stable) so any
    divergence in these tests comes only from the deliberately extreme parameter
    draw, never from integrator instability."""
    data = CommonData()
    data.n_tot = 365 + 10
    data.date = np.zeros((data.n_tot, 3), dtype=np.int32)
    data.tt = np.zeros(data.n_tot, dtype=np.float64)
    data.Tair = np.zeros(data.n_tot, dtype=np.float64)
    data.Q = np.zeros(data.n_tot, dtype=np.float64)
    data.Twat_obs = np.full(data.n_tot, -999.0, dtype=np.float64)
    data.Twat_mod = np.zeros(data.n_tot, dtype=np.float64)

    data.version = 8
    data.mod_num = 'CRN'
    data.time_res = '1d'
    data.fun_obj = 'RMS'
    data.runmode = 'FORWARD'
    data.station = 'test_station'
    data.series = 'test_series'
    data.Qmedia = np.float64(10.0)
    data.Tice_cover = np.float64(0.0)
    data.max_plausible_twat = np.float64(60.0)
    data.stability_error_fraction = np.float64(0.10)

    for i in range(data.n_tot):
        data.tt[i] = np.float64(i / 365.0)
        data.Tair[i] = 15.0 + 10.0 * np.sin(2.0 * PI * data.tt[i])
        data.Q[i] = 10.0 + 5.0 * np.cos(2.0 * PI * data.tt[i])

    data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)
    data.par_best = data.par.copy()
    data.flag_par = np.ones(8, dtype=np.bool_)

    data.folder = folder
    os.makedirs(data.folder, exist_ok=True)
    return data


# A stable version-8 parameter set (matches _build_forward_data's own par), and a
# deliberately divergent one (a1=1e6 blows the CRN-computed equilibrium temperature
# far past max_plausible_twat on the very first step -- no reliance on integrator
# instability, since CRN is unconditionally stable).
STABLE_PAR = [1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1]
DIVERGENT_PAR = [1.0e6, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1]


def _write_chain_csv(path, rows):
    pd.DataFrame(rows, columns=[f"par_{j+1}" for j in range(8)]).to_csv(path, index=False)


class TestForwardModeEnsembleDivergence(unittest.TestCase):
    def setUp(self):
        self.folder = 'test_ensemble_divergence_output'
        self.data = _build_forward_data(self.folder)

    def tearDown(self):
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)

    def _configure(self, chain_rows, on_divergent_draw='drop', max_divergent_fraction=0.5,
                    save_ensemble=True):
        chain_path = os.path.join(self.folder, 'chain.csv')
        _write_chain_csv(chain_path, chain_rows)
        self.data.forward_options = {
            'enable_prediction_intervals': True,
            'mcmc_chain_path': chain_path,
            'residual_sigma': 1.0,
            'n_samples': len(chain_rows),
            'random_seed': 42,
        }
        self.data.uncertainty_options = {
            'noise_model': 'iid',
            'save_ensemble': save_ensemble,
            'on_divergent_draw': on_divergent_draw,
            'max_divergent_fraction': max_divergent_fraction,
        }
        return chain_path

    def test_single_divergent_draw_is_dropped_and_reported(self):
        rows = [STABLE_PAR] * 4 + [DIVERGENT_PAR]
        self._configure(rows, on_divergent_draw='drop', max_divergent_fraction=0.5)

        forward_mode(self.data)  # must not raise

        ensemble_path = os.path.join(
            self.folder, "Forward_Prediction_Ensemble_test_station_test_series_1d.npz"
        )
        with np.load(ensemble_path) as npz:
            ensemble = npz['simulations']
        self.assertEqual(ensemble.shape[0], 4)  # the divergent draw was excluded
        self.assertTrue(np.all(np.isfinite(ensemble)))

        meta_path = os.path.join(
            self.folder, "Forward_Prediction_Ensemble_test_station_test_series_1d_meta.json"
        )
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta['n_draws_requested'], 5)
        self.assertEqual(meta['n_divergent_draws_excluded'], 1)
        self.assertAlmostEqual(meta['divergent_draw_fraction'], 0.2)
        self.assertEqual(len(meta['excluded_draws']), 1)
        self.assertEqual(meta['on_divergent_draw'], 'drop')

    def test_on_divergent_draw_raise_raises_instead_of_dropping(self):
        rows = [STABLE_PAR] * 4 + [DIVERGENT_PAR]
        self._configure(rows, on_divergent_draw='raise', max_divergent_fraction=0.5)

        with self.assertRaises(NumericalDivergenceError):
            forward_mode(self.data)

    def test_all_draws_divergent_raises_rather_than_empty_success(self):
        rows = [DIVERGENT_PAR] * 3
        self._configure(rows, on_divergent_draw='drop', max_divergent_fraction=0.10)

        with self.assertRaises(NumericalDivergenceError):
            forward_mode(self.data)

    def test_no_divergent_draws_behaves_as_before(self):
        rows = [STABLE_PAR] * 5
        self._configure(rows, on_divergent_draw='drop', max_divergent_fraction=0.10)

        forward_mode(self.data)  # must not raise

        meta_path = os.path.join(
            self.folder, "Forward_Prediction_Ensemble_test_station_test_series_1d_meta.json"
        )
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta['n_divergent_draws_excluded'], 0)
        self.assertEqual(meta['excluded_draws'], [])


if __name__ == '__main__':
    unittest.main()
