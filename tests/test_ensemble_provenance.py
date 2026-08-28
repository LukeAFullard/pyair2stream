"""
Regression tests for docs/audit/12_ensemble_provenance_and_pairing.md.

`scenario.paired_difference(ens_a, ens_b)` is documented as requiring both
ensembles to have been generated from the SAME parameter draws in the SAME order,
but nothing enforced it: two `forward_mode()` runs with omitted or mismatched
`forward_options.random_seed` would silently pass the shape-only check and produce
a plausible-shaped but statistically meaningless "paired" difference. These tests
check that:

1. `forward_options.reuse_sample_indices_from` makes a second `forward_mode()` call
   reuse the exact `sample_indices` a prior run saved, byte-for-byte, regardless of
   global numpy random state at the time of the second call.
2. `scenario.paired_difference_from_files()` succeeds and returns a sensible result
   when both ensembles' saved provenance actually matches.
3. It raises `ValueError` when the two ensembles' saved `sample_indices`, requested
   sample count, or source chain identity do not match.
4. The plain, shape-only `scenario.paired_difference()` is unaffected (still used
   internally, and covered directly by tests/test_report04_uncertainty_and_mcmc.py).
"""

import json
import os
import shutil
import unittest

import numpy as np
import pandas as pd

from pyair2stream.config import CommonData, PI
from pyair2stream.optimization import forward_mode
from pyair2stream import scenario

STABLE_PAR = [1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1]


def _build_forward_data(folder, q_scale=1.0):
    """A small, otherwise-normal FORWARD-mode CommonData (CRN, unconditionally
    stable -- no divergence is exercised in these tests, only provenance)."""
    data = CommonData()
    data.n_tot = 365 + 10
    dates = pd.date_range('2000-01-01', periods=data.n_tot, freq='D')
    data.date = np.column_stack([dates.year, dates.month, dates.day]).astype(np.int32)
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
        data.Q[i] = q_scale * (10.0 + 5.0 * np.cos(2.0 * PI * data.tt[i]))

    data.par = np.array(STABLE_PAR, dtype=np.float64)
    data.par_best = data.par.copy()
    data.flag_par = np.ones(8, dtype=np.bool_)

    data.folder = folder
    os.makedirs(data.folder, exist_ok=True)
    return data


def _write_chain_csv(path, n_rows, base=STABLE_PAR, jitter=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.array(base) + jitter * rng.standard_normal((n_rows, 8))
    pd.DataFrame(rows, columns=[f"par_{j+1}" for j in range(8)]).to_csv(path, index=False)


class TestReuseSampleIndicesFrom(unittest.TestCase):
    def setUp(self):
        self.folder_a = 'test_prov_output_a'
        self.folder_b = 'test_prov_output_b'
        self.data_a = _build_forward_data(self.folder_a, q_scale=1.0)
        self.data_b = _build_forward_data(self.folder_b, q_scale=1.3)  # scenario B: different flow

        self.chain_path = os.path.join(self.folder_a, 'chain.csv')
        _write_chain_csv(self.chain_path, n_rows=20)

    def tearDown(self):
        for folder in (self.folder_a, self.folder_b):
            if os.path.exists(folder):
                shutil.rmtree(folder)

    def _run(self, data, n_samples=6, reuse_from=None, seed=42):
        data.forward_options = {
            'enable_prediction_intervals': True,
            'mcmc_chain_path': self.chain_path,
            'residual_sigma': 1.0,
            'n_samples': n_samples,
            'random_seed': seed,
        }
        if reuse_from is not None:
            data.forward_options['reuse_sample_indices_from'] = reuse_from
        data.uncertainty_options = {'noise_model': 'iid', 'save_ensemble': True}
        forward_mode(data)
        ensemble_path = os.path.join(
            data.folder,
            f"Forward_Prediction_Ensemble_{data.station}_{data.series}_{data.time_res}.npz",
        )
        meta_path = ensemble_path.replace('.npz', '_meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        return ensemble_path, meta

    def test_reuse_produces_byte_identical_sample_indices_regardless_of_global_state(self):
        ensemble_a, meta_a = self._run(self.data_a, n_samples=6, seed=42)

        # Perturb global numpy random state between the two runs -- reuse must not
        # depend on it at all (the whole point of the mechanism).
        np.random.seed(12345)
        np.random.rand(500)

        ensemble_b, meta_b = self._run(
            self.data_b, reuse_from=ensemble_a.replace('.npz', '_meta.json'),
        )

        self.assertEqual(meta_a['sample_indices'], meta_b['sample_indices'])
        self.assertEqual(meta_a['chain_content_sha256'], meta_b['chain_content_sha256'])
        self.assertEqual(meta_b['reused_sample_indices_from'], ensemble_a.replace('.npz', '_meta.json'))

    def test_reuse_ignores_requested_n_samples(self):
        # Once reusing, the number of samples is whatever the prior run saved, not
        # whatever n_samples this call happens to request.
        ensemble_a, meta_a = self._run(self.data_a, n_samples=6, seed=1)
        meta_a_path = ensemble_a.replace('.npz', '_meta.json')

        ensemble_b, meta_b = self._run(self.data_b, n_samples=999, reuse_from=meta_a_path)
        self.assertEqual(len(meta_b['sample_indices']), 6)
        self.assertEqual(meta_a['sample_indices'], meta_b['sample_indices'])

    def test_reuse_from_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self._run(self.data_a, reuse_from=os.path.join(self.folder_a, 'does_not_exist_meta.json'))

    def test_reuse_from_different_chain_raises(self):
        ensemble_a, meta_a = self._run(self.data_a, n_samples=6, seed=1)
        meta_a_path = ensemble_a.replace('.npz', '_meta.json')

        other_chain_path = os.path.join(self.folder_b, 'other_chain.csv')
        _write_chain_csv(other_chain_path, n_rows=20, seed=999)  # different content

        with self.assertRaisesRegex(ValueError, r'different MCMC chain'):
            self.data_b.forward_options = {
                'enable_prediction_intervals': True,
                'mcmc_chain_path': other_chain_path,
                'residual_sigma': 1.0,
                'n_samples': 6,
                'random_seed': 42,
                'reuse_sample_indices_from': meta_a_path,
            }
            self.data_b.uncertainty_options = {'noise_model': 'iid', 'save_ensemble': True}
            forward_mode(self.data_b)


class TestPairedDifferenceFromFiles(unittest.TestCase):
    def setUp(self):
        self.folder_a = 'test_prov_pair_a'
        self.folder_b = 'test_prov_pair_b'
        self.chain_path = os.path.join('.', 'test_prov_pair_chain.csv')
        _write_chain_csv(self.chain_path, n_rows=20)

    def tearDown(self):
        for folder in (self.folder_a, self.folder_b):
            if os.path.exists(folder):
                shutil.rmtree(folder)
        if os.path.exists(self.chain_path):
            os.remove(self.chain_path)

    def _run(self, folder, q_scale, n_samples=6, reuse_from=None, seed=42, chain_path=None):
        data = _build_forward_data(folder, q_scale=q_scale)
        data.forward_options = {
            'enable_prediction_intervals': True,
            'mcmc_chain_path': chain_path or self.chain_path,
            'residual_sigma': 1.0,
            'n_samples': n_samples,
            'random_seed': seed,
        }
        if reuse_from is not None:
            data.forward_options['reuse_sample_indices_from'] = reuse_from
        data.uncertainty_options = {'noise_model': 'iid', 'save_ensemble': True}
        forward_mode(data)
        return os.path.join(
            folder, f"Forward_Prediction_Ensemble_{data.station}_{data.series}_{data.time_res}.npz"
        )

    def test_matching_provenance_succeeds_and_returns_sensible_result(self):
        ensemble_a = self._run(self.folder_a, q_scale=1.0, n_samples=6, seed=7)
        meta_a_path = ensemble_a.replace('.npz', '_meta.json')
        ensemble_b = self._run(self.folder_b, q_scale=1.4, reuse_from=meta_a_path)

        diff = scenario.paired_difference_from_files(ensemble_a, ensemble_b)
        ens_a, _ = scenario.load_ensemble(ensemble_a)
        ens_b, _ = scenario.load_ensemble(ensemble_b)
        self.assertEqual(diff.shape, ens_a.shape)
        np.testing.assert_allclose(diff, ens_a - ens_b)
        # Scenario B has systematically higher flow -> a real, non-trivial temperature
        # difference is expected (sanity: not all-zero).
        self.assertGreater(np.max(np.abs(diff)), 1e-6)

    def test_mismatched_sample_indices_raises(self):
        # Two independent draws (no reuse) will almost certainly disagree.
        ensemble_a = self._run(self.folder_a, q_scale=1.0, n_samples=6, seed=1)
        ensemble_b = self._run(self.folder_b, q_scale=1.0, n_samples=6, seed=2)

        with self.assertRaisesRegex(ValueError, r'sample_indices'):
            scenario.paired_difference_from_files(ensemble_a, ensemble_b)

    def test_mismatched_n_samples_raises(self):
        ensemble_a = self._run(self.folder_a, q_scale=1.0, n_samples=5, seed=1)
        ensemble_b = self._run(self.folder_b, q_scale=1.0, n_samples=6, seed=1)

        with self.assertRaisesRegex(ValueError, r'requested sample count'):
            scenario.paired_difference_from_files(ensemble_a, ensemble_b)

    def test_mismatched_chain_raises(self):
        other_chain_path = os.path.join('.', 'test_prov_pair_other_chain.csv')
        _write_chain_csv(other_chain_path, n_rows=20, seed=999)
        try:
            ensemble_a = self._run(self.folder_a, q_scale=1.0, n_samples=6, seed=1)
            ensemble_b = self._run(self.folder_b, q_scale=1.0, n_samples=6, seed=1,
                                    chain_path=other_chain_path)
            with self.assertRaisesRegex(ValueError, r'source chain'):
                scenario.paired_difference_from_files(ensemble_a, ensemble_b)
        finally:
            if os.path.exists(other_chain_path):
                os.remove(other_chain_path)

    def test_missing_sidecar_raises_file_not_found(self):
        ensemble_a = self._run(self.folder_a, q_scale=1.0, n_samples=6, seed=1)
        meta_path = ensemble_a.replace('.npz', '_meta.json')
        os.remove(meta_path)
        ensemble_b = self._run(self.folder_b, q_scale=1.0, n_samples=6, seed=1)

        with self.assertRaises(FileNotFoundError):
            scenario.paired_difference_from_files(ensemble_a, ensemble_b)


class TestPlainPairedDifferenceUnaffected(unittest.TestCase):
    """The shape-only helper must keep working exactly as before for advanced/
    same-process callers -- see tests/test_report04_uncertainty_and_mcmc.py for the
    primary coverage; this is a minimal smoke check in this file's own context."""

    def test_paired_difference_shape_only_still_works(self):
        ens_a = np.array([[1.0, 2.0], [3.0, 4.0]])
        ens_b = np.array([[0.5, 0.5], [0.5, 0.5]])
        diff = scenario.paired_difference(ens_a, ens_b)
        np.testing.assert_allclose(diff, ens_a - ens_b)

    def test_paired_difference_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            scenario.paired_difference(np.zeros((2, 3)), np.zeros((2, 4)))


if __name__ == '__main__':
    unittest.main()
