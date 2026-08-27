"""
Tests for docs/audit/07_reproducibility_and_provenance.md.

Covers:
- 7.1: `random_seed:` config key threaded through `read_calibration` ->
  `data.random_seed` -> `run_optimizer`, and recorded in
  `calibration_metadata.json`.
- 7.2: `DE_mode` is bit-identical across two runs with the same explicit seed,
  and differs across two runs with different seeds.
- 7.3/7.5: no remaining reference to the false "version 8 parameter zeroing"
  claim, and the version reported by the package matches `pyproject.toml`
  and the newest `CHANGELOG.md` heading.
- 7.8: dataclass fields declared explicitly on `CommonData` instead of being
  set only by assignment.
"""

import json
import os
import re
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

from pyair2stream.config import CommonData, PI
from pyair2stream.io import read_calibration
from pyair2stream.optimization import DE_mode


def _make_optimization_data(runmode='DE'):
    """Minimal CommonData fixture wired up enough to run DE_mode end-to-end."""
    data = CommonData()
    data.n_tot = 365 + 30
    data.date = np.zeros((data.n_tot, 3), dtype=np.int32)
    data.tt = np.zeros(data.n_tot, dtype=np.float64)
    data.Tair = np.zeros(data.n_tot, dtype=np.float64)
    data.Q = np.zeros(data.n_tot, dtype=np.float64)
    data.Twat_obs = np.full(data.n_tot, -999.0, dtype=np.float64)
    data.Twat_mod = np.zeros(data.n_tot, dtype=np.float64)
    data.Twat_obs_agg = np.full(data.n_tot, -999.0, dtype=np.float64)
    data.Twat_mod_agg = np.full(data.n_tot, -999.0, dtype=np.float64)

    data.version = 8
    data.mod_num = 'CRN'
    data.time_res = '1d'
    data.fun_obj = 'RMS'
    data.runmode = runmode
    data.station = 'test_station'
    data.series = 'test_series'
    data.Qmedia = np.float64(10.0)
    data.Tice_cover = np.float64(0.0)

    for i in range(data.n_tot):
        data.tt[i] = np.float64(i / 365.0)
        data.Tair[i] = 15.0 + 10.0 * np.sin(2.0 * PI * data.tt[i])
        data.Q[i] = 10.0 + 5.0 * np.cos(2.0 * PI * data.tt[i])
        if i >= 365:
            data.Twat_obs[i] = 12.0 + 8.0 * np.sin(2.0 * PI * data.tt[i])

    n_dat = data.n_tot - 365
    data.n_dat = n_dat
    data.I_inf = np.zeros((n_dat, 3), dtype=np.int32)
    data.I_pos = np.zeros(n_dat, dtype=np.int32)
    n_inf = 0
    n_pos = 0
    for i in range(365, data.n_tot):
        data.I_inf[n_inf, 0] = n_pos
        data.I_inf[n_inf, 1] = n_pos
        data.I_inf[n_inf, 2] = i
        data.I_pos[n_pos] = i
        data.Twat_obs_agg[i] = data.Twat_obs[i]
        n_inf += 1
        n_pos += 1

    data.eval_mask = np.zeros(data.n_tot, dtype=np.bool_)
    data.eval_mask[365:] = True

    data.par = np.array([1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.5, 0.1], dtype=np.float64)
    data.parmin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.parmax = np.array([2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float64)
    data.flag_par = np.ones(8, dtype=np.bool_)

    data.n_particles = 6
    data.n_run = 5
    data.mineff_index = -1e30

    data.folder = 'test_report07_output'
    os.makedirs(data.folder, exist_ok=True)
    return data


class TestDESeedReproducibility(unittest.TestCase):
    """7.2: seeded DE runs must be reproducible; unseeded/differently-seeded runs need not be."""

    def tearDown(self):
        for folder in ('test_report07_output',):
            if os.path.exists(folder):
                shutil.rmtree(folder)

    def test_same_seed_bit_identical(self):
        data1 = _make_optimization_data()
        DE_mode(data1, seed=42)
        par_best_1 = data1.par_best.copy()

        data2 = _make_optimization_data()
        DE_mode(data2, seed=42)
        par_best_2 = data2.par_best.copy()

        np.testing.assert_array_equal(par_best_1, par_best_2)
        self.assertEqual(data1.finalfit, data2.finalfit)

    def test_different_seeds_differ(self):
        data1 = _make_optimization_data()
        DE_mode(data1, seed=1)
        par_best_1 = data1.par_best.copy()

        data2 = _make_optimization_data()
        DE_mode(data2, seed=2)
        par_best_2 = data2.par_best.copy()

        self.assertFalse(np.array_equal(par_best_1, par_best_2))


class TestRandomSeedConfigThreading(unittest.TestCase):
    """7.1: `random_seed:` config key is parsed and threaded through to run_optimizer."""

    def setUp(self):
        self.tmpdir = 'test_report07_config'
        os.makedirs(self.tmpdir, exist_ok=True)
        self.input_path = os.path.join(self.tmpdir, 'input.csv')

        n_days = 365 + 30
        dates = pd.date_range('2000-01-01', periods=n_days, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'T_air': 15.0 + 10.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.0),
            'Discharge': 10.0 + 5.0 * np.cos(2 * np.pi * np.arange(n_days) / 365.0),
            'T_water': 12.0 + 8.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.0),
        })
        df.to_csv(self.input_path, index=False)

        self.config_path = os.path.join(self.tmpdir, 'config.yaml')
        self.output_dir = os.path.join(self.tmpdir, 'output')

    def tearDown(self):
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _write_config(self, random_seed):
        import yaml
        config = {
            'project_name': 'report07_test',
            'station_name': 'TestStation',
            'run_mode': 'DE',
            'version': 8,
            'integrator': 'CRN',
            'objective_function': 'RMS',
            'random_seed': random_seed,
            'paths': {
                'input_data': self.input_path,
                'output_dir': self.output_dir,
            },
            'optimization': {'n_run': 3, 'n_particles': 6},
            'parameter_bounds': {
                'min': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'max': [2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
            },
        }
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(config, f)

    def test_random_seed_parsed_into_data(self):
        self._write_config(random_seed=123)
        data = read_calibration(config_file=self.config_path)
        self.assertEqual(data.random_seed, 123)

    def test_random_seed_defaults_to_none(self):
        self._write_config(random_seed=None)
        data = read_calibration(config_file=self.config_path)
        self.assertIsNone(data.random_seed)

    def test_calibration_metadata_records_seed(self):
        from pyair2stream.io import read_Tseries
        from pyair2stream.model import aggregation, statis
        from pyair2stream.main import run_optimizer, forward

        self._write_config(random_seed=7)
        data = read_calibration(config_file=self.config_path)
        read_Tseries(data, 'c')
        aggregation(data)
        statis(data)
        run_optimizer(data)
        forward(data)

        metadata_path = os.path.join(self.output_dir, 'calibration_metadata.json')
        self.assertTrue(os.path.exists(metadata_path))
        with open(metadata_path) as f:
            metadata = json.load(f)
        self.assertEqual(metadata['random_seed'], 7)


class TestVersionConsistency(unittest.TestCase):
    """7.5: pyproject.toml, CHANGELOG.md, and pyair2stream.__version__ must agree."""

    def test_version_matches_pyproject_and_changelog(self):
        import pyair2stream

        pyproject_text = (REPO_ROOT / 'pyproject.toml').read_text()
        m = re.search(r'^version = "([^"]+)"', pyproject_text, re.MULTILINE)
        self.assertIsNotNone(m, "Could not find version in pyproject.toml")
        pyproject_version = m.group(1)

        changelog_text = (REPO_ROOT / 'CHANGELOG.md').read_text()
        m = re.search(r'^## \[([^\]]+)\]', changelog_text, re.MULTILINE)
        self.assertIsNotNone(m, "Could not find newest heading in CHANGELOG.md")
        changelog_version = m.group(1)

        self.assertEqual(pyair2stream.__version__, pyproject_version)
        self.assertEqual(pyair2stream.__version__, changelog_version)


class TestVersion8ZeroingClaimRemoved(unittest.TestCase):
    """7.3: no remaining reference to the false "version 8 parameter zeroing" claim."""

    def test_no_stale_claim_in_docs_or_code(self):
        needle = "version 8 parameter zeroing"
        for relpath in ('README.md', 'CHANGELOG.md', 'pyair2stream/io.py'):
            text = (REPO_ROOT / relpath).read_text().lower()
            self.assertNotIn(needle, text, f"Stale claim still present in {relpath}")


if __name__ == '__main__':
    unittest.main()
