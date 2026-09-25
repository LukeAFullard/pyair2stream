"""
Smoke tests for the examples under examples/.

Running the examples takes minutes, so CI runs them in a separate job
(.github/workflows/tests.yml, examples). These fast checks catch the simple
breakages: a script that no longer parses, a missing README or run.py, a
configuration that is not valid YAML, or a reference to the envelope columns
`Twat_mod_p5`/`Twat_mod_p95` (the real names are `Twat_mod_lower`/`Twat_mod_upper`).
"""

import ast
import os
import re
import unittest

import yaml

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
_REMOVED_COLUMN_RE = re.compile(r"Twat_mod_p(?:5|95)(?!\d)['\"]")


def _example_dirs():
    return sorted(os.path.join(EXAMPLES_DIR, d) for d in os.listdir(EXAMPLES_DIR)
                  if re.match(r"\d\d_", d) and os.path.isdir(os.path.join(EXAMPLES_DIR, d)))


def _files(ext):
    for d in _example_dirs():
        for name in sorted(os.listdir(d)):
            if name.endswith(ext):
                yield os.path.join(d, name)


class TestExamples(unittest.TestCase):
    def test_every_example_has_readme_and_run_script(self):
        dirs = _example_dirs()
        self.assertGreater(len(dirs), 0)
        for d in dirs:
            for name in ("README.md", "run.py"):
                self.assertTrue(os.path.exists(os.path.join(d, name)), f"{d} has no {name}")

    def test_every_script_parses(self):
        for path in _files(".py"):
            with open(path) as f:
                source = f.read()
            ast.parse(source, filename=path)
            self.assertIsNone(_REMOVED_COLUMN_RE.search(source), f"{path} uses Twat_mod_p5/p95")

    def test_every_config_is_valid_yaml(self):
        for path in _files(".yaml"):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            self.assertIn("run_mode", cfg, path)
            self.assertIn("paths", cfg, path)


if __name__ == '__main__':
    unittest.main()
