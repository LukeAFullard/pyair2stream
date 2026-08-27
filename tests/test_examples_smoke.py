"""
Smoke tests for the example scripts under examples/ (docs/audit/05_cli_and_io_correctness.md,
Defect E / required change 5.5).

`Twat_mod_p5`/`Twat_mod_p95` column-name typos in two example scripts (referencing
columns the code has never written -- the real names are `Twat_mod_lower`/
`Twat_mod_upper`) went undetected because nothing in CI ever ran or even parsed
these scripts. This does not execute the examples (several need proprietary or
generated data files not present in the repo, or take minutes to run), but it at
least parses every one so a syntax error or an obviously undefined top-level name
typo is caught automatically by the existing `pytest tests/` CI job -- the "at
minimum" bar the report sets.
"""

import ast
import os
import re
import unittest

# Match the removed column names exactly (as a quoted string/dict key), not as a
# prefix of the still-real 'Twat_mod_p50' column.
_REMOVED_COLUMN_RE = re.compile(r"Twat_mod_p(?:5|95)(?!\d)['\"]")

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _iter_example_scripts():
    for root, _dirs, files in os.walk(EXAMPLES_DIR):
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


class TestExampleScriptsSmoke(unittest.TestCase):
    def test_every_example_script_parses(self):
        scripts = list(_iter_example_scripts())
        self.assertGreater(len(scripts), 0, "Expected at least one example script under examples/")

        failures = []
        for path in scripts:
            with open(path, 'r') as f:
                source = f.read()
            try:
                ast.parse(source, filename=path)
            except SyntaxError as e:
                failures.append(f"{path}: {e}")

        self.assertEqual(failures, [], "Example script(s) failed to parse:\n" + "\n".join(failures))

    def test_no_example_references_removed_envelope_columns(self):
        # Twat_mod_p5/Twat_mod_p95 were never the real column names (the code has
        # always written Twat_mod_lower/Twat_mod_p50/Twat_mod_upper) and the
        # dual-name fallback that used to paper over this in post_processing.py
        # has been removed (report 05, Defect E). Guard against the typo coming back.
        offenders = []
        for path in _iter_example_scripts():
            with open(path, 'r') as f:
                source = f.read()
            if _REMOVED_COLUMN_RE.search(source):
                offenders.append(path)

        self.assertEqual(offenders, [], f"Example script(s) reference removed columns Twat_mod_p5/p95: {offenders}")


if __name__ == '__main__':
    unittest.main()
