"""
Compiles and runs the original air2stream Fortran model for the golden
regression tests.

The Fortran source is not vendored in this repository. Instead it is pulled
in as a git submodule pointing at the upstream reference implementation:

    https://github.com/spiccolroaz/air2stream

pinned to a specific commit (see .gitmodules) so that results stay
reproducible regardless of what happens on the upstream `main` branch.

The upstream source targets Intel Fortran on Windows in a few places
(`ifport`/`makedirqq`, and `form='binary'`, both Intel-specific extensions
gfortran does not implement). Rather than silently maintaining a modified
fork, we keep the upstream source untouched in the submodule and apply a
small, auditable patch (fortran/patches/gfortran-portability.patch) at build
time. See fortran/patches/NOTICE.md for exactly what the patch changes and
why, and for licensing attribution (upstream is CC BY-SA 3.0).
"""

import subprocess
import tempfile
import os
import shutil
import atexit
import numpy as np
import datetime
from datetime import timedelta

# Repo root, resolved from this file's location rather than the working
# directory, so tests work regardless of where pytest is invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UPSTREAM_SRC = os.path.join(_REPO_ROOT, "fortran", "upstream", "src")
_PATCH_FILE = os.path.join(_REPO_ROOT, "fortran", "patches", "gfortran-portability.patch")

_SOURCE_FILES = [
    "AIR2STREAM_MODULES.f90",
    "AIR2STREAM_MAIN.f90",
    "AIR2STREAM_READ.f90",
    "AIR2STREAM_RUNMODE.f90",
    "AIR2STREAM_SUBROUTINES.f90",
]

# Cached compiled binary, so repeated calls within a test session don't
# recompile identical source over and over. The build dir is cleaned up at
# interpreter exit rather than per-call, since it needs to outlive any
# single call to run_fortran_model().
_cached_binary = None


def _build_fortran_binary():
    """Copy the upstream submodule source into a scratch dir, apply the
    gfortran-portability patch, compile, and return the binary path.
    Cached for the process lifetime."""
    global _cached_binary

    if _cached_binary is not None:
        return _cached_binary

    if not os.path.isdir(_UPSTREAM_SRC) or not os.listdir(_UPSTREAM_SRC):
        raise RuntimeError(
            f"Fortran submodule source not found at {_UPSTREAM_SRC}.\n"
            "The upstream air2stream source is a git submodule, not a "
            "vendored copy. Run:\n\n"
            "    git submodule update --init --recursive\n\n"
            "and try again."
        )

    build_dir = tempfile.mkdtemp(prefix="air2stream_build_")
    atexit.register(shutil.rmtree, build_dir, ignore_errors=True)

    src_dir = os.path.join(build_dir, "src")
    os.makedirs(src_dir)

    for fname in _SOURCE_FILES:
        shutil.copy(os.path.join(_UPSTREAM_SRC, fname), os.path.join(src_dir, fname))

    # Patch paths are written as a/src/FILE.f90 -> b/src/FILE.f90
    patch_result = subprocess.run(
        ["patch", "-p1", "-i", _PATCH_FILE],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    if patch_result.returncode != 0:
        raise RuntimeError(
            "Failed to apply fortran/patches/gfortran-portability.patch "
            f"against the upstream submodule source:\n"
            f"{patch_result.stdout}\n{patch_result.stderr}\n"
            "This usually means the pinned submodule commit no longer "
            "matches the patch context. Check .gitmodules / "
            "fortran/patches/NOTICE.md."
        )

    binary_path = os.path.join(build_dir, "air2stream")
    subprocess.run(
        ["gfortran", "-ffree-line-length-none", "-o", binary_path] + [os.path.join("src", f) for f in _SOURCE_FILES],
        cwd=build_dir,
        check=True,
    )

    _cached_binary = binary_path
    return _cached_binary


def run_fortran_model(version, mod_num, n_tot_raw, Tair, Q, par, Qmedia, Twat_initial=4.0):
    fortran_bin_src = _build_fortran_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        fortran_bin = os.path.join(tmpdir, "air2stream")
        shutil.copy(fortran_bin_src, fortran_bin)
        os.chmod(fortran_bin, 0o755)

        input_txt = f"""1
test
test_air
test_water
c
1d
{version}
0.0
NSE
{mod_num}
FORWARD
0.0
1
0.0
"""
        with open(os.path.join(tmpdir, "input.txt"), "w") as f:
            f.write(input_txt)

        output_folder = os.path.join(tmpdir, "test")
        os.makedirs(output_folder, exist_ok=True)

        with open(os.path.join(output_folder, "parameters_forward.txt"), "w") as f:
            f.write(" ".join(map(str, par)) + "\n")

        with open(os.path.join(output_folder, "test_air_test_water_cc.txt"), "w") as f:
            padded_n_tot = max(365, n_tot_raw)

            start_date = datetime.date(2000, 1, 1)
            for i in range(padded_n_tot):
                current_date = start_date + timedelta(days=i)
                day = current_date.day
                month = current_date.month
                year = current_date.year

                ta = Tair[i] if i < n_tot_raw else Tair[-1]
                q = Q[i] if i < n_tot_raw else Q[-1]

                if i == 0:
                    tw = Twat_initial
                else:
                    tw = -999.0

                f.write(f"{day} {month} {year} {ta:.6f} {tw:.6f} {q:.6f}\n")

        # run
        res = subprocess.run([fortran_bin], cwd=tmpdir, input="go\n", capture_output=True, text=True)

        output_file = os.path.join(output_folder, f"output_{version}", f"2_FORWARD_NSE_test_air_test_water_cc_1d.out")
        with open(output_file, 'r') as f:
            lines = f.readlines()
            out_data = np.genfromtxt(lines, missing_values="NaN", filling_values=np.nan)

        # Fortran duplicates the first 365 days. So output length is padded_n_tot + 365.
        # We return the first n_tot_raw days AFTER the warmup.
        return out_data[365:365+n_tot_raw, 5]
