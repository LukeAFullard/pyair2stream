import subprocess
import tempfile
import os
import shutil
import numpy as np
import datetime
from datetime import timedelta

def run_fortran_model(version, mod_num, n_tot_raw, Tair, Q, par, Qmedia, Twat_initial=4.0):
    with tempfile.TemporaryDirectory() as tmpdir:
        fortran_bin = os.path.join(tmpdir, "air2stream")
        src_dir = os.path.abspath("fortran/src")

        # Compile Fortran into the temp directory so we don't pollute the repo
        subprocess.run(["gfortran", "-ffree-line-length-none", "-o", fortran_bin,
                        os.path.join(src_dir, "AIR2STREAM_MODULES.f90"),
                        os.path.join(src_dir, "AIR2STREAM_MAIN.f90"),
                        os.path.join(src_dir, "AIR2STREAM_READ.f90"),
                        os.path.join(src_dir, "AIR2STREAM_RUNMODE.f90"),
                        os.path.join(src_dir, "AIR2STREAM_SUBROUTINES.f90")],
                        cwd=tmpdir, check=True)

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
