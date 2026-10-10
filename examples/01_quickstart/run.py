"""
Run example 01 and refresh the figure its README shows.

    python examples/01_quickstart/run.py

The run is the one call the README shows, `pyair2stream.run`; from a terminal,
`pyair2stream --config examples/01_quickstart/config.yaml` does the same.
"""
import os
import shutil

import pyair2stream

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
os.chdir(REPO)      # the paths in the settings file are relative to the repository's top folder

result = pyair2stream.run("examples/01_quickstart/config.yaml")
for period in ("calibration", "validation"):
    s = result.scores[period]
    print(f"{period}: NSE {s['NSE']:.3f}, RMSE {s['RMSE']:.2f} °C")
print("Fitted parameters:", {name: round(value, 4) for name, value in result.parameters.items()})

os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
shutil.copy(os.path.join(result.output_dir, "validation_DE_NSE_Mentue.png"),
            os.path.join(HERE, "figures", "validation.png"))
