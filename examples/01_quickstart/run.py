"""
Run example 01 and refresh the figure its README shows.

    python examples/01_quickstart/run.py

This does exactly what the README's one command does:
    pyair2stream --config examples/01_quickstart/config.yaml
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

subprocess.run([sys.executable, "-m", "pyair2stream.main", "--config", "examples/01_quickstart/config.yaml"],
               cwd=REPO, check=True)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
shutil.copy(os.path.join(HERE, "output", "validation_DE_NSE_Mentue.png"),
            os.path.join(HERE, "figures", "validation.png"))
