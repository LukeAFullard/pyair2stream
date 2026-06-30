# pyair2stream

A Python port of the **air2stream** hybrid model for river water temperature.

## Description

`pyair2stream` provides a hybrid physics-based and empirical modeling approach for estimating river water temperature based on air temperature and discharge data. This Python port offers a modular, easy-to-use interface, complete with various optimization techniques (PSO, DE, MCMC, etc.) for parameter calibration.

## Installation

You can install `pyair2stream` directly from the source repository:

```bash
git clone https://github.com/example/pyair2stream.git
cd pyair2stream
pip install .
```

## Minimal Usage Example

Ensure you have a configuration file (`config.yaml`) and your time-series CSV data ready. Then run:

```bash
pyair2stream --config path/to/your/config.yaml
```

Alternatively, you can import it in your Python scripts:

```python
from pyair2stream.config import CommonData
from pyair2stream.io import read_calibration
from pyair2stream.main import main

# Read configuration
data = read_calibration(config_file="config.yaml")

# You can manipulate data or call optimizers directly
# from pyair2stream.optimization import DE_mode
# DE_mode(data)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
