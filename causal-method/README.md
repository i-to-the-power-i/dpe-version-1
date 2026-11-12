# Causal History Extraction

`causal-history-extraction` is a Python package for analyzing the direction of
causal influence between two binary sequences. It builds causal histories at
bit flips, extracts shared patterns, evaluates their contribution to changes,
and compares the two directions.

## Requirements

- Python 3.11 or newer
- A C compiler, used to build the Cython extension

## Installation

From the repository root, install the package in editable mode:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ./causal-method

# Cythonize
python causal-method/setup.py build_ext
```

Alternatively, enter this directory and install it with:

```bash
python -m pip install .
```

The package build compiles its Cython extension. If your system does not have
a compiler configured, the supplied Conda environment installs one:

```bash
conda env create -f environment.yml
conda activate dpe-env
python -m pip install .
```

## Command-line usage

After installation, start the interactive analysis:

```bash
analyze-causality
```

Enter the two binary sequences when prompted. For example:

```text
Enter sequence Y
000001000010000000000100001000
Enter sequence X
011101111010011001110101101001
```

The command prints causal-history dictionaries, extracted patterns, pattern
statistics, and a detected direction. Charts and the causal network graph are
written to a `results/` directory relative to the current working directory.

## Python API

The underlying functions are available from `model.cy_utils`. The sequences
should be equal-length binary strings (or compatible arrays of binary values).

```python
from model.cy_utils import (
    calculate_causal_history,
    generate_pattern_dictionary,
    calculate_contribution_analysis,
    identify_causality,
)

x = "011101111010011001110101101001"
y = "000001000010000000000100001000"

# Each history uses target first and source second.
history_x_to_y = calculate_causal_history(y, x)
history_y_to_x = calculate_causal_history(x, y)

patterns_x_to_y = generate_pattern_dictionary(history_x_to_y)
patterns_y_to_x = generate_pattern_dictionary(history_y_to_x)

stats_x_to_y = calculate_contribution_analysis(patterns_x_to_y, x, y)
stats_y_to_x = calculate_contribution_analysis(patterns_y_to_x, y, x)

direction, gap = identify_causality(stats_x_to_y, stats_y_to_x, x, y)
print(f"Detected direction: {direction} (gap={gap})")
```

`identify_causality` estimates an independence threshold using shuffled
surrogates; its `n_surrogates` argument defaults to 100. Smaller values can
speed up exploratory runs, while larger values increase the work performed.

## Development and tests

Install the development extras and run the tests from this directory:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

The package source is under `src/model/`; the console command is defined by the
`analyze-causality` entry point.
