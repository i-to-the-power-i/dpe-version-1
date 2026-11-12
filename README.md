# Causal History Extraction

Identify a causal direction between two binary sequences using causal-history
patterns. The installable Python package is in `causal-method/`; experiment
scripts and their dependencies are in `experiments/`.

## Installation

Requirements: Python 3.11 or newer, Git, and a C compiler (needed to build the
Cython extension).

Clone the repository and enter it:

```bash
git clone <repository-url>
cd <repository-directory>
```

Replace `<repository-url>` with this repository's Git URL and
`<repository-directory>` with the directory created by `git clone`.

Create and activate a virtual environment, then install the package in editable
mode from the repository root:

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

## Basic usage

Run the interactive analysis command:

```bash
analyze-causality
```

When prompted, enter the two binary sequences (strings containing `0` and `1`):

```text
Enter sequence Y
000001000010000000000100001000
Enter sequence X
011101111010011001110101101001
```

The command prints pattern statistics and the detected direction. It also
creates charts and a network graph under a `results/` directory in the current
working directory.

## Experiments

To install the additional dependencies used by the experiment scripts, from
the repository root run:

```bash
python -m pip install -r experiments/requirements-experiments.txt
```

The scripts are in `experiments/`. Run a script from the repository root, for
example:

```bash
python experiments/experiment-1.py
```
