# Method reference

This document describes the methods implemented in `src/model/cy_utils.pyx`
(the compiled implementation used by the CLI) and `src/model/utils.py` (a
Python implementation of many of the same routines, plus plotting helpers).
Import the compiled routines from `model.cy_utils`; import the Python
counterparts and `barchart_analysis` from `model.utils`.

> **Input convention:** the analysis methods operate on binary sequences. For
> directional methods, the target sequence is passed before the source when
> building a causal history. Keep paired sequences the same length. The
> compiled methods expect byte/binary-compatible values; pass strings made of
> `0` and `1` or arrays with values `0` and `1`.

## Causal-pattern analysis

### `_ensure_uint8(seq)`

Internal conversion helper. Encodes strings as ASCII bytes, converts lists to
`numpy.uint8`, and converts other array-like inputs with `numpy.asanyarray`.
It is not intended as a user-facing validation method; it does not verify
that values are binary.

### `calculate_causal_history(seq_1, seq_2)`

Builds a set of source-sequence segments associated with transitions in the
target. It scans `seq_1` for adjacent value changes; at each change it records
the corresponding segment of `seq_2` since the previous change.

- **Arguments:** `seq_1` is the target; `seq_2` is the source.
- **Returns:** a set of unique source segments (strings for string input).
- **Example:** `calculate_causal_history(y, x)` builds the history for `X -> Y`.

The compiled implementation in `cy_utils.pyx` does not currently enforce equal
input lengths; use equal-length sequences to avoid invalid comparisons or
unexpected results. The Python version in `utils.py` raises `ValueError` when
the lengths differ.

### `sliding_xnor_comparison(p1, p2)`

Slides the shorter input across the longer input and collects contiguous runs
of equal-position bits. Runs shorter than two bits are ignored.

- **Arguments:** two binary strings or compatible array-like patterns.
- **Returns:** a set of matching subsequences (strings), each at least two bits
  long.

### `generate_pattern_dictionary(G_input)` / `generate_pattern_dictionary(G)`

Compares every distinct pair of causal-history segments using
`sliding_xnor_comparison` and combines the discovered patterns.

- **Argument:** a collection of causal-history segments.
- **Returns:** a set of patterns. If no shared pattern is found, returns the
  input history collection unchanged (so the result may be a set or list).

### `calculate_contribution_analysis(patterns, X_in, Y_in)`

Counts occurrences of each pattern in source `X_in` and checks whether the
corresponding window in target `Y_in` contains at least one bit flip.

- **Arguments:** patterns, source sequence `X_in`, target sequence `Y_in`.
- **Returns:** a pandas DataFrame indexed by `Pattern`, with `Change`,
  `No-change`, and `Ratio` columns. `Ratio` is the fraction of matched windows
  containing a target flip, rounded to four decimal places. Overlapping pattern
  occurrences are counted.

### `run_causal_analysis(seq_x_bin, seq_y_bin)`

Runs both directional histories, pattern extraction, contribution analysis,
and final direction identification.

- **Arguments:** binary sequences/arrays `seq_x_bin` and `seq_y_bin`.
- **Returns:** the `(direction, gap)` pair from `identify_causality` (for
  example, `('X -> Y', value)` or `('Independence', value)`).

This convenience method uses the default 100 surrogates for its threshold
estimate and does not generate the CLI plots.

## Discretization and entropy

### `discrete(sequence, n_bins=2)`

Maps a numeric sequence into uniformly spaced integer bins from `0` to
`n_bins - 1`. An empty input produces an empty integer array; a constant input
produces an array of zeros.

### `weighted_binary_entropy(ratio, count, seq_len, pattern_len)`

Computes binary entropy for a pattern's flip ratio and weights it by the
pattern's frequency among all possible windows. Returns `(weighted_entropy,
weight)`. Entropy is zero for ratios at either endpoint (`0` or `1`).

### `compute_scale_entropy(stats_df, seq_len)`

Groups contribution statistics by pattern length and computes summary entropy
and weight information for each scale. It expects a contribution-analysis
DataFrame with pattern strings as its index and `Ratio`, `Change`, and
`No-change` columns. Returns a DataFrame of per-scale summaries, including the
pattern set and average/normalized weighted entropy.

### `normalized_weighted_entropy(stats, seq_len)`

Applies `weighted_binary_entropy` to each row of a contribution DataFrame.
Returns `(mean_weighted_entropy, per_pattern_weighted_entropies)`. For `None`,
an empty DataFrame, or zero total weight, returns the scalar `0.0`.

### `surrogate_threshold(X, Y, n_surrogates=100, percentile=95)`

Estimates a null threshold for directional entropy gaps by independently
permuting the two sequences, recomputing both directions, and taking the
requested percentile of the absolute gaps. Returns a numeric threshold.
Because each surrogate reruns the full analysis, this method can be
computationally expensive. It uses NumPy's random generator and does not accept
a seed argument.

### `identify_causality(stats_x_y, stats_y_x, X, Y, n_surrogates=100)`

Compares the normalized weighted entropies for the two directions and checks
the absolute difference against `surrogate_threshold`.

- **Returns:** `(direction, absolute_gap)`. The direction is `Independence` if
  the gap is below the surrogate threshold; otherwise it is `X -> Y` or
  `Y -> X` according to which direction has lower entropy.
- **Note:** the default threshold calculation performs 100 surrogates. Reduce
  `n_surrogates` for quicker exploratory analysis.

## Plotting

### `barchart_analysis(stats, file_name='causal_analysis_chart', direction='x->y')`

Available from `model.utils`. Plots the `Ratio` for each pattern, saves a
high-resolution PNG to `results/{file_name}.png`, and displays the chart. It
creates the `results/` directory if needed. Empty statistics or missing
`Ratio` columns cause it to print a skip message and return without plotting.

### `draw_network(W_p_x, W_p_y, file_name='causal_network_dual')`

Draws a directed network connecting pattern nodes to the `X` and `Y` hubs,
with edge labels showing the supplied values. Saves a PNG to
`results/{file_name}.png` and displays the figure. Both arguments should be
Series-like objects indexed by pattern. The routine expects the `results/`
directory to exist; the CLI creates it before invoking this method.

The compiled and Python implementations have slightly different layout and
edge-mapping details; use the implementation matching your import path.

## Synthetic data and file loading

### `ar1_pair_generator(length=1000, noise_intensity=0.01, a=0.8, b=0.8, phi=0)`

Generates two numeric autoregressive sequences. The implemented recurrence is
`X[t] = a * X[t-1] + phi * Y[t-1] + noise_x[t]` and
`Y[t] = b * Y[t-1] + noise_y[t]`. A positive `phi` couples past `Y` into `X`.
Returns the pair `(X, Y)` as NumPy arrays. This creates continuous values;
apply `discrete` before using binary-pattern methods.

### `ar1_data_generator(...)`

Runs a parameter sweep over coupling strengths and repeated AR(1) simulations.
The compiled version accepts `num_sequences=2000`, `sequence_length=2000`,
and `transients=500`; the Python version accepts `num_sequences=1000` and
`sequence_length=1000`. They differ in parameter grid, output representation,
and transient handling. Both write generated data to `dataset/ar1_data.json`
and print a completion message. Ensure the `dataset/` directory exists before
running; these are batch jobs and can take a long time with default sizes.

### `skew_tent_map(x, b)`

Calculates one iteration of the skew-tent map on `[0, 1]`:
`x / b` when `0 <= x < b`, `(1 - x) / (1 - b)` when `b <= x < 1`, and `0`
otherwise. `b` is the map's skew/peak parameter.

### `skewtent_data_generator(b1=0.65, b2=0.47, initial_values=None)`

Compiled implementation of a coupled master/slave skew-tent simulation. It
sweeps coupling strengths `eta` from `0.0` through `0.9`, generates trials,
discards initial transients, and writes the results to
`dataset/skewtent_maps_data.json`. It returns no value. Defaults run 2,000
trials per coupling with 2,000 retained observations after 500 transient
steps. `initial_values`, when provided, maps an eta string such as `"0.1"` to
`{"M": initial_master, "S": initial_slave}`.

### `generate_coupled_data(b1=0.65, b2=0.47, initial_values=None)`

Python implementation of the coupled skew-tent generator. It sweeps the same
eta values, runs 1,500 trials with 1,000 retained observations after 500
transient steps, and writes to `dataset/akewtent_maps_data.json` (note this
filename differs from the compiled generator). It returns no value. Initial
conditions use the same mapping format as `skewtent_data_generator`.

### `extract_json(file_path='dataset/skewtent_maps_data.json')`

Compiled loader for the skew-tent dataset. Reads and returns parsed JSON data;
raises `FileNotFoundError` if the path is absent and reports loading status to
stdout.

### `extract_json_data(file_path, as_numpy=True)`

Python loader for the coupled-map JSON dataset. Reads and returns the parsed
nested dictionary; when `as_numpy=True`, converts the `X` and `Y` trial lists
to NumPy arrays. Raises `FileNotFoundError` for a missing path and may raise
`json.JSONDecodeError` for invalid JSON.

## Interactive entry point

`model.main:main` is exposed as the `analyze-causality` console command. It
prompts for `Y` and `X`, runs both directions, prints intermediate pattern
statistics and the final detected direction, then writes contribution charts
and a network graph under `results/` in the current working directory.
