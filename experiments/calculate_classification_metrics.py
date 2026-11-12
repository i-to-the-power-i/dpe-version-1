"""Calculate positive-direction precision, recall, and F1 for experiments.

Run from the experiments directory:
    python calculate_classification_metrics.py

Writes results/classification-metrics.csv. The benchmark's true causal direction
is the positive class; all other verdicts (including independence) are negative.
"""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
def scores(actual, predicted, positive_label):
    """Compute binary one-vs-rest scores for the specified true direction."""
    tp = sum(a == positive_label and p == positive_label for a, p in zip(actual, predicted))
    fp = sum(a != positive_label and p == positive_label for a, p in zip(actual, predicted))
    fn = sum(a == positive_label and p != positive_label for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def experiment_data():
    """Yield experiment name, result frame, and its expected-label function."""
    for name, folder, filename, condition, positive_label in (
        ("Experiment 1: delayed bitflip", "experiment-1", "experiment-1-all_results.csv", "delay", "X -> Y"),
        ("Experiment 1b: stochastic delayed bitflip", "experiment-1-random-delay-sweep", "experiment-1-random-delay-sweep-all_results.csv", "delay", "X -> Y"),
        ("Experiment 2: AR(1)", "experiment-2", "experiment-2-all_results.csv", "phi", "Y -> X"),
        ("Experiment 3: coupled skewtent", "experiment-3", "experiment-3-all_results.csv", "eta", "Y -> X"),
    ):
        path = RESULTS / folder / filename
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        frame = pd.read_csv(path)
        frame[condition] = pd.to_numeric(frame[condition])
        if positive_label == "X -> Y":
            expected = lambda row: positive_label
        else:
            expected = lambda row: "Independence" if row[condition] == 0 else positive_label
        yield name, frame, expected, positive_label

    # Experiment 4 contains a single result per model (no trial dimension).
    path = RESULTS / "experiment-4" / "results.csv"
    if path.exists():
        frame = pd.read_csv(path)
        yield "Experiment 4: predator-prey", frame, lambda row: "Y -> X", "Y -> X"
    else:
        print(f"Skipping missing file: {path}")


def main():
    rows = []
    for name, frame, expected, positive_label in experiment_data():
        for model, group in frame.groupby("model", sort=True):
            actual = [expected(row) for _, row in group.iterrows()]
            predicted = group["direction"].fillna("Independence").tolist()
            precision, recall, f1 = scores(actual, predicted, positive_label)
            rows.append({
                "experiment": name,
                "model": model,
                "n": len(group),
                "positive_class": positive_label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            })

    output = RESULTS / "classification-metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.6f")
    print(f"Wrote {output}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
