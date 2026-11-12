"""Build a consolidated accuracy figure and timing/pattern artifacts."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.utils import calculate_causal_history, discrete, generate_pattern_dictionary

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def wilson(k, n, z=1.96):
    p = np.asarray(k, dtype=float) / np.asarray(n, dtype=float)
    n = np.asarray(n, dtype=float)
    center = (p + z*z/(2*n))/(1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return np.clip(center-half, 0, 1), np.clip(center+half, 0, 1)


def make_figure():
    specs = [
        ("Experiment 1: Delayed bitflip", "experiment-1", "delay", "xy_count"),
        ("Experiment 2: AR(1)", "experiment-2", "phi", None),
        ("Experiment 3: Coupled skewtent", "experiment-3", "eta", "yx_count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.flat
    colors = {"DPE":"C0", "GC":"C1", "TE":"C2", "LZP":"C3"}
    for ax, (title, folder, xcol, success_col) in zip(axes, specs):
        path = RESULTS / folder
        accuracy = pd.read_csv(path / f"{folder}-accuracy.csv")
        dist = pd.read_csv(path / f"{folder}-proportions-count.csv")
        accuracy = accuracy.loc[:, ~accuracy.columns.str.startswith("Unnamed")]
        dist = dist.loc[:, ~dist.columns.str.startswith("Unnamed")]
        key = xcol
        accuracy[key] = pd.to_numeric(accuracy[key])
        dist[key] = pd.to_numeric(dist[key])
        if success_col is None:
            success_col = "xy_count"
        elif folder in ("experiment-2", "experiment-3"):
            # At zero coupling, the correct verdict is independence.
            if folder == "experiment-2":
                dist["correct_count"] = np.where(dist[key] == 0, dist.independent_count, dist.yx_count)
            else:
                dist["correct_count"] = np.where(dist[key] == 0, dist.independent_count, dist.yx_count)
            success_col = "correct_count"
        for model in ["DPE", "GC", "TE", "LZP"]:
            a = accuracy[accuracy.model == model].sort_values(key)
            d = dist[dist.model == model].sort_values(key)
            if a.empty:
                continue
            n = d.total_trials.to_numpy()
            k = d[success_col].to_numpy()
            low, high = wilson(k, n)
            y = a.accuracy.to_numpy()
            x = a[key].to_numpy()
            ax.fill_between(x, low, high, color=colors[model], alpha=.12)
            ax.errorbar(x, y, yerr=[np.maximum(0, y-low), np.maximum(0, high-y)], color=colors[model], marker="o",
                        linewidth=1.2, markersize=3, capsize=2, label=model)
        ax.set(title=title, xlabel={"delay":"Delay", "phi":"Coupling strength (φ)", "eta":"Coupling strength (η)"}[key], ylabel="Accuracy", ylim=(-.03,1.05))
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    axes[3].axis("off")
    axes[3].text(.02,.95,"Verdict breakdowns",weight="bold",va="top",fontsize=12)
    axes[3].text(.02,.82,"Full counts are retained in the experiment-1 to experiment-3 proportions-count CSV files (appendix tables).\n\nWilson 95% confidence intervals shown.",va="top",wrap=True)
    out = RESULTS / "consolidated-accuracy-wilson-ci.png"
    fig.savefig(out, dpi=300)
    fig.savefig(RESULTS / "consolidated-accuracy-wilson-ci.pdf")
    plt.close(fig)


def export_patterns():
    data = json.loads((ROOT.parent / "dataset" / "skewtent_maps_data.json").read_text())
    entry = data["0.3"]
    # One paired realization (index 0), discretized as in the experiment pipeline.
    x = np.asarray(discrete(np.asarray(entry["X"][0]), n_bins=2), dtype=np.uint8)
    y = np.asarray(discrete(np.asarray(entry["Y"][0]), n_bins=2), dtype=np.uint8)
    history_x_to_y = calculate_causal_history(y, x)
    history_y_to_x = calculate_causal_history(x, y)
    patterns_x_to_y = generate_pattern_dictionary(history_x_to_y)
    patterns_y_to_x = generate_pattern_dictionary(history_y_to_x)
    out = RESULTS / "experiment-3" / "skewtent-eta-0.3-sequence-0-patterns.txt"
    with out.open("w") as f:
        f.write("1D coupled skewtent maps — extracted patterns\n")
        f.write("eta=0.3; paired sequence index=0; dataset=dataset/skewtent_maps_data.json\n")
        f.write(f"Length after discretization: {len(x)}\n\n")
        for label, history, patterns in [("X -> Y", history_x_to_y, patterns_x_to_y), ("Y -> X", history_y_to_x, patterns_y_to_x)]:
            f.write(f"{label}\nHistory segments: {len(history)}\nPatterns: {len(patterns)}\n")
            for p in sorted(patterns):
                f.write(p + "\n")
            f.write("\n")


if __name__ == "__main__":
    make_figure()
    export_patterns()
