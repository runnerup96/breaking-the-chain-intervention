import argparse
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.titlesize": 16,
    "figure.titlesize": 16,
    "text.usetex": False,
    "font.family": "serif",
})

SUBMETRICS = ["HSVT", "Local Edits", "Global"]
CONDITIONS = ["with_gold_structure", "with_predicted_structure"]
BAR_WIDTH = 0.35
HATCH_COLOR = "black"
GRID_ALPHA = 0.3
DPI = 300


def load_faithfulness(filepath):
    with open(filepath) as f:
        data = json.load(f)
    try:
        return data["metrics"]["faithfullness"]
    except KeyError:
        return data["metrics"]["faithfulness"]


DEFAULT_OUTPUT_DIR = "figures/faithfulness_metrics"


def plot_faithfulness(filepath, output_dir=DEFAULT_OUTPUT_DIR):
    faith = load_faithfulness(filepath)
    import re
    stem = os.path.splitext(os.path.basename(filepath))[0]
    # Keep only the model name/size portion before any date (YYYY-MM-DD)
    title = re.split(r"_\d{4}-\d{2}-\d{2}", stem)[0]

    gold_means = [faith["with_gold_structure"][m]["mean"] for m in SUBMETRICS]
    pred_means = [faith["with_predicted_structure"][m]["mean"] for m in SUBMETRICS]

    x = np.arange(len(SUBMETRICS))

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(x - BAR_WIDTH / 2, gold_means, BAR_WIDTH,
           color="steelblue", alpha=0.85, edgecolor="black", label="Gold")
    ax.bar(x + BAR_WIDTH / 2, pred_means, BAR_WIDTH,
           color="none", edgecolor=HATCH_COLOR, hatch="//", linewidth=0.7, label="Predicted")

    ax.set_xticks(x)
    ax.set_xticklabels(SUBMETRICS)
    ax.set_ylabel("Faithfulness Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)

    gold_patch = mpatches.Patch(facecolor="steelblue", edgecolor="black", alpha=0.85, label="Gold")
    pred_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="Predicted")
    ax.legend(handles=[gold_patch, pred_patch], loc="upper right")

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{title}.png"), dpi=DPI, bbox_inches="tight")
    print(f"Saved to {output_dir}/{title}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot faithfulness metrics from a single JSON file.")
    parser.add_argument("filepath", help="Path to the JSON results file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save PNG/PDF (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()
    plot_faithfulness(args.filepath, args.output_dir)


if __name__ == "__main__":
    main()
