import argparse
import json
import os
import re

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

BAR_WIDTH = 0.35
HATCH_COLOR = "black"
GRID_ALPHA = 0.3
DPI = 300


def load_metrics(filepath):
    with open(filepath) as f:
        data = json.load(f)
    perf = data["metrics"]["performance"]
    bad = perf["with_bad_structure"]["sql_match"]["mean"]
    corrected = perf["with_corrected_structure"]["sql_match"]["mean"]
    return bad, corrected


DEFAULT_OUTPUT_DIR = "figures/correction_metrics"


def plot_correction(filepath, output_dir=DEFAULT_OUTPUT_DIR):
    bad_mean, corrected_mean = load_metrics(filepath)

    stem = os.path.splitext(os.path.basename(filepath))[0]
    title = re.split(r"_\d{4}-\d{2}-\d{2}", stem)[0]

    x = np.array([0])

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(x - BAR_WIDTH / 2, [bad_mean], BAR_WIDTH,
           color="steelblue", alpha=0.85, edgecolor="black")
    ax.bar(x + BAR_WIDTH / 2, [corrected_mean], BAR_WIDTH,
           color="orange", alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(["SQL Match"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)

    bad_patch = mpatches.Patch(facecolor="steelblue", edgecolor="black", alpha=0.85, label="Bad Structure")
    corrected_patch = mpatches.Patch(facecolor="orange", edgecolor="black", alpha=0.85, label="Corrected Structure")
    ax.legend(handles=[bad_patch, corrected_patch], loc="upper right")

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{title}.png"), dpi=DPI, bbox_inches="tight")
    print(f"Saved to {output_dir}/{title}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot correction metrics from a single JSON file.")
    parser.add_argument("filepath", help="Path to the JSON results file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save PNG/PDF (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()
    plot_correction(args.filepath, args.output_dir)


if __name__ == "__main__":
    main()
