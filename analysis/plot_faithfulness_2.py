import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titlesize": 16,
        "figure.titlesize": 16,
        "text.usetex": False,
        "font.family": "serif",
    }
)

BAR_WIDTH = 0.25
GRID_ALPHA = 0.3
DPI = 300
DEFAULT_OUTPUT_DIR = "figures/faithfulness_metrics"
DEFAULT_OUTPUT_NAME = "faithfulness.png"


def get_model_short_name(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    # Cut timestamp tail if present: *_YYYY-MM-DD...
    stem = re.split(r"_\d{4}-\d{2}-\d{2}", stem)[0]
    return stem


def _extract_metric_mean(pred: dict, key_candidates: list[str], field_name: str) -> float:
    metric_obj = None
    for key in key_candidates:
        metric_obj = pred.get(key)
        if metric_obj is not None:
            break

    if not isinstance(metric_obj, dict) or "mean" not in metric_obj:
        keys = ", ".join(f"`{k}`" for k in key_candidates)
        raise KeyError(f"Missing {field_name}.mean in with_predicted_structure (expected one of: {keys})")

    return metric_obj["mean"]


def load_scores(filepath):
    with open(filepath) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    faith = metrics.get("faithfullness") or metrics.get("faithfulness")
    if faith is None:
        raise KeyError("Missing `metrics.faithfullness` (or `metrics.faithfulness`) section")

    pred = faith.get("with_predicted_structure")
    if pred is None:
        raise KeyError("Missing `with_predicted_structure` in faithfulness section")

    faithfulness_id = _extract_metric_mean(
        pred,
        ["faithfulness_id"],
        "faithfulness_id",
    )
    strong_local = _extract_metric_mean(
        pred,
        ["faithfulness_strong_Local Edits", "faithfulness_strong_Local_Edits"],
        "faithfulness_strong_Local Edits",
    )
    strong_global = _extract_metric_mean(
        pred,
        [
            "faithfulness_strong_global_edits",
            "faithfulness_strong_Global Edits",
            "faithfulness_strong_Global_Edits",
            "faithfulness_strong_Global",
        ],
        "faithfulness_strong_global_edits",
    )

    return faithfulness_id, strong_local, strong_global


def plot_multi_model(filepaths, output_dir=DEFAULT_OUTPUT_DIR, output_name=DEFAULT_OUTPUT_NAME):
    model_names = []
    id_scores = []
    strong_local_scores = []
    strong_global_scores = []

    for filepath in filepaths:
        model_names.append(get_model_short_name(filepath))
        fid, fstrong_local, fstrong_global = load_scores(filepath)
        id_scores.append(fid)
        strong_local_scores.append(fstrong_local)
        strong_global_scores.append(fstrong_global)

    x = np.arange(len(model_names))
    fig_width = max(8, 1.1 * len(model_names) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    ax.bar(
        x - BAR_WIDTH,
        id_scores,
        BAR_WIDTH,
        color="steelblue",
        alpha=0.9,
        edgecolor="black",
        label="faithfulness_id",
    )
    ax.bar(
        x,
        strong_local_scores,
        BAR_WIDTH,
        color="indianred",
        alpha=0.9,
        edgecolor="black",
        label="faithfulness_strong_small_edits",
    )
    ax.bar(
        x + BAR_WIDTH,
        strong_global_scores,
        BAR_WIDTH,
        color="darkseagreen",
        alpha=0.9,
        edgecolor="black",
        label="faithfulness_strong_big_edits",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05)
    ax.set_title("Faithfulness comparison across models")
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    ax.legend(loc="upper right")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build one bar plot for several model result files. "
            "For each model, draws three bars: faithfulness_id, "
            "faithfulness_strong_Local Edits, and faithfulness_strong_global_edits."
        )
    )
    parser.add_argument(
        "filepaths",
        nargs="+",
        help="Paths to JSON result files (one file per model)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output figure (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output PNG filename (default: {DEFAULT_OUTPUT_NAME})",
    )

    args = parser.parse_args()
    plot_multi_model(args.filepaths, args.output_dir, args.output_name)


if __name__ == "__main__":
    main()
