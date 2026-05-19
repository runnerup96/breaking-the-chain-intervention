"""
plot_delta_bar.py
-----------------
RQ3 Delta-bar plot: Delta (ID − OO(I)D) for standard vs tool per model × dataset.
Transparent wide bar = standard, opaque narrow bar = tool.
Green arrow / line shows the change, delta value annotated above.

Output: delta_bar.png

Usage:
    python fig_delta.py \\
        --root /path/to/intervention_predictions \\
        --project_path /path/to/project \\
        --datasets ricechem averitec tabfact \\
        --output_dir ./artifacts

    # AVeriTeC without explanations:
    python fig_delta.py ... --averitec_run standard_no_expl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns

from _common import (
    MODEL_ORDER, DATASET_ORDER, NICE2RAW_DS,
    MODEL_COLORS, PLOT_PARAMS,
    base_parser, setup_tools,
    load_predictions, build_df_exps,
    save_figure,
)


def compute_pivot(df_exps: pd.DataFrame) -> pd.DataFrame:
    """Mirrors notebook RQ3 aggregation."""
    grouped = df_exps.groupby(["dataset_name", "model_name_list", "run_type_list"])

    def compute_delta(g):
        id_m   = g["id_faith_list"].mean()
        ooid_m = g["id_ooId_faith_list"].mean()
        return pd.Series({"id_mean": id_m, "ooid_mean": ooid_m, "delta": id_m - ooid_m})

    agg = grouped.apply(compute_delta).reset_index()
    pivot = agg.pivot_table(
        index=["dataset_name", "model_name_list"],
        columns="run_type_list",
        values="delta",
        aggfunc="first",
    ).reset_index()
    for col in ["standard", "tool"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    return pivot


def build_plot(df_exps: pd.DataFrame, datasets_nice: list[str]) -> plt.Figure:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    P = PLOT_PARAMS

    pivot = compute_pivot(df_exps)

    fig, axes = plt.subplots(1, len(datasets_nice),
                             figsize=(5 * len(datasets_nice), 4))
    if len(datasets_nice) == 1:
        axes = [axes]

    for i, (ax, ds_nice) in enumerate(zip(axes, datasets_nice)):
        data = pivot[pivot["dataset_name"] == ds_nice].copy()
        data = data[data["model_name_list"].isin(MODEL_ORDER)]
        data["model_name_list"] = pd.Categorical(
            data["model_name_list"], categories=MODEL_ORDER, ordered=True)
        data = data.sort_values("model_name_list").dropna(
            subset=["standard", "tool"], how="all")

        if data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, style="italic")
            ax.set_title(ds_nice, fontsize=P["TITLE_SIZE"])
            continue

        models   = data["model_name_list"].tolist()
        x_pos    = np.arange(len(models))
        std_vals = data["standard"].values
        tool_vals = data["tool"].values

        for j, model in enumerate(models):
            color = MODEL_COLORS.get(model, "#999999")
            ax.bar(x_pos[j], std_vals[j],  width=0.8,       color=color,
                   alpha=0.25, zorder=2, edgecolor="black", linewidth=0.5)
            ax.bar(x_pos[j], tool_vals[j], width=0.8 * 0.7, color=color,
                   alpha=1.0,  zorder=2, edgecolor="black", linewidth=0.5)

        for j, (s, t) in enumerate(zip(std_vals, tool_vals)):
            if pd.isna(s) or pd.isna(t):
                continue
            delta_change = t - s
            x = x_pos[j]
            if abs(delta_change) < 0.05:
                ax.plot([x, x], [s, t], color="forestgreen", linewidth=2.5, zorder=3)
            else:
                ax.add_patch(FancyArrowPatch(
                    (x, s), (x, t),
                    color="forestgreen", arrowstyle="->",
                    linewidth=2.0, mutation_scale=18, zorder=3,
                ))
            ax.text(x, max(s, t) + 0.05, f"{delta_change:+.2f}",
                    color="forestgreen", va="bottom", ha="center",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="forestgreen", alpha=0.85),
                    zorder=4)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, fontsize=P["TICKS_SIZE"], rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=P["TICKS_SIZE"])
        ax.set_title(ds_nice, fontsize=P["TITLE_SIZE"])
        if i == 0:
            ax.set_ylabel("Δ", fontsize=P["YLABEL_SIZE"], labelpad=10)
        ax.axhline(0, color="gray", linewidth=0.6, alpha=0.4, zorder=0)
        ax.grid(True); ax.set_ylim(-0.02, 0.75); ax.set_axisbelow(True)
        sns.despine(ax=ax)

    plt.tight_layout()
    return fig


def main():
    args = base_parser(
        "RQ3 Delta-bar plot: standard vs tool Delta per model × dataset."
    ).parse_args()

    if not setup_tools(args.project_path):
        print("ERROR: tools required.")
        return

    all_pred = load_predictions(
        root=Path(args.root),
        datasets=args.datasets,
        run_types=["standard", "tool"],
        averitec_run=args.averitec_run,
    )
    if "averitec" in all_pred and args.averitec_run != "standard":
        bl = all_pred["averitec"].pop(args.averitec_run, {})
        all_pred["averitec"]["standard"] = bl

    df = build_df_exps(all_pred, args.datasets, ["standard", "tool"])
    if df.empty:
        print("No data.")
        return

    ds_nice = [d for d in DATASET_ORDER if NICE2RAW_DS[d] in args.datasets
               and d in df["dataset_name"].unique()]
    fig = build_plot(df, ds_nice)
    save_figure(fig, Path(args.output_dir), "fig_tool_comparison")


if __name__ == "__main__":
    main()
