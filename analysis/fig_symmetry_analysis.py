"""
plot_symmetry.py
----------------
RQ2 Symmetry plot: Correction OO(I)D vs Counterfact OO(I)D scatter per dataset.
Each point = one model. Diagonal = perfect symmetry.

Output: symmetry_analysis.png

Usage:
    python fig_symmetry.py \\
        --root /path/to/intervention_predictions \\
        --project_path /path/to/project \\
        --datasets ricechem averitec tabfact \\
        --output_dir ./artifacts

    # AVeriTeC without explanations:
    python fig_symmetry.py ... --averitec_run standard_no_expl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from _common import (
    MODEL_ORDER, DATASET_ORDER, NICE2RAW_DS,
    MODEL_COLORS, PLOT_PARAMS,
    base_parser, setup_tools,
    load_predictions, build_df_exps,
    save_figure,
)


def build_plot(df_exps, datasets_nice: list[str]) -> plt.Figure:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    P = PLOT_PARAMS
    fig, axes = plt.subplots(1, len(datasets_nice),
                             figsize=(4 * len(datasets_nice), 4))
    if len(datasets_nice) == 1:
        axes = [axes]

    df_std = df_exps[df_exps["run_type_list"] == "standard"]

    for ax, ds_nice in zip(axes, datasets_nice):
        sub = df_std[df_std["dataset_name"] == ds_nice]
        rows = []
        for model in MODEL_ORDER:
            m_df = sub[sub["model_name_list"] == model]
            cor_df  = m_df[m_df["intervention_type_list"] == "correction"]
            cnt_df  = m_df[m_df["intervention_type_list"] == "counterfact"]
            if len(cor_df) > 10 and len(cnt_df) > 10:
                rows.append({
                    "model":       model,
                    "correction":  cor_df["id_ooId_faith_list"].mean(),
                    "counterfact": cnt_df["id_ooId_faith_list"].mean(),
                })
        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, style="italic")
            ax.set_title(ds_nice, fontsize=P["TITLE_SIZE"])
            continue

        import pandas as pd
        summary = pd.DataFrame(rows)

        ax.plot([0, 1], [0, 1], "--", linewidth=1.3,
                color="#555555", alpha=0.6, zorder=1, label="Symmetry")

        for _, row in summary.iterrows():
            color = MODEL_COLORS.get(row["model"], "#999999")
            ax.scatter(row["correction"], row["counterfact"],
                       s=110, color=color,
                       edgecolors="black", linewidth=0.5,
                       alpha=0.95, zorder=5)
            ax.annotate(row["model"],
                        xy=(row["correction"], row["counterfact"]),
                        xytext=(10, 5), textcoords="offset points",
                        fontsize=P["LEGEND_FONT"], color="#222222", alpha=0.9)

        ax.set_xlim(-0.02, 1); ax.set_ylim(-0.02, 1)
        ax.set_xlabel("Correction",  fontsize=P["XLABEL_SIZE"])
        ax.set_ylabel("Counterfact", fontsize=P["YLABEL_SIZE"])
        ax.set_title(ds_nice, fontsize=P["TITLE_SIZE"])
        ax.tick_params(axis="both", labelsize=P["TICKS_SIZE"])
        ax.grid(True); ax.set_axisbelow(True)
        sns.despine(ax=ax)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=10,
                   markeredgecolor="black", markeredgewidth=0.5, label=m)
        for m, c in MODEL_COLORS.items() if m in MODEL_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0), ncol=5, frameon=False,
               fontsize=P["LEGEND_FONT"], columnspacing=1.5, handletextpad=0.5)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    return fig


def main():
    args = base_parser("RQ2 Symmetry plot: Correction vs Counterfact OO(I)D.").parse_args()

    if not setup_tools(args.project_path):
        print("ERROR: tools required.")
        return

    all_pred = load_predictions(
        root=Path(args.root),
        datasets=args.datasets,
        run_types=["standard"],
        averitec_run=args.averitec_run,
    )
    if "averitec" in all_pred and args.averitec_run != "standard":
        bl = all_pred["averitec"].pop(args.averitec_run, {})
        all_pred["averitec"]["standard"] = bl

    df = build_df_exps(all_pred, args.datasets, ["standard"])
    if df.empty:
        print("No data.")
        return

    ds_nice = [d for d in DATASET_ORDER if NICE2RAW_DS[d] in args.datasets
               and d in df["dataset_name"].unique()]
    fig = build_plot(df, ds_nice)
    save_figure(fig, Path(args.output_dir), "fig_symmetry_analysis")


if __name__ == "__main__":
    main()