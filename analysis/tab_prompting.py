"""
table_prompt_influence.py
--------------------------
RQ4 Prompt influence table: OO(I)D per prompting regime × model × dataset.

Produces two tables:
  - per_model: rows = models, columns = (dataset, regime)
  - summary:   rows = regimes, columns = datasets, values = "mean ± std"

Output: prompt_per_model_table.csv / .md
        prompt_summary_table.csv / .md

Usage:
    python tab_prompting.py \\
        --root /path/to/intervention_predictions \\
        --project_path /path/to/project \\
        --datasets ricechem averitec tabfact \\
        --output_dir ./artifacts

    # Custom prompt regimes (default: standard detailed max_detailed):
    python tab_prompting.py ... --regimes standard detailed max_detailed

    # AVeriTeC without explanations:
    python tab_prompting.py ... --averitec_run standard_no_expl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from _common import (
    MODEL_ORDER, DATASET_ORDER, NICE2RAW_DS,
    base_parser, setup_tools,
    load_predictions, build_df_exps,
    save_table,
)

PROMPT_ORDER_DEFAULT = ["standard", "detailed", "max_detailed"]


def build_tables(df_exps: pd.DataFrame, datasets_nice: list[str], prompt_order: list[str]):
    # ── per-model ─────────────────────────────────────────────────────────────
    per_model = df_exps.pivot_table(
        index="model_name_list",
        columns=["dataset_name", "run_type_list"],
        values="id_ooId_faith_list",
        aggfunc="mean",
    ).round(2)
    per_model = per_model.reindex(
        index=[m for m in MODEL_ORDER if m in per_model.index])
    new_cols = [
        (ds, p) for ds in datasets_nice for p in prompt_order
        if (ds, p) in per_model.columns
    ]
    if new_cols:
        per_model = per_model.reindex(
            columns=pd.MultiIndex.from_tuples(new_cols))
    per_model.index.name = "Model"

    # ── summary ───────────────────────────────────────────────────────────────
    raw = (
        df_exps.groupby(["run_type_list", "dataset_name"])["id_ooId_faith_list"]
        .agg(["mean", "std"])
        .round(3)
    )
    raw["value"] = raw.apply(lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1)
    summary = raw["value"].unstack(level="dataset_name")
    summary = summary.reindex(
        index=[p for p in prompt_order if p in summary.index],
        columns=[d for d in datasets_nice if d in summary.columns],
    )
    summary.index.name = "Prompting regime"
    return per_model, summary


def main():
    p = base_parser(
        "RQ4 Prompt influence: OO(I)D per prompting regime × model × dataset."
    )
    p.add_argument("--regimes", nargs="+", default=PROMPT_ORDER_DEFAULT,
                   help="Prompting regimes to include (in display order).")
    args = p.parse_args()

    if not setup_tools(args.project_path):
        print("ERROR: tools required.")
        return

    all_pred = load_predictions(
        root=Path(args.root),
        datasets=args.datasets,
        run_types=args.regimes,
        averitec_run=args.averitec_run,
    )
    # remap averitec ablation run → its regime name so df_exps works
    if "averitec" in all_pred and args.averitec_run != "standard":
        # move it under "standard" key if "standard" is in regimes
        if "standard" in args.regimes:
            bl = all_pred["averitec"].pop(args.averitec_run, {})
            all_pred["averitec"]["standard"] = bl

    df = build_df_exps(all_pred, args.datasets, args.regimes)
    if df.empty:
        print("No data.")
        return

    ds_nice = [d for d in DATASET_ORDER if NICE2RAW_DS[d] in args.datasets
               and d in df["dataset_name"].unique()]
    per_model, summary = build_tables(df, ds_nice, args.regimes)

    out = Path(args.output_dir)
    save_table(per_model, out, "tab_prompting")
    save_table(summary,   out, "tab_prompting_summary")

    print("\n── Per-model ──")
    print(per_model.to_string())
    print("\n── Summary (mean ± std) ──")
    print(summary.to_string())


if __name__ == "__main__":
    main()
