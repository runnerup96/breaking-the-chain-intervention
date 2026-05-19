"""
table_rq1_faith.py
------------------
Generates Table 2: F_ID, F_OOD, F_strong (OO(I)D), Delta per model × dataset.

Columns per dataset: ID | OOD | OO(I)D | Delta

Output: rq1_faith_table.csv / .md

Usage:
    python tab_faithfulness.py \\
        --root /path/to/intervention_predictions \\
        --project_path /path/to/project \\
        --datasets ricechem averitec tabfact \\
        --output_dir ./artifacts

    # AVeriTeC without explanations:
    python tab_faithfulness.py ... --averitec_run standard_no_expl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from _common import (
    MODEL_ORDER, DATASET_ORDER, NICE2RAW_DS,
    base_parser, setup_tools, load_predictions,
    build_df_exps, save_table, save_latex,
)


def build_table(df_exps: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    df_std = df_exps[df_exps["run_type_list"] == "standard"]

    ds_order = [d for d in DATASET_ORDER if d in datasets]
    data: dict = {}
    for model in MODEL_ORDER:
        for ds_nice in ds_order:
            mask = (
                (df_std["model_name_list"] == model) &
                (df_std["dataset_name"]    == ds_nice)
            )
            sub = df_std[mask]
            if sub.empty:
                id_m = ood_m = ooid_m = np.nan
            else:
                id_m   = round(sub["id_faith_list"].mean(),      2)
                ood_m  = round(sub["f_ood_list"].mean(),          2)
                ooid_m = round(sub["id_ooId_faith_list"].mean(),  2)
            delta = round(id_m - ooid_m, 2) if not (np.isnan(id_m) or np.isnan(ooid_m)) else np.nan
            data.setdefault((ds_nice, "F_ID"),     []).append(id_m)
            data.setdefault((ds_nice, "F_OOD"),    []).append(ood_m)
            data.setdefault((ds_nice, "F_strong"), []).append(ooid_m)
            data.setdefault((ds_nice, "Δ"),  []).append(delta)

    df = pd.DataFrame(data, index=[m for m in MODEL_ORDER if m != "Llama-3.2 1B"])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "Model"
    return df


def main():
    args = base_parser(
        "Table 2: F_ID, F_OOD, F_strong (OO(I)D), Delta per model × dataset."
    ).parse_args()

    if not setup_tools(args.project_path):
        print("ERROR: tools required for faith metrics.")
        return

    all_pred = load_predictions(
        root=Path(args.root),
        datasets=args.datasets,
        run_types=["standard"],
        averitec_run=args.averitec_run,
    )

    # remap averitec ablation run → "standard" key so df_exps sees it as "standard"
    if "averitec" in all_pred and args.averitec_run != "standard":  # remap ablation key
        bl = all_pred["averitec"].pop(args.averitec_run, {})
        all_pred["averitec"]["standard"] = bl

    df = build_df_exps(all_pred, args.datasets, ["standard"])
    if df.empty:
        print("No data loaded.")
        return

    ds_nice = [d for d in DATASET_ORDER if NICE2RAW_DS[d] in args.datasets]
    table = build_table(df, ds_nice)
    save_table(table, Path(args.output_dir), "tab_faithfulness")
    save_latex(table, Path(args.output_dir), "tab_faithfulness",
               datasets=[d for d in DATASET_ORDER if NICE2RAW_DS[d] in args.datasets])
    print(table.to_string())


if __name__ == "__main__":
    main()
