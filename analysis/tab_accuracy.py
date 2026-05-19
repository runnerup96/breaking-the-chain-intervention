"""
table_accuracy_agreement.py
----------------------------
Generates the accuracy & agreement table:

    Acc (w/ M)  — accuracy_total when model generates mediator first (standard run)
    Acc (no M)  — accuracy_total from baseline (no mediator) JSON files
    F1          — macro F1 from baseline JSON (NaN for RiceChem)
    Agreement   — fraction of samples where Y_baseline == Y_standard
                  (merged by idx; both None predictions excluded from denominator)

Output: accuracy_agreement_table.csv / .md

Usage:
    python tab_accuracy.py \\
        --root         /path/to/intervention_predictions \\
        --baseline_root /path/to/baseline_predictions \\
        --project_path /path/to/project \\
        --datasets ricechem averitec tabfact \\
        --output_dir ./artifacts

    # AVeriTeC without explanations:
    python tab_accuracy.py ... --averitec_run standard_no_expl
"""

import sys
from math import isclose
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import pandas as pd

from _common import (
    MODEL_ORDER, DATASET_ORDER, NICE2RAW_DS, NICE2SHORT_MODEL,
    Y_BEFORE_KEY, TOOLS,
    base_parser, setup_tools,
    load_predictions, load_baseline,
    save_table,
)


def gold_y(sample: dict, ds_raw: str):
    if ds_raw == "ricechem" and "ricechem" in TOOLS:
        return TOOLS["ricechem"](
            {"rubric": sample["gold_rubric"]},
            {"task_idx": sample["task_idx"]},
        )
    return sample.get("gold_target")


def approx_eq(a, b, ds_raw: str) -> bool:
    if a is None or b is None:
        return False
    if ds_raw == "ricechem":
        try:
            return isclose(float(a), float(b), abs_tol=0.01)
        except (TypeError, ValueError):
            return False
    return a == b


def acc_with_mediator(all_pred: dict, ds_raw: str, model_short: str) -> float:
    results = (
        all_pred.get(ds_raw, {})
                .get("standard", {})
                .get(model_short, {})
                .get("result", [])
    )
    if not results:
        return float("nan")
    y_key = Y_BEFORE_KEY[ds_raw]
    correct = sum(
        approx_eq(s.get(y_key), gold_y(s, ds_raw), ds_raw)
        for s in results
    )
    return round(correct / len(results), 2)


def bl_metric(baseline: dict, ds_raw: str, ms: str, key: str) -> float:
    d = baseline.get(ds_raw, {}).get(ms)
    if d is None:
        return float("nan")
    v = d.get("metrics", {}).get(key)
    return round(v, 2) if v is not None else float("nan")


def agreement_score(all_pred: dict, baseline: dict, ds_raw: str, ms: str) -> float:
    bl_data = baseline.get(ds_raw, {}).get(ms)
    if bl_data is None:
        return float("nan")
    idx2bl = {s["idx"]: s.get("predicted_answer") for s in bl_data.get("result", [])}
    std_results = (
        all_pred.get(ds_raw, {})
                .get("standard", {})
                .get(ms, {})
                .get("result", [])
    )
    if not std_results:
        return float("nan")
    y_key = Y_BEFORE_KEY[ds_raw]
    pairs = [
        (idx2bl[s["idx"]], s.get(y_key))
        for s in std_results
        if s["idx"] in idx2bl
        and idx2bl[s["idx"]] is not None
        and s.get(y_key) is not None
    ]
    if not pairs:
        return float("nan")
    matches = sum(approx_eq(bl, std, ds_raw) for bl, std in pairs)
    return round(matches / len(pairs), 2)


def build_table(
    all_pred: dict,
    baseline: dict,
    datasets_raw: list[str],
) -> pd.DataFrame:
    ds_order = [d for d in DATASET_ORDER if NICE2RAW_DS[d] in datasets_raw]
    data: dict = {}

    for model_nice in MODEL_ORDER:
        ms = NICE2SHORT_MODEL.get(model_nice)
        for ds_nice in ds_order:
            dr = NICE2RAW_DS[ds_nice]
            data.setdefault((ds_nice, "Acc↑(M)"), []).append(
                acc_with_mediator(all_pred, dr, ms))
            data.setdefault((ds_nice, "Acc↑"), []).append(
                bl_metric(baseline, dr, ms, "accuracy_total"))
            data.setdefault((ds_nice, "F1"),          []).append(
                bl_metric(baseline, dr, ms, "macro_f1"))
            data.setdefault((ds_nice, "Agr."),   []).append(
                agreement_score(all_pred, baseline, dr, ms))

    df = pd.DataFrame(data, index=MODEL_ORDER)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "Model"
    return df


def main():
    p = base_parser(
        "Accuracy & Agreement table: Acc(w/M), Acc(no M), F1, Agreement."
    )
    p.add_argument("--baseline_root", default=None,
                   help="Root of baseline_predictions dir.")
    args = p.parse_args()

    setup_tools(args.project_path)   # needed for gold_y in ricechem

    all_pred = load_predictions(
        root=Path(args.root),
        datasets=args.datasets,
        run_types=["standard"],
        averitec_run=args.averitec_run,
    )
    # remap averitec ablation run → "standard"
    if "averitec" in all_pred and args.averitec_run != "standard":
        bl = all_pred["averitec"].pop(args.averitec_run, {})
        all_pred["averitec"]["standard"] = bl

    baseline = load_baseline(
        Path(args.baseline_root) if args.baseline_root else None,
        args.datasets,
    )

    table = build_table(all_pred, baseline, args.datasets)
    save_table(table, Path(args.output_dir), "tab_accuracy")
    print(table.to_string())


if __name__ == "__main__":
    main()
