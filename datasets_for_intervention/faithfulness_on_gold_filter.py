"""
Recalculates faithfulness metrics restricted to samples that pass the
gold-structure filter from make_gold_intervention_pairs.py:
  - completion_type == 'structure_prediction'
  - has skeleton + slots fields
  - schema_links is not empty
  - schema_links match gold  (compare_schema_links)
  - slots match gold         (compare_slots)

For the filtered indices both completion types (structure_prediction and
gold_structure) are kept so the evaluator can report metrics for both.
"""

import os
import sys
import json
import argparse

from pauq_dataset import PAUQDataset
from pauq_evaluation import PAUQEvaluation
from utils import compare_schema_links, compare_slots


def passes_gold_filter(sample: dict, idx2golden: dict) -> bool:
    if sample.get("completion_type") == "gold_structure":
        return False
    if "skeleton" not in sample or "slots" not in sample:
        return False
    schema_links = sample.get("schema_links", {})
    if not schema_links:
        return False

    golden = idx2golden.get(sample["index"])
    if golden is None:
        return False

    return (
        compare_schema_links(schema_links, golden["true_schema_links"])
        and compare_slots(sample["slots"], golden["true_slots"])
    )


def recalculate(prediction_path: str, dataset_path: str = "./pauq"):
    with open(prediction_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "result" not in data:
        print(f"No 'result' key in {prediction_path}, skipping.")
        return None

    dataset = PAUQDataset(dataset_path)
    idx2golden = {s["index"]: s for s in dataset}

    # Collect indices that pass the gold filter
    gold_indices = set()
    for sample in data["result"]:
        if passes_gold_filter(sample, idx2golden):
            gold_indices.add(sample["index"])

    print(f"Samples passing gold filter: {len(gold_indices)}")

    # Keep all result entries (both completion types) for those indices
    filtered_samples = [s for s in data["result"] if s["index"] in gold_indices]

    evaluator = PAUQEvaluation(dataset)
    metrics = evaluator.evaluate(filtered_samples)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate faithfulness on gold-filtered samples."
    )
    parser.add_argument(
        "prediction_files",
        nargs="*",
        help="Path(s) to prediction JSON file(s). "
             "If omitted, all files in the default predictions directory are processed.",
    )
    parser.add_argument(
        "--predictions-dir",
        default="/Users/kmvafin/research/breaking-the-chain-intervention"
                "/intervention_analysis/intervention_predictions/pauq",
        help="Directory with prediction files (used when no files are specified).",
    )
    parser.add_argument(
        "--dataset-path",
        default="./pauq",
        help="Path to the PAUQ dataset directory.",
    )
    parser.add_argument(
        "--output",
        default="faithfulness_gold_filter_metrics.json",
        help="Path to save the aggregated metrics JSON.",
    )
    args = parser.parse_args()

    files = args.prediction_files
    if not files:
        files = [
            os.path.join(args.predictions_dir, f)
            for f in os.listdir(args.predictions_dir)
        ]

    all_metrics = {}
    for path in files:
        name = os.path.basename(path)
        print(f"\n{'='*60}")
        print(f"File: {name}")
        metrics = recalculate(path, args.dataset_path)
        if metrics is not None:
            all_metrics[name] = metrics

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)
    print(f"\nMetrics saved to {args.output}")

    return all_metrics


if __name__ == "__main__":
    main()
