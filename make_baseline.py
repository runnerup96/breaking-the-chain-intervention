"""
make_baseline.py
----------------
Baseline runner: direct X → Y without any intermediate structure (mediator).

The model receives only the input (X) and must predict the final answer (Y)
directly, without generating any checklist, DSL query, or Q&A structure.

This produces a simple accuracy / MAE metric that serves as a reference
for the intervention experiments.

Usage:
    python make_baseline.py \\
        --model_name "Qwen/Qwen3-4B" \\
        --evaluation_dataset "ricechem" \\
        --batch_size 16

Output is saved to:
    intervention_analysis/baseline_predictions/{dataset}/{model}_{timestamp}.json
"""

import argparse
import json
import os
import random
from math import isclose
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import llm_model

from datasets_for_intervention import (
    ricechem_dataset,
    averitec_dataset,
    tabfact_dataset,
)
from datasets_for_intervention.ricechem_baseline import RiceChemBaseline
from datasets_for_intervention.averitec_baseline import AVeriTeCBaseline
from datasets_for_intervention.tabfact_baseline   import TabFactBaseline


# ── reproducibility ───────────────────────────────────────────────────────────
def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── model name → short name (mirrors make_intervention.py) ───────────────────
MODEL_NAME2SIMPLE = {
    "Qwen/Qwen3-1.7B":                    "qwen3-1.7B",
    "Qwen/Qwen3-4B":                      "qwen3-4B",
    "Qwen/Qwen3-8B":                      "qwen3-8B",
    "tiiuae/Falcon3-3B-Instruct":         "falcon3-3B",
    "tiiuae/Falcon3-7B-Instruct":         "falcon3-7B",
    "alpindale/Llama-3.2-1B-Instruct":    "llama32-1B",
    "alpindale/Llama-3.2-3B-Instruct":    "llama32-3B",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "llama31-8B",
    "google/gemma-2-2b-it":               "gemma2-2B",
    "Meta-llama/Llama-3.1-70B-Instruct":  "llama-70B",
}

# ── token budget — one pass only, model outputs just the answer ───────────────
GEN_MAX_NEW_TOKENS_BASELINE = {
    "ricechem": 32,   # a float like "7.0"
    "averitec": 16,   # "Supported" or "Refuted"
    "tabfact":  16,   # "True" or "False"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline experiment: direct X→Y prediction without mediator."
    )
    parser.add_argument("--model_name",         type=str, required=True)
    parser.add_argument("--evaluation_dataset", type=str, required=True,
                        choices=["ricechem", "averitec", "tabfact"])
    parser.add_argument("--no_explanations", action="store_true",
                        help="AVeriTeC ablation: strip explanations from prompts. "
                             "Ignored for other datasets.")
    parser.add_argument("--batch_size",         type=int, default=16)
    parser.add_argument("--try_one_batch",      action="store_true",
                        help="Run only the first batch (quick debug).")
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--use_api",            action="store_true")
    parser.add_argument("--api_base_url",       type=str,
                        default="https://inference.airi.net:46783/v1")
    parser.add_argument("--tokenizer_name",     type=str, default=None)

    args = parser.parse_args()
    fix_seed(args.seed)

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm = llm_model.LLMModel(
        args.model_name,
        use_api=args.use_api,
        api_base_url=args.api_base_url,
    )

    project_path = os.environ["PROJECT_PATH"]

    # ── dataset + baseline logic ──────────────────────────────────────────────
    if args.evaluation_dataset == "ricechem":
        dataset  = ricechem_dataset.RiceChemDataset(
            data_path=os.path.join(project_path, "statics/datasets/RiceChem/data"),
        )
        baseline = RiceChemBaseline(dataset, llm)

    elif args.evaluation_dataset == "averitec":
        include_explanations = not args.no_explanations
        dataset  = averitec_dataset.AVeriTeCDataset(
            os.path.join(project_path, "statics/datasets/AVeriTeC/data"),
            include_explanations=include_explanations,
        )
        baseline = AVeriTeCBaseline(dataset, llm,
                                    include_explanations=include_explanations)

    elif args.evaluation_dataset == "tabfact":
        dataset  = tabfact_dataset.TabFactDataset(
            queries_json_path=os.path.join(
                project_path, "statics/datasets/TabFact/bootstrap_full.json"
            ),
            tables_dir=os.path.join(
                project_path, "statics/datasets/TabFact/data/all_csv"
            ),
        )
        baseline = TabFactBaseline(dataset, llm)

    else:
        raise NotImplementedError(f"Unknown dataset: {args.evaluation_dataset}")

    print(f"Loaded {args.evaluation_dataset} | baseline (no mediator) | "
          f"n={len(dataset)}")

    # ── dataloader ────────────────────────────────────────────────────────────
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: b,
        shuffle=False,
    )
    if args.try_one_batch:
        dataloader = [next(iter(dataloader))]

    max_new_tokens = GEN_MAX_NEW_TOKENS_BASELINE[args.evaluation_dataset]

    # ── inference loop ────────────────────────────────────────────────────────
    processed_samples: list = []

    for batch in tqdm(dataloader, desc=f"Baseline {args.evaluation_dataset}"):
        prompts = [baseline.make_prompt(s) for s in batch]

        outputs = llm.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=False,
        )

        for orig_sample, model_out in zip(batch, outputs):
            sample = deepcopy(orig_sample)

            raw_completion = baseline.clean_output(model_out["completion"])
            predicted      = baseline.parse_answer(raw_completion)

            # gold key differs by dataset
            gold = sample["gold_score"] if args.evaluation_dataset == "ricechem" \
                   else sample["gold_target"]

            # correct: None when parse failed
            if predicted is None:
                correct = None
            elif args.evaluation_dataset == "ricechem":
                correct = isclose(float(predicted), float(gold), abs_tol=0.01)
            else:
                correct = (predicted == gold)

            result_sample = {
                "idx":              sample["idx"],
                "raw_generation":   raw_completion,
                "predicted_answer": predicted,
                "gold_answer":      gold,
                "correct":          correct,
            }

            # keep X fields for downstream inspection
            if args.evaluation_dataset == "ricechem":
                result_sample["task"]           = sample["task"]
                result_sample["student_answer"] = sample["student_answer"]
                result_sample["score_range"]    = sample["score_range"]
            elif args.evaluation_dataset == "averitec":
                result_sample["claim"]          = sample["claim"]
            elif args.evaluation_dataset == "tabfact":
                result_sample["statement"]      = sample["statement"]
                result_sample["table_id"]       = sample["table_id"]

            processed_samples.append(result_sample)

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("\n=== Running evaluation ===")
    try:
        metrics = baseline.evaluate(processed_samples)
    except Exception as e:
        print(f"[WARNING] Evaluation failed: {type(e).__name__}: {e}")
        metrics = {"error": str(e)}

    # ── save ──────────────────────────────────────────────────────────────────
    ds_subdir = args.evaluation_dataset
    if args.evaluation_dataset == "averitec" and args.no_explanations:
        ds_subdir = "averitec_no_expl"
    save_dir = os.path.join(
        project_path,
        "intervention_analysis",
        "baseline_predictions",
        ds_subdir,
    )
    os.makedirs(save_dir, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
    simple_name = MODEL_NAME2SIMPLE.get(args.model_name,
                                        args.model_name.split("/")[-1])
    filename    = f"{simple_name}_{timestamp}.json"

    n_total   = len(processed_samples)
    n_correct = sum(1 for s in processed_samples if s.get("correct") is True)
    n_error   = sum(1 for s in processed_samples if s.get("correct") is None)

    final_dict = {
        "meta": {
            "experiment_type": "baseline_no_mediator",
            "model":           args.model_name,
            "model_simple":    simple_name,
            "use_api":         args.use_api,
            "api_base_url":    args.api_base_url if args.use_api else None,
            "tokenizer_name":  args.tokenizer_name,
            "dataset":              args.evaluation_dataset,
            "include_explanations": not args.no_explanations,
            "batch_size":           args.batch_size,
            "seed":            args.seed,
            "try_one_batch":   args.try_one_batch,
            "timestamp":       timestamp,
            "total_samples":   n_total,
            "n_correct":       n_correct,
            "n_parse_error":   n_error,
        },
        "metrics": metrics,
        "result":  processed_samples,
    }

    out_path = os.path.join(save_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {n_total} samples + metrics → {out_path}")
