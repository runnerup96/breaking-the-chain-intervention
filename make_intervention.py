import argparse
import json
import os
import random
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
import hashlib

import llm_model

from datasets_for_intervention import (
    ricechem_intervention, ricechem_dataset, ricechem_evaluation, ricechem_structure_processor,
    entailment_intervention, entailment_dataset, entailment_evaluation, entailment_structure_processor,
    averitec_intervention, averitec_dataset, averitec_evaluation, averitec_structure_processor,
    tabfact_intervention, tabfact_dataset, tabfact_evaluation, tabfact_dsl_engine, tabfact_structure_processor
)

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

model_name2simple = {
    "Qwen/Qwen3-1.7B": "qwen3-1.7B",
    "Qwen/Qwen3-4B": "qwen3-4B",
    "Qwen/Qwen3-8B": "qwen3-8B",
    "tiiuae/Falcon3-3B-Instruct": "falcon3-3B",
    "tiiuae/Falcon3-7B-Instruct": "falcon3-7B",
    "alpindale/Llama-3.2-1B-Instruct": "llama32-1B",
    "alpindale/Llama-3.2-3B-Instruct": "llama32-3B",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "llama31-8B",
    "google/gemma-2-2b-it": "gemma2-2B",
    "Meta-llama/Llama-3.1-70B-Instruct": "llama-70B",
}


GEN_MAX_NEW_TOKENS = {
    "default": {
        "pred": {"none": 512, "simple": 512, "structured": 512},
        "interv": {"none": 10, "simple": 200, "structured": 200},
    },
    "ricechem": {
        "pred": {"none": 350, "simple": 350, "structured": 350},
        "interv": {"none": 10, "simple": 200, "structured": 200},
    },
    "averitec": {
        "pred": {"none": 512, "simple": 512, "structured": 512},
        "interv": {"none": 10, "simple": 200, "structured": 200},
    },
    "entailment": {
        "pred": {"none": 350, "simple": 700, "structured": 700},
        "interv": {"none": 16, "simple": 350, "structured": 350},
    },
    "tabfact": {
        "pred": {"none": 200, "simple": 512, "structured": 512},
        "interv": {"none": 10, "simple": 200, "structured": 200},
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--evaluation_dataset", type=str, required=True,
                        choices=["ricechem", "entailment", "averitec", "tabfact"])

    parser.add_argument("--prompting_regime", type=str, choices=["standard", "detailed", "max_detailed"], default="standard")
    parser.add_argument("--tool_mode", type=str, choices=["none", "simple", "structured"], default="none")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--try_one_batch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--api_base_url", type=str, default='https://inference.airi.net:46783/v1')
    parser.add_argument("--tokenizer_name", type=str, default=None)

    args = parser.parse_args()
    fix_seed(args.seed)

    llm = llm_model.LLMModel(args.model_name, use_api=args.use_api, api_base_url=args.api_base_url)

    project_path = os.environ["PROJECT_PATH"]

    evaluator = None

    if args.evaluation_dataset == "ricechem":
        dataset = ricechem_dataset.RiceChemDataset(
            data_path=os.path.join(project_path, "statics/datasets/RiceChem/data"),
        )
        tool = ricechem_structure_processor.RiceChemTool(dataset, args.tool_mode)
        processor = ricechem_structure_processor.RiceChemStructureProcessor(dataset, args.tool_mode)
        intervention_logic = ricechem_intervention.RiceChemIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=args.prompting_regime,
            tool_mode=args.tool_mode
        )
        evaluator = ricechem_evaluation.RiceChemEvaluation(dataset, processor, args.tool_mode)

    elif args.evaluation_dataset == "entailment":
        train_dataset_path = os.path.join(project_path, "statics/datasets/EntailmentBank/data/train.jsonl")
        dataset_path = os.path.join(project_path, "statics/datasets/EntailmentBank/data/test.jsonl")
        # paraphrases_path = os.path.join(project_path, "statics/datasets/EntailmentBank/aligned_test_question_paraphases.json")
 
        dataset = entailment_dataset.EntailmentDataset(dataset_path)
 
        # Build few-shot examples from train split (logic lives in EntailmentIntervention)
        few_shot_examples = entailment_intervention.EntailmentIntervention.get_few_shot_examples(
            train_dataset_path=train_dataset_path,
            prompting_regime=args.prompting_regime,
            n_few_shot_examples=4,
        )
 
        tool = entailment_structure_processor.EntailmentTool()
        processor = entailment_structure_processor.EntailmentStructureProcessor(args.tool_mode)
        intervention_logic = entailment_intervention.EntailmentIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            few_shot_examples=few_shot_examples,
            prompting_regime=args.prompting_regime,
            tool_mode=args.tool_mode,
        )
        evaluator = entailment_evaluation.EntailmentEvaluation(dataset, processor, args.tool_mode)
    elif args.evaluation_dataset == "averitec":
        dataset_path = os.path.join(project_path, "statics/datasets/AVeriTeC/data")
        dataset = averitec_dataset.AVeriTeCDataset(dataset_path)
        tool = averitec_structure_processor.AVeriTeCTool(dataset, args.tool_mode)
        processor = averitec_structure_processor.AVeriTeCStructureProcessor(dataset, args.tool_mode)
        intervention_logic = averitec_intervention.AVeriTeCIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=args.prompting_regime,
            tool_mode=args.tool_mode,
        )
        evaluator = averitec_evaluation.AVeriTeCEvaluation(dataset, processor, args.tool_mode)
    elif args.evaluation_dataset == "tabfact":
        dataset_path = os.path.join(project_path, "statics/datasets/TabFact")
        dataset = tabfact_dataset.TabFactDataset(
            queries_json_path=os.path.join(dataset_path, "bootstrap_full.json"),
            tables_dir=os.path.join(dataset_path, "data/all_csv"),
        )
        engine = tabfact_dsl_engine.TabFactEngine()
        tool = tabfact_structure_processor.TabFactTool(engine)
        processor = tabfact_structure_processor.TabFactStructureProcessor(engine)
        intervention_logic = tabfact_intervention.TabFactIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=args.prompting_regime,
            tool_mode=args.tool_mode,
        )
        evaluator = tabfact_evaluation.TabFactEvaluation(
            dataset=dataset,
            processor=processor,
            tool=tool,
            tool_mode=args.tool_mode,
        )
    else:
        raise NotImplementedError(f"No implementation for {args.evaluation_dataset} dataset"
                                  f"Currently -- [ricechem, entailment, averitec, tabfact]")

    print(f"Loaded {args.evaluation_dataset} | prompting_regime={args.prompting_regime}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: b, shuffle=False)
    if args.try_one_batch:
        dataloader = [next(iter(dataloader))]

    processed_samples_list = []

    for batch in tqdm(dataloader, desc=f"Intervention {args.evaluation_dataset}"):
        pred_prompts = [intervention_logic.make_prompt(s, include_gold_structure=False) for s in batch]
        pred_outputs = llm.generate(
            pred_prompts,
            max_new_tokens=GEN_MAX_NEW_TOKENS[args.evaluation_dataset]['pred'][args.tool_mode],
            skip_special_tokens=False
        )

        for orig_sample, model_out in zip(batch, pred_outputs):
            sample = deepcopy(orig_sample)

            sample_with_interv = intervention_logic.make_intervention(sample, model_out)

            if sample_with_interv.get('generation_status') == 'error':
                processed_samples_list.append(sample_with_interv)
                continue

            prompt_list = intervention_logic.interventions_to_prompt(sample_with_interv)
            if prompt_list:
                interv_outputs = llm.generate(
                    prompt_list,
                    max_new_tokens=GEN_MAX_NEW_TOKENS[args.evaluation_dataset]['interv'][args.tool_mode],
                    skip_special_tokens=False
                )
                final_sample = intervention_logic.collect_intervention_completion(sample_with_interv, interv_outputs)
            else:
                final_sample = sample_with_interv

            processed_samples_list.append(final_sample)

    print("\n=== Running evaluation ===")
    if evaluator is not None:
        try:
            evaluation_metrics = evaluator.evaluate(processed_samples_list)
            print("Evaluation completed successfully")
        except Exception as e:
            print(f"[WARNING] Evaluation failed: {type(e).__name__}: {e}")
            evaluation_metrics = {"error": str(e)}
    else:
        evaluation_metrics = {"note": "No evaluator configured for this dataset"}
        print("No evaluator for this dataset")

    subdir = args.prompting_regime if args.tool_mode == 'none' else 'tool'
    save_dir = os.path.join(project_path, "intervention_analysis", "intervention_predictions", args.evaluation_dataset, subdir)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"{model_name2simple[args.model_name]}_{args.prompting_regime}_{timestamp}.json"

    n_total     = len(processed_samples_list)
    n_correct   = sum(1 for s in processed_samples_list if s.get("generation_status") == "correct")
    n_incorrect = sum(1 for s in processed_samples_list if s.get("generation_status") == "incorrect")
    n_error     = sum(1 for s in processed_samples_list if s.get("generation_status") == "error")

    final_dict = {
        "meta": {
            "model":          args.model_name,
            "model_simple":   model_name2simple.get(args.model_name, args.model_name),
            "use_api":        args.use_api,
            "api_base_url":   args.api_base_url if args.use_api else None,
            "tokenizer_name": args.tokenizer_name,

            "dataset":          args.evaluation_dataset,
            "prompting_regime": args.prompting_regime,
            "tool_mode":        args.tool_mode,

            "batch_size":    args.batch_size,
            "seed":          args.seed,
            "try_one_batch": args.try_one_batch,
            "timestamp":     timestamp,

            "total_samples":  n_total,
            "n_correct":      n_correct,
            "n_incorrect":    n_incorrect,
            "n_error":        n_error,
            "correct_rate":   round(n_correct   / max(1, n_correct + n_incorrect), 3),
            "incorrect_rate": round(n_incorrect / max(1, n_correct + n_incorrect), 3),
            "error_rate":     round(n_error     / max(1, n_total), 3),
        },
        "metrics": evaluation_metrics,
        "result": processed_samples_list,
    }

    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(processed_samples_list)} samples + metrics → {filename}")