# make_intervention.py
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

import llm_model

# RiceChem — новый unified класс
from datasets_for_intervention.ricechem_dataset import RiceChemDataset

from datasets_for_intervention import (
    ricechem_intervention, ricechem_dataset, ricechem_evaluation, ricechem_structure_processor,
    entailment_intervention, entailment_dataset, entailment_evaluation,
    averitec_intervention, averitec_dataset, averitec_evaluation,
    tabfact_intervention, tabfact_dataset, tabfact_evaluation,
)

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

model_name2simple = {
    "Qwen/Qwen3-1.7B": "qwen3-17B",
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--evaluation_dataset", type=str, required=True,
                        choices=["ricechem", "entailment", "averitec", "tabfact"])

    # Новый флаг только для ricechem
    parser.add_argument("--prompting_regime", type=str, choices=["standard", "detailed"], default="standard")
    # Старый флаг для других датасетов
    # parser.add_argument("--prompting_regime", type=str,
    #                     choices=["baseline_structure_faithfulness", "detailed_instruction"],
    #                     default="baseline_structure_faithfulness")
    parser.add_argument("--tool_mode", type=str, choices=["simple", "structured"], default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--try_one_batch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--api_base_url", type=str, default="https://inference.airi.net:46783/v1")

    args = parser.parse_args()
    fix_seed(args.seed)

    llm = llm_model.LLMModel(args.model_name, use_api=args.use_api, api_base_url=args.api_base_url)

    project_path = os.environ["PROJECT_PATH"]

    evaluator = None

    if args.evaluation_dataset == "ricechem":
        dataset = ricechem_dataset.RiceChemDataset(
            data_path=os.path.join(project_path, "statics/datasets/RiceChem/data"),
            correction_path=os.path.join(project_path, f"intervention_analysis/intervention_predictions/ricechem/{model_name2simple[args.model_name]}.json"),
            use_corrections=True,
            correction_only=True
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
        pass

    elif args.evaluation_dataset == "averitec":
        pass

    elif args.evaluation_dataset == "tabfact":
        pass

    print(f"Loaded {args.evaluation_dataset} | prompting_regime={args.prompting_regime if args.evaluation_dataset=='ricechem' else args.prompting_regime}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: b, shuffle=False)
    if args.try_one_batch:
        dataloader = [next(iter(dataloader))]

    processed_samples_list = []

    for batch in tqdm(dataloader, desc=f"Intervention {args.evaluation_dataset}"):
        pred_prompts = [intervention_logic.make_prompt(s, include_gold_structure=False) for s in batch]
        pred_outputs = llm.generate(pred_prompts, max_new_tokens=1024, skip_special_tokens=False)

        gold_prompts = [intervention_logic.make_prompt(s, include_gold_structure=True) for s in batch]
        gold_outputs = llm.generate(gold_prompts, max_new_tokens=512 if args.tool_mode else 10, skip_special_tokens=False)

        all_outputs = pred_outputs + gold_outputs
        all_samples = [deepcopy(s) for s in batch] + [deepcopy(s) for s in batch]
        completion_types = ["structure_prediction"] * len(batch) + ["gold_structure"] * len(batch)

        for orig_sample, model_out, completion_type in zip(all_samples, all_outputs, completion_types):
            sample = deepcopy(orig_sample)
            sample['completion_type'] = completion_type

            # try:
            sample_with_interv = intervention_logic.make_intervention(sample, model_out)
            prompt_list = intervention_logic.interventions_to_prompt(sample_with_interv)
            interv_outputs = llm.generate(prompt_list, max_new_tokens=512 if args.tool_mode else 10, skip_special_tokens=True)
            final_sample = intervention_logic.collect_intervention_completion(sample_with_interv, interv_outputs)
            processed_samples_list.append(final_sample)

            # except Exception as e:
            #     print(f"[ERROR] {e}")
                

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

    save_dir = os.path.join(project_path, "intervention_analysis", "intervention_predictions", args.evaluation_dataset)
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{model_name2simple[args.model_name]}_{args.prompting_regime}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    final_dict = {
        "meta": {
            "model": args.model_name,
            "dataset": args.evaluation_dataset,
            "prompting_regime": args.prompting_regime,
            "total_samples": len(processed_samples_list)
        },
        "metrics": evaluation_metrics,
        "results": processed_samples_list
    }

    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(processed_samples_list)} samples + metrics → {filename}")
