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
import hashlib

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

def get_few_shot_examples(train_dataset_path: str, prompting_regime: str, n_few_shot_examples: int):
    train_dataset = entailment_dataset.EntailmentDataset(train_dataset_path)
    if prompting_regime == "standard":
        stride = max(1, len(train_dataset) // (n_few_shot_examples * 2))
        examples = [deepcopy(train_dataset[idx]) for idx in range(0, len(train_dataset), stride)]
        examples = examples[:n_few_shot_examples]
    elif prompting_regime == "detailed" or prompting_regime == "max_detailed":
        if len(train_dataset) == 0:
            raise ValueError("The train dataset is empty, cannot build few-shot examples.")

        def stable_seed(sample_id: str, mode: str) -> int:
            digest = hashlib.sha256(f"{sample_id}::{mode}".encode("utf-8")).digest()
            return int.from_bytes(digest[:4], "big")

        # intervention_modes = ["delete", "replace", "rewire", "global"]
        intervention_modes = ["rewire", "global", "delete", "replace"]
        mode_idx = 0
        examples = []
        n_pairs = (n_few_shot_examples + 1) // 2
        stride = max(1, len(train_dataset) // max(1, n_pairs))

        for idx in range(0, len(train_dataset), stride):
            if len(examples) >= n_few_shot_examples:
                break

            original_sample = deepcopy(train_dataset[idx])
            original_id = original_sample["id"]
            original_sample["id"] = f"{original_id}::orig"
            examples.append(original_sample)
            if len(examples) >= n_few_shot_examples:
                break

            intervened_sample = deepcopy(train_dataset[idx])
            mode = intervention_modes[mode_idx % len(intervention_modes)]
            mode_idx += 1
            intervened_sample["id"] = f"{original_id}::{mode}"
            intervened_sample["proof"] = entailment_intervention.intervene_step_proof(
                step_proof=intervened_sample["proof"],
                hypothesis_id=intervened_sample["hypothesis_id"],
                distractors=intervened_sample["distractors"],
                mode=mode,
                seed=stable_seed(original_id, mode),
                verbose=False
            )
            intervened_sample["score"] = not intervened_sample["score"]
            examples.append(intervened_sample)
    else:
        raise ValueError(f"Invalid prompting regime: {prompting_regime}")
    assert len(examples) == n_few_shot_examples
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--evaluation_dataset", type=str, required=True,
                        choices=["ricechem", "entailment", "averitec", "tabfact"])

    # Новый флаг только для ricechem
    parser.add_argument("--prompting_regime", type=str, choices=["standard", "detailed", "max_detailed"], default="standard")
    # Старый флаг для других датасетов
    # parser.add_argument("--prompting_regime", type=str,
    #                     choices=["baseline_structure_faithfulness", "detailed_instruction"],
    #                     default="baseline_structure_faithfulness")
    parser.add_argument("--tool_mode", type=str, choices=["simple", "structured"], default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--try_one_batch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--prompting_regime", type=str,
    #                     choices=["baseline_structure_faithfulness", "detailed_instruction", "maxumum_mediator_faithfulness"], default="baseline_structure_faithfulness")
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--api_base_url", type=str, default='https://inference.airi.net:46783/v1')
    parser.add_argument("--tokenizer_name", type=str, default=None)
    """
    We consider two prompting regimes.
    First explores faithfulness / reasoning transparency (as opposed to e.g. steganography) as is.
    Main question: how does the model handle contradictions WITHOUT clear instructions / demonstration?
    - In system prompt, we don't include a phrase about possibility of intervention.
    - We only show non-intervened few-shots (without contradictions)

    Second regime explores the ability of an LLM to follow explicit faithfulness instructions.
    Main question: how does the model handle contradictions WITH clear instructions / demonstration?
    - In system prompt, we include an explicit phrase about possibility of intervention.
    - We show intervened few-shot examples (with contradictions)

    Third regime is an extension of the second regime with focus on maximum compliance with the interventions.
    """
    args = parser.parse_args()

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
            correction_only=False
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

    elif args.evaluation_dataset == "entailment": # WARNING! Not updated yet
        train_dataset_path = os.path.join(project_path, "statics/datasets/EntailmentBank/data/train.jsonl")
        few_shot_examples = get_few_shot_examples(
            train_dataset_path=train_dataset_path,
            prompting_regime=args.prompting_regime,
            n_few_shot_examples=5,
        )
        dataset_path = os.path.join(project_path, "statics/datasets/EntailmentBank/data/test.jsonl")
        paraphrases_path = os.path.join(project_path, "statics/datasets/EntailmentBank/aligned_test_question_paraphases.json")
        dataset = entailment_dataset.EntailmentDataset(dataset_path, paraphrases_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = entailment_intervention.EntailmentIntervention(dataset, llm_model, few_shot_examples=few_shot_examples, hsvt_mode="paraphrase", prompting_regime=args.prompting_regime)
        evaluator = entailment_evaluation.EntailmentEvaluation(dataset, intervention_logic)
    elif args.evaluation_dataset == "averitec": # WARNING! Not updated yet
        dataset_path = os.path.join(project_path, "statics/datasets/AVeriTeC/data")
        dataset = averitec_dataset.AVeriTeCDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = averitec_intervention.AVeriTeCIntervention(dataset, llm_model, prompting_regime=args.prompting_regime)
        evaluator = averitec_evaluation.AVeriTeCEvaluation(dataset, intervention_logic)
    elif args.evaluation_dataset == "tabfact": # WARNING! Not updated yet
        dataset_path = os.path.join(project_path, "statics/datasets/TabFact")
        dataset = tabfact_dataset.TabFactDataset(f'{dataset_path}/bootstrap_full.json',
                                                 f'{dataset_path}/data/all_csv')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = tabfact_intervention.TabFactIntervention(dataset, llm_model, args.prompting_regime)
        evaluator = tabfact_evaluation.TabFactEvaluation(dataset, intervention_logic)
    else:
        raise NotImplementedError(f"No implementation for {args.evaluation_dataset} dataset"
                                  f"Currently -- [ricechem, entailment, averitec, tabfact]")

    print(f"Loaded {args.evaluation_dataset} | prompting_regime={args.prompting_regime if args.evaluation_dataset=='ricechem' else args.prompting_regime}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: b, shuffle=False)
    if args.try_one_batch:
        dataloader = [next(iter(dataloader))]

    processed_samples_list = []

    for batch in tqdm(dataloader, desc=f"Intervention {args.evaluation_dataset}"):
        pred_prompts = [intervention_logic.make_prompt(s, include_gold_structure=False) for s in batch]
        pred_outputs = llm.generate(pred_prompts, max_new_tokens=350, skip_special_tokens=False)

        gold_prompts = [intervention_logic.make_prompt(s, include_gold_structure=True) for s in batch]
        gold_outputs = llm.generate(gold_prompts, max_new_tokens=200 if args.tool_mode else 10, skip_special_tokens=False)

        all_outputs = pred_outputs + gold_outputs
        all_samples = [deepcopy(s) for s in batch] + [deepcopy(s) for s in batch]
        completion_types = ["structure_prediction"] * len(batch) + ["gold_structure"] * len(batch)

        for orig_sample, model_out, completion_type in zip(all_samples, all_outputs, completion_types):
            sample = deepcopy(orig_sample)
            sample['completion_type'] = completion_type

            # try:
            sample_with_interv = intervention_logic.make_intervention(sample, model_out)
            prompt_list = intervention_logic.interventions_to_prompt(sample_with_interv)
            interv_outputs = llm.generate(prompt_list, max_new_tokens=200 if args.tool_mode else 10, skip_special_tokens=False)
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
