
import argparse
import llm_model
from datasets_for_intervention import ricechem_intervention, ricechem_dataset, ricechem_evaluation
from datasets_for_intervention import averitec_intervention, averitec_dataset, averitec_evaluation
from datasets_for_intervention import pauq_intervention, pauq_dataset, pauq_evaluation
import hashlib
import json
import os
from tqdm import tqdm
from datetime import datetime
import json
from torch.utils.data import DataLoader
from copy import deepcopy
from transformers.utils import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.utils import logging

import llm_model
from datasets_for_intervention import entailment_intervention, entailment_dataset, entailment_evaluation
from datasets_for_intervention import ricechem_intervention, ricechem_dataset, ricechem_evaluation
from datasets_for_intervention import averitec_intervention, averitec_dataset, averitec_evaluation
from datasets_for_intervention import tabfact_intervention, tabfact_dataset, tabfact_evaluation

logging.set_verbosity_error()

def fix_seed(seed=42):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) 

model_name2simple_model_name = {
        "Qwen/Qwen3-1.7B": "qwen3-17B",
        "Qwen/Qwen3-4B": "qwen3-4B",
        "Qwen/Qwen3-8B": "qwen3-8B",
        "tiiuae/Falcon3-3B-Instruct": "falcon3-3B",
        "tiiuae/Falcon3-7B-Instruct": "falcon3-7B",
        "alpindale/Llama-3.2-3B-Instruct": "llama32-3B",
        "alpindale/Llama-3.2-1B-Instruct": "llama32-1B",
        "unsloth/Meta-Llama-3.1-8B-Instruct": "llama31-8B",
        "google/gemma-2-2b-it": "gemma2-2B",
        "google/gemma-2-9b-it": "gemma2-9B",
        "Openai/Gpt-oss-120b": "gpt-oss-120b",
        "qwen/qwen3-235b-a22b": "qwen3-235b-a22b",
        "Meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b"
    }

def get_few_shot_examples(train_dataset_path: str, prompting_regime: str, n_few_shot_examples: int):
    train_dataset = entailment_dataset.EntailmentDataset(train_dataset_path)
    if prompting_regime == "baseline_structure_faithfulness":
        stride = max(1, len(train_dataset) // (n_few_shot_examples * 2))
        examples = [deepcopy(train_dataset[idx]) for idx in range(0, len(train_dataset), stride)]
        examples = examples[:n_few_shot_examples]
    elif prompting_regime == "detailed_instruction":
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
    parser.add_argument("--evaluation_dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--try_one_batch", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--prompting_regime", type=str,
                        choices=["baseline_structure_faithfulness", "detailed_instruction"], default="baseline_structure_faithfulness")
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--api_base_url", type=str, default='https://inference.airi.net:46783/v1')
    parser.add_argument("--tokenizer_name", type=str, default=None)
    """We consider two prompting regimes.
    First explores faithfulness / reasoning transparency (as opposed to e.g. steganography) as is.
    Main question: how does the model handle contradictions WITHOUT clear instructions / demonstration?
    - In system prompt, we don't include a phrase about possibility of intervention.
    - We only show non-intervened few-shots (without contradictions)

    Second regime explores the ability of an LLM to follow explicit faithfulness instructions.
    Main question: how does the model handle contradictions WITH clear instructions / demonstration?
    - In system prompt, we include an explicit phrase about possibility of intervention.
    - We show intervened few-shot examples (with contradictions)
    """
    args = parser.parse_args()

    if args.model_name not in model_name2simple_model_name:
        raise ValueError(f'Unknown model: {args.model_name}. Check model_name2simple_model_name dict in make_intervention.py')
    
    fix_seed(args.seed)

    torch._dynamo.config.cache_size_limit = 8192

    llm_model = llm_model.LLMModel(
        args.model_name,
        use_api=args.use_api,
        api_base_url=args.api_base_url,
        tokenizer_name=args.tokenizer_name,
    )

    project_path = os.environ["PROJECT_PATH"]

    dataset = None
    intervention_logic = None
    evaluator = None
    if args.evaluation_dataset == "ricechem":
        dataset_path = os.path.join(project_path, "statics/result_splits/RiceChem")
        dataset = ricechem_dataset.RiceChemDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = ricechem_intervention.RiceChemIntervention(dataset, llm_model, prompt_type=args.prompting_regime)
        evaluator = ricechem_evaluation.RiceChemEvaluation(dataset, intervention_logic)
    elif args.evaluation_dataset == "entailment":
        train_dataset_path = os.path.join(project_path, "statics/result_splits/entailment_bank/dataset/task_2/train.jsonl")
        few_shot_examples = get_few_shot_examples(
            train_dataset_path=train_dataset_path,
            prompting_regime=args.prompting_regime,
            n_few_shot_examples=5,
        )
        dataset_path = os.path.join(project_path, "statics/result_splits/entailment_bank/dataset/task_2/test.jsonl")
        paraphrases_path = os.path.join(project_path, "statics/result_splits/entailment_bank/dataset/task_2/aligned_test_question_paraphases.json")
        dataset = entailment_dataset.EntailmentDataset(dataset_path, paraphrases_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = entailment_intervention.EntailmentIntervention(dataset, llm_model, few_shot_examples=few_shot_examples, hsvt_mode="paraphrase", prompting_regime=args.prompting_regime)
        evaluator = entailment_evaluation.EntailmentEvaluation(dataset, intervention_logic)
    elif args.evaluation_dataset == "averitec":
        dataset_path = os.path.join(project_path, "statics/result_splits/AVeriTeC/data")
        dataset = averitec_dataset.AVeriTeCDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = averitec_intervention.AVeriTeCIntervention(dataset, llm_model, prompt_type=args.prompting_regime)
        evaluator = averitec_evaluation.AVeriTeCEvaluation(dataset, intervention_logic)
    elif args.evaluation_dataset == "pauq":
        dataset = pauq_dataset.PAUQDataset(args.data_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = pauq_intervention.PAUQIntervention(dataset, llm_model)
        evaluator = pauq_evaluation.PAUQEvaluation(dataset)
    elif args.evaluation_dataset == "tabfact":
        dataset_path = os.path.join(project_path, "statics/result_splits/Table-Fact-Checking")
        dataset = tabfact_dataset.TabFactDataset(f'{dataset_path}/bootstrap/bootstrap_full.json',
                                                 f'{dataset_path}/data/all_csv')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = tabfact_intervention.TabFactIntervention(dataset, llm_model, args.prompting_regime)
        evaluator = tabfact_evaluation.TabFactEvaluation(dataset, intervention_logic)
    else:
        raise NotImplementedError(f"No implementation for {args.evaluation_dataset} dataset"
                                  f"Currently -- [ricechem, entailment, averitec, tabfact]")

    print(f"Loaded dataset {args.evaluation_dataset}")

    if args.try_one_batch:
        dataloader = [next(iter(dataloader))]
        # dataloader = [list(dataloader)[-1]]

    processed_samples_list, fails_list = [], []
    for batch in tqdm(dataloader, desc="Running inference", total=len(dataloader)):
        
        # batch_idx_list = [sample["idx"] for sample in batch]
        prompted_batch_with_structure_prediction = [intervention_logic.make_prompt(sample, include_gold_structure=False) for sample in batch]
        structure_prediction_outputs = llm_model.generate(prompted_batch_with_structure_prediction,
                                                          max_new_tokens=1024,
                                                          skip_special_tokens=False)
        promted_batch_with_gold_structure = [intervention_logic.make_prompt(sample, include_gold_structure=True) for sample in batch]
        gold_structure_outputs = llm_model.generate(promted_batch_with_gold_structure,
                                                    max_new_tokens=10,
                                                    skip_special_tokens=False)
        # Combine outputs and completion types
        batched_model_outputs = structure_prediction_outputs + gold_structure_outputs
        all_batch = prompted_batch_with_structure_prediction + promted_batch_with_gold_structure
        completion_type_list = ["structure_prediction"] * len(prompted_batch_with_structure_prediction) + ["gold_structure"] * len(promted_batch_with_gold_structure)
        # here we have just generation, we do the intervention independent from the gold/predicted structure
        doubled_batch = batch + [deepcopy(s) for s in batch]
        for sample, model_output, completion_type in zip(doubled_batch, batched_model_outputs, completion_type_list):
            sample['completion_type'] = completion_type
            # Mediator(DO_X)
            try:
                sample_with_interventions = intervention_logic.make_intervention(sample, model_output)
                prompt_list = intervention_logic.interventions_to_prompt(sample_with_interventions)
                intervened_completion_outputs = llm_model.generate(prompt_list, max_new_tokens=100,
                                                                skip_special_tokens=True)
                # parse completions to final structure
                final_sample = intervention_logic.collect_intervention_completion(sample_with_interventions, intervened_completion_outputs)
                processed_samples_list.append(final_sample)
            except Exception as e:# here only KeyError
                error_type, error_message = type(e).__name__, str(e)
                error_string = f"{error_type}: {error_message}"
                fails_list.append([sample, error_string])

    evaluation_metrics = None
    try:
        evaluation_metrics = evaluator.evaluate(processed_samples_list)
    except Exception as e:
        error_type, error_message = type(e).__name__, str(e)
        print(f"[WARNING] Evaluation failed with {error_type}: {error_message}")
        evaluation_metrics = {
            "error": f"{error_type}: {error_message}"
        }

    final_dataset_dict = {"metrics": evaluation_metrics, "result": processed_samples_list, "fails": fails_list}
    print('Processed: ', len(processed_samples_list))
    print('Failed: ', len(fails_list))
    dataset_name = args.evaluation_dataset
    path2save = os.path.join(project_path, "intervention_analysis", "intervention_predictions", dataset_name)
    os.makedirs(path2save, exist_ok=True)

    model_name = model_name2simple_model_name[args.model_name]

    curr_time = datetime.now().strftime("%Y-%m-%d@%H:%M")
    prompt_regime = "dt" if args.prompting_regime == "detailed_instruction" else "bsf"
    file_name = f"{model_name}_{curr_time}_{prompt_regime}_one_batch.json" if args.try_one_batch else f"{model_name}_{curr_time}_{prompt_regime}.json"
    path2save = os.path.join(path2save, file_name)

    with open(path2save, "w") as f:
        json.dump(final_dataset_dict, f, ensure_ascii=False, indent=4)
    print(f"The results are saved to {path2save}!")