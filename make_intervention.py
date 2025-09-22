
import argparse
import json
import os
import time
import random
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.utils import logging

import llm_model
from datasets_for_intervention import entailment_intervention, entailment_dataset, entailment_evaluation
from datasets_for_intervention import ricechem_intervention, ricechem_dataset, ricechem_evaluation
from datasets_for_intervention import averitec_intervention, averitec_dataset, averitec_evaluation

load_dotenv()

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
        "Qwen/Qwen3-4B": "qwen3-4B",
        "Qwen/Qwen3-8B": "qwen3-8B",
        "tiiuae/Falcon3-3B-Instruct": "falcon3-3B",
        "tiiuae/Falcon3-7B-Instruct": "falcon3-7B",
        "alpindale/Llama-3.2-3B-Instruct": "llama32-3B",
        "alpindale/Llama-3.2-1B-Instruct": "llama32-1B",
        "google/gemma-2-2b-it": "gemma2-2B",
        "google/gemma-2-9b-it": "gemma2-9B",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--evaluation_dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--try_one_batch", action="store_true", default=False)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    fix_seed(args.seed)

    llm_model = llm_model.LLMModel(args.model_name, device_map="cuda:0")

    project_path = os.environ["PROJECT_PATH"]

    dataset = None
    intervention_logic = None
    evaluator = None
    if args.evaluation_dataset == "ricechem":
        dataset_path = os.path.join(project_path, "statics/result_splits/RiceChem")
        dataset = ricechem_dataset.RiceChemDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = ricechem_intervention.RiceChemIntervention(dataset, llm_model)
        evaluator = ricechem_evaluation.RiceChemEvaluation(dataset, intervention_logic)
    elif args.evaluation_dataset == "entailment":
        train_dataset_path = os.path.join(project_path, "entailment_trees_emnlp2021_data_v3/dataset/task_2/train.jsonl")
        train_dataset = entailment_dataset.EntailmentDataset(train_dataset_path)
        few_shot_examples = train_dataset[::128][:5]
        assert len(few_shot_examples) == 5

        dataset_path = os.path.join(project_path, "entailment_trees_emnlp2021_data_v3/dataset/task_2/test.jsonl")
        paraphrases_path = os.path.join(project_path, "entailment_trees_emnlp2021_data_v3/dataset/task_2/aligned_test_question_paraphases.json")
        dataset = entailment_dataset.EntailmentDataset(dataset_path, paraphrases_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = entailment_intervention.EntailmentIntervention(dataset, llm_model, few_shot_examples=few_shot_examples, hsvt_mode="paraphrase")
        evaluator = entailment_evaluation.EntailmentEvaluation(dataset, intervention_logic)
    elif args.evaluation_dataset == "averitec":
        dataset_path = os.path.join(project_path, "statics/result_splits/AVeriTeC/data")
        dataset = averitec_dataset.AVeriTeCDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = averitec_intervention.AVeriTeCIntervention(dataset, llm_model)
        evaluator = averitec_evaluation.AVeriTeCEvaluation(dataset, intervention_logic)
    else:
        raise NotImplementedError(f"No implementation for {args.evaluation_dataset} dataset"
                                  f"Currently -- [ricechem, averitec]")

    print(f"Loaded dataset {args.evaluation_dataset}")

    if args.try_one_batch:
        dataloader = [next(iter(dataloader))]
        # dataloader = [list(dataloader)[-1]]

    curr_time = datetime.now().strftime("%Y-%m-%d@%H:%M")

    processed_samples_list, fails_list = [], []
    start_time = time.perf_counter()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        batch_start = time.perf_counter()
        batch_ids = [sample["id"] for sample in batch]
        print(f"Processing batch {batch_idx}: {len(batch)} samples, IDs: {batch_ids}")

        # Process structure prediction separately (1024 tokens)
        prompted_batch_with_structure_prediction = [intervention_logic.make_prompt(sample, include_gold_structure=False) for sample in batch]
        structure_prediction_outputs = llm_model.generate(prompted_batch_with_structure_prediction,
                                                         max_new_tokens=1024,
                                                         skip_special_tokens=False)

        # Process gold structure separately with smaller max_new_tokens
        prompted_batch_with_gold_structure = [intervention_logic.make_prompt(sample, include_gold_structure=True) for sample in batch]
        gold_structure_outputs = llm_model.generate(prompted_batch_with_gold_structure, 
                                                   max_new_tokens=10,
                                                   skip_special_tokens=False)

        # Combine outputs and completion types
        batched_model_outputs = structure_prediction_outputs + gold_structure_outputs
        completion_type_list = ["structure_prediction"] * len(structure_prediction_outputs) + ["gold_structure"] * len(gold_structure_outputs)

        # Log outputs with correct IDs
        print(f"Raw model outputs for batch {batch_idx}:")
        raw_outputs_path = os.path.join(args.logging_dir, f"raw_outputs_{curr_time}.jsonl")

        # Log structure prediction outputs
        for i, output in enumerate(structure_prediction_outputs):
            print(f"  Sample {batch_ids[i]} (structure_prediction): {len(output['completion'])} chars")
            with open(raw_outputs_path, "a") as f:
                json.dump({**output, "sample_id": batch_ids[i], "completion_type": "structure_prediction"}, f, ensure_ascii=False)
                f.write('\n')

        # Log gold structure outputs
        for i, output in enumerate(gold_structure_outputs):
            print(f"  Sample {batch_ids[i]} (gold_structure): {len(output['completion'])} chars")
            with open(raw_outputs_path, "a") as f:
                json.dump({**output, "sample_id": batch_ids[i], "completion_type": "gold_structure"}, f, ensure_ascii=False)
                f.write('\n')

        # here we have just generation, we do the intervention independent from the gold/predicted structure
        doubled_batch = batch + [deepcopy(s) for s in batch]
        for sample, model_output, completion_type in zip(doubled_batch, batched_model_outputs, completion_type_list):
            sample['completion_type'] = completion_type
            # Mediator(DO_X)
            try:
                sample_with_interventions = intervention_logic.make_intervention(sample, model_output)
                prompt_list = intervention_logic.interventions_to_prompt(sample_with_interventions)
                intervened_completion_outputs = llm_model.generate(prompt_list, max_new_tokens=10,
                                                                skip_special_tokens=True)
                # parse completions to final structure
                final_sample = intervention_logic.collect_intervention_completion(sample_with_interventions, intervened_completion_outputs)
                processed_samples_list.append(final_sample)
            except Exception as e:# here only KeyError
                error_type, error_message = type(e).__name__, str(e)
                error_context = {
                    'sample_id': sample.get('id', 'unknown'),
                    'completion_type': completion_type,
                    'error_type': error_type,
                    'error_message': error_message,
                    'model_output_length': len(model_output.get('completion', '')),
                    'has_proof': 'proof' in sample,
                    'has_distractors': 'distractors' in sample
                }
                print(f"ERROR processing sample {error_context['sample_id']}: {error_context}")
                fails_list.append([sample, error_context])

        batch_time = time.time() - batch_start
        print(f"Batch {batch_idx} completed in {batch_time:.2f}s")


    evaluation_metrics = evaluator.evaluate(processed_samples_list)

    final_dataset_dict = {"metrics": evaluation_metrics, "result": processed_samples_list, "fails": fails_list}
    print('Processed: ', len(processed_samples_list))
    print('Failed: ', len(fails_list))
    dataset_name = args.evaluation_dataset
    path2save = os.path.join(project_path, "intervention_analysis", "intervention_predictions", dataset_name)
    os.makedirs(path2save, exist_ok=True)

    model_name = model_name2simple_model_name[args.model_name]

    file_name = f"{model_name}_{curr_time}_one_batch.json" if args.try_one_batch else f"{model_name}_{curr_time}.json"
    path2save = os.path.join(path2save, file_name)

    with open(path2save, "w") as f:
        json.dump(final_dataset_dict, f, ensure_ascii=False, indent=4)
    print(f"The results are saved to {path2save}!")

