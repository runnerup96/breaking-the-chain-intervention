# make_correction.py
import argparse
import json
import os
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import logging

import llm_model

from datasets_for_intervention import ricechem_intervention
from datasets_for_intervention import averitec_intervention

from datasets_for_intervention import ricechem_dataset
from datasets_for_intervention import averitec_dataset

from datasets_for_intervention import ricechem_evaluation
# from datasets_for_intervention import averitec_evaluation

logging.set_verbosity_error()


def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


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
    "Meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--evaluation_dataset", type=str, required=True,
                        choices=["ricechem", "averitec"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--try_one_batch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompting_regime",
        type=str,
        choices=["baseline_structure_faithfulness", "detailed_instruction", "baseline_structure_tool_call"],
        default="baseline_structure_faithfulness",
    )

    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--api_base_url", type=str, default="https://inference.airi.net:46783/v1")
    parser.add_argument("--tokenizer_name", type=str, default=None)


    args = parser.parse_args()

    if args.model_name not in model_name2simple_model_name:
        raise ValueError(f"Unknown model: {args.model_name}")

    fix_seed(args.seed)
    torch._dynamo.config.cache_size_limit = 8192

    llm = llm_model.LLMModel(
        args.model_name,
        use_api=args.use_api,
        api_base_url=args.api_base_url,
        tokenizer_name=args.tokenizer_name,
    )

    project_path = os.environ["PROJECT_PATH"]

    dataset = None
    intervention_logic = None
    evaluator = None

    corrections_path = f'{project_path}/intervention_analysis/intervention_predictions/{args.evaluation_dataset}/{model_name2simple_model_name[args.model_name]}.json'

    if args.evaluation_dataset == "ricechem":
        dataset_path = os.path.join(project_path, "statics/datasets/RiceChem/data")
        dataset = ricechem_dataset.RiceChemDataset(data_path=dataset_path, correction_path=corrections_path, use_corrections=True, correction_only=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: b, shuffle=False)
        intervention_logic = ricechem_intervention.RiceChemIntervention(dataset, llm, prompt_type=args.prompting_regime)
        evaluator = ricechem_evaluation.RiceChemCorrectionEvaluation(dataset)

    elif args.evaluation_dataset == "averitec":
        dataset = averitec_dataset.AVeriTeCCorrectionDataset(path=corrections_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: b, shuffle=False)
        intervention_logic = averitec_intervention.AVeriTeCIntervention(dataset, llm, prompt_type=args.prompting_regime)
        # evaluator = averitec_evaluation.AVeriTeCCorrectionEvaluation(dataset, intervention_logic)

    else:
        raise NotImplementedError(args.evaluation_dataset)

    if args.try_one_batch:
        dataloader = [next(iter(dataloader))]

    processed_samples_list = []

    max_new_tokens = 1024 if args.prompting_regime == "baseline_structure_tool_call" else 10

    for batch in tqdm(dataloader, desc=f"Correction ({args.evaluation_dataset})", total=len(dataloader)):
        # 1) BAD mediator pass
        bad_prompts = []
        for s in batch:
            tmp = deepcopy(s)
            if args.evaluation_dataset == "ricechem":
                tmp["filled_rubric"] = tmp["bad_rubric"]
            else:  # averitec_correction
                tmp["supporting_questions"] = tmp["bad_supporting_questions"]
            bad_prompts.append(intervention_logic.make_prompt(tmp, include_gold_structure=True))
        bad_outputs = llm.generate(bad_prompts, max_new_tokens=max_new_tokens, skip_special_tokens=True)

        # 2) Build corrected prompts + run again with GOLD mediator
        corrected_prompts = []
        samples_with_interventions = []

        for s, out in zip([deepcopy(x) for x in batch], bad_outputs):
            s2 = intervention_logic.make_correction_intervention(s, out)
            samples_with_interventions.append(s2)
            corrected_prompts.append(intervention_logic.make_prompt(s2['structure_intervention']['correction'][0], include_gold_structure=True))

        corrected_outputs = llm.generate(corrected_prompts, max_new_tokens=max_new_tokens, skip_special_tokens=True)

        for s2, out2 in zip(samples_with_interventions, corrected_outputs):
            final = intervention_logic.collect_correction_completion(s2, [out2])
            processed_samples_list.append(final)

    evaluation_metrics = {}
    if evaluator:
        try:
            evaluation_metrics = evaluator.evaluate(processed_samples_list)
        except Exception as e:
            error_type, error_message = type(e).__name__, str(e)
            print(f"[WARNING] Evaluation failed with {error_type}: {error_message}")
            evaluation_metrics = {
                "error": f"{error_type}: {error_message}"
            }

    final_dict = {
        "metrics": evaluation_metrics,
        "result": processed_samples_list,
        "fails": [],
    }

    print('Processed: ', len(processed_samples_list))
    # print('Failed: ', len(fails_list))
    dataset_name = args.evaluation_dataset
    path2save = os.path.join(project_path, "intervention_analysis", "correction_predictions", dataset_name)
    os.makedirs(path2save, exist_ok=True)

    model_name = model_name2simple_model_name[args.model_name]

    curr_time = datetime.now().strftime("%Y-%m-%d@%H:%M")
    prompt_regime = "dt" if args.prompting_regime == "detailed_instruction" else "bsf"
    file_name = f"{model_name}_{curr_time}_{prompt_regime}_one_batch.json" if args.try_one_batch else f"{model_name}_{curr_time}_{prompt_regime}.json"
    path2save = os.path.join(path2save, file_name)

    with open(path2save, "w", encoding='utf-8') as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=4)
    print(f"The results are saved to {path2save}!")
