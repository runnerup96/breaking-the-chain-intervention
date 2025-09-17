
import argparse
import llm_model
from datasets_for_intervention import wilds_reviews_intervention, wilds_reviews_dataset
from datasets_for_intervention import ricechem_intervention, ricechem_dataset, ricechem_evaluation
from datasets_for_intervention import averitec_intervention, averitec_dataset, averitec_evaluation
import os
from tqdm import tqdm
from datetime import datetime
import json
from torch.utils.data import DataLoader
from copy import deepcopy


model_name2simple_model_name = {
        "Qwen/Qwen3-4B": "qwen3-4B",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--evaluation_dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--try_one_batch", type=bool, default=False)

    args = parser.parse_args()

    llm_model = llm_model.LLMModel(args.model_name)

    project_path = os.environ["PROJECT_PATH"]

    dataset = None
    if args.evaluation_dataset == "amazon_reviews":
        dataset_path = os.path.join(project_path, "statics/result_splits/test_balanced.json")
        dataset = wilds_reviews_dataset.WildsReviewsDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = wilds_reviews_intervention.WildsReviewsIntervention(llm_model.stop_token)
        evaluator = None
    elif args.evaluation_dataset == "ricechem":
        dataset_path = os.path.join(project_path, "statics/result_splits/RiceChem")
        dataset = ricechem_dataset.RiceChemDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = ricechem_intervention.RiceChemIntervention(dataset, llm_model.tokenizer)
        evaluator = ricechem_evaluation.RiceChemEvaluation(dataset, intervention_logic)
    elif args.evaluation_dataset == "averitec":
        dataset_path = os.path.join(project_path, "statics/result_splits/AVeriTeC/data")
        dataset = averitec_dataset.AVeriTeCDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = averitec_intervention.AVeriTeCIntervention(dataset, llm_model.tokenizer)
        evaluator = averitec_evaluation.AVeriTeCEvaluation(dataset, intervention_logic)
    else:
        raise NotImplementedError(f"No implementation for {args.evaluation_dataset} dataset"
                                  f"Currently -- [amazon_reviews, ricechem, averitec]")

    print(f"Loaded dataset {args.evaluation_dataset}")

    if args.try_one_batch:
        dataloader = [next(iter(dataloader))]
        # dataloader = [list(dataloader)[-1]]

    processed_samples_list, fails_list = [], []
    for batch in tqdm(dataloader, desc="Running inference", total=len(dataloader)):
        
        # batch_idx_list = [sample["idx"] for sample in batch]
        prompted_batch_with_structure_prediction = [intervention_logic.make_prompt(sample, include_gold_structure=False) for sample in batch]
        promted_batch_with_gold_structure = [intervention_logic.make_prompt(sample, include_gold_structure=True) for sample in batch]
        all_batch = prompted_batch_with_structure_prediction + promted_batch_with_gold_structure
        completion_type_list = ["structure_prediction"] * len(prompted_batch_with_structure_prediction) + ["gold_structure"] * len(promted_batch_with_gold_structure)
        # DO_X -- if we have ground truth, we wont need to fill it by the model
        batched_model_outputs = llm_model.generate(all_batch, max_new_tokens=1024,# X2 from batch
                                                   skip_special_tokens=False)
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
                error_string = f"{error_type}: {error_message}"
                fails_list.append([sample, error_string])


    evaluation_metrics = evaluator.evaluate(processed_samples_list)

    final_dataset_dict = {"metrics": evaluation_metrics, "result": processed_samples_list, "fails": fails_list}
    print('Processed: ', len(processed_samples_list))
    print('Failed: ', len(fails_list))
    dataset_name = args.evaluation_dataset
    path2save = os.path.join(project_path, "intervention_analysis", "intervention_predictions", dataset_name)
    os.makedirs(path2save, exist_ok=True)

    model_name = model_name2simple_model_name[args.model_name]

    curr_time = datetime.now().strftime("%Y-%m-%d@%H:%M")
    file_name = f"{model_name}_{curr_time}_one_batch.json" if args.try_one_batch else f"{model_name}_{curr_time}.json"
    path2save = os.path.join(path2save, file_name)

    with open(path2save, "w") as f:
        json.dump(final_dataset_dict, f, ensure_ascii=False, indent=4)
    print(f"The results are saved to {path2save}!")



            
        
        



