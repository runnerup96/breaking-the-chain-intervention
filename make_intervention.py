
import argparse
import llm_model
# from datasets_for_intervention import wilds_reviews_intervention, wilds_reviews_dataset
from datasets_for_intervention import ricechem_intervention, ricechem_dataset, entailment_intervention, entailment_dataset
import os
from tqdm import tqdm
from datetime import datetime
import json
from torch.utils.data import DataLoader


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
        raise NotImplementedError(f"No implementation for {args.evaluation_dataset} dataset")
        # dataset_path = os.path.join(project_path, "statics/result_splits/test_balanced.json")
        # dataset = wilds_reviews_dataset.WildsReviewsDataset(dataset_path)
        # dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        # intervention_logic = wilds_reviews_intervention.WildsReviewsIntervention(llm_model.stop_token)
    elif args.evaluation_dataset == "ricechem":
        dataset_path = os.path.join(project_path, "statics/result_splits/RiceChem")
        dataset = ricechem_dataset.RiceChemDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = ricechem_intervention.RiceChemIntervention(dataset, llm_model.stop_token)
    elif args.evaluation_dataset == "entailment":
        dataset_path = os.path.join(project_path, "entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl")
        dataset = entailment_dataset.EntailmentDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda batch: batch, shuffle=False)
        intervention_logic = entailment_intervention.EntailmentIntervention(dataset, llm_model.stop_token)
    else:
        raise NotImplementedError(f"No implementation for {args.evaluation_dataset} dataset"
                                  f"Currently -- [amazon_reviews, ricechem]")

    print(f"Loaded dataset {args.evaluation_dataset}")

    if args.try_one_batch:
        dataloader = [next(iter(dataloader))]

    intervention_dict = {"interventions": dict(), "fails": []}
    for batch in tqdm(dataloader, desc="Running inference", total=len(dataloader)):
        batch_idx_list = [sample["idx"] for sample in batch]
        prompted_batch = [intervention_logic.make_prompt(sample) for sample in batch]
        # DO_X -- if we have ground truth, we wont need to fill it by the model
        # TODO: Add a possiblity to run ground truth generation
        batched_model_outputs = llm_model.generate(prompted_batch, max_new_tokens=1024,
                                                   include_chat_template=True,
                                                   skip_special_tokens=False)
        for idx, (sample_idx, model_output) in enumerate(zip(batch_idx_list, batched_model_outputs)):
            # Mediator(DO_X)
            intervened_batch_for_completion = intervention_logic.make_intervention(model_output)
            # i may have multiple interventions per one sample -- so i check all of them for correctness
            if intervened_batch_for_completion:
                # we do that validation only on the completion
                # TODO: check on size of the intervention
                intervention_check = intervention_logic.validate_all_interventions(model_output, intervened_batch_for_completion)
                if intervention_check:
                    # reconstruct the prompt & prepare for completion
                    prompt_list = intervention_logic.reconstruct_interventions_to_prompt(model_output, intervened_batch_for_completion)
                    intervened_completion_outputs = llm_model.generate(prompt_list, max_new_tokens=10,
                                                                       include_chat_template=False,
                                                                       skip_special_tokens=True)
                    # keep the original as 0
                    do_prediction = intervention_logic.extract_target_from_prompt(model_output)
                    mediator_prediction = [intervention_logic.infer_completion(output) for output in intervened_completion_outputs]
                    # additionaly check that we get clean results
                    if do_prediction is not None or None not in mediator_prediction:
                        # TODO: Collect all dataset specific generations here and save to separate field for fast eval
                        intervention_dict["interventions"][sample_idx] = {"predictions": [do_prediction] + mediator_prediction,# what we have extracted
                                                                                 "original_generation": model_output,
                                                                                 "intervened_generations": intervened_completion_outputs}# what we need for analysis
                    else:
                        intervention_dict['fails'].append(sample_idx)
                        continue
                else:
                    intervention_dict['fails'].append(sample_idx)#just add index of failed sample for intervention
            else:
                intervention_dict['fails'].append(sample_idx)  # just add index of failed sample for intervention

    dataset_name = args.evaluation_dataset
    path2save = os.path.join(project_path, "intervention_analysis", "intervention_predictions", dataset_name)
    os.makedirs(path2save, exist_ok=True)

    model_name = model_name2simple_model_name[args.model_name]

    curr_time = datetime.now().strftime("%Y-%m-%d@%H:%M")
    file_name = f"{model_name}_{curr_time}_one_batch.json" if args.try_one_batch else f"{model_name}_{curr_time}.json"
    path2save = os.path.join(path2save, file_name)

    with open(path2save, "w") as f:
        json.dump(intervention_dict, f, ensure_ascii=False, indent=4)
    print(f"Saved predictions to {path2save}!")


            
        
        



