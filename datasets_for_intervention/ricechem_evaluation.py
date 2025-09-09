from statistics import mean, pstdev
from math import isclose

class RiceChemEvaluation:

    def __init__(self, dataset, intervention_logic):
        self.dataset = dataset
        self.intervention_logic = intervention_logic

        self.idx2gold_checklist = {sample['idx']: sample['filled_rubric'] for sample in dataset}
        self.idx2gold_score = {sample['idx']: sample['score'] for sample in dataset}

    def compare_checklists(self, gold_checklist, predicted_checklist):
        for item, answer in gold_checklist.items():
            if item not in predicted_checklist or predicted_checklist[item] != answer:
                return 0
        return 1

    def compare_scores(self, gold_score, predicted_score, *, atol=1e-6):
        if gold_score is None or predicted_score is None:
            return 0
        return 1 if isclose(gold_score, predicted_score, abs_tol=atol) else 0

    def summarize_nested_lists(self, tree):
        if isinstance(tree, dict):
            return {k: self.summarize_nested_lists(v) for k, v in tree.items()}
        elif isinstance(tree, list):
            if not all(isinstance(x, (int, float)) for x in tree):
                raise TypeError("All list elements must be int or float.")
            if len(tree) == 0:
                return {"mean": None, "std": None}
            return {"mean": round(mean(tree), 3), "std": round(pstdev(tree), 3)}
        else:
            raise TypeError("Leaf values must be lists; found non-list leaf instead.")

    def evaluate(self, processed_samples_list):
        evaluation_metrics = {
            "performance": {
                "with_gold_structure": {
                    "score_match": []
                },
                "with_predicted_structure": {
                    "checklist_match": [],
                    "score_match": []
                }
            },
            "faithfullness": {
                "with_gold_structure": {
                    "HSVT": [],
                    "Local Edits": [],  # avg here,
                    "Global": []
                },
                "with_predicted_structure": {
                    "HSVT": [],
                    "Local Edits": [],
                    "Global": []
                }
            },
            "local_edit_influence": {
                "with_gold_structure": {task_idx: {intervention_idx: []
                                                   for intervention_idx in
                                                   range(len(self.dataset.task2rubric_weights[task_idx]))}
                                        for task_idx in self.dataset.task2rubric_weights},
                "with_predicted_structure": {task_idx: {intervention_idx: []
                                                        for intervention_idx in
                                                        range(len(self.dataset.task2rubric_weights[task_idx]))}
                                             for task_idx in self.dataset.task2rubric_weights}
            }
        }
        for sample in processed_samples_list:
            sample_idx = sample['idx']
            task_idx = sample['task_idx']
            completion_type = sample['completion_type']
            # original checklist and corresponding score
            gold_checklist, gold_score = self.idx2gold_checklist[sample_idx], self.idx2gold_score[sample_idx]
            # predicted checklist and corresponding score
            predicted_checklist, predicted_score = sample['filled_rubric'], sample['score']

            checklist_match = self.compare_checklists(gold_checklist, predicted_checklist)
            score_match = self.compare_scores(gold_score, predicted_score)

            if completion_type == "gold_structure":
                evaluation_metrics["performance"]["with_gold_structure"]["score_match"].append(score_match)
            elif completion_type == "structure_prediction":
                evaluation_metrics["performance"]["with_predicted_structure"]["checklist_match"].append(checklist_match)
                evaluation_metrics["performance"]["with_predicted_structure"]["score_match"].append(score_match)

            # faithfullness metrics
            structure_intervention = sample['structure_intervention']

            hsvt_intervention = structure_intervention['HSVT'][0]
            expected_hsvt_edit_score = hsvt_intervention['score']
            hsvt_result_after_intervention = hsvt_intervention['score_after_intervention']
            hsvt_intervention_score = self.compare_scores(expected_hsvt_edit_score, hsvt_result_after_intervention)

            if completion_type == "gold_structure":
                evaluation_metrics["faithfullness"]["with_gold_structure"]["HSVT"].append(hsvt_intervention_score)
            elif completion_type == "structure_prediction":
                evaluation_metrics["faithfullness"]["with_predicted_structure"]["HSVT"].append(hsvt_intervention_score)

            # Local edits intervention
            local_edits_intervention = structure_intervention['Local Edits']
            for intervention_idx, local_edit_intervention in enumerate(local_edits_intervention):
                expected_local_edit_score = local_edit_intervention['score']
                local_edit_result_after_intervention = local_edit_intervention['score_after_intervention']

                local_edit_intervention_match = self.compare_scores(expected_local_edit_score,
                                                                    local_edit_result_after_intervention)

                if completion_type == "gold_structure":

                    evaluation_metrics["faithfullness"]["with_gold_structure"]["Local Edits"].append(
                        local_edit_intervention_match)
                    evaluation_metrics["local_edit_influence"]["with_gold_structure"][task_idx][
                        intervention_idx].append(local_edit_intervention_match)

                elif completion_type == "structure_prediction":

                    evaluation_metrics["faithfullness"]["with_predicted_structure"]["Local Edits"].append(
                        local_edit_intervention_match)
                    evaluation_metrics["local_edit_influence"]["with_predicted_structure"][task_idx][
                        intervention_idx].append(local_edit_intervention_match)

            # Global intervention
            global_intervention = structure_intervention['Global'][0]
            expected_global_edit_score = global_intervention['score']
            global_result_after_intervention = global_intervention['score_after_intervention']
            global_intervention_match = self.compare_scores(expected_global_edit_score, global_result_after_intervention)

            if completion_type == "gold_structure":
                evaluation_metrics["faithfullness"]["with_gold_structure"]["Global"].append(
                    global_intervention_match)
            elif completion_type == "structure_prediction":
                evaluation_metrics["faithfullness"]["with_predicted_structure"]["Global"].append(
                    global_intervention_match)

        aggregated_evaluation_metrics = self.summarize_nested_lists(evaluation_metrics)
        self.print_evaluation_metrics(aggregated_evaluation_metrics)

        return aggregated_evaluation_metrics
    
    def print_evaluation_metrics(self, evaluation_metrics):
        print("\nEvaluation Results:")
        print("===================")

        print("\nPerformance Metrics:")
        print("-------------------")
        for structure_type, metrics in evaluation_metrics["performance"].items():
            print(f"\n{structure_type}:")
            for metric_name, value in metrics.items():
                if None not in value.values():
                    print(f"  {metric_name}: mean = {value['mean']}, std = {value['std']}")
                else:
                    print(f"  {metric_name}: mean = No, std = No")
        
        print("\nFaithfulness Metrics:")
        print("--------------------")
        for structure_type, metrics in evaluation_metrics["faithfullness"].items():
            print(f"\n{structure_type}:")
            for intervention_type, value in metrics.items():
                if None not in value.values():
                    print(f"  {intervention_type}: mean = {value['mean']}, std = {value['std']}")
                else:
                    print(f"  {intervention_type}: mean = No , std = No ")
                
        print("\nLocal Edit Influence:")
        print("--------------------") 
        for structure_type, task_metrics in evaluation_metrics["local_edit_influence"].items():
            print(f"\n{structure_type}:")
            for task_id, scores in task_metrics.items():
                print(f"  Task {task_id}:")
                for edit_id, value in scores.items():
                    if None not in value.values():
                        print(f"    Edit {edit_id}: mean = {value['mean']}, std = {value['std']}")
                    else:
                        print(f"    Edit {edit_id}: mean = No, std = No")
