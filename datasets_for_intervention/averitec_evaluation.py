from statistics import mean, pstdev
from math import isclose

class AVeriTeCEvaluation:

    def __init__(self, dataset, intervention_logic):
        self.dataset = dataset
        self.intervention_logic = intervention_logic

        self.idx2gold_structure = {sample['idx']: sample['supporting_questions'] for sample in dataset}
        self.idx2gold_verdict = {sample['idx']: sample['label'] for sample in dataset}


    def compare_checklists(self, gold_supporting_questions, predicted_supporting_questions):
        if gold_supporting_questions is None or predicted_supporting_questions is None:
            return 0
        for item, answer in gold_supporting_questions.items():
            if item not in predicted_supporting_questions or predicted_supporting_questions[item] != answer:
                return 0
        return 1

    def compare_verdicts(self, gold_verdict, predicted_verdict):
        if gold_verdict is None or predicted_verdict is None:
            return 0
        return 1 if gold_verdict == predicted_verdict else 0


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
                    "verdict_match": []
                },
                "with_predicted_structure": {
                    "structure_match": [],
                    "verdict_match": []
                }
            },
            "faithfullness": {
                "with_gold_structure": {
                    "HSVT": [],
                    "Local Edits": [],
                    "Global": []
                },
                "with_predicted_structure": {
                    "HSVT": [],
                    "Local Edits": [],
                    "Global": []
                }
            },
            "local_edit_influence": {
                "with_gold_structure": {},
                "with_predicted_structure": {}
            }
        }

        for sample in processed_samples_list:
            sample_idx = sample['idx']
            completion_type = sample['completion_type']
            
            # original structure and corresponding verdict
            gold_structure, gold_verdict = self.idx2gold_structure[sample_idx], self.idx2gold_verdict[sample_idx]
            # predicted structure and corresponding verdict
            predicted_structure, predicted_verdict = sample['supporting_questions'], sample['label']

            structure_match = self.compare_checklists(gold_structure, predicted_structure)
            verdict_match = self.compare_verdicts(gold_verdict, predicted_verdict)

            if completion_type == "gold_structure":
                evaluation_metrics["performance"]["with_gold_structure"]["verdict_match"].append(verdict_match)
            elif completion_type == "structure_prediction":
                evaluation_metrics["performance"]["with_predicted_structure"]["structure_match"].append(structure_match)
                evaluation_metrics["performance"]["with_predicted_structure"]["verdict_match"].append(verdict_match)

            # faithfullness metrics
            structure_intervention = sample['structure_intervention']

            hsvt_intervention = structure_intervention['HSVT'][0]
            expected_hsvt_verdict = hsvt_intervention['label']
            hsvt_result_after_intervention = hsvt_intervention['label_after_intervention']
            hsvt_intervention_match = self.compare_verdicts(expected_hsvt_verdict, hsvt_result_after_intervention)

            if completion_type == "gold_structure":
                evaluation_metrics["faithfullness"]["with_gold_structure"]["HSVT"].append(hsvt_intervention_match)
            elif completion_type == "structure_prediction":
                evaluation_metrics["faithfullness"]["with_predicted_structure"]["HSVT"].append(hsvt_intervention_match)

            # Local edits intervention
            if predicted_verdict == "Supported" or len(predicted_structure) == 1:
                local_edits_intervention = structure_intervention['Local Edits']
                for intervention_idx, local_edit_intervention in enumerate(local_edits_intervention):
                    expected_local_edit_verdict = local_edit_intervention['label']
                    local_edit_result_after_intervention = local_edit_intervention['label_after_intervention']
                    # if predicted_structure == 'Supported' or len(gold_structure) == 1:
                    local_edit_intervention_match = self.compare_verdicts(expected_local_edit_verdict,
                                                                        local_edit_result_after_intervention)

                    if completion_type == "gold_structure":
                        evaluation_metrics["faithfullness"]["with_gold_structure"]["Local Edits"].append(
                            local_edit_intervention_match)
                        if intervention_idx not in evaluation_metrics["local_edit_influence"]["with_gold_structure"]:
                            evaluation_metrics["local_edit_influence"]["with_gold_structure"][intervention_idx] = []
                        evaluation_metrics["local_edit_influence"]["with_gold_structure"][intervention_idx].append(local_edit_intervention_match)

                    elif completion_type == "structure_prediction":
                        evaluation_metrics["faithfullness"]["with_predicted_structure"]["Local Edits"].append(
                            local_edit_intervention_match)
                        if intervention_idx not in evaluation_metrics["local_edit_influence"]["with_predicted_structure"]:
                            evaluation_metrics["local_edit_influence"]["with_predicted_structure"][intervention_idx] = []
                        evaluation_metrics["local_edit_influence"]["with_predicted_structure"][intervention_idx].append(local_edit_intervention_match)

                # Global intervention (we do it only if more then one local edit and we look only at Supported class) 
                if len(local_edits_intervention) > 1:
                    global_intervention = structure_intervention['Global'][0]
                    expected_global_verdict = global_intervention['label']
                    global_result_after_intervention = global_intervention['label_after_intervention']
                    global_intervention_match = self.compare_verdicts(expected_global_verdict, global_result_after_intervention)

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
        for structure_type, edit_metrics in evaluation_metrics["local_edit_influence"].items():
            print(f"\n{structure_type}:")
            for edit_id, value in edit_metrics.items():
                if None not in value.values():
                    print(f"  Edit {edit_id}: mean = {value['mean']}, std = {value['std']}")
                else:
                    print(f"  Edit {edit_id}: mean = No, std = No")

