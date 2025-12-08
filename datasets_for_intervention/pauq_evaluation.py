from .utils import extract_tables_and_columns
from statistics import mean, pstdev


class PAUQEvaluation:
    def __init__(self, dataset=None):
        self.dataset = dataset

    def compare_sql_queries(self, query_before: str, query_after: str, intervention: dict[str, str]) -> bool:
        type = intervention["type"]
        before = intervention["before"]
        after = intervention["after"]

        # parsed_query_before = extract_tables_and_columns(query_before)
        # parsed_query_after = extract_tables_and_columns(query_after)

        # if before not in parsed_query_before[type]:
        #     print(before)
        #     print(parsed_query_before[type])
        #     print(query_before)
        #     print("aAAA" * 50)
        #     print(query_after)
        #     raise ValueError("Query before and intervention do not match!")

        # if before in parsed_query_after[type]:
        #     return False

        # before_idx = parsed_query_before[type].index(before)
        # parsed_query_before[type][before_idx] = after

        # return set(parsed_query_after[type]) == set(parsed_query_before[type])
        if before in query_after:
            return False
        
        return after in query_after


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
            # "local_edit_influence": {
            #     "with_gold_structure": {task_idx: {intervention_idx: []
            #                                        for intervention_idx in
            #                                        range(len(self.dataset.task2rubric_weights[task_idx]))}
            #                             for task_idx in self.dataset.task2rubric_weights},
            #     "with_predicted_structure": {task_idx: {intervention_idx: []
            #                                             for intervention_idx in
            #                                             range(len(self.dataset.task2rubric_weights[task_idx]))}
            #                                  for task_idx in self.dataset.task2rubric_weights}
            # }
        }
        for sample in processed_samples_list:
            sample_idx = sample['index']
            completion_type = sample['completion_type']
            gold_sql = sample['query']
            # predicted checklist and corresponding score
            predicted_sql = sample['generated_sql']

            # faithfullness metrics
            structure_intervention = sample['structure_intervention']

            hsvt_intervention = structure_intervention['HSVT'][0]
            hsvt_result_after_intervention = hsvt_intervention['generated_sql']
            hsvt_intervention_score = int(hsvt_result_after_intervention == predicted_sql)

            # if completion_type == "gold_structure":
            #     evaluation_metrics["faithfullness"]["with_gold_structure"]["HSVT"].append(hsvt_intervention_score)
            # elif completion_type == "structure_prediction":
            evaluation_metrics["faithfullness"]["with_predicted_structure"]["HSVT"].append(hsvt_intervention_score)

            # Local edits intervention
            local_edits_intervention = structure_intervention['Local Edits']
            for intervention_idx, local_edit_intervention in enumerate(local_edits_intervention):
                local_edit_result_after_intervention = local_edit_intervention['generated_sql']

                local_edit_intervention_match = self.compare_sql_queries(predicted_sql,
                                                                         local_edit_result_after_intervention,
                                                                         local_edit_intervention['local_intervention'])

                evaluation_metrics["faithfullness"]["with_predicted_structure"]["Local Edits"].append(
                    local_edit_intervention_match)

            # Global intervention
            global_intervention = structure_intervention['Global'][0]
            global_result_after_intervention = global_intervention['generated_sql']

            global_intervention_score = 0
            for global_int in global_intervention['global_intervention']:
                global_intervention_match = self.compare_sql_queries(predicted_sql, global_result_after_intervention, global_int)
                global_intervention_score += global_intervention_match
            global_intervention_score /= len(global_intervention['global_intervention'])

            evaluation_metrics["faithfullness"]["with_predicted_structure"]["Global"].append(
                global_intervention_score)

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
        # for structure_type, task_metrics in evaluation_metrics["local_edit_influence"].items():
        #     print(f"\n{structure_type}:")
        #     for task_id, scores in task_metrics.items():
        #         print(f"  Task {task_id}:")
        #         for edit_id, value in scores.items():
        #             if None not in value.values():
        #                 print(f"    Edit {edit_id}: mean = {value['mean']}, std = {value['std']}")
        #             else:
        #                 print(f"    Edit {edit_id}: mean = No, std = No")



if __name__ == "__main__":
    sql_before = "SELECT name FROM students WHERE age > 20"
    sql_after = "SELECT full_name FROM students WHERE age > 20"
    intervention = {"type": "columns", "before": "name", "after": "full_name"}

    info_before = extract_tables_and_columns(sql_before)
    info_after = extract_tables_and_columns(sql_after)

    print("Before:", info_before)
    print("After:", info_after)

    eval = PAUQEvaluation()
    print(eval.compare_sql_queries(sql_before, sql_after, intervention))
