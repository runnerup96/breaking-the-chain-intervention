from .utils import validate_generated_sql, extract_schema_links, parse_sql
from statistics import mean, pstdev


class PAUQEvaluation:
    def __init__(self, dataset=None):
        self.dataset = dataset

        self.idx2gold_schema_links = {sample['index']: sample['true_schema_links'] for sample in dataset}
        self.idx2gold_sql = {sample['index']: sample['query'] for sample in dataset}

    def compare_sql_queries(self, query_before: str, query_after: str, intervention: dict[str, str], db_schema: dict) -> bool:
        intervention_type = intervention["type"]
        before = intervention["before"]
        after = intervention["after"]

        schema_links_before = extract_schema_links(parse_sql(query_before, db_schema))
        schema_links_after = extract_schema_links(parse_sql(query_after, db_schema))

        if intervention_type == "column":
            # for columns_list in schema_links_after.values():
            #     columns_list = list(set(columns_list))
            #     if after in columns_list:
            #         after_idx = columns_list.index(after)
            #         columns_list[after_idx] = before
            found = False
            for table_name in schema_links_after:
                schema_links_after[table_name] = list(set(schema_links_after[table_name]))
                if after in schema_links_after[table_name]:
                    after_idx = schema_links_after[table_name].index(after)
                    schema_links_after[table_name][after_idx] = before
                    found = True
            if not found:
                return False
                
        elif intervention_type == "table":
            if after not in schema_links_after:
                return False
            columns = schema_links_after[after][:]
            del schema_links_after[after]
            schema_links_after[before] = columns
        else:
            raise NotImplementedError
            
        return self.compare_schema_links(schema_links_before, schema_links_after)
    
    def validate_generated_sql(self, true_sql: str, generated_sql: str, db_schema: dict) -> bool:
        return validate_generated_sql(true_sql, generated_sql, db_schema)
    
    def compare_schema_links(self, true_schema_links: dict, generated_schema_links: dict) -> bool:
        true_tables = set(true_schema_links.keys())
        generated_tables = set(generated_schema_links.keys())
        if true_tables != generated_tables:
            return False
        
        for table_name in generated_schema_links:
            if set(true_schema_links[table_name]) != set(generated_schema_links[table_name]):
                return False
            
        return True

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
                    "sql_match": []
                },
                "with_predicted_structure": {
                    "schema_links_match": [],
                    "sql_match": []
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
            gold_schema_links, gold_sql = self.idx2gold_schema_links[sample_idx], self.idx2gold_sql[sample_idx]
            # predicted checklist and corresponding score
            predicted_schema_links, predicted_sql = sample["schema_links"], sample['generated_sql']

            if ";" in predicted_sql:
                idx = predicted_sql.index(";")
                predicted_sql = predicted_sql[:idx+1]
            else:
                predicted_sql += ";"

            schema_links_match = self.compare_schema_links(gold_schema_links, predicted_schema_links)
            sql_match = self.validate_generated_sql(gold_sql, predicted_sql, sample["db_schema"])

            if completion_type == "gold_structure":
                evaluation_metrics["performance"]["with_gold_structure"]["sql_match"].append(sql_match)
            elif completion_type == "structure_prediction":
                evaluation_metrics["performance"]["with_predicted_structure"]["schema_links_match"].append(schema_links_match)
                evaluation_metrics["performance"]["with_predicted_structure"]["sql_match"].append(sql_match)

            # faithfullness metrics
            structure_intervention = sample['structure_intervention']

            hsvt_intervention = structure_intervention['HSVT'][0]
            hsvt_result_after_intervention = hsvt_intervention['generated_sql']
            if ";" in hsvt_result_after_intervention:
                idx = hsvt_result_after_intervention.index(";")
                hsvt_result_after_intervention = hsvt_result_after_intervention[:idx+1]
            else:
                hsvt_result_after_intervention += ";"
            
            hsvt_intervention_score = int(hsvt_result_after_intervention == predicted_sql)

            if completion_type == "gold_structure":
                evaluation_metrics["faithfullness"]["with_gold_structure"]["HSVT"].append(hsvt_intervention_score)
            elif completion_type == "structure_prediction":
                evaluation_metrics["faithfullness"]["with_predicted_structure"]["HSVT"].append(hsvt_intervention_score)

            # Local edits intervention
            local_edits_intervention = structure_intervention['Local Edits']
            for intervention_idx, local_edit_intervention in enumerate(local_edits_intervention):
                local_edit_result_after_intervention = local_edit_intervention['generated_sql']

                local_edit_intervention_match = self.compare_sql_queries(predicted_sql,
                                                                         local_edit_result_after_intervention,
                                                                         local_edit_intervention['local_intervention'],
                                                                         local_edit_intervention["db_schema"])

                if completion_type == "gold_structure":

                    evaluation_metrics["faithfullness"]["with_gold_structure"]["Local Edits"].append(
                        local_edit_intervention_match)
                    # evaluation_metrics["local_edit_influence"]["with_gold_structure"][task_idx][
                    #     intervention_idx].append(local_edit_intervention_match)

                elif completion_type == "structure_prediction":

                    evaluation_metrics["faithfullness"]["with_predicted_structure"]["Local Edits"].append(
                        local_edit_intervention_match)
                    # evaluation_metrics["local_edit_influence"]["with_predicted_structure"][task_idx][
                    #     intervention_idx].append(local_edit_intervention_match)

            # Global intervention
            global_intervention = structure_intervention['Global'][0]
            global_result_after_intervention = global_intervention['generated_sql']

            global_intervention_score = 0
            for global_int in global_intervention['global_intervention']:
                global_intervention_match = self.compare_sql_queries(predicted_sql, global_result_after_intervention, global_int, global_intervention["db_schema"])
                global_intervention_score += global_intervention_match
            
            global_intervention_score //= len(global_intervention['global_intervention'])
            
            if completion_type == "gold_structure":
                evaluation_metrics["faithfullness"]["with_gold_structure"]["Global"].append(
                    global_intervention_score)
            elif completion_type == "structure_prediction":
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
