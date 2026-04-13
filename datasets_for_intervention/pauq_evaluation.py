try:
    from utils import validate_generated_sql, extract_schema_links, parse_sql, extract_skeleton_and_slots, compare_skeletons, compare_slots
except ImportError:
    from .utils import validate_generated_sql, extract_schema_links, parse_sql, extract_skeleton_and_slots, compare_skeletons, compare_slots
    
from statistics import mean, pstdev


def _normalize_schema_links(schema_links: dict) -> dict:
    normalized = {}
    for table, cols in schema_links.items():
        table_lc = table.lower()
        cols_lc = [c.lower() for c in cols]
        normalized[table_lc] = sorted(set(cols_lc))
    return normalized


def _compare_schema_links_normalized(true_schema_links: dict, generated_schema_links: dict) -> bool:
    true_schema_links = _normalize_schema_links(true_schema_links)
    generated_schema_links = _normalize_schema_links(generated_schema_links)

    true_tables = set(true_schema_links.keys())
    generated_tables = set(generated_schema_links.keys())
    if true_tables != generated_tables:
        return False

    for table_name in generated_schema_links:
        if set(true_schema_links[table_name]) != set(generated_schema_links[table_name]):
            return False

    return True


def faithfulness_id(mediator_schema_links, mediator_skeleton, mediator_slots, generated_sql, db_schema) -> bool:
    try:
        gen_schema_links = extract_schema_links(parse_sql(generated_sql, db_schema))
        gen_skeleton, gen_slots = extract_skeleton_and_slots(generated_sql, db_schema)
    except Exception:
        return False
    mediator_skeleton = "".join(mediator_skeleton.split())
    gen_skeleton = "".join(gen_skeleton.split())
    result = (
        # _compare_schema_links_normalized(mediator_schema_links, gen_schema_links)
        compare_skeletons(mediator_skeleton, gen_skeleton)
        and compare_slots(mediator_slots, gen_slots)
    )
    if not result:
        print(f"Faithfulness ID failed for {generated_sql}")
        print(f"Mediator schema links: {mediator_schema_links}")
        print(f"Generated schema links: {gen_schema_links}")
        print(f"Mediator skeleton: {mediator_skeleton}")
        print(f"Generated skeleton: {gen_skeleton}")
        print(f"Mediator slots: {mediator_slots}")
        print(f"Generated slots: {gen_slots}")
        print("--------------------------------")
    return result

class PAUQEvaluation:
    def __init__(self, dataset):
        self.dataset = dataset

        self.idx2gold_schema_links = {sample['index']: sample['true_schema_links'] for sample in dataset}
        self.idx2gold_sql = {sample['index']: sample['query'] for sample in dataset}

    def compare_sql_queries(self, query_before: str, query_after: str, intervention_list: dict[str, str], db_schema: dict) -> bool:
        schema_links_before = extract_schema_links(parse_sql(query_before, db_schema))
        schema_links_after = extract_schema_links(parse_sql(query_after, db_schema))

        for intervention in intervention_list:
            intervention_type = intervention["type"]
            before = intervention["before"]
            after = intervention["after"]

            if intervention_type == "column":
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

        return _compare_schema_links_normalized(schema_links_before, schema_links_after)

    def validate_generated_sql(self, true_sql: str, generated_sql: str, db_schema: dict) -> bool:
        return validate_generated_sql(true_sql, generated_sql, db_schema)

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
                    "Global": [],
                    "faithfulness_id": [],
                    "faithfulness_strong_HSVT": [],
                    "faithfulness_strong_Local Edits": [],
                    "faithfulness_strong_Global": [],
                }
            },
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

            schema_links_match = _compare_schema_links_normalized(gold_schema_links, predicted_schema_links)
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

            hsvt_intervention_score = self.validate_generated_sql(predicted_sql, hsvt_result_after_intervention, sample["db_schema"])

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
                                                                         [local_edit_intervention['local_intervention']],
                                                                         local_edit_intervention["db_schema"])

                if completion_type == "gold_structure":
                    evaluation_metrics["faithfullness"]["with_gold_structure"]["Local Edits"].append(
                        local_edit_intervention_match)

                elif completion_type == "structure_prediction":
                    evaluation_metrics["faithfullness"]["with_predicted_structure"]["Local Edits"].append(
                        local_edit_intervention_match)

            # Global intervention
            global_intervention = structure_intervention['Global'][0]
            global_result_after_intervention = global_intervention['generated_sql']

            global_intervention_match = self.compare_sql_queries(
                predicted_sql,
                global_result_after_intervention,
                global_intervention['global_intervention'],
                global_intervention["db_schema"]
            )
            global_intervention_score = int(global_intervention_match)
            
            if completion_type == "gold_structure":
                evaluation_metrics["faithfullness"]["with_gold_structure"]["Global"].append(
                    global_intervention_score)
            elif completion_type == "structure_prediction":
                evaluation_metrics["faithfullness"]["with_predicted_structure"]["Global"].append(
                    global_intervention_score)

            # faithfulness_id and faithfulness_strong (only for structure_prediction)
            if completion_type == "structure_prediction":
                f_id = faithfulness_id(
                    predicted_schema_links, sample["skeleton"], sample["slots"],
                    predicted_sql, sample["db_schema"]
                )
                evaluation_metrics["faithfullness"]["with_predicted_structure"]["faithfulness_id"].append(int(f_id))

                f_hsvt = faithfulness_id(
                    hsvt_intervention["schema_links"], hsvt_intervention["skeleton"], hsvt_intervention["slots"],
                    hsvt_intervention["generated_sql"], sample["db_schema"]
                )
                evaluation_metrics["faithfullness"]["with_predicted_structure"]["faithfulness_strong_HSVT"].append(
                    int(f_id and f_hsvt)
                )

                for local_edit_intervention in local_edits_intervention:
                    f_local = faithfulness_id(
                        local_edit_intervention["schema_links"], local_edit_intervention["skeleton"], local_edit_intervention["slots"],
                        local_edit_intervention["generated_sql"], local_edit_intervention["db_schema"]
                    )
                    evaluation_metrics["faithfullness"]["with_predicted_structure"]["faithfulness_strong_Local Edits"].append(
                        int(f_id and f_local)
                    )

                f_global = faithfulness_id(
                    global_intervention["schema_links"], global_intervention["skeleton"], global_intervention["slots"],
                    global_intervention["generated_sql"], global_intervention["db_schema"]
                )
                evaluation_metrics["faithfullness"]["with_predicted_structure"]["faithfulness_strong_Global"].append(
                    int(f_id and f_global)
                )

        aggregated_evaluation_metrics = self.summarize_nested_lists(evaluation_metrics)

        pred = aggregated_evaluation_metrics["faithfullness"]["with_predicted_structure"]
        f_id_mean = pred["faithfulness_id"]["mean"]
        for itype in ["HSVT", "Local Edits", "Global"]:
            f_strong_mean = pred[f"faithfulness_strong_{itype}"]["mean"]
            if f_id_mean is not None and f_strong_mean is not None:
                pred[f"faithfulness_gap_{itype}"] = round(f_id_mean - f_strong_mean, 3)
            else:
                pred[f"faithfulness_gap_{itype}"] = None

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
        gap_keys = {"faithfulness_gap_HSVT", "faithfulness_gap_Local Edits", "faithfulness_gap_Global"}
        for structure_type, metrics in evaluation_metrics["faithfullness"].items():
            print(f"\n{structure_type}:")
            for key, value in metrics.items():
                if key in gap_keys:
                    print(f"  {key}: {value}")
                elif None not in value.values():
                    print(f"  {key}: mean = {value['mean']}, std = {value['std']}")
                else:
                    print(f"  {key}: mean = No , std = No ")
                
        print("\nLocal Edit Influence:")
        print("--------------------") 


class PAUQCorrectionEvaluation:
    def __init__(self, dataset):
        self.dataset = dataset

        self.idx2gold_schema_links = {s["idx"]: s["true_schema_links"] for s in dataset}
        self.idx2gold_skeleton = {s["idx"]: s["true_skeleton"] for s in dataset}
        self.idx2gold_slots = {s["idx"]: s["true_slots"] for s in dataset}
        self.idx2gold_sql = {s["idx"]: s["query"] for s in dataset}
        self.idx2bad_schema_links = {s["idx"]: s["bad_schema_links"] for s in dataset}
        self.idx2bad_skeleton = {s["idx"]: s["bad_skeleton"] for s in dataset}
        self.idx2bad_slots = {s["idx"]: s["bad_slots"] for s in dataset}
        self.idx2db_schema = {s["idx"]: s["db_schema"] for s in dataset}

    def compare_sql_queries(self, query_before: str, query_after: str, db_schema: dict) -> bool:
        schema_links_before = extract_schema_links(parse_sql(query_before, db_schema))
        schema_links_after = extract_schema_links(parse_sql(query_after, db_schema))

        return _compare_schema_links_normalized(schema_links_before, schema_links_after)

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
                "with_bad_structure": {
                    "sql_match": [],
                },
                "with_corrected_structure": {
                    "sql_match": [],
                },
            },
            "faithfulness": {
                "correction": [],
                "faithfulness_id": [],
                "faithfulness_strong_correction": [],
            },
        }

        for sample in processed_samples_list:
            idx = sample["idx"]

            db_schema = self.idx2db_schema[idx]
            gold_schema_links = self.idx2gold_schema_links[idx]
            gold_skeleton = self.idx2gold_skeleton[idx]
            gold_slots = self.idx2gold_slots[idx]
            
            gold_sql = self.idx2gold_sql[idx]

            bad_schema_links = self.idx2bad_schema_links[idx]
            bad_skeleton = self.idx2bad_skeleton[idx]
            bad_slots = self.idx2bad_slots[idx]

            bad_sql_pred = sample.get("generated_sql_before_intervention")
 
            evaluation_metrics["performance"]["with_bad_structure"]["sql_match"].append(
                self.compare_sql_queries(gold_sql, bad_sql_pred, db_schema)
            )

            corrected = sample["structure_intervention"]["correction"][0]
            corrected_sql_pred = corrected.get("sql_after_intervention")
            after = self.compare_sql_queries(gold_sql, corrected_sql_pred, db_schema)
            evaluation_metrics["performance"]["with_corrected_structure"]["sql_match"].append(after)
            evaluation_metrics["faithfulness"]["correction"].append(after)

            f_id = faithfulness_id(bad_schema_links, bad_skeleton, bad_slots, bad_sql_pred, db_schema)
            evaluation_metrics["faithfulness"]["faithfulness_id"].append(int(f_id))

            f_correction = faithfulness_id(gold_schema_links, gold_skeleton, gold_slots, corrected_sql_pred, db_schema)
            evaluation_metrics["faithfulness"]["faithfulness_strong_correction"].append(int(f_id and f_correction))

        aggregated = self.summarize_nested_lists(evaluation_metrics)

        faith = aggregated["faithfulness"]
        f_id_mean = faith["faithfulness_id"]["mean"]
        f_strong_mean = faith["faithfulness_strong_correction"]["mean"]
        if f_id_mean is not None and f_strong_mean is not None:
            faith["faithfulness_gap_correction"] = round(f_id_mean - f_strong_mean, 3)
        else:
            faith["faithfulness_gap_correction"] = None

        self.print_evaluation_metrics(aggregated)
        return aggregated

    def print_evaluation_metrics(self, evaluation_metrics):
        print("\nEvaluation Results (PAUQ Correction):")
        print("========================================")

        print("\nPerformance:")
        for structure_type, metrics in evaluation_metrics["performance"].items():
            print(f"\n{structure_type}:")
            for metric_name, value in metrics.items():
                if None not in value.values():
                    print(f"  {metric_name}: mean = {value['mean']}, std = {value['std']}")
                else:
                    print(f"  {metric_name}: mean = No, std = No")

        print("\nFaithfulness:")
        gap_keys = {"faithfulness_gap_correction"}
        for metric_name, value in evaluation_metrics["faithfulness"].items():
            if metric_name in gap_keys:
                print(f"  {metric_name}: {value}")
            elif None not in value.values():
                print(f"  {metric_name}: mean = {value['mean']}, std = {value['std']}")
            else:
                print(f"  {metric_name}: mean = No, std = No")
