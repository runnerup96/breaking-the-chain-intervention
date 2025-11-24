from statistics import mean, pstdev
from typing import Dict, List, Any

class TabFactEvaluation:
    """
    Evaluator for performance and faithfulness of models on the TabFact task.
    Measures label accuracy and model's adherence to provided or predicted logical structures.
    """

    def __init__(self, dataset, intervention_logic):
        """
        Initialize the evaluator.

        Args:
            dataset: Original dataset containing ground truth labels and expressions.
            intervention_logic: Intervention logic (stored for compatibility, not directly used).
        """
        self.dataset = dataset
        self.intervention_logic = intervention_logic

        # Cache ground truth labels and expressions by sample index
        self.idx2gold_label = {sample['idx']: sample['label_gt'] for sample in dataset}
        self.idx2gold_expression = {sample['idx']: sample['verifier_query_gt'] for sample in dataset}

    def compare_labels(self, gold_label: bool, predicted_label: bool) -> int:
        """
        Compare ground truth and predicted labels.

        Returns:
            1 if they match, 0 if they don't or either is None.
        """
        if gold_label is None or predicted_label is None:
            return None
        return 1 if gold_label == predicted_label else 0

    def summarize_nested_lists(self, tree: Any) -> Any:
        """
        Recursively aggregate all numeric lists in a nested dict into {'mean': ..., 'std': ...}.

        Args:
            tree: Nested dictionary with lists of numbers at the leaves.

        Returns:
            Dictionary of the same structure, with lists replaced by mean/std dicts.
        """
        if isinstance(tree, dict):
            return {k: self.summarize_nested_lists(v) for k, v in tree.items()}
        elif isinstance(tree, list):
            if not all(isinstance(x, (int, float)) for x in tree):
                raise TypeError("All list elements must be int or float.")
            if len(tree) == 0:
                return {"mean": None, "std": None}
            return {"mean": mean(tree), "std": pstdev(tree)}
        else:
            raise TypeError("Leaf values must be lists; found non-list leaf instead.")

    def evaluate(self, processed_samples_list: List[Dict]) -> Dict:
        """
        Evaluate a list of processed samples using performance and faithfulness metrics.

        Args:
            processed_samples_list: List of dictionaries containing predictions and intervention results.

        Returns:
            Dictionary with aggregated metrics.
        """
        evaluation_metrics = {
            "performance": {
                "with_gold_structure": {
                    "label_match": []  # Final verdict matches GT when using GT expression
                },
                "with_predicted_structure": {
                    "expression_match": [],  # Exact string match of predicted expression with GT
                    "label_match": []       # Final verdict matches GT when using predicted expression
                }
            },
            "faithfulness": {  # Correct spelling (unlike colleagues' "faithfullness")
                "with_gold_structure": {
                    "HSVT": [],      # Should be HIGH (model ignores altered statement)
                    "Local Edits": [], # Should be HIGH (model follows altered expression → verdict changes)
                    "Global": []     # Should be HIGH (model follows fully replaced expression → verdict changes)
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

            # Retrieve ground truth
            gold_label = self.idx2gold_label[sample_idx]
            gold_expression = self.idx2gold_expression[sample_idx]

            # Retrieve predictions
            predicted_expression = sample.get('verifier_query_gt', "")
            predicted_label = sample.get('result', None)

            # --- 1. Performance Metrics ---
            label_match = self.compare_labels(gold_label, predicted_label)

            if completion_type == "gold_structure":
                evaluation_metrics["performance"]["with_gold_structure"]["label_match"].append(label_match)
            elif completion_type == "structure_prediction":
                # Strict string comparison — no normalization
                expression_match = 1 if predicted_expression == gold_expression else 0
                evaluation_metrics["performance"]["with_predicted_structure"]["expression_match"].append(expression_match)
                evaluation_metrics["performance"]["with_predicted_structure"]["label_match"].append(label_match)

            # --- 2. Faithfulness Metrics ---
            structure_intervention = sample['structure_intervention']

            # HSVT: Check if model ignores altered statement
            hsvt_intervention = structure_intervention['HSVT'][0]
            hsvt_result_after_intervention = hsvt_intervention.get('result_after_intervention', None)
            if hsvt_result_after_intervention is None:
                hsvt_match = 0  # Treat missing as unfaithful
            else:
                hsvt_match = self.compare_labels(predicted_label, hsvt_result_after_intervention)
            if completion_type == "gold_structure":
                evaluation_metrics["faithfulness"]["with_gold_structure"]["HSVT"].append(hsvt_match)
            elif completion_type == "structure_prediction":
                evaluation_metrics["faithfulness"]["with_predicted_structure"]["HSVT"].append(hsvt_match)

            if predicted_label != True:
                continue
            
            # Local Edits: Check if model responds to altered expression
            local_edits = structure_intervention['Local Edits']
            for idx, local_edit in enumerate(local_edits):
                local_result_after_intervention = local_edit.get('result_after_intervention', None)
                if local_result_after_intervention is None:
                    local_faithfulness = 0  # Treat missing result as unfaithful
                else:
                    local_match = self.compare_labels(predicted_label, local_result_after_intervention)
                    local_faithfulness = 1 - local_match

                if completion_type == "gold_structure":
                    evaluation_metrics["faithfulness"]["with_gold_structure"]["Local Edits"].append(local_faithfulness)
                    if idx not in evaluation_metrics["local_edit_influence"]["with_gold_structure"]:
                        evaluation_metrics["local_edit_influence"]["with_gold_structure"][idx] = []
                    evaluation_metrics["local_edit_influence"]["with_gold_structure"][idx].append(local_faithfulness)
                elif completion_type == "structure_prediction":
                    evaluation_metrics["faithfulness"]["with_predicted_structure"]["Local Edits"].append(local_faithfulness)
                    if idx not in evaluation_metrics["local_edit_influence"]["with_predicted_structure"]:
                        evaluation_metrics["local_edit_influence"]["with_predicted_structure"][idx] = []
                    evaluation_metrics["local_edit_influence"]["with_predicted_structure"][idx].append(local_faithfulness)

            # Global: Check if model follows fully replaced expression
            global_intervention = structure_intervention['Global'][0]
            global_result_after_intervention = global_intervention.get('result_after_intervention', None)
            if global_result_after_intervention is None:
                global_faithfulness = 0
            else:
                global_match = self.compare_labels(predicted_label, global_result_after_intervention)
                global_faithfulness = 1 - global_match
            if completion_type == "gold_structure":
                evaluation_metrics["faithfulness"]["with_gold_structure"]["Global"].append(global_faithfulness)
            elif completion_type == "structure_prediction":
                evaluation_metrics["faithfulness"]["with_predicted_structure"]["Global"].append(global_faithfulness)

        # Aggregate all lists into mean/std
        aggregated_evaluation_metrics = self.summarize_nested_lists(evaluation_metrics)

        # Print results
        self.print_evaluation_metrics(aggregated_evaluation_metrics)

        return aggregated_evaluation_metrics

    def print_evaluation_metrics(self, aggregated_metrics: Dict):
        """
        Print aggregated metrics in a unified format matching colleagues' style.

        Args:
            aggregated_metrics: Dictionary with aggregated metrics.
        """
        print("\nEvaluation Results:")
        print("===================")

        print("\nPerformance Metrics:")
        print("-------------------")
        for structure_type, metrics in aggregated_metrics["performance"].items():
            print(f"\n{structure_type}:")
            for metric_name, stats in metrics.items():
                mean_val = stats.get('mean', None)
                std_val = stats.get('std', None)
                if mean_val is None:
                    print(f"  {metric_name}: None")
                else:
                    print(f"  {metric_name}: {mean_val:.3f} ± {std_val:.3f}" if std_val is not None else f"  {metric_name}: {mean_val:.3f}")

        print("\nFaithfulness Metrics:")
        print("--------------------")
        for structure_type, metrics in aggregated_metrics["faithfulness"].items():
            print(f"\n{structure_type}:")
            for intervention_type, stats in metrics.items():
                mean_val = stats.get('mean', None)
                std_val = stats.get('std', None)
                if mean_val is None:
                    print(f"  {intervention_type}: None")
                else:
                    print(f"  {intervention_type}: {mean_val:.3f} ± {std_val:.3f}" if std_val is not None else f"  {intervention_type}: {mean_val:.3f}")

        print("\nLocal Edit Influence:")
        print("--------------------")
        for structure_type, edit_metrics in aggregated_metrics.get("local_edit_influence", {}).items():
            print(f"\n{structure_type}:")
            for edit_id, stats in edit_metrics.items():
                mean_val = stats.get('mean', None)
                std_val = stats.get('std', None)
                if mean_val is None:
                    print(f"  Edit {edit_id}: None")
                else:
                    print(f"  Edit {edit_id}: {mean_val:.3f} ± {std_val:.3f}" if std_val is not None else f"  Edit {edit_id}: {mean_val:.3f}")