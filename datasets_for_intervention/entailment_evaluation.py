from statistics import mean, pstdev

from datasets_for_intervention.entailment_intervention import (
    parse_step_proof,
    serialize_step_proof,
)


class EntailmentEvaluation:

    def __init__(self, dataset, intervention_logic):
        self.dataset = dataset
        self.intervention_logic = intervention_logic

        # Cache gold labels and normalized proofs by id
        self.id2gold_score = {sample["id"]: sample["score"] for sample in dataset}
        self.id2gold_proof_norm = {
            sample["id"]: self._normalize_proof(sample["proof"]) for sample in dataset
        }

        # Prefer modes from intervention logic if present; otherwise default order
        self.modes = getattr(intervention_logic, "modes", ["delete", "replace", "rewire"])

    # ---------- helpers ----------
    def _normalize_proof(self, proof_text):
        if proof_text is None:
            return None
        rules = parse_step_proof(proof_text)
        return serialize_step_proof(rules)

    def compare_proofs(self, gold_proof_text, predicted_proof_text):
        gold_norm = self._normalize_proof(gold_proof_text)
        pred_norm = self._normalize_proof(predicted_proof_text)
        if gold_norm is None or pred_norm is None:
            return 0
        return 1 if gold_norm == pred_norm else 0

    def _coerce_binary_state(self, value):
        """
        Map various inputs to one of: True, False, None, or the string 'invalid'.
        -1 => 'invalid' (counts as incorrect)
         0 => False
         1 => True
        Strings like 'yes'/'true'/'1' => True; 'no'/'false'/'0' => False
        Anything else => None
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == -1:
                return "invalid"
            if value == 0:
                return False
            if value == 1:
                return True
            return None
        if isinstance(value, str):
            v = value.strip().lower()
            contains_positive = any([ans in v for ans in ["1", "true", "yes"]])
            contains_negative = any([ans in v for ans in ["0", "false", "no"]])
            
            if contains_negative and contains_positive:
                return "invalid"
            elif contains_positive:
                return True
            elif contains_negative:
                return False
            return "invalid"
        return None

    def compare_binary_targets(self, gold_bool, predicted_value):
        state = self._coerce_binary_state(predicted_value)
        if state == "invalid":
            return 0
        if state is None:
            return 0
        return 1 if bool(gold_bool) == bool(state) else 0

    def summarize_nested_lists(self, tree):
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

    # ---------- main eval ----------
    def evaluate(self, processed_samples_list):
        evaluation_metrics = {
            "performance": {
                "with_gold_structure": {
                    "score_match": []
                },
                "with_predicted_structure": {
                    "proof_match": [],
                    "score_match": []
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
                "with_gold_structure": {mode: [] for mode in self.modes},
                "with_predicted_structure": {mode: [] for mode in self.modes}
            }
        }

        for sample in processed_samples_list:
            sample_id = sample["id"]
            completion_type = sample["completion_type"]

            gold_score = self.id2gold_score[sample_id]
            gold_proof_norm = self.id2gold_proof_norm[sample_id]

            predicted_score_raw = sample.get("score", None)

            # Performance metrics
            score_match = self.compare_binary_targets(gold_score, predicted_score_raw)
            if completion_type == "gold_structure":
                evaluation_metrics["performance"]["with_gold_structure"]["score_match"].append(score_match)
            elif completion_type == "structure_prediction":
                predicted_proof = sample.get("proof", None)
                proof_match = self.compare_proofs(gold_proof_norm, predicted_proof)
                evaluation_metrics["performance"]["with_predicted_structure"]["proof_match"].append(proof_match)
                evaluation_metrics["performance"]["with_predicted_structure"]["score_match"].append(score_match)

            # Faithfulness & local edit influence
            structure_intervention = sample["structure_intervention"]

            # HSVT
            hsvt = structure_intervention["HSVT"][0]
            expected_hsvt_score = hsvt["score"]
            hsvt_after = hsvt["result_after_intervention"]
            hsvt_match = self.compare_binary_targets(expected_hsvt_score, hsvt_after)
            if completion_type == "gold_structure":
                evaluation_metrics["faithfullness"]["with_gold_structure"]["HSVT"].append(hsvt_match)
            elif completion_type == "structure_prediction":
                evaluation_metrics["faithfullness"]["with_predicted_structure"]["HSVT"].append(hsvt_match)

            # Local edits
            local_edits = structure_intervention["Local Edits"]
            for idx, local_edit in enumerate(local_edits):
                expected_local = local_edit["score"]
                local_after = local_edit["result_after_intervention"]
                local_match = self.compare_binary_targets(expected_local, local_after)

                if completion_type == "gold_structure":
                    evaluation_metrics["faithfullness"]["with_gold_structure"]["Local Edits"].append(local_match)
                    if idx < len(self.modes):
                        evaluation_metrics["local_edit_influence"]["with_gold_structure"][self.modes[idx]].append(local_match)
                elif completion_type == "structure_prediction":
                    evaluation_metrics["faithfullness"]["with_predicted_structure"]["Local Edits"].append(local_match)
                    if idx < len(self.modes):
                        evaluation_metrics["local_edit_influence"]["with_predicted_structure"][self.modes[idx]].append(local_match)

            # Global
            glob = structure_intervention["Global"][0]
            expected_global = glob["score"]
            global_after = glob["result_after_intervention"]
            global_match = self.compare_binary_targets(expected_global, global_after)
            if completion_type == "gold_structure":
                evaluation_metrics["faithfullness"]["with_gold_structure"]["Global"].append(global_match)
            elif completion_type == "structure_prediction":
                evaluation_metrics["faithfullness"]["with_predicted_structure"]["Global"].append(global_match)

        aggregated = self.summarize_nested_lists(evaluation_metrics)
        self.print_evaluation_metrics(aggregated)
        return aggregated

    def print_evaluation_metrics(self, evaluation_metrics):
        print("\nEvaluation Results:")
        print("===================")

        print("\nPerformance Metrics:")
        print("-------------------")
        for structure_type, metrics in evaluation_metrics["performance"].items():
            print(f"\n{structure_type}:")
            for metric_name, stats in metrics.items():
                mean_val = stats["mean"] if isinstance(stats, dict) else stats
                if mean_val is None:
                    print(f"  {metric_name}: None")
                else:
                    print(f"  {metric_name}: {mean_val:.3f}")

        print("\nFaithfulness Metrics:")
        print("--------------------")
        for structure_type, metrics in evaluation_metrics["faithfullness"].items():
            print(f"\n{structure_type}:")
            for intervention_type, stats in metrics.items():
                mean_val = stats["mean"] if isinstance(stats, dict) else stats
                if mean_val is None:
                    print(f"  {intervention_type}: None")
                else:
                    print(f"  {intervention_type}: {mean_val:.3f}")

        print("\nLocal Edit Influence:")
        print("--------------------")
        for structure_type, mode_metrics in evaluation_metrics["local_edit_influence"].items():
            print(f"\n{structure_type}:")
            for mode_name, stats in mode_metrics.items():
                mean_val = stats["mean"] if isinstance(stats, dict) else stats
                if mean_val is None:
                    print(f"  {mode_name}: None")
                else:
                    print(f"  {mode_name}: {mean_val:.3f}")


