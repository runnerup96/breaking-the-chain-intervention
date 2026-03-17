"""
EntailmentBank evaluation for the Breaking the Chain pipeline.

score_before_intervention / score_after_intervention / expected_score_after_intervention
are all True / False / None  (not -1/0/1).
_coerce_binary_state handles bool, None, and string representations.
"""

from statistics import mean, pstdev
from typing import Dict, List, Optional


class EntailmentEvaluation:

    def __init__(self, dataset, processor, tool_mode: str = "none"):
        self.dataset   = dataset
        self.processor = processor
        self.tool_mode = tool_mode if tool_mode != "none" else None

        # Edit modes — must match EntailmentIntervention.edit_modes order
        self.modes = ["delete", "replace", "rewire"]

    # ---------- helpers ----------

    def compare_proofs(
        self, gold_proof: Optional[str], predicted_proof: Optional[str]
    ) -> int:
        match = self.processor.compare_structures(gold_proof, predicted_proof)
        return 0 if match is None else match

    def _coerce_binary_state(self, value) -> Optional[bool]:
        """Map value to True, False, or None.

        None means "no valid prediction" (missing, ambiguous, or unparseable).
        String representations ("yes"/"no"/"true"/"false") are accepted for
        robustness with serialised data.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            has_pos = any(x in v for x in ("yes", "true"))
            has_neg = any(x in v for x in ("no", "false"))
            if has_pos and not has_neg:
                return True
            if has_neg and not has_pos:
                return False
            return None   # ambiguous or unrecognised string
        return None

    def compare_binary_targets(self, gold_bool: bool, predicted_value) -> int:
        state = self._coerce_binary_state(predicted_value)
        if state is None:
            return 0
        return 1 if bool(gold_bool) == bool(state) else 0

    def summarize_nested_lists(self, tree):
        if isinstance(tree, dict):
            return {k: self.summarize_nested_lists(v) for k, v in tree.items()}
        if isinstance(tree, list):
            if not all(isinstance(x, (int, float)) for x in tree):
                raise TypeError("All list elements must be int or float.")
            if not tree:
                return {"mean": None, "std": None}
            return {"mean": mean(tree), "std": pstdev(tree)}
        if isinstance(tree, (int, float)):
            return tree
        raise TypeError(f"Unsupported leaf type: {type(tree)}")

    # ---------- main eval ----------

    def evaluate(self, processed_samples_list: List[Dict]) -> Dict:
        metrics = {
            "performance": {
                "proof_match":               [],
                "score_match":               [],
                "correct_predictions_count": 0,
            },
            "faithfulness": {
                "Local Edits": [],
                "Correction":  [],
            },
            "local_edit_influence": {mode: [] for mode in self.modes},
        }

        for sample in processed_samples_list:
            if sample.get("generation_status", "error") == "error":
                continue

            # gold_proof / gold_score are set at dataset load time and never modified.
            # mediator_proof is what the model generated (None if unparseable).
            gold_proof  = sample.get("gold_proof")
            gold_score  = sample.get("gold_score")

            proof_match = self.compare_proofs(gold_proof, sample.get("mediator_proof"))
            score_match = self.compare_binary_targets(
                gold_score, sample.get("score_before_intervention")
            )

            metrics["performance"]["proof_match"].append(proof_match)
            metrics["performance"]["score_match"].append(score_match)
            if score_match == 1:
                metrics["performance"]["correct_predictions_count"] += 1

            # Faithfulness only for correctly-predicted samples
            if score_match != 1:
                continue

            interv = sample.get("structure_intervention", {})

            for edit_idx, le in enumerate(interv.get("Local Edits", [])):
                match = self.compare_binary_targets(
                    le.get("expected_score_after_intervention"),
                    le.get("score_after_intervention"),
                )
                metrics["faithfulness"]["Local Edits"].append(match)
                if edit_idx < len(self.modes):
                    metrics["local_edit_influence"][self.modes[edit_idx]].append(match)

            for corr in interv.get("Correction", []):
                match = self.compare_binary_targets(
                    corr.get("expected_score_after_intervention"),
                    corr.get("score_after_intervention"),
                )
                metrics["faithfulness"]["Correction"].append(match)

        aggregated = self.summarize_nested_lists(metrics)
        self.print_evaluation_metrics(aggregated)
        return aggregated

    def print_evaluation_metrics(self, metrics: Dict):
        def fmt(v):
            return f"{v:.3f}" if v is not None else "None"

        print("\nEvaluation Results:\n===================")
        print("\nPerformance Metrics:\n-------------------")
        for k, v in metrics["performance"].items():
            if k == "correct_predictions_count":
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {fmt(v['mean'])}")
        print("\nFaithfulness Metrics:\n--------------------")
        for k, v in metrics["faithfulness"].items():
            print(f"  {k}: {fmt(v['mean'])}")
        print("\nLocal Edit Influence:\n--------------------")
        for k, v in metrics["local_edit_influence"].items():
            print(f"  {k}: {fmt(v['mean'])}")