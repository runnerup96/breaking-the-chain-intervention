from tkinter import E
import unittest
from copy import deepcopy

from datasets_for_intervention.entailment_dataset import EntailmentDataset
from datasets_for_intervention.entailment_evaluation import EntailmentEvaluation
from datasets_for_intervention.entailment_intervention import serialize_step_proof, parse_step_proof
from datasets_for_intervention.test_intervention.entailment_mocks import EntailmentBankDatasetMock

class TestEntailmentEvaluation(unittest.TestCase):
    def setUp(self):
        self.dataset = EntailmentBankDatasetMock()
        # intervention_logic is only used for modes in local_edit_influence; pass a minimal stub
        class Stub:
            modes = ["delete", "replace", "rewire"]
        self.ev = EntailmentEvaluation(self.dataset, intervention_logic=Stub())

    # ---- proof normalization + compare_proofs ----
    def test_compare_proofs_exact_and_mismatch(self):
        sample = self.dataset[0]
        proof = sample["proof"]
        # Normalizing the same proof should match
        self.assertEqual(self.ev.compare_proofs(proof, proof), 1)
        # A modified proof that changes RHS should not match
        rules = parse_step_proof(proof)
        if rules:
            # change the final rhs id to a bogus token to force mismatch
            rules[-1].rhs_id = "bogus_rhs"
            modified = serialize_step_proof(rules)
            self.assertEqual(self.ev.compare_proofs(proof, modified), 0)

    # ---- binary coercion and target comparison ----
    def test_compare_binary_targets_cases(self):
        self.assertEqual(self.ev.compare_binary_targets(True, 1), 1)
        self.assertEqual(self.ev.compare_binary_targets(True, "yes"), 1)
        self.assertEqual(self.ev.compare_binary_targets(False, 0), 1)
        self.assertEqual(self.ev.compare_binary_targets(False, "no"), 1)
        self.assertEqual(self.ev.compare_binary_targets(True, -1), 0)  # invalid
        self.assertEqual(self.ev.compare_binary_targets(True, None), 0)
        self.assertEqual(self.ev.compare_binary_targets(True, "maybe"), 0)
        self.assertEqual(self.ev.compare_binary_targets(True, "Yes and no"), 0)

    # ---- summarize_nested_lists ----
    def test_summarize_nested_lists_happy_path_and_errors(self):
        tree = {"a": [1, 1, 1], "b": [], "c": {"d": [0, 2]}}
        out = self.ev.summarize_nested_lists(tree)
        self.assertEqual(out["a"]["mean"], 1)
        self.assertEqual(out["a"]["std"], 0)
        self.assertIsNone(out["b"]["mean"])
        self.assertIsNone(out["b"]["std"])
        self.assertEqual(out["c"]["d"]["mean"], 1)
        self.assertGreater(out["c"]["d"]["std"], 0)

        with self.assertRaises(TypeError):
            self.ev.summarize_nested_lists({"bad": {"leaf": "not-a-list"}})

    # ---- evaluate integration (happy path) ----
    def test_evaluate_aggregates_metrics(self):
        gold = deepcopy(self.dataset[0])
        gold["completion_type"] = "gold_structure"
        gold["structure_intervention"] = {
            "HSVT": [{"score": True, "result_after_intervention": 1}],
            "Local Edits": [
                {"score": False, "result_after_intervention": 0},
                {"score": False, "result_after_intervention": 0},
                {"score": False, "result_after_intervention": 0},
            ],
            "Global": [{"score": False, "result_after_intervention": 0}],
        }

        pred = deepcopy(self.dataset[1])
        pred["completion_type"] = "structure_prediction"
        # Match gold proof and score
        pred["proof"] = self.dataset[1]["proof"]
        pred["score"] = True
        pred["structure_intervention"] = {
            "HSVT": [{"score": True, "result_after_intervention": 1}],
            "Local Edits": [
                {"score": False, "result_after_intervention": 0},
                {"score": False, "result_after_intervention": 0},
                {"score": False, "result_after_intervention": 0},
            ],
            "Global": [{"score": False, "result_after_intervention": 0}],
        }

        agg = self.ev.evaluate([gold, pred])

        perf = agg["performance"]
        self.assertEqual(perf["with_gold_structure"]["score_match"]["mean"], 1)
        # For predicted structure, proof_match should be 1 if normalized proofs match
        self.assertEqual(perf["with_predicted_structure"]["proof_match"]["mean"], 1)
        self.assertEqual(perf["with_predicted_structure"]["score_match"]["mean"], 1)

        faith = agg["faithfullness"]
        for side in ("with_gold_structure", "with_predicted_structure"):
            self.assertEqual(faith[side]["HSVT"]["mean"], 1)
            self.assertEqual(faith[side]["Global"]["mean"], 1)
            self.assertEqual(faith[side]["Local Edits"]["mean"], 1)

        lei = agg["local_edit_influence"]["with_predicted_structure"]
        for mode in ("delete", "replace", "rewire"):
            self.assertEqual(lei[mode]["mean"], 1)

    # ---- mismatch case ----
    def test_evaluate_handles_incorrect_sample(self):
        pred = deepcopy(self.dataset[0])
        pred["completion_type"] = "structure_prediction"
        # Wrong score and wrong proof (change rhs)
        rules = parse_step_proof(pred["proof"]) if pred["proof"] else []
        if rules:
            rules[-1].rhs_id = "wrong_rhs"
            pred["proof"] = serialize_step_proof(rules)
        pred["score"] = 0
        pred["structure_intervention"] = {
            "HSVT": [{"score": True, "result_after_intervention": 0}],  # mismatch
            "Local Edits": [
                {"score": False, "result_after_intervention": 0},
                {"score": False, "result_after_intervention": 1},  # one mismatch
                {"score": False, "result_after_intervention": 0},
            ],
            "Global": [{"score": False, "result_after_intervention": 1}],  # mismatch
        }

        gold = deepcopy(self.dataset[0])
        gold["completion_type"] = "gold_structure"
        gold["structure_intervention"] = {
            "HSVT": [{"score": True, "result_after_intervention": 1}],
            "Local Edits": [
                {"score": False, "result_after_intervention": 0},
                {"score": False, "result_after_intervention": 0},
                {"score": False, "result_after_intervention": 0},
            ],
            "Global": [{"score": False, "result_after_intervention": 0}],
        }

        agg = self.ev.evaluate([gold, pred])

        perf = agg["performance"]["with_predicted_structure"]
        self.assertEqual(perf["proof_match"]["mean"], 0)
        self.assertEqual(perf["score_match"]["mean"], 0)

        faith = agg["faithfullness"]["with_predicted_structure"]
        # With the new logic, faithfulness is only computed for correct predictions.
        # Since the predicted answer is incorrect here, these should be None.
        self.assertIsNone(faith["HSVT"]["mean"])
        self.assertIsNone(faith["Local Edits"]["mean"])
        self.assertIsNone(faith["Global"]["mean"])


    def test_evaluate_correct_prediction_with_intervention_mismatches(self):
        # Predicted answer is correct and proof matches, but interventions have mismatches
        pred = deepcopy(self.dataset[0])
        pred["completion_type"] = "structure_prediction"
        pred["proof"] = self.dataset[0]["proof"]  # match proof
        pred["score"] = True  # correct original prediction for dataset[0]
        pred["structure_intervention"] = {
            "HSVT": [{"score": True, "result_after_intervention": 0}],  # mismatch vs gold
            "Local Edits": [
                {"score": False, "result_after_intervention": 0},
                {"score": False, "result_after_intervention": 1},  # one mismatch
                {"score": False, "result_after_intervention": 0},
            ],
            "Global": [{"score": False, "result_after_intervention": 1}],  # mismatch vs gold
        }

        gold = deepcopy(self.dataset[0])
        gold["completion_type"] = "gold_structure"
        gold["structure_intervention"] = {
            "HSVT": [{"score": True, "result_after_intervention": 1}],
            "Local Edits": [
                {"score": False, "result_after_intervention": 0},
                {"score": False, "result_after_intervention": 0},
                {"score": False, "result_after_intervention": 0},
            ],
            "Global": [{"score": False, "result_after_intervention": 0}],
        }

        agg = self.ev.evaluate([gold, pred])

        perf = agg["performance"]["with_predicted_structure"]
        self.assertEqual(perf["proof_match"]["mean"], 1)
        self.assertEqual(perf["score_match"]["mean"], 1)

        faith = agg["faithfullness"]["with_predicted_structure"]
        # Since original prediction is correct, faithfulness is computed and reflects mismatches
        self.assertEqual(faith["HSVT"]["mean"], 0)
        self.assertGreater(faith["Local Edits"]["mean"], 0.6)
        self.assertLess(faith["Local Edits"]["mean"], 0.7)
        self.assertEqual(faith["Global"]["mean"], 0)

if __name__ == "__main__":
    unittest.main()


