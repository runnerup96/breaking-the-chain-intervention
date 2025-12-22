import unittest
from copy import deepcopy
from math import isclose

from datasets_for_intervention.tabfact_evaluation import TabFactEvaluation
from tabfact_mocks import TabFactDatasetMock

class TestTabFactEvaluation(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        # intervention_logic is not used in evaluate, can pass None
        self.ev = TabFactEvaluation(self.dataset, intervention_logic=None)
        # Disable printing during tests
        self.ev.print_evaluation_metrics = lambda *_args, **_kwargs: None

    def test_compare_labels(self):
        """Checks compare_labels."""
        self.assertEqual(self.ev.compare_labels(True, True), 1)
        self.assertEqual(self.ev.compare_labels(False, False), 1)
        self.assertEqual(self.ev.compare_labels(True, False), 0)
        self.assertEqual(self.ev.compare_labels(None, True), 0)
        self.assertEqual(self.ev.compare_labels(True, None), 0)

    def test_summarize_nested_lists(self):
        """Checks summarize_nested_lists."""
        tree = {
            "a": [1, 1, 1],
            "b": [],
            "c": {"d": [0, 1]}
        }
        out = self.ev.summarize_nested_lists(tree)
        self.assertEqual(out["a"]["mean"], 1)
        self.assertEqual(out["a"]["std"], 0)
        self.assertIsNone(out["b"]["mean"])
        self.assertIsNone(out["b"]["std"])
        self.assertTrue(isclose(out["c"]["d"]["mean"], 0.5))
        self.assertTrue(isclose(out["c"]["d"]["std"], 0.5))

    def test_evaluate_faithfulness_high_when_expected(self):
        """Checks that faithfulness is high when the model behaves as expected."""
        # Gold sample: all interventions lead to expected behavior
        gold = deepcopy(self.dataset[0])
        gold["completion_type"] = "gold_structure"
        gold["result"] = True  # Model's final answer
        gold["structure_intervention"] = {
            "HSVT": [{"result_after_intervention": True}],  # Unchanged -> faithful
            "Local Edits": [
                {"result_after_intervention": False},  # Changed -> faithful (1 - 0 = 1)
                {"result_after_intervention": False},
                {"result_after_intervention": False},
            ],
            "Global": [
                {"result_after_intervention": False},  # Changed -> faithful (1 - 0 = 1)
                {"result_after_intervention": False},  # Changed -> faithful (1 - 0 = 1)
            ],
        }

        agg = self.ev.evaluate([gold])

        # Faithfulness should be 1.0 for all intervention types
        faith = agg["faithfulness"]["with_gold_structure"]
        self.assertEqual(faith["HSVT"]["mean"], 1.0)
        self.assertEqual(faith["Local Edits"]["mean"], 1.0)
        self.assertEqual(faith["Global"]["mean"], 1.0)

    def test_evaluate_faithfulness_low_when_expected(self):
        """Checks that faithfulness is low when the model ignores interventions."""
        pred = deepcopy(self.dataset[0])
        pred["completion_type"] = "structure_prediction"
        pred["result"] = True  # Model's original answer
        pred["structure_intervention"] = {
            "HSVT": [{"result_after_intervention": False}],  # Changed -> unfaithful for HSVT (0)
            "Local Edits": [
                {"result_after_intervention": True},  # Unchanged -> unfaithful (1 - 1 = 0)
                {"result_after_intervention": True},
                {"result_after_intervention": True},
            ],
            "Global": [
                {"result_after_intervention": True},  # Unchanged -> unfaithful (1 - 1 = 0)
                {"result_after_intervention": True},  # Unchanged -> unfaithful (1 - 1 = 0)
            ],
        }

        agg = self.ev.evaluate([pred])

        faith = agg["faithfulness"]["with_predicted_structure"]
        self.assertEqual(faith["HSVT"]["mean"], 0.0)
        self.assertEqual(faith["Local Edits"]["mean"], 0.0)
        self.assertEqual(faith["Global"]["mean"], 0.0)

    def test_evaluate_performance(self):
        """Checks performance metrics."""
        # Gold structure: answer matches GT
        gold = deepcopy(self.dataset[0])
        gold["completion_type"] = "gold_structure"
        gold["result"] = True  # Matches label_gt
        gold["structure_intervention"] = {
            "HSVT": [{"result_after_intervention": True}],
            "Local Edits": [{"result_after_intervention": False}],
            "Global": [{"result_after_intervention": False}],
        }

        # Predicted structure: expression and answer match GT
        pred = deepcopy(self.dataset[0])
        pred["completion_type"] = "structure_prediction"
        pred["verifier_query_gt"] = self.dataset[0]["verifier_query_gt"]  # Exact match
        pred["result"] = True  # Matches label_gt
        pred["structure_intervention"] = {
            "HSVT": [{"result_after_intervention": True}],
            "Local Edits": [{"result_after_intervention": False}],
            "Global": [{"result_after_intervention": False}],
        }

        agg = self.ev.evaluate([gold, pred])

        perf = agg["performance"]
        self.assertEqual(perf["with_gold_structure"]["label_match"]["mean"], 1.0)
        self.assertEqual(perf["with_predicted_structure"]["expression_match"]["mean"], 1.0)
        self.assertEqual(perf["with_predicted_structure"]["label_match"]["mean"], 1.0)

    def test_evaluate_faithfulness_local_edits_inversion(self):
        """
        Integration test: checks that faithfulness for Local Edits is correctly inverted.
        If model's answer changed after local_edit -> faithfulness = 1.
        If unchanged -> faithfulness = 0.
        """
        pred = deepcopy(self.dataset[0])
        pred["completion_type"] = "structure_prediction"
        pred["result"] = True  # Model's original answer

        # Create intervention structure where Local Edits give different results
        pred["structure_intervention"] = {
            "HSVT": [{"result_after_intervention": True}],  # Unchanged
            "Local Edits": [
                {"result_after_intervention": False},  # Changed -> faithful (1)
                {"result_after_intervention": True},   # Unchanged -> unfaithful (0)
                {"result_after_intervention": False},  # Changed -> faithful (1)
            ],
            "Global": [
                {"result_after_intervention": False},  # Changed -> faithful (1)
                {"result_after_intervention": False},  # Changed -> faithful (1)
            ],
        }

        agg = self.ev.evaluate([pred])

        faith = agg["faithfulness"]["with_predicted_structure"]
        # Local Edits: [1, 0, 1] -> mean = (1+0+1)/3 = 0.666...
        self.assertAlmostEqual(faith["Local Edits"]["mean"], 2/3, places=5)

    def test_evaluate_treats_missing_result_as_unfaithful(self):
        """Checks that missing 'result_after_intervention' is treated as unfaithful (score 0)."""
        pred = deepcopy(self.dataset[0])
        pred["completion_type"] = "structure_prediction"
        pred["result"] = True
        pred["structure_intervention"] = {
            "HSVT": [{}],  # missing → should be 0
            "Local Edits": [{}],  # missing → should be 0 (not 1!)
            "Global": [{}],  # missing → should be 0
        }

        agg = self.ev.evaluate([pred])

        faith = agg["faithfulness"]["with_predicted_structure"]
        self.assertEqual(faith["HSVT"]["mean"], 0.0)
        self.assertEqual(faith["Local Edits"]["mean"], 0.0)  # not 1.0!
        self.assertEqual(faith["Global"]["mean"], 0.0)

    def test_evaluate_aggregation_multiple_samples(self):
        """Checks that metrics are correctly aggregated over multiple samples."""
        sample1 = deepcopy(self.dataset[0])
        sample1["completion_type"] = "gold_structure"
        sample1["result"] = True
        sample1["structure_intervention"] = {
            "HSVT": [{"result_after_intervention": True}],  # unchanged → faithful (1)
            "Local Edits": [{"result_after_intervention": False}],  # changed → faithful (1)
            "Global": [{"result_after_intervention": False}],  # changed → faithful (1)
        }

        sample2 = deepcopy(self.dataset[0])
        sample2["completion_type"] = "gold_structure"
        sample2["result"] = True  # ← same as sample1
        sample2["structure_intervention"] = {
            "HSVT": [{"result_after_intervention": False}],  # changed → unfaithful (0)
            "Local Edits": [{"result_after_intervention": True}],  # unchanged → unfaithful (0)
            "Global": [{"result_after_intervention": True}],  # unchanged → unfaithful (0)
        }

        agg = self.ev.evaluate([sample1, sample2])

        perf = agg["performance"]["with_gold_structure"]
        self.assertAlmostEqual(perf["label_match"]["mean"], 1.0, places=5)  # both True vs gold True

        faith = agg["faithfulness"]["with_gold_structure"]
        self.assertAlmostEqual(faith["HSVT"]["mean"], 0.5, places=5)        # [1, 0]
        self.assertAlmostEqual(faith["Local Edits"]["mean"], 0.5, places=5) # [1, 0]
        self.assertAlmostEqual(faith["Global"]["mean"], 0.5, places=5)      # [1, 0]

    def test_evaluate_local_edit_influence_structure(self):
        """Checks that local_edit_influence is populated correctly."""
        pred = deepcopy(self.dataset[0])
        pred["completion_type"] = "structure_prediction"
        pred["result"] = True
        pred["structure_intervention"] = {
            "HSVT": [{"result_after_intervention": True}],
            "Local Edits": [
                {"result_after_intervention": False},  # Changed → faithful (1)
                {"result_after_intervention": True},   # Unchanged → unfaithful (0)
            ],
            "Global": [{"result_after_intervention": False}],
        }

        agg = self.ev.evaluate([pred])

        # Check structure and values in local_edit_influence
        lei = agg["local_edit_influence"]["with_predicted_structure"]
        self.assertIn(0, lei)
        self.assertIn(1, lei)
        self.assertAlmostEqual(lei[0]["mean"], 1.0, places=5)  # index 0: changed → 1
        self.assertAlmostEqual(lei[1]["mean"], 0.0, places=5)  # index 1: unchanged → 0


if __name__ == '__main__':
    unittest.main()