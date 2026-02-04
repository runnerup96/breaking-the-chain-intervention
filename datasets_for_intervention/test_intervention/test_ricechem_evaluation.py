import unittest
from copy import deepcopy
from math import isclose

from datasets_for_intervention.ricechem_evaluation import RiceChemEvaluation
from datasets_for_intervention.test_intervention.ricechem_mocks import RiceChemDatasetMock



class TestRiceChemEvaluation(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        # intervention_logic is not used inside evaluate for calculations, safe to pass None
        self.ev = RiceChemEvaluation(self.dataset, intervention_logic=None)
        # Avoid printing during tests (and avoid formatting issues on lists)
        self.ev.print_evaluation_metrics = lambda *_args, **_kwargs: None

    # ---- compare_checklists ----
    def test_compare_checklists_exact_match_and_mismatch(self):
        gold = {"A": True, "B": False, "C": True}
        pred_same = {"A": True, "B": False, "C": True}
        pred_diff = {"A": True, "B": True, "C": True}
        pred_partial = {"A": True, "B": False}  # missing C

        self.assertEqual(self.ev.compare_checklists(gold, pred_same), 1)
        self.assertEqual(self.ev.compare_checklists(gold, pred_diff), 0)
        self.assertEqual(self.ev.compare_checklists(gold, pred_partial), 0)

    # ---- compare_scores ----
    def test_compare_scores_exact_close_none(self):
        self.assertEqual(self.ev.compare_scores(4.0, 4.0), 1)
        self.assertEqual(self.ev.compare_scores(4.0, 4.0 + 1e-7, atol=1e-6), 1)  # within atol
        self.assertEqual(self.ev.compare_scores(4.0, 4.001, atol=1e-6), 0)       # outside atol
        self.assertEqual(self.ev.compare_scores(None, 4.0), 0)
        self.assertEqual(self.ev.compare_scores(4.0, None), 0)

    # ---- summarize_nested_lists ----
    def test_summarize_nested_lists_happy_path(self):
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

    def test_summarize_nested_lists_raises_on_non_list_leaf(self):
        with self.assertRaises(TypeError):
            self.ev.summarize_nested_lists({"bad": {"leaf": "not-a-list"}})

    # ---- evaluate (integration) ----
    def test_evaluate_aggregates_metrics(self):
        # Gold sample baseline from dataset
        gold = deepcopy(self.dataset[0])
        gold["completion_type"] = "gold_structure"
        # Interventions where expected score == result => all ones
        gold["structure_intervention"] = {
            "HSVT": [{"score": 10.0, "score_after_intervention": 10.0}],
            "Local Edits": [
                {"score": 1.0, "score_after_intervention": 1.0},
                {"score": 2.0, "score_after_intervention": 2.0},
                {"score": 3.0, "score_after_intervention": 3.0},
            ],
            "Global": [{"score": 5.0, "score_after_intervention": 5.0}],
        }

        # Predicted sample matches gold checklist & score (=> checklist_match=1, score_match=1)
        pred = deepcopy(self.dataset[1])
        pred["completion_type"] = "structure_prediction"
        pred["filled_rubric"] = deepcopy(self.dataset[1]["filled_rubric"])  # identical to gold in mock
        pred["score"] = self.dataset[1]["score"]  # 4.0
        pred["structure_intervention"] = {
            "HSVT": [{"score": 7.0, "score_after_intervention": 7.0}],
            "Local Edits": [
                {"score": 10.0, "score_after_intervention": 10.0},
                {"score": 0.0,  "score_after_intervention": 0.0},
                {"score": 3.0,  "score_after_intervention": 3.0},
            ],
            "Global": [{"score": 9.0, "score_after_intervention": 9.0}],
        }

        agg = self.ev.evaluate([gold, pred])

        # ----- Assertions on aggregated structure -----
        # Performance
        perf = agg["performance"]
        self.assertEqual(perf["with_gold_structure"]["score_match"]["mean"], 1)
        self.assertEqual(perf["with_predicted_structure"]["checklist_match"]["mean"], 1)
        self.assertEqual(perf["with_predicted_structure"]["score_match"]["mean"], 1)

        # Faithfulness
        faith = agg["faithfullness"]
        for side in ("with_gold_structure", "with_predicted_structure"):
            self.assertEqual(faith[side]["HSVT"]["mean"], 1)
            self.assertEqual(faith[side]["Global"]["mean"], 1)
            self.assertEqual(faith[side]["Local Edits"]["mean"], 1)

        # Local edit influence: means are 1 per edit index
        lei = agg["local_edit_influence"]
        for side in ("with_gold_structure", "with_predicted_structure"):
            for edit_idx in range(3):
                self.assertEqual(
                    lei[side][1][edit_idx]["mean"],  # task_idx = 1
                    1
                )

    def test_evaluate_handles_mismatch(self):
        # Create a predicted sample that mismatches score and one checklist item
        pred = deepcopy(self.dataset[0])
        pred["idx"] = "mock_1@Task1"
        pred["completion_type"] = "structure_prediction"
        # Flip one checklist item
        pred["filled_rubric"] = dict(pred["filled_rubric"])
        pred["filled_rubric"]["B"] = True  # gold has False
        # Change score
        pred["score"] = pred["score"] + 1.0
        # Interventions with a deliberate mismatch
        pred["structure_intervention"] = {
            "HSVT": [{"score": 10.0, "score_after_intervention": 9.0}],  # mismatch
            "Local Edits": [
                {"score": 1.0, "score_after_intervention": 1.0},
                {"score": 2.0, "score_after_intervention": 1.0},  # one mismatch
                {"score": 3.0, "score_after_intervention": 3.0},
            ],
            "Global": [{"score": 5.0, "score_after_intervention": 4.0}],   # mismatch
        }

        # Need a gold sample present in dataset indices mapping; use the original
        gold = deepcopy(self.dataset[0])
        gold["completion_type"] = "gold_structure"
        gold["structure_intervention"] = {
            "HSVT": [{"score": 10.0, "score_after_intervention": 10.0}],
            "Local Edits": [
                {"score": 1.0, "score_after_intervention": 1.0},
                {"score": 2.0, "score_after_intervention": 2.0},
                {"score": 3.0, "score_after_intervention": 3.0},
            ],
            "Global": [{"score": 5.0, "score_after_intervention": 5.0}],
        }

        agg = self.ev.evaluate([gold, pred])

        perf = agg["performance"]["with_predicted_structure"]
        # checklist mismatch -> 0, score mismatch -> 0
        self.assertEqual(perf["checklist_match"]["mean"], 0)
        self.assertEqual(perf["score_match"]["mean"], 0)

        faith = agg["faithfullness"]["with_predicted_structure"]
        self.assertEqual(faith["HSVT"]["mean"], 0)
        # Local edits: 2 matches, 1 mismatch -> mean = 2/3
        self.assertTrue(isclose(faith["Local Edits"]["mean"], 2/3, abs_tol=1e-3))
        self.assertEqual(faith["Global"]["mean"], 0)


