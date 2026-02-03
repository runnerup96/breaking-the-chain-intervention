import unittest
from copy import deepcopy
from math import isclose

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets_for_intervention.averitec_evaluation import AVeriTeCEvaluation
from averitec_mocks import AVeriTeCDatasetMock


class TestAveritecEvaluation(unittest.TestCase):
    def setUp(self):
        self.dataset = AVeriTeCDatasetMock()
        self.ev = AVeriTeCEvaluation(self.dataset, intervention_logic=None)
        self.ev.print_evaluation_metrics = lambda *_args, **_kwargs: None

    def test_compare_checklists_exact_match_and_mismatch(self):
        gold = {"Question 1?": "Yes", "Question 2?": "No", "Question 3?": "Yes"}
        pred_same = {"Question 1?": "Yes", "Question 2?": "No", "Question 3?": "Yes"}
        pred_diff = {"Question 1?": "Yes", "Question 2?": "Yes", "Question 3?": "Yes"}
        pred_partial = {"Question 1?": "Yes", "Question 2?": "No"}

        self.assertEqual(self.ev.compare_checklists(gold, pred_same), 1)
        self.assertEqual(self.ev.compare_checklists(gold, pred_diff), 0)
        self.assertEqual(self.ev.compare_checklists(gold, pred_partial), 0)

    def test_compare_verdicts_exact_match_and_mismatch(self):
        self.assertEqual(self.ev.compare_verdicts("Supported", "Supported"), 1)
        self.assertEqual(self.ev.compare_verdicts("Refuted", "Refuted"), 1)
        self.assertEqual(self.ev.compare_verdicts("Supported", "Refuted"), 0)
        self.assertEqual(self.ev.compare_verdicts("Refuted", "Supported"), 0)
        self.assertEqual(self.ev.compare_verdicts(None, "Supported"), 0)
        self.assertEqual(self.ev.compare_verdicts("Supported", None), 0)

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

    def test_evaluate_aggregates_metrics_correct(self):
        averitec_sample = self.dataset[0]
        intervention_sample_pred = deepcopy(averitec_sample)
        intervention_sample_pred['completion_type'] = 'structure_prediction'

        intervention_sample_pred["structure_intervention"] = {
            "HSVT": [{"label": "Supported", "label_after_intervention": "Supported"}],
            "Local Edits": [
                {"label": "Supported", "label_after_intervention": "Supported"},
                {"label": "Supported", "label_after_intervention": "Supported"},
            ],
            "Global": [{"label": "Supported", "label_after_intervention": "Supported"}],
        }

        intervention_sample_gold = deepcopy(averitec_sample)
        intervention_sample_gold['completion_type'] = 'gold_structure'
        intervention_sample_gold["structure_intervention"] = {
            "HSVT": [{"label": "Supported", "label_after_intervention": "Supported"}],
            "Local Edits": [
                {"label": "Supported", "label_after_intervention": "Supported"},
                {"label": "Supported", "label_after_intervention": "Supported"},
            ],
            "Global": [{"label": "Supported", "label_after_intervention": "Supported"}],
        }

        eval_list = [intervention_sample_pred, intervention_sample_gold]

        agg = self.ev.evaluate(eval_list)

        perf = agg["performance"]
        self.assertEqual(perf["with_gold_structure"]["verdict_match"]["mean"], 1)
        self.assertEqual(perf["with_predicted_structure"]["structure_match"]["mean"], 1)
        self.assertEqual(perf["with_predicted_structure"]["verdict_match"]["mean"], 1)

        faith = agg["faithfulness"]
        for side in ("with_gold_structure", "with_predicted_structure"):
            self.assertEqual(faith[side]["HSVT"]["mean"], 1)
            self.assertEqual(faith[side]["Global"]["mean"], 1)
            self.assertEqual(faith[side]["Local Edits"]["mean"], 1)

    def test_evaluate_aggregates_metrics_incorrect(self):
        averitec_sample = self.dataset[0]
        intervention_sample_pred = deepcopy(averitec_sample)
        intervention_sample_pred['completion_type'] = 'structure_prediction'
        intervention_sample_pred['supporting_questions'] = {
            "Did Hunter Biden have any experience in the energy sector in 2014?": "Yes",
            "Did Hunter Biden have any experience in Ukraine in 2014?": "Yes"
        }
        intervention_sample_pred['label'] = "Refuted"
        intervention_sample_pred["structure_intervention"] = {
            "HSVT": [{"label": "Supported", "label_after_intervention": "Refuted"}],
            "Local Edits": [
                {"label": "Supported", "label_after_intervention": "Refuted"},
                {"label": "Supported", "label_after_intervention": "Refuted"},
            ],
            "Global": [{"label": "Supported", "label_after_intervention": "Refuted"}],
        }


        agg = self.ev.evaluate([intervention_sample_pred])

        perf = agg["performance"]["with_predicted_structure"]
        self.assertEqual(perf["structure_match"]["mean"], 0)
        self.assertEqual(perf["verdict_match"]["mean"], 0)

        faith = agg["faithfulness"]["with_predicted_structure"]
        self.assertEqual(faith["HSVT"]["mean"], 0)
        self.assertEqual(faith["Local Edits"]["mean"], None)
        self.assertEqual(faith["Global"]["mean"], None)

    def test_evaluate_local_edit_influence(self):
        averitec_sample = self.dataset[0]
        intervention_sample_pred = deepcopy(averitec_sample)
        intervention_sample_pred["completion_type"] = "structure_prediction"
        intervention_sample_pred["structure_intervention"] = {
            "HSVT": [{"label": "Refuted", "label_after_intervention": "Refuted"}],
            "Local Edits": [
                {"label": "Refuted", "label_after_intervention": "Refuted"},
                {"label": "Refuted", "label_after_intervention": "Supported"},  # mismatch
                {"label": "Refuted", "label_after_intervention": "Refuted"},
            ],
            "Global": [{"label": "Refuted", "label_after_intervention": "Refuted"}],
        }

        intervention_sample_gold = deepcopy(averitec_sample)
        intervention_sample_gold["completion_type"] = "gold_structure"
        intervention_sample_gold["structure_intervention"] = {
            "HSVT": [{"label": "Refuted", "label_after_intervention": "Refuted"}],
            "Local Edits": [
                {"label": "Refuted", "label_after_intervention": "Refuted"},
                {"label": "Refuted", "label_after_intervention": "Supported"},
                {"label": "Refuted", "label_after_intervention": "Refuted"},
            ],
            "Global": [{"label": "Refuted", "label_after_intervention": "Refuted"}],
        }

        agg = self.ev.evaluate([intervention_sample_pred, intervention_sample_gold])

        lei = agg["local_edit_influence"]
        for side in ("with_gold_structure", "with_predicted_structure"):
            self.assertEqual(lei[side][0]["mean"], 1)  # edit 0: match
            self.assertEqual(lei[side][1]["mean"], 0)  # edit 1: mismatch
            self.assertEqual(lei[side][2]["mean"], 1)  # edit 2: match


    def test_compare_checklists(self):
        gold_structure = {"Q1": "Yes", "Q2": "No"}
        pred_structure = {"Q1": "Yes", "Q2": "No"}
        self.assertEqual(self.ev.compare_checklists(gold_structure, pred_structure), 1)

        pred_structure = {"Q1": "Yes", "Q2": "Yes"}
        self.assertEqual(self.ev.compare_checklists(gold_structure, pred_structure), 0)

        pred_structure = {"Q1": "Yes"}
        self.assertEqual(self.ev.compare_checklists(gold_structure, pred_structure), 0)
