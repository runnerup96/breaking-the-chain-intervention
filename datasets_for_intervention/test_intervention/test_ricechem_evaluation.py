import unittest
from copy import deepcopy
from math import isclose

from datasets_for_intervention.ricechem_evaluation import RiceChemEvaluation
from datasets_for_intervention.ricechem_structure_processor import RiceChemStructureProcessor
from datasets_for_intervention.test_intervention.ricechem_mocks import RiceChemDatasetMock


class TestRiceChemEvaluation(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.processor = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        self.ev = RiceChemEvaluation(self.dataset, self.processor, tool_mode='none')
        self.ev.print_evaluation_metrics = lambda *_args, **_kwargs: None

    def test_compare_scores_none_and_close(self):
        self.assertEqual(self.ev.compare_scores(1.0, 1.0), 1)
        self.assertEqual(self.ev.compare_scores(1.0, 1.0 + 5e-4, atol=1e-3), 1)
        self.assertEqual(self.ev.compare_scores(1.0, 1.0 + 2e-3, atol=1e-3), 0)
        self.assertEqual(self.ev.compare_scores(None, 1.0), 0)
        self.assertEqual(self.ev.compare_scores(1.0, None), 0)

    def test_summarize_counts_and_none(self):
        out = self.ev.summarize([1, 0, None, 1])
        self.assertEqual(out["n_total"], 4)
        self.assertEqual(out["n_none"], 1)
        self.assertEqual(out["n_valid"], 3)
        self.assertTrue(isclose(out["mean"], 2/3, abs_tol=1e-3))

        out2 = self.ev.summarize([None, None])
        self.assertIsNone(out2["mean"])
        self.assertIsNone(out2["std"])
        self.assertEqual(out2["n_valid"], 0)
        self.assertEqual(out2["n_none"], 2)

    def test_evaluate_happy_path_non_tool(self):
        base = deepcopy(self.dataset[0])

        gold = deepcopy(base)
        gold["completion_type"] = "gold_structure"
        gold["score_before_intervention"] = gold["gold_score"]
        gold["tool_rubric"] = {}
        gold["structure_intervention"] = {
            "HSVT": [{"score_after_intervention": gold["gold_score"], "tool_rubric_after_intervention": {}}],
            "Local Edits": [
                {"expected_score_after_intervention": 0.0, "score_after_intervention": 0.0, "tool_rubric_after_intervention": {}},
                {"expected_score_after_intervention": 1.0, "score_after_intervention": 1.0, "tool_rubric_after_intervention": {}},
            ],
            "Correction": [{"score_after_intervention": gold["gold_score"], "tool_rubric_after_intervention": {}, "mediator_rubric": gold["gold_rubric"]}],
        }

        pred = deepcopy(base)
        pred["completion_type"] = "structure_prediction"
        pred["mediator_rubric"] = deepcopy(base["gold_rubric"])
        pred["score_before_intervention"] = base["gold_score"]
        pred["tool_rubric"] = {}
        pred["structure_intervention"] = {
            "HSVT": [{"score_after_intervention": base["gold_score"], "tool_rubric_after_intervention": {}}],
            "Local Edits": [
                {"expected_score_after_intervention": 0.0, "score_after_intervention": 0.0, "tool_rubric_after_intervention": {}},
                {"expected_score_after_intervention": 1.0, "score_after_intervention": 1.0, "tool_rubric_after_intervention": {}},
            ],
            "Correction": [],
        }

        agg = self.ev.evaluate([gold, pred])

        p_gold = agg["performance"]["with_gold_structure"]["score_match"]
        self.assertEqual(p_gold["mean"], 1)

        p_pred = agg["performance"]["with_predicted_structure"]
        self.assertEqual(p_pred["checklist_match"]["mean"], 1)
        self.assertEqual(p_pred["score_match"]["mean"], 1)

        f_gold = agg["faithfulness"]["with_gold_structure"]
        self.assertEqual(f_gold["HSVT"]["mean"], 1)
        self.assertEqual(f_gold["Local Edits"]["mean"], 1)
        # self.assertEqual(f_gold["Correction"]["mean"], 1)

        f_pred = agg["faithfulness"]["with_predicted_structure"]
        self.assertEqual(f_pred["HSVT"]["mean"], 1)
        self.assertEqual(f_pred["Local Edits"]["mean"], 1)

    def test_evaluate_propagates_none_and_counts(self):
        base = deepcopy(self.dataset[0])

        pred = deepcopy(base)
        pred["completion_type"] = "structure_prediction"
        pred["mediator_rubric"] = None
        pred["score_before_intervention"] = None
        pred["tool_rubric"] = None
        pred["structure_intervention"] = {"HSVT": [], "Local Edits": [], "Correction": []}

        agg = self.ev.evaluate([pred])

        perf = agg["performance"]["with_predicted_structure"]
        self.assertEqual(perf["checklist_match"]["n_none"], 1)
        self.assertEqual(perf["score_match"]["mean"], 0)

    def test_evaluate_ignores_unknown_completion_types(self):
        ev = RiceChemEvaluation(self.dataset, self.processor, tool_mode='none')
        ev.print_evaluation_metrics = lambda *_a, **_k: None

        s = deepcopy(self.dataset[0])
        s["completion_type"] = "something_else"
        agg = ev.evaluate([s])

        self.assertIsNone(agg["performance"]["with_gold_structure"]["score_match"]["mean"])
        self.assertIsNone(agg["performance"]["with_predicted_structure"]["score_match"]["mean"])

    def test_mediator_tool_match_counts_none_when_missing_fields(self):
        ev = RiceChemEvaluation(self.dataset, self.processor, tool_mode="simple")
        ev.print_evaluation_metrics = lambda *_a, **_k: None

        s = deepcopy(self.dataset[0])
        s["completion_type"] = "structure_prediction"
        s["mediator_rubric"] = None
        s["tool_rubric"] = {"A item": True}
        s["score_before_intervention"] = s["gold_score"]
        s["structure_intervention"] = {"HSVT": [], "Local Edits": [], "Correction": []}

        agg = ev.evaluate([s])
        mt = agg["mediator_tool_match"]["with_predicted_structure"]["predicted"]
        self.assertEqual(mt["n_none"], 1)
        self.assertIsNone(mt["mean"])

    def test_mediator_tool_match_happy_path_tool_mode(self):
        proc = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        ev = RiceChemEvaluation(self.dataset, proc, tool_mode="simple")
        ev.print_evaluation_metrics = lambda *_a, **_k: None

        base = deepcopy(self.dataset[0])
        s = deepcopy(base)
        s["completion_type"] = "structure_prediction"
        s["mediator_rubric"] = deepcopy(base["gold_rubric"])
        s["tool_rubric"] = deepcopy(base["gold_rubric"])
        s["score_before_intervention"] = base["gold_score"]
        s["structure_intervention"] = {
            "HSVT": [{"score_after_intervention": base["gold_score"], "mediator_rubric": deepcopy(base["gold_rubric"]), "tool_rubric_after_intervention": deepcopy(base["gold_rubric"])}],
            "Local Edits": [],
            "Correction": [],
        }

        agg = ev.evaluate([s])
        self.assertEqual(agg["mediator_tool_match"]["with_predicted_structure"]["predicted"]["mean"], 1)
        self.assertEqual(agg["mediator_tool_match"]["with_predicted_structure"]["HSVT"]["mean"], 1)

    @unittest.expectedFailure
    def test_correction_faithfulness_must_use_score_after_intervention(self):
        ev = RiceChemEvaluation(self.dataset, self.processor, tool_mode='none')
        ev.print_evaluation_metrics = lambda *_a, **_k: None

        base = deepcopy(self.dataset[0])
        s = deepcopy(base)
        s["completion_type"] = "gold_structure"
        s["score_before_intervention"] = -999.0
        s["tool_rubric"] = {}
        s["structure_intervention"] = {
            "HSVT": [],
            "Local Edits": [],
            "Correction": [{
                "score_before_intervention": -123.0,
                "score_after_intervention": base["gold_score"],
                "tool_rubric_after_intervention": {},
                "mediator_rubric": deepcopy(base["gold_rubric"]),
            }],
        }

        agg = ev.evaluate([s])
        self.assertEqual(agg["faithfulness"]["with_gold_structure"]["Correction"]["mean"], 1)