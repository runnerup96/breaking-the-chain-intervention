import unittest
from copy import deepcopy
from math import isclose

from datasets_for_intervention.ricechem_evaluation import RiceChemEvaluation
from datasets_for_intervention.ricechem_structure_processor import RiceChemStructureProcessor
from datasets_for_intervention.test_intervention.ricechem_mocks import RiceChemDatasetMock


def _base(dataset, i=0):
    return deepcopy(dataset[i])


def _correct_sample(base):
    s = deepcopy(base)
    s["generation_status"] = "correct"
    s["mediator_rubric"] = deepcopy(base["gold_rubric"])
    s["score_before_intervention"] = base["gold_score"]
    s["tool_rubric"] = {}
    # Two local edits: both faithful (expected == actual)
    s["structure_intervention"] = {
        "Local Edits": [
            {
                "mediator_rubric": deepcopy(base["gold_rubric"]),
                "expected_score_after_intervention": 0.0,
                "score_after_intervention": 0.0,
                "tool_rubric_after_intervention": {},
            },
            {
                "mediator_rubric": deepcopy(base["gold_rubric"]),
                "expected_score_after_intervention": base["gold_score"],
                "score_after_intervention": base["gold_score"],
                "tool_rubric_after_intervention": {},
            },
        ],
        "Correction": [],
    }
    return s


def _incorrect_sample(base):
    s = deepcopy(base)
    s["generation_status"] = "incorrect"
    s["mediator_rubric"] = {k: not v for k, v in base["gold_rubric"].items()}
    s["score_before_intervention"] = 0.0
    s["tool_rubric"] = {}
    s["structure_intervention"] = {
        "Local Edits": [],
        "Correction": [
            {
                "mediator_rubric": deepcopy(base["gold_rubric"]),
                "score_after_intervention": base["gold_score"],
                "tool_rubric_after_intervention": {},
            }
        ],
    }
    return s


def _error_sample(base):
    s = deepcopy(base)
    s["generation_status"] = "error"
    s["mediator_rubric"] = {}
    s["score_before_intervention"] = None
    s["tool_rubric"] = {}
    s["structure_intervention"] = {"Local Edits": [], "Correction": []}
    return s


# ===========================================================================
# 1. Helpers
# ===========================================================================

class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        proc = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        self.ev = RiceChemEvaluation(self.dataset, proc, tool_mode='none')
        self.ev.print_evaluation_metrics = lambda *_a, **_k: None

    def test_compare_scores_equal(self):
        self.assertEqual(self.ev.compare_scores(1.0, 1.0), 1)

    def test_compare_scores_within_atol(self):
        self.assertEqual(self.ev.compare_scores(1.0, 1.0 + 5e-4, atol=1e-3), 1)

    def test_compare_scores_outside_atol(self):
        self.assertEqual(self.ev.compare_scores(1.0, 1.0 + 2e-3, atol=1e-3), 0)

    def test_compare_scores_none_gold(self):
        self.assertEqual(self.ev.compare_scores(None, 1.0), 0)

    def test_compare_scores_none_pred(self):
        self.assertEqual(self.ev.compare_scores(1.0, None), 0)

    def test_summarize_with_nones(self):
        out = self.ev.summarize([1, 0, None, 1])
        self.assertEqual(out["n_total"], 4)
        self.assertEqual(out["n_none"], 1)
        self.assertEqual(out["n_valid"], 3)
        self.assertTrue(isclose(out["mean"], 2 / 3, abs_tol=1e-3))

    def test_summarize_all_none(self):
        out = self.ev.summarize([None, None])
        self.assertIsNone(out["mean"])
        self.assertIsNone(out["std"])
        self.assertEqual(out["n_valid"], 0)
        self.assertEqual(out["n_none"], 2)

    def test_summarize_empty(self):
        out = self.ev.summarize([])
        self.assertIsNone(out["mean"])
        self.assertEqual(out["n_total"], 0)


# ===========================================================================
# 2. evaluate — counts
# ===========================================================================

class TestEvaluateCounts(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        proc = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        self.ev = RiceChemEvaluation(self.dataset, proc, tool_mode='none')
        self.ev.print_evaluation_metrics = lambda *_a, **_k: None
        self.base = _base(self.dataset)

    def test_counts_correct_only(self):
        agg = self.ev.evaluate([_correct_sample(self.base)])
        counts = agg["performance"]["counts"]
        self.assertEqual(counts["n_correct"], 1)
        self.assertEqual(counts["n_incorrect"], 0)
        self.assertEqual(counts["n_error"], 0)
        self.assertEqual(counts["n_total"], 1)

    def test_counts_incorrect_only(self):
        agg = self.ev.evaluate([_incorrect_sample(self.base)])
        counts = agg["performance"]["counts"]
        self.assertEqual(counts["n_correct"], 0)
        self.assertEqual(counts["n_incorrect"], 1)
        self.assertEqual(counts["n_error"], 0)

    def test_counts_error_only(self):
        agg = self.ev.evaluate([_error_sample(self.base)])
        counts = agg["performance"]["counts"]
        self.assertEqual(counts["n_correct"], 0)
        self.assertEqual(counts["n_incorrect"], 0)
        self.assertEqual(counts["n_error"], 1)

    def test_counts_mixed(self):
        samples = [
            _correct_sample(self.base),
            _incorrect_sample(self.base),
            _error_sample(self.base),
        ]
        agg = self.ev.evaluate(samples)
        counts = agg["performance"]["counts"]
        self.assertEqual(counts["n_correct"], 1)
        self.assertEqual(counts["n_incorrect"], 1)
        self.assertEqual(counts["n_error"], 1)
        self.assertEqual(counts["n_total"], 3)

    def test_rates_add_up(self):
        samples = [_correct_sample(self.base), _incorrect_sample(self.base)]
        agg = self.ev.evaluate(samples)
        counts = agg["performance"]["counts"]
        self.assertAlmostEqual(counts["correct_rate"] + counts["incorrect_rate"], 1.0, places=3)

    def test_error_rate(self):
        samples = [_correct_sample(self.base), _error_sample(self.base)]
        agg = self.ev.evaluate(samples)
        counts = agg["performance"]["counts"]
        self.assertAlmostEqual(counts["error_rate"], 0.5, places=3)


# ===========================================================================
# 3. evaluate — performance metrics
# ===========================================================================

class TestEvaluatePerformance(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        proc = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        self.ev = RiceChemEvaluation(self.dataset, proc, tool_mode='none')
        self.ev.print_evaluation_metrics = lambda *_a, **_k: None
        self.base = _base(self.dataset)

    def test_checklist_match_correct_sample_is_1(self):
        agg = self.ev.evaluate([_correct_sample(self.base)])
        self.assertEqual(agg["performance"]["checklist_match"]["mean"], 1)

    def test_checklist_match_incorrect_sample_is_0(self):
        agg = self.ev.evaluate([_incorrect_sample(self.base)])
        self.assertEqual(agg["performance"]["checklist_match"]["mean"], 0)

    def test_score_match_correct_sample_is_1(self):
        agg = self.ev.evaluate([_correct_sample(self.base)])
        self.assertEqual(agg["performance"]["score_match"]["mean"], 1)

    def test_score_match_incorrect_sample_may_be_0(self):
        agg = self.ev.evaluate([_incorrect_sample(self.base)])
        # score_before_intervention = 0.0, gold_score = 1.5 → no match
        self.assertEqual(agg["performance"]["score_match"]["mean"], 0)

    def test_error_sample_excluded_from_performance_metrics(self):
        agg = self.ev.evaluate([_error_sample(self.base)])
        # No data for checklist/score
        self.assertIsNone(agg["performance"]["checklist_match"]["mean"])
        self.assertIsNone(agg["performance"]["score_match"]["mean"])

    def test_performance_metrics_average_over_multiple_samples(self):
        s1 = _correct_sample(self.base)    # checklist=1, score=1
        s2 = _incorrect_sample(self.base)  # checklist=0, score=0
        agg = self.ev.evaluate([s1, s2])
        self.assertAlmostEqual(agg["performance"]["checklist_match"]["mean"], 0.5, places=3)
        self.assertAlmostEqual(agg["performance"]["score_match"]["mean"], 0.5, places=3)


# ===========================================================================
# 4. evaluate — faithfulness
# ===========================================================================

class TestEvaluateFaithfulness(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        proc = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        self.ev = RiceChemEvaluation(self.dataset, proc, tool_mode='none')
        self.ev.print_evaluation_metrics = lambda *_a, **_k: None
        self.base = _base(self.dataset)

    def test_local_edits_faithfulness_perfect(self):
        agg = self.ev.evaluate([_correct_sample(self.base)])
        self.assertEqual(agg["faithfulness"]["Local Edits"]["mean"], 1)

    def test_local_edits_faithfulness_partial(self):
        s = _correct_sample(self.base)
        s["structure_intervention"]["Local Edits"][1]["score_after_intervention"] = 999.0
        agg = self.ev.evaluate([s])
        self.assertAlmostEqual(agg["faithfulness"]["Local Edits"]["mean"], 0.5, places=3)

    def test_correction_faithfulness_gold_score_vs_score_after(self):
        agg = self.ev.evaluate([_incorrect_sample(self.base)])
        self.assertEqual(agg["faithfulness"]["Correction"]["mean"], 1)

    def test_correction_unfaithful_when_score_mismatch(self):
        s = _incorrect_sample(self.base)
        s["structure_intervention"]["Correction"][0]["score_after_intervention"] = -1.0
        agg = self.ev.evaluate([s])
        self.assertEqual(agg["faithfulness"]["Correction"]["mean"], 0)

    def test_correct_sample_has_no_correction_data(self):
        agg = self.ev.evaluate([_correct_sample(self.base)])
        self.assertIsNone(agg["faithfulness"]["Correction"]["mean"])

    def test_incorrect_sample_has_no_local_edits_data(self):
        agg = self.ev.evaluate([_incorrect_sample(self.base)])
        self.assertIsNone(agg["faithfulness"]["Local Edits"]["mean"])

    def test_error_sample_contributes_no_faithfulness_data(self):
        agg = self.ev.evaluate([_error_sample(self.base)])
        self.assertIsNone(agg["faithfulness"]["Local Edits"]["mean"])
        self.assertIsNone(agg["faithfulness"]["Correction"]["mean"])

    def test_faithfulness_correct_counts(self):
        s1 = _correct_sample(self.base)  # 2 local edits
        s2 = _correct_sample(self.base)  # 2 local edits
        agg = self.ev.evaluate([s1, s2])
        self.assertEqual(agg["faithfulness"]["Local Edits"]["n_total"], 4)


# ===========================================================================
# 5. evaluate — mediator_tool_match (tool_mode='simple')
# ===========================================================================

class TestEvaluateMediatrToolMatch(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        proc = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        self.ev_tool = RiceChemEvaluation(self.dataset, proc, tool_mode='simple')
        self.ev_tool.print_evaluation_metrics = lambda *_a, **_k: None
        self.base = _base(self.dataset)

    def _correct_with_matching_tool_rubric(self):
        s = _correct_sample(self.base)
        s["tool_rubric"] = deepcopy(self.base["gold_rubric"])
        for edit in s["structure_intervention"]["Local Edits"]:
            edit["tool_rubric_after_intervention"] = deepcopy(edit["mediator_rubric"])
        return s

    def test_mediator_tool_match_predicted_perfect(self):
        agg = self.ev_tool.evaluate([self._correct_with_matching_tool_rubric()])
        self.assertEqual(agg["mediator_tool_match"]["predicted"]["mean"], 1)

    def test_mediator_tool_match_predicted_mismatch(self):
        s = _correct_sample(self.base)
        s["tool_rubric"] = {k: not v for k, v in self.base["gold_rubric"].items()}
        agg = self.ev_tool.evaluate([s])
        self.assertEqual(agg["mediator_tool_match"]["predicted"]["mean"], 0)

    def test_mediator_tool_match_none_when_tool_rubric_missing(self):
        s = _correct_sample(self.base)
        s["tool_rubric"] = None
        agg = self.ev_tool.evaluate([s])
        self.assertIsNone(agg["mediator_tool_match"]["predicted"]["mean"])
        self.assertEqual(agg["mediator_tool_match"]["predicted"]["n_none"], 1)

    def test_mediator_tool_match_local_edits_perfect(self):
        agg = self.ev_tool.evaluate([self._correct_with_matching_tool_rubric()])
        self.assertEqual(agg["mediator_tool_match"]["Local Edits"]["mean"], 1)

    def test_mediator_tool_match_correction_perfect(self):
        s = _incorrect_sample(self.base)
        corr = s["structure_intervention"]["Correction"][0]
        corr["tool_rubric_after_intervention"] = deepcopy(corr["mediator_rubric"])
        agg = self.ev_tool.evaluate([s])
        self.assertEqual(agg["mediator_tool_match"]["Correction"]["mean"], 1)

    def test_mediator_tool_match_empty_when_tool_mode_none(self):
        proc = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        ev = RiceChemEvaluation(self.dataset, proc, tool_mode='none')
        ev.print_evaluation_metrics = lambda *_a, **_k: None
        agg = ev.evaluate([_correct_sample(self.base)])
        self.assertEqual(agg["mediator_tool_match"], {})


# ===========================================================================
# 6. Edge cases
# ===========================================================================

class TestEvaluateEdgeCases(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        proc = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        self.ev = RiceChemEvaluation(self.dataset, proc, tool_mode='none')
        self.ev.print_evaluation_metrics = lambda *_a, **_k: None
        self.base = _base(self.dataset)

    def test_empty_list_returns_zero_counts(self):
        agg = self.ev.evaluate([])
        counts = agg["performance"]["counts"]
        self.assertEqual(counts["n_total"], 0)
        self.assertEqual(counts["n_correct"], 0)
        self.assertIsNone(agg["performance"]["checklist_match"]["mean"])
        self.assertIsNone(agg["faithfulness"]["Local Edits"]["mean"])

    def test_unknown_generation_status_skipped(self):
        s = deepcopy(self.base)
        s["generation_status"] = "something_else"
        s["structure_intervention"] = {"Local Edits": [], "Correction": []}
        agg = self.ev.evaluate([s])
        self.assertEqual(agg["performance"]["counts"]["n_total"], 0)

    def test_none_score_before_intervention_handled(self):
        s = _correct_sample(self.base)
        s["score_before_intervention"] = None
        agg = self.ev.evaluate([s])
        # compare_scores(gold, None) = 0
        self.assertEqual(agg["performance"]["score_match"]["mean"], 0)

    def test_idx_not_in_dataset_yields_none_gold(self):
        s = _correct_sample(self.base)
        s["idx"] = "nonexistent_idx@Task1"
        agg = self.ev.evaluate([s])
        # gold_score = None → compare_scores(None, ...) = 0
        self.assertEqual(agg["performance"]["score_match"]["mean"], 0)

    def test_multiple_correct_samples_local_edits_aggregated(self):
        s0 = _correct_sample(_base(self.dataset, 0))
        s1 = _correct_sample(_base(self.dataset, 1))
        agg = self.ev.evaluate([s0, s1])
        # Both samples faithful → mean = 1
        self.assertEqual(agg["faithfulness"]["Local Edits"]["mean"], 1)
        # n_total = 2 samples × 2 edits each = 4
        self.assertEqual(agg["faithfulness"]["Local Edits"]["n_total"], 4)