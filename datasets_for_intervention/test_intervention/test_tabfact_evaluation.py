"""
test_tabfact_evaluation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for TabFactEvaluation under the new architecture.

Architecture:
  - evaluate(samples) -> {performance, faithfulness, mediator_tool_match}
  - performance: {counts, query_match, execution_match, target_match}
  - faithfulness:
      Local Edits  (correct samples) : compare_targets(expected_target_after_intervention, target_after_intervention)
      Correction   (incorrect samples): compare_targets(gold_target, target_after_intervention)
  - error samples: counted but excluded from all metric lists
  - compare_targets(gold, pred) -> int(0|1); 0 when either is None
  - summarize(lst) -> {mean, std, n_total, n_valid, n_none}
"""

import unittest
from copy import deepcopy
from math import isclose

from datasets_for_intervention.test_intervention.tabfact_mocks import TabFactDatasetMock
from datasets_for_intervention.tabfact_evaluation import TabFactEvaluation


# -----------------------------------------------------------------------
# Minimal fakes so we don't need a real engine for most tests
# -----------------------------------------------------------------------

class _FakeProcessor:
    def compare_structures(self, a, b):
        if a is None or b is None:
            return None
        return 1 if a == b else 0

    def extract_columns_values(self, q):
        return set(), set()


class _FakeTool:
    def calculate_score(self, args, sample):
        return None  # execution_match will always be None (excluded from mean)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_ev(dataset=None, tool_mode="none"):
    ds = dataset or TabFactDatasetMock()
    return TabFactEvaluation(ds, _FakeProcessor(), _FakeTool(), tool_mode=tool_mode)


def _base_sample(dataset, idx=0, status="correct"):
    """Minimal processed sample dict for evaluation."""
    s = deepcopy(dataset[idx])
    s["generation_status"] = status
    s["mediator_query"] = s["gold_query"] if status == "correct" else "eq{1; 2}=True"
    s["target_before_intervention"] = True if status != "error" else None
    s["tool_query"] = None
    s["structure_intervention"] = {"Local Edits": [], "Correction": []}
    return s


def _with_le(sample, edits):
    """Attach Local Edit entries: list of (expected, actual) tuples."""
    sample["structure_intervention"]["Local Edits"] = [
        {
            "expected_target_after_intervention": exp,
            "target_after_intervention": act,
            "mediator_query": "le_query=True",
        }
        for exp, act in edits
    ]
    return sample


def _with_correction(sample, target_after):
    """Attach one Correction entry."""
    sample["structure_intervention"]["Correction"] = [
        {
            "target_after_intervention": target_after,
            "mediator_query": sample["gold_query"],
        }
    ]
    return sample


# =======================================================================
# compare_targets
# =======================================================================

class TestCompareTargets(unittest.TestCase):
    def setUp(self):
        self.ev = _make_ev()

    def test_both_true(self):
        self.assertEqual(self.ev.compare_targets(True, True), 1)

    def test_both_false(self):
        self.assertEqual(self.ev.compare_targets(False, False), 1)

    def test_mismatch_true_false(self):
        self.assertEqual(self.ev.compare_targets(True, False), 0)

    def test_mismatch_false_true(self):
        self.assertEqual(self.ev.compare_targets(False, True), 0)

    def test_gold_none_returns_0(self):
        self.assertEqual(self.ev.compare_targets(None, True), 0)

    def test_pred_none_returns_0(self):
        self.assertEqual(self.ev.compare_targets(True, None), 0)

    def test_both_none_returns_0(self):
        self.assertEqual(self.ev.compare_targets(None, None), 0)


# =======================================================================
# summarize
# =======================================================================

class TestSummarize(unittest.TestCase):
    def setUp(self):
        self.ev = _make_ev()

    def test_empty_list(self):
        s = self.ev.summarize([])
        self.assertIsNone(s["mean"])
        self.assertIsNone(s["std"])
        self.assertEqual(s["n_total"], 0)
        self.assertEqual(s["n_valid"], 0)
        self.assertEqual(s["n_none"], 0)

    def test_all_ones(self):
        s = self.ev.summarize([1, 1, 1])
        self.assertAlmostEqual(s["mean"], 1.0)
        self.assertAlmostEqual(s["std"], 0.0)
        self.assertEqual(s["n_total"], 3)
        self.assertEqual(s["n_valid"], 3)
        self.assertEqual(s["n_none"], 0)

    def test_all_zeros(self):
        s = self.ev.summarize([0, 0])
        self.assertAlmostEqual(s["mean"], 0.0)

    def test_mixed_values(self):
        s = self.ev.summarize([1, 0, 1])
        self.assertAlmostEqual(s["mean"], 2 / 3, places=5)
        self.assertEqual(s["n_valid"], 3)

    def test_with_none_values(self):
        s = self.ev.summarize([1, None, 0])
        self.assertAlmostEqual(s["mean"], 0.5, places=5)
        self.assertEqual(s["n_total"], 3)
        self.assertEqual(s["n_valid"], 2)
        self.assertEqual(s["n_none"], 1)

    def test_all_none(self):
        s = self.ev.summarize([None, None])
        self.assertIsNone(s["mean"])
        self.assertEqual(s["n_total"], 2)
        self.assertEqual(s["n_valid"], 0)
        self.assertEqual(s["n_none"], 2)

    def test_single_item(self):
        s = self.ev.summarize([1])
        self.assertAlmostEqual(s["mean"], 1.0)
        self.assertAlmostEqual(s["std"], 0.0)

    def test_std_calculation(self):
        # [0, 1] -> mean=0.5, variance=0.25, std=0.5
        s = self.ev.summarize([0, 1])
        self.assertAlmostEqual(s["mean"], 0.5, places=5)
        self.assertAlmostEqual(s["std"], 0.5, places=5)


# =======================================================================
# evaluate — counts
# =======================================================================

class TestEvaluateCounts(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ev = _make_ev(self.dataset)
        self.ev._print_metrics = lambda _: None

    def test_correct_counted(self):
        s = _base_sample(self.dataset, 0, "correct")
        m = self.ev.evaluate([s])
        self.assertEqual(m["performance"]["counts"]["n_correct"], 1)
        self.assertEqual(m["performance"]["counts"]["n_total"], 1)
        self.assertEqual(m["performance"]["counts"]["n_error"], 0)

    def test_incorrect_counted(self):
        s = _base_sample(self.dataset, 0, "incorrect")
        m = self.ev.evaluate([s])
        self.assertEqual(m["performance"]["counts"]["n_incorrect"], 1)

    def test_error_counted_and_excluded_from_metrics(self):
        s = _base_sample(self.dataset, 0, "error")
        m = self.ev.evaluate([s])
        self.assertEqual(m["performance"]["counts"]["n_error"], 1)
        # Error sample excluded: query_match list should be empty
        self.assertIsNone(m["performance"]["query_match"]["mean"])

    def test_multiple_statuses(self):
        samples = [
            _base_sample(self.dataset, 0, "correct"),
            _base_sample(self.dataset, 1, "incorrect"),
            _base_sample(self.dataset, 2, "error"),
        ]
        m = self.ev.evaluate(samples)
        c = m["performance"]["counts"]
        self.assertEqual(c["n_total"], 3)
        self.assertEqual(c["n_correct"], 1)
        self.assertEqual(c["n_incorrect"], 1)
        self.assertEqual(c["n_error"], 1)

    def test_rates_sum_to_one(self):
        samples = [
            _base_sample(self.dataset, 0, "correct"),
            _base_sample(self.dataset, 1, "incorrect"),
        ]
        m = self.ev.evaluate(samples)
        c = m["performance"]["counts"]
        total = c["correct_rate"] + c["incorrect_rate"] + c["error_rate"]
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_empty_list(self):
        m = self.ev.evaluate([])
        self.assertEqual(m["performance"]["counts"]["n_total"], 0)
        self.assertIsNone(m["performance"]["counts"]["correct_rate"])


# =======================================================================
# evaluate — query_match and target_match
# =======================================================================

class TestEvaluatePerformance(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ev = _make_ev(self.dataset)
        self.ev._print_metrics = lambda _: None

    def test_query_match_correct_sample_exact(self):
        s = _base_sample(self.dataset, 0, "correct")
        # mediator_query == gold_query -> compare_structures returns 1
        m = self.ev.evaluate([s])
        self.assertEqual(m["performance"]["query_match"]["mean"], 1.0)

    def test_query_match_incorrect_sample(self):
        s = _base_sample(self.dataset, 0, "incorrect")
        # mediator_query = "eq{1; 2}=True" != gold_query -> 0
        m = self.ev.evaluate([s])
        self.assertEqual(m["performance"]["query_match"]["mean"], 0.0)

    def test_target_match_when_pred_equals_gold_target(self):
        s = _base_sample(self.dataset, 0, "correct")
        s["target_before_intervention"] = True  # equals gold_target=True
        m = self.ev.evaluate([s])
        self.assertEqual(m["performance"]["target_match"]["mean"], 1.0)

    def test_target_match_when_pred_wrong(self):
        s = _base_sample(self.dataset, 0, "correct")
        s["target_before_intervention"] = False  # != gold_target=True
        m = self.ev.evaluate([s])
        self.assertEqual(m["performance"]["target_match"]["mean"], 0.0)

    def test_target_match_none_gives_0(self):
        s = _base_sample(self.dataset, 0, "correct")
        s["target_before_intervention"] = None
        m = self.ev.evaluate([s])
        self.assertEqual(m["performance"]["target_match"]["mean"], 0.0)

    def test_execution_match_none_when_tool_returns_none(self):
        # FakeTool.calculate_score always returns None -> execution_match=None -> n_valid=0
        s = _base_sample(self.dataset, 0, "correct")
        m = self.ev.evaluate([s])
        self.assertIsNone(m["performance"]["execution_match"]["mean"])
        self.assertEqual(m["performance"]["execution_match"]["n_none"], 1)

    def test_aggregation_across_correct_and_incorrect(self):
        s1 = _base_sample(self.dataset, 0, "correct")   # query_match=1
        s2 = _base_sample(self.dataset, 1, "incorrect") # query_match=0
        m = self.ev.evaluate([s1, s2])
        self.assertAlmostEqual(m["performance"]["query_match"]["mean"], 0.5, places=5)


# =======================================================================
# evaluate — faithfulness (Local Edits)
# =======================================================================

class TestEvaluateFaithfulnessLocalEdits(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ev = _make_ev(self.dataset)
        self.ev._print_metrics = lambda _: None

    def test_faithful_when_expected_matches_actual(self):
        # expected=False, actual=False -> compare_targets(False, False)=1 -> faithful
        s = _base_sample(self.dataset, 0, "correct")
        _with_le(s, [(False, False)])
        m = self.ev.evaluate([s])
        self.assertEqual(m["faithfulness"]["Local Edits"]["mean"], 1.0)

    def test_unfaithful_when_expected_mismatches_actual(self):
        # expected=False, actual=True -> 0
        s = _base_sample(self.dataset, 0, "correct")
        _with_le(s, [(False, True)])
        m = self.ev.evaluate([s])
        self.assertEqual(m["faithfulness"]["Local Edits"]["mean"], 0.0)

    def test_mixed_faithfulness(self):
        s = _base_sample(self.dataset, 0, "correct")
        # (False,False)=1, (False,True)=0, (False,False)=1 -> mean=2/3
        _with_le(s, [(False, False), (False, True), (False, False)])
        m = self.ev.evaluate([s])
        self.assertAlmostEqual(m["faithfulness"]["Local Edits"]["mean"], 2 / 3, places=5)

    def test_missing_actual_target_treated_as_unfaithful(self):
        # target_after_intervention missing -> None -> compare_targets(False, None)=0
        s = _base_sample(self.dataset, 0, "correct")
        s["structure_intervention"]["Local Edits"] = [
            {"expected_target_after_intervention": False}  # no target_after_intervention
        ]
        m = self.ev.evaluate([s])
        self.assertEqual(m["faithfulness"]["Local Edits"]["mean"], 0.0)

    def test_missing_expected_target_treated_as_unfaithful(self):
        # expected_target_after_intervention missing -> None
        s = _base_sample(self.dataset, 0, "correct")
        s["structure_intervention"]["Local Edits"] = [
            {"target_after_intervention": False}  # no expected_target_after_intervention
        ]
        m = self.ev.evaluate([s])
        self.assertEqual(m["faithfulness"]["Local Edits"]["mean"], 0.0)

    def test_local_edits_only_on_correct_samples(self):
        # incorrect sample's Local Edits (if any) should NOT contribute
        s = _base_sample(self.dataset, 0, "incorrect")
        _with_le(s, [(False, False)])  # normally empty for incorrect, but let's be explicit
        m = self.ev.evaluate([s])
        # LE faithfulness list should still be empty for incorrect samples
        self.assertIsNone(m["faithfulness"]["Local Edits"]["mean"])

    def test_no_local_edits_gives_none_mean(self):
        s = _base_sample(self.dataset, 0, "correct")
        # no local edits attached
        m = self.ev.evaluate([s])
        self.assertIsNone(m["faithfulness"]["Local Edits"]["mean"])

    def test_aggregation_across_samples(self):
        s1 = _base_sample(self.dataset, 0, "correct")
        _with_le(s1, [(False, False)])  # faithful=1
        s2 = _base_sample(self.dataset, 1, "correct")
        _with_le(s2, [(False, True)])   # faithful=0
        m = self.ev.evaluate([s1, s2])
        self.assertAlmostEqual(m["faithfulness"]["Local Edits"]["mean"], 0.5, places=5)


# =======================================================================
# evaluate — faithfulness (Correction)
# =======================================================================

class TestEvaluateFaithfulnessCorrection(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ev = _make_ev(self.dataset)
        self.ev._print_metrics = lambda _: None

    def test_faithful_when_restored_to_gold_target(self):
        # gold_target=True; after correction model outputs True -> faithful=1
        s = _base_sample(self.dataset, 0, "incorrect")
        _with_correction(s, True)
        m = self.ev.evaluate([s])
        self.assertEqual(m["faithfulness"]["Correction"]["mean"], 1.0)

    def test_unfaithful_when_still_wrong_after_correction(self):
        s = _base_sample(self.dataset, 0, "incorrect")
        _with_correction(s, False)  # gold_target=True, actual=False -> 0
        m = self.ev.evaluate([s])
        self.assertEqual(m["faithfulness"]["Correction"]["mean"], 0.0)

    def test_none_actual_treated_as_unfaithful(self):
        s = _base_sample(self.dataset, 0, "incorrect")
        _with_correction(s, None)
        m = self.ev.evaluate([s])
        self.assertEqual(m["faithfulness"]["Correction"]["mean"], 0.0)

    def test_correction_only_on_incorrect_samples(self):
        # Correct sample's Correction list (if any) should NOT contribute
        s = _base_sample(self.dataset, 0, "correct")
        _with_correction(s, True)
        m = self.ev.evaluate([s])
        self.assertIsNone(m["faithfulness"]["Correction"]["mean"])

    def test_no_correction_gives_none_mean(self):
        s = _base_sample(self.dataset, 0, "incorrect")
        m = self.ev.evaluate([s])
        self.assertIsNone(m["faithfulness"]["Correction"]["mean"])

    def test_aggregation_across_samples(self):
        s1 = _base_sample(self.dataset, 0, "incorrect")
        _with_correction(s1, True)   # faithful=1
        s2 = _base_sample(self.dataset, 1, "incorrect")
        _with_correction(s2, False)  # faithful=0
        m = self.ev.evaluate([s1, s2])
        self.assertAlmostEqual(m["faithfulness"]["Correction"]["mean"], 0.5, places=5)


# =======================================================================
# evaluate — output structure
# =======================================================================

class TestEvaluateOutputStructure(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ev = _make_ev(self.dataset)
        self.ev._print_metrics = lambda _: None

    def test_top_level_keys_present(self):
        m = self.ev.evaluate([_base_sample(self.dataset, 0, "correct")])
        self.assertIn("performance", m)
        self.assertIn("faithfulness", m)
        self.assertIn("mediator_tool_match", m)

    def test_performance_sub_keys(self):
        m = self.ev.evaluate([_base_sample(self.dataset, 0, "correct")])
        perf = m["performance"]
        self.assertIn("counts", perf)
        self.assertIn("query_match", perf)
        self.assertIn("execution_match", perf)
        self.assertIn("target_match", perf)

    def test_faithfulness_sub_keys(self):
        m = self.ev.evaluate([_base_sample(self.dataset, 0, "correct")])
        faith = m["faithfulness"]
        self.assertIn("Local Edits", faith)
        self.assertIn("Correction", faith)

    def test_summarize_output_shape(self):
        m = self.ev.evaluate([_base_sample(self.dataset, 0, "correct")])
        qm = m["performance"]["query_match"]
        for key in ("mean", "std", "n_total", "n_valid", "n_none"):
            self.assertIn(key, qm)

    def test_mediator_tool_match_empty_when_no_tool(self):
        m = self.ev.evaluate([_base_sample(self.dataset, 0, "correct")])
        self.assertEqual(m["mediator_tool_match"], {})

    def test_mediator_tool_match_present_in_tool_mode(self):
        ev = _make_ev(self.dataset, tool_mode="simple")
        ev._print_metrics = lambda _: None
        s = _base_sample(self.dataset, 0, "correct")
        m = ev.evaluate([s])
        self.assertIn("predicted", m["mediator_tool_match"])
        self.assertIn("Local Edits", m["mediator_tool_match"])
        self.assertIn("Correction", m["mediator_tool_match"])


# =======================================================================
# evaluate — error samples excluded
# =======================================================================

class TestErrorSamplesExcluded(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ev = _make_ev(self.dataset)
        self.ev._print_metrics = lambda _: None

    def test_error_sample_not_in_query_match(self):
        # Only error sample -> query_match list is empty
        s = _base_sample(self.dataset, 0, "error")
        m = self.ev.evaluate([s])
        self.assertIsNone(m["performance"]["query_match"]["mean"])
        self.assertEqual(m["performance"]["query_match"]["n_total"], 0)

    def test_error_sample_not_in_faithfulness(self):
        s = _base_sample(self.dataset, 0, "error")
        m = self.ev.evaluate([s])
        self.assertIsNone(m["faithfulness"]["Local Edits"]["mean"])
        self.assertIsNone(m["faithfulness"]["Correction"]["mean"])

    def test_mixed_error_and_correct(self):
        good = _base_sample(self.dataset, 0, "correct")
        _with_le(good, [(False, False)])
        bad = _base_sample(self.dataset, 1, "error")
        m = self.ev.evaluate([good, bad])
        # Error excluded: only one good sample contributes
        self.assertEqual(m["performance"]["query_match"]["n_valid"], 1)
        self.assertEqual(m["faithfulness"]["Local Edits"]["n_valid"], 1)


# =======================================================================
# evaluate — idx2gold index lookup
# =======================================================================

class TestGoldIndexLookup(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ev = _make_ev(self.dataset)
        self.ev._print_metrics = lambda _: None

    def test_gold_target_looked_up_from_dataset(self):
        # All mock samples have gold_target=True
        # If we claim target=True, target_match=1
        s = _base_sample(self.dataset, 0, "correct")
        s["target_before_intervention"] = True
        m = self.ev.evaluate([s])
        self.assertEqual(m["performance"]["target_match"]["mean"], 1.0)

    def test_unknown_idx_uses_default(self):
        # Sample with unknown idx: gold_query defaults to "" -> query_match depends on mediator
        s = _base_sample(self.dataset, 0, "correct")
        s["idx"] = "unknown_idx_xyz"
        s["mediator_query"] = ""
        # both empty -> compare_structures("", "") = 1
        m = self.ev.evaluate([s])
        # should not crash; result depends on implementation
        self.assertIn("mean", m["performance"]["query_match"])


if __name__ == "__main__":
    unittest.main(verbosity=2)