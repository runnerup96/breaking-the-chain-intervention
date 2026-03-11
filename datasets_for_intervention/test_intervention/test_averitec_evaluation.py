"""
Tests for AVeriTeCEvaluation.

Covers:
  - compare_targets
  - summarize
  - evaluate:
      counts (correct / incorrect / error)
      performance: checklist_match, verdict_match
      faithfulness: Local Edits (with filter), Correction
      mediator_tool_match (tool_mode only)
  - Local Edits filter: Supported → included, multi-Refuted → excluded, single-Refuted → included
"""
import sys
import unittest
from copy import deepcopy
from math import isclose

from datasets_for_intervention.test_intervention.averitec_mocks import AVeriTeCDatasetMock, GOLD_RUBRIC_2Q, GOLD_RUBRIC_1Q, GOLD_RUBRIC_3Q
from datasets_for_intervention.averitec_structure_processor import AVeriTeCStructureProcessor
from datasets_for_intervention.averitec_evaluation import AVeriTeCEvaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATASET = AVeriTeCDatasetMock()
PROC    = AVeriTeCStructureProcessor(DATASET, "none")


def make_eval(tool_mode="none"):
    return AVeriTeCEvaluation(DATASET, PROC, tool_mode)


def make_processed(
    idx,
    generation_status,
    mediator_rubric,
    target_before,
    local_edits,       # list of {"expected": str, "actual": str}
    correction,        # list of {"actual": str}
    tool_rubric=None,
    local_edit_tool_rubric=None,   # list of dict|None, parallel to local_edits
    correction_tool_rubric=None,   # list of dict|None, parallel to correction
):
    """Build a processed sample in the format expected by AVeriTeCEvaluation.evaluate()."""
    local_edit_items = []
    for i, e in enumerate(local_edits):
        item = {
            "mediator_rubric": mediator_rubric,
            "expected_target_after_intervention": e["expected"],
            "target_after_intervention": e["actual"],
        }
        if local_edit_tool_rubric and local_edit_tool_rubric[i] is not None:
            item["tool_rubric_after_intervention"] = local_edit_tool_rubric[i]
        local_edit_items.append(item)

    correction_items = []
    for i, c in enumerate(correction):
        item = {
            "mediator_rubric": mediator_rubric,
            "target_after_intervention": c["actual"],
        }
        if correction_tool_rubric and correction_tool_rubric[i] is not None:
            item["tool_rubric_after_intervention"] = correction_tool_rubric[i]
        correction_items.append(item)

    s = {
        "idx": idx,
        "generation_status": generation_status,
        "mediator_rubric": mediator_rubric,
        "tool_rubric": tool_rubric,
        "target_before_intervention": target_before,
        "structure_intervention": {
            "Local Edits": local_edit_items,
            "Correction": correction_items,
        },
    }
    return s


# ===========================================================================
# 1. compare_targets
# ===========================================================================

class TestCompareTargets(unittest.TestCase):

    def setUp(self):
        self.ev = make_eval()

    def test_matching_strings_returns_1(self):
        self.assertEqual(self.ev.compare_targets("Supported", "Supported"), 1)
        self.assertEqual(self.ev.compare_targets("Refuted", "Refuted"), 1)

    def test_mismatching_strings_returns_0(self):
        self.assertEqual(self.ev.compare_targets("Supported", "Refuted"), 0)
        self.assertEqual(self.ev.compare_targets("Refuted", "Supported"), 0)

    def test_none_gold_returns_0(self):
        self.assertEqual(self.ev.compare_targets(None, "Supported"), 0)

    def test_none_pred_returns_0(self):
        self.assertEqual(self.ev.compare_targets("Supported", None), 0)

    def test_both_none_returns_0(self):
        self.assertEqual(self.ev.compare_targets(None, None), 0)


# ===========================================================================
# 2. summarize
# ===========================================================================

class TestSummarize(unittest.TestCase):

    def setUp(self):
        self.ev = make_eval()

    def test_all_ones(self):
        r = self.ev.summarize([1, 1, 1])
        self.assertAlmostEqual(r["mean"], 1.0)
        self.assertAlmostEqual(r["std"], 0.0)
        self.assertEqual(r["n_total"], 3)
        self.assertEqual(r["n_valid"], 3)
        self.assertEqual(r["n_none"], 0)

    def test_mixed_with_none(self):
        r = self.ev.summarize([1, None])
        self.assertAlmostEqual(r["mean"], 1.0)
        self.assertEqual(r["n_valid"], 1)
        self.assertEqual(r["n_none"], 1)

    def test_with_zeros(self):
        r = self.ev.summarize([1, 1, 0])
        self.assertAlmostEqual(r["mean"], round(2/3, 3), places=3)
        self.assertEqual(r["n_valid"], 3)

    def test_empty_list(self):
        r = self.ev.summarize([])
        self.assertIsNone(r["mean"])
        self.assertIsNone(r["std"])
        self.assertEqual(r["n_total"], 0)

    def test_all_none(self):
        r = self.ev.summarize([None, None])
        self.assertIsNone(r["mean"])
        self.assertEqual(r["n_none"], 2)
        self.assertEqual(r["n_valid"], 0)


# ===========================================================================
# 3. evaluate -- counts
# ===========================================================================

class TestEvaluateCounts(unittest.TestCase):

    def setUp(self):
        self.ev = make_eval()

    def test_counts_correct_incorrect_error(self):
        samples = [
            make_processed("0", "correct",   GOLD_RUBRIC_2Q, "Supported",
                           [{"expected": "Refuted", "actual": "Refuted"},
                            {"expected": "Refuted", "actual": "Refuted"}], []),
            make_processed("1", "incorrect", GOLD_RUBRIC_1Q, "Supported",
                           [], [{"actual": "Refuted"}]),
            make_processed("0", "error",     {}, None, [], []),
        ]
        r = self.ev.evaluate(samples)
        c = r["performance"]["counts"]
        self.assertEqual(c["n_total"],     3)
        self.assertEqual(c["n_correct"],   1)
        self.assertEqual(c["n_incorrect"], 1)
        self.assertEqual(c["n_error"],     1)

    def test_error_samples_excluded_from_performance(self):
        samples = [
            make_processed("0", "error", {}, None, [], []),
        ]
        r = self.ev.evaluate(samples)
        self.assertEqual(r["performance"]["checklist_match"]["n_total"], 0)
        self.assertEqual(r["performance"]["verdict_match"]["n_total"],   0)


# ===========================================================================
# 4. evaluate -- performance metrics
# ===========================================================================

class TestEvaluatePerformance(unittest.TestCase):

    def setUp(self):
        self.ev = make_eval()

    def test_checklist_match_correct(self):
        # correct sample: mediator_rubric == gold_rubric -> match = 1
        s = make_processed("0", "correct", deepcopy(GOLD_RUBRIC_2Q), "Supported",
                           [{"expected": "Refuted", "actual": "Refuted"},
                            {"expected": "Refuted", "actual": "Refuted"}], [])
        r = self.ev.evaluate([s])
        self.assertEqual(r["performance"]["checklist_match"]["mean"], 1.0)

    def test_checklist_match_incorrect(self):
        bad = deepcopy(GOLD_RUBRIC_2Q)
        bad[list(bad.keys())[0]] = True
        s = make_processed("0", "incorrect", bad, "Refuted", [], [{"actual": "Supported"}])
        r = self.ev.evaluate([s])
        self.assertEqual(r["performance"]["checklist_match"]["mean"], 0.0)

    def test_verdict_match_correct(self):
        s = make_processed("0", "correct", deepcopy(GOLD_RUBRIC_2Q), "Supported",
                           [{"expected": "Refuted", "actual": "Refuted"},
                            {"expected": "Refuted", "actual": "Refuted"}], [])
        r = self.ev.evaluate([s])
        self.assertEqual(r["performance"]["verdict_match"]["mean"], 1.0)

    def test_verdict_match_incorrect(self):
        s = make_processed("0", "incorrect", deepcopy(GOLD_RUBRIC_2Q), "Refuted",
                           [], [{"actual": "Supported"}])
        r = self.ev.evaluate([s])
        self.assertEqual(r["performance"]["verdict_match"]["mean"], 0.0)


# ===========================================================================
# 5. evaluate -- faithfulness: Local Edits filter
# ===========================================================================

class TestLocalEditsFilter(unittest.TestCase):

    def setUp(self):
        self.ev = make_eval()

    def test_supported_sample_included_in_local_edits(self):
        # gold_target="Supported", target_before="Supported" -> eligible
        s = make_processed("0", "correct", deepcopy(GOLD_RUBRIC_2Q), "Supported",
                           [{"expected": "Refuted", "actual": "Refuted"},
                            {"expected": "Refuted", "actual": "Refuted"}], [])
        r = self.ev.evaluate([s])
        self.assertEqual(r["faithfulness"]["Local Edits"]["n_total"], 2)
        self.assertAlmostEqual(r["faithfulness"]["Local Edits"]["mean"], 1.0)

    def test_single_refuted_included_in_local_edits(self):
        # 1Q Refuted -> len==1 -> eligible
        s = make_processed("1", "correct", deepcopy(GOLD_RUBRIC_1Q), "Refuted",
                           [{"expected": "Supported", "actual": "Supported"}], [])
        r = self.ev.evaluate([s])
        self.assertEqual(r["faithfulness"]["Local Edits"]["n_total"], 1)
        self.assertAlmostEqual(r["faithfulness"]["Local Edits"]["mean"], 1.0)

    def test_multi_refuted_excluded_from_local_edits(self):
        # 3Q Refuted, target_before="Refuted", len>1 -> NOT eligible
        s = make_processed("2", "correct", deepcopy(GOLD_RUBRIC_3Q), "Refuted",
                           [{"expected": "Supported", "actual": "Refuted"},
                            {"expected": "Supported", "actual": "Supported"},
                            {"expected": "Supported", "actual": "Refuted"}], [])
        r = self.ev.evaluate([s])
        self.assertEqual(r["faithfulness"]["Local Edits"]["n_total"], 0)

    def test_unfaithful_local_edit_lowers_mean(self):
        # 2 faithful, 1 unfaithful
        s = make_processed("0", "correct", deepcopy(GOLD_RUBRIC_2Q), "Supported",
                           [{"expected": "Refuted", "actual": "Refuted"},
                            {"expected": "Refuted", "actual": "Supported"}], [])
        r = self.ev.evaluate([s])
        self.assertAlmostEqual(r["faithfulness"]["Local Edits"]["mean"], 0.5)


# ===========================================================================
# 6. evaluate -- faithfulness: Correction
# ===========================================================================

class TestCorrectionFaithfulness(unittest.TestCase):

    def setUp(self):
        self.ev = make_eval()

    def test_correct_correction_mean_1(self):
        # incorrect sample, gold_target="Supported", correction gives "Supported"
        s = make_processed("0", "incorrect", deepcopy(GOLD_RUBRIC_2Q), "Refuted",
                           [], [{"actual": "Supported"}])
        r = self.ev.evaluate([s])
        self.assertAlmostEqual(r["faithfulness"]["Correction"]["mean"], 1.0)

    def test_incorrect_correction_mean_0(self):
        s = make_processed("0", "incorrect", deepcopy(GOLD_RUBRIC_2Q), "Refuted",
                           [], [{"actual": "Refuted"}])
        r = self.ev.evaluate([s])
        self.assertAlmostEqual(r["faithfulness"]["Correction"]["mean"], 0.0)

    def test_none_target_after_intervention_counts_as_0(self):
        s = make_processed("0", "incorrect", deepcopy(GOLD_RUBRIC_2Q), "Refuted",
                           [], [{"actual": None}])
        r = self.ev.evaluate([s])
        self.assertAlmostEqual(r["faithfulness"]["Correction"]["mean"], 0.0)


# ===========================================================================
# 7. evaluate -- mediator_tool_match (tool_mode=structured)
# ===========================================================================

class TestMediatorToolMatch(unittest.TestCase):

    def setUp(self):
        self.ev = make_eval(tool_mode="structured")

    def test_tool_rubric_matches_mediator_predicted(self):
        s = make_processed(
            "0", "correct", deepcopy(GOLD_RUBRIC_2Q), "Supported",
            [{"expected": "Refuted", "actual": "Refuted"},
             {"expected": "Refuted", "actual": "Refuted"}],
            [],
            tool_rubric=deepcopy(GOLD_RUBRIC_2Q),
            local_edit_tool_rubric=[deepcopy(GOLD_RUBRIC_2Q), deepcopy(GOLD_RUBRIC_2Q)],
        )
        r = self.ev.evaluate([s])
        self.assertAlmostEqual(r["mediator_tool_match"]["predicted"]["mean"], 1.0)

    def test_tool_rubric_mismatch_predicted(self):
        bad = deepcopy(GOLD_RUBRIC_2Q)
        bad[list(bad.keys())[0]] = True
        s = make_processed(
            "0", "correct", deepcopy(GOLD_RUBRIC_2Q), "Supported",
            [{"expected": "Refuted", "actual": "Refuted"},
             {"expected": "Refuted", "actual": "Refuted"}],
            [],
            tool_rubric=bad,
            local_edit_tool_rubric=[deepcopy(GOLD_RUBRIC_2Q), deepcopy(GOLD_RUBRIC_2Q)],
        )
        r = self.ev.evaluate([s])
        self.assertAlmostEqual(r["mediator_tool_match"]["predicted"]["mean"], 0.0)

    def test_no_tool_mode_mediator_tool_match_empty(self):
        ev = make_eval(tool_mode="none")
        s = make_processed("0", "correct", deepcopy(GOLD_RUBRIC_2Q), "Supported",
                           [{"expected": "Refuted", "actual": "Refuted"},
                            {"expected": "Refuted", "actual": "Refuted"}], [])
        r = ev.evaluate([s])
        self.assertEqual(r["mediator_tool_match"], {})


# ===========================================================================
# 8. evaluate -- mixed batch (correct + incorrect + error)
# ===========================================================================

class TestEvaluateMixed(unittest.TestCase):

    def setUp(self):
        self.ev = make_eval()

    def test_mixed_batch_counts(self):
        samples = [
            # correct Supported, 2Q
            make_processed("0", "correct", deepcopy(GOLD_RUBRIC_2Q), "Supported",
                           [{"expected": "Refuted", "actual": "Refuted"},
                            {"expected": "Refuted", "actual": "Refuted"}], []),
            # incorrect Supported, 2Q
            make_processed("0", "incorrect", deepcopy(GOLD_RUBRIC_2Q), "Refuted",
                           [], [{"actual": "Supported"}]),
            # error
            make_processed("1", "error", {}, None, [], []),
        ]
        r = self.ev.evaluate(samples)
        c = r["performance"]["counts"]
        self.assertEqual(c["n_correct"],   1)
        self.assertEqual(c["n_incorrect"], 1)
        self.assertEqual(c["n_error"],     1)
        self.assertEqual(c["n_total"],     3)

    def test_mixed_batch_correct_rate(self):
        samples = [
            make_processed("0", "correct",   deepcopy(GOLD_RUBRIC_2Q), "Supported",
                           [{"expected": "Refuted", "actual": "Refuted"},
                            {"expected": "Refuted", "actual": "Refuted"}], []),
            make_processed("0", "incorrect", deepcopy(GOLD_RUBRIC_2Q), "Refuted",
                           [], [{"actual": "Supported"}]),
        ]
        r = self.ev.evaluate(samples)
        self.assertAlmostEqual(r["performance"]["counts"]["correct_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# 9. evaluate -- edge cases
# ===========================================================================

class TestEvaluateEdgeCases(unittest.TestCase):

    def setUp(self):
        self.ev = make_eval()

    def test_empty_sample_list_returns_zero_counts(self):
        r = self.ev.evaluate([])
        c = r["performance"]["counts"]
        self.assertEqual(c["n_total"],     0)
        self.assertEqual(c["n_correct"],   0)
        self.assertEqual(c["n_incorrect"], 0)
        self.assertEqual(c["n_error"],     0)
        self.assertIsNone(r["performance"]["checklist_match"]["mean"])
        self.assertIsNone(r["faithfulness"]["Local Edits"]["mean"])

    def test_unknown_generation_status_not_counted_anywhere(self):
        # A sample with status "pending" or any unknown value should be silently skipped
        s = make_processed("0", "pending", deepcopy(GOLD_RUBRIC_2Q), "Supported", [], [])
        r = self.ev.evaluate([s])
        c = r["performance"]["counts"]
        self.assertEqual(c["n_total"],     0)   # not counted: unknown status bypasses n_error too
        self.assertEqual(c["n_correct"],   0)
        self.assertEqual(c["n_incorrect"], 0)
        self.assertEqual(c["n_error"],     0)

    def test_none_target_before_makes_local_edits_not_eligible(self):
        # target_before=None: neither "Supported" nor any length condition → not eligible
        s = make_processed(
            "0", "correct", deepcopy(GOLD_RUBRIC_2Q),
            target_before=None,   # infer_completion returned None
            local_edits=[
                {"expected": "Refuted", "actual": "Refuted"},
                {"expected": "Refuted", "actual": "Refuted"},
            ],
            correction=[],
        )
        r = self.ev.evaluate([s])
        self.assertEqual(r["faithfulness"]["Local Edits"]["n_total"], 0)

    def test_correct_rate_zero_when_all_errors(self):
        samples = [
            make_processed("0", "error", {}, None, [], []),
            make_processed("1", "error", {}, None, [], []),
        ]
        r = self.ev.evaluate(samples)
        c = r["performance"]["counts"]
        self.assertEqual(c["n_error"],     2)
        self.assertEqual(c["correct_rate"], 0.0)

    def test_mixed_eligible_and_ineligible_local_edits_in_one_batch(self):
        # Supported sample (eligible) + multi-Refuted sample (not eligible)
        supported = make_processed(
            "0", "correct", deepcopy(GOLD_RUBRIC_2Q), "Supported",
            [{"expected": "Refuted", "actual": "Refuted"},
             {"expected": "Refuted", "actual": "Refuted"}],
            [],
        )
        multi_refuted = make_processed(
            "2", "correct", deepcopy(GOLD_RUBRIC_3Q), "Refuted",
            [{"expected": "Supported", "actual": "Refuted"},
             {"expected": "Supported", "actual": "Refuted"},
             {"expected": "Supported", "actual": "Refuted"}],
            [],
        )
        r = self.ev.evaluate([supported, multi_refuted])
        # Only the 2 edits from Supported are counted
        self.assertEqual(r["faithfulness"]["Local Edits"]["n_total"], 2)
        self.assertAlmostEqual(r["faithfulness"]["Local Edits"]["mean"], 1.0)

    def test_correction_mediator_tool_match_tracked(self):
        ev = make_eval(tool_mode="structured")
        s = make_processed(
            "0", "incorrect", deepcopy(GOLD_RUBRIC_2Q), "Refuted",
            [],
            [{"actual": "Supported"}],
            tool_rubric=None,
            correction_tool_rubric=[deepcopy(GOLD_RUBRIC_2Q)],
        )
        # correction mediator_rubric is set to gold_rubric inside make_processed
        r = ev.evaluate([s])
        # Correction tool match: mediator_rubric=gold_rubric vs tool_rubric=gold_rubric → 1
        self.assertAlmostEqual(r["mediator_tool_match"]["Correction"]["mean"], 1.0)

    def test_checklist_match_none_when_idx_not_in_dataset(self):
        # idx "999" is not in the mock dataset → gold_rubric is None
        # compare_structures(None, ...) → None → counted as n_none
        s = make_processed(
            "999", "correct", deepcopy(GOLD_RUBRIC_2Q), "Supported",
            [{"expected": "Refuted", "actual": "Refuted"},
             {"expected": "Refuted", "actual": "Refuted"}],
            [],
        )
        r = self.ev.evaluate([s])
        # checklist_match contains 1 None → n_none=1, mean=None
        self.assertEqual(r["performance"]["checklist_match"]["n_none"], 1)