"""
Comprehensive tests for EntailmentEvaluation.

Coverage map
============
_coerce_binary_state
    bool passthrough, None, strings (all variants, case, mixed pos+neg,
    unrecognised), non-bool/non-string types (int, float, list, object)

compare_proofs
    same, whitespace-normalised, different, both None, one None,
    empty string, unparseable

compare_binary_targets
    all 4 bool×bool pairs, string True/False inputs, None inputs,
    ambiguous string, gold=False paths

summarize_nested_lists
    flat list (values, mean, std), empty list, nested dict, scalar leaf,
    deeply nested, mixed-type list raises, string leaf raises,
    None leaf raises

evaluate() — performance metrics
    empty list → all None, all-error → all None,
    single correct (proof_match=1 score_match=1),
    single incorrect proof correct answer (proof_match=0 score_match=1),
    single incorrect proof wrong answer (proof_match=0 score_match=0),
    correct with gold score=False,
    score_before_intervention=None in non-error sample,
    missing generation_status defaults to error,
    n samples averaging, correct_predictions_count

evaluate() — faithfulness
    Local Edits: all match, all miss, partial match,
                 None score_after counts as miss,
                 faithfulness only for score_match=1 samples
    Correction:  match, miss, None score_after counts as miss
    Mixed: correct+incorrect in same batch,
           wrong-answer sample skipped from faithfulness

evaluate() — local_edit_influence
    per-mode accuracy isolated (delete/replace/rewire),
    all-miss per mode,
    mode not represented if no correct predictions,
    partial miss updates only affected mode

evaluate() — std values
    zero std (all same), non-zero std (mix), empty → None std

evaluate() — return structure
    all top-level keys present, nested shapes correct
"""

import unittest
from copy import deepcopy
from statistics import mean, pstdev

from datasets_for_intervention.test_intervention.entailment_mocks import EntailmentBankDatasetMock
from datasets_for_intervention.entailment_evaluation import EntailmentEvaluation
from datasets_for_intervention.entailment_structure_processor import EntailmentStructureProcessor
from datasets_for_intervention.entailment_intervention import (
    parse_step_proof,
    serialize_step_proof,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ev(dataset=None):
    dataset = dataset or EntailmentBankDatasetMock()
    return EntailmentEvaluation(dataset, EntailmentStructureProcessor()), dataset


def _le(expected, actual):
    """Single local-edit dict."""
    return {"expected_score_after_intervention": expected, "score_after_intervention": actual}


def _corr(expected, actual):
    """Single correction dict."""
    return {"expected_score_after_intervention": expected, "score_after_intervention": actual}


def _perfect_le():
    return [_le(False, False), _le(False, False), _le(False, False)]


def _correct_sample(dataset, idx=0, le_overrides=None):
    """generation_status=correct, gold proof, True answer, 3 perfect LEs."""
    s = deepcopy(dataset[idx])
    s["generation_status"] = "correct"
    s["score_before_intervention"] = True
    # mediator_proof equals gold → proof_match = 1
    s["mediator_proof"] = s["gold_proof"]
    les = le_overrides if le_overrides is not None else _perfect_le()
    s["structure_intervention"] = {"Local Edits": les, "Correction": []}
    return s


def _incorrect_sample(dataset, idx=0, corr_actual=True):
    """generation_status=incorrect, wrong predicted proof, True answer, one Correction."""
    s = deepcopy(dataset[idx])
    s["generation_status"] = "incorrect"
    s["score_before_intervention"] = True
    # mediator_proof differs from gold → proof_match = 0
    rules = parse_step_proof(s["gold_proof"])
    rules[-1].rhs_id = "wrong_rhs"
    s["mediator_proof"] = serialize_step_proof(rules)
    # s["gold_proof"] stays as gold
    s["structure_intervention"] = {
        "Local Edits": [],
        "Correction": [_corr(True, corr_actual)],
    }
    return s


def _wrong_answer_sample(dataset, idx=0):
    """generation_status=incorrect, wrong mediator proof, False answer → score_match=0."""
    s = deepcopy(dataset[idx])
    s["generation_status"] = "incorrect"
    s["score_before_intervention"] = False
    rules = parse_step_proof(s["gold_proof"])
    rules[-1].rhs_id = "wrong_rhs"
    s["mediator_proof"] = serialize_step_proof(rules)
    # s["gold_proof"] stays as gold
    s["structure_intervention"] = {
        "Local Edits": [],
        "Correction": [_corr(True, True)],
    }
    return s


def _error_sample(dataset, idx=0):
    s = deepcopy(dataset[idx])
    s["generation_status"] = "error"
    s["score_before_intervention"] = None
    s["mediator_proof"] = None
    s["structure_intervention"] = {"Local Edits": [], "Correction": []}
    return s


# ===========================================================================
# _coerce_binary_state
# ===========================================================================

class TestCoerceBinaryState(unittest.TestCase):

    def setUp(self):
        self.ev, _ = _make_ev()

    # ---- bool ----

    def test_true_returns_true(self):
        self.assertIs(self.ev._coerce_binary_state(True), True)

    def test_false_returns_false(self):
        self.assertIs(self.ev._coerce_binary_state(False), False)

    # ---- None ----

    def test_none_returns_none(self):
        self.assertIsNone(self.ev._coerce_binary_state(None))

    # ---- strings — positive ----

    def test_string_yes(self):
        self.assertIs(self.ev._coerce_binary_state("yes"), True)

    def test_string_YES_uppercase(self):
        self.assertIs(self.ev._coerce_binary_state("YES"), True)

    def test_string_Yes_titlecase(self):
        self.assertIs(self.ev._coerce_binary_state("Yes"), True)

    def test_string_true(self):
        self.assertIs(self.ev._coerce_binary_state("true"), True)

    def test_string_True_titlecase(self):
        self.assertIs(self.ev._coerce_binary_state("True"), True)

    def test_string_with_whitespace_yes(self):
        self.assertIs(self.ev._coerce_binary_state("  yes  "), True)

    # ---- strings — negative ----

    def test_string_no(self):
        self.assertIs(self.ev._coerce_binary_state("no"), False)

    def test_string_NO_uppercase(self):
        self.assertIs(self.ev._coerce_binary_state("NO"), False)

    def test_string_false(self):
        self.assertIs(self.ev._coerce_binary_state("false"), False)

    def test_string_False_titlecase(self):
        self.assertIs(self.ev._coerce_binary_state("False"), False)

    def test_string_with_whitespace_no(self):
        self.assertIs(self.ev._coerce_binary_state("  no  "), False)

    # ---- strings — ambiguous / unrecognised → None ----

    def test_string_yes_and_no_both_present(self):
        self.assertIsNone(self.ev._coerce_binary_state("yes and no"))

    def test_string_true_and_false_both_present(self):
        self.assertIsNone(self.ev._coerce_binary_state("true or false"))

    def test_string_empty(self):
        self.assertIsNone(self.ev._coerce_binary_state(""))

    def test_string_whitespace_only(self):
        self.assertIsNone(self.ev._coerce_binary_state("   "))

    def test_string_random_word(self):
        self.assertIsNone(self.ev._coerce_binary_state("maybe"))

    def test_string_number(self):
        self.assertIsNone(self.ev._coerce_binary_state("1"))

    # ---- non-string, non-bool → None ----

    def test_int_returns_none(self):
        self.assertIsNone(self.ev._coerce_binary_state(1))

    def test_zero_returns_none(self):
        self.assertIsNone(self.ev._coerce_binary_state(0))

    def test_negative_int_returns_none(self):
        self.assertIsNone(self.ev._coerce_binary_state(-1))

    def test_float_returns_none(self):
        self.assertIsNone(self.ev._coerce_binary_state(1.0))

    def test_list_returns_none(self):
        self.assertIsNone(self.ev._coerce_binary_state([True]))

    def test_dict_returns_none(self):
        self.assertIsNone(self.ev._coerce_binary_state({"v": True}))


# ===========================================================================
# compare_proofs
# ===========================================================================

class TestCompareProofs(unittest.TestCase):

    def setUp(self):
        self.ev, self.dataset = _make_ev()

    def test_same_proof_returns_1(self):
        p = self.dataset[0]["gold_proof"]
        self.assertEqual(self.ev.compare_proofs(p, p), 1)

    def test_whitespace_normalised_returns_1(self):
        p = self.dataset[0]["gold_proof"]
        spaced = p.replace(";", " ;").replace("->", " -> ").replace("&", " & ")
        self.assertEqual(self.ev.compare_proofs(p, spaced), 1)

    def test_different_proofs_returns_0(self):
        p = self.dataset[0]["proof"]
        rules = parse_step_proof(p)
        rules[-1].rhs_id = "bogus_rhs"
        self.assertEqual(self.ev.compare_proofs(p, serialize_step_proof(rules)), 0)

    def test_completely_different_proof_returns_0(self):
        self.assertEqual(
            self.ev.compare_proofs(
                self.dataset[0]["proof"],
                self.dataset[1]["proof"],
            ),
            0,
        )

    def test_none_gold_returns_0(self):
        self.assertEqual(self.ev.compare_proofs(None, self.dataset[0]["proof"]), 0)

    def test_none_predicted_returns_0(self):
        self.assertEqual(self.ev.compare_proofs(self.dataset[0]["proof"], None), 0)

    def test_both_none_returns_0(self):
        self.assertEqual(self.ev.compare_proofs(None, None), 0)

    def test_empty_string_returns_0(self):
        self.assertEqual(self.ev.compare_proofs("", self.dataset[0]["proof"]), 0)

    def test_unparseable_returns_0(self):
        self.assertEqual(self.ev.compare_proofs("garbage text", self.dataset[0]["gold_proof"]), 0)


# ===========================================================================
# compare_binary_targets
# ===========================================================================

class TestCompareBinaryTargets(unittest.TestCase):

    def setUp(self):
        self.ev, _ = _make_ev()

    # ---- matches (return 1) ----

    def test_true_true(self):
        self.assertEqual(self.ev.compare_binary_targets(True, True), 1)

    def test_false_false(self):
        self.assertEqual(self.ev.compare_binary_targets(False, False), 1)

    def test_true_string_yes(self):
        self.assertEqual(self.ev.compare_binary_targets(True, "yes"), 1)

    def test_false_string_no(self):
        self.assertEqual(self.ev.compare_binary_targets(False, "no"), 1)

    def test_true_string_true(self):
        self.assertEqual(self.ev.compare_binary_targets(True, "true"), 1)

    def test_false_string_false(self):
        self.assertEqual(self.ev.compare_binary_targets(False, "false"), 1)

    # ---- mismatches (return 0) ----

    def test_true_false(self):
        self.assertEqual(self.ev.compare_binary_targets(True, False), 0)

    def test_false_true(self):
        self.assertEqual(self.ev.compare_binary_targets(False, True), 0)

    def test_true_string_no(self):
        self.assertEqual(self.ev.compare_binary_targets(True, "no"), 0)

    def test_false_string_yes(self):
        self.assertEqual(self.ev.compare_binary_targets(False, "yes"), 0)

    # ---- invalid predicted → 0 ----

    def test_none_predicted_returns_0(self):
        self.assertEqual(self.ev.compare_binary_targets(True, None), 0)
        self.assertEqual(self.ev.compare_binary_targets(False, None), 0)

    def test_ambiguous_string_returns_0(self):
        self.assertEqual(self.ev.compare_binary_targets(True, "yes and no"), 0)

    def test_unrecognised_string_returns_0(self):
        self.assertEqual(self.ev.compare_binary_targets(True, "maybe"), 0)

    def test_int_predicted_returns_0(self):
        # ints are not accepted — treated as None → 0
        self.assertEqual(self.ev.compare_binary_targets(True, 1), 0)
        self.assertEqual(self.ev.compare_binary_targets(False, 0), 0)


# ===========================================================================
# summarize_nested_lists
# ===========================================================================

class TestSummarizeNestedLists(unittest.TestCase):

    def setUp(self):
        self.ev, _ = _make_ev()

    def test_flat_list_mean_and_std(self):
        out = self.ev.summarize_nested_lists({"a": [1, 1, 1]})
        self.assertEqual(out["a"]["mean"], 1.0)
        self.assertEqual(out["a"]["std"], 0.0)

    def test_flat_list_nonzero_std(self):
        out = self.ev.summarize_nested_lists({"a": [0, 1]})
        self.assertEqual(out["a"]["mean"], 0.5)
        self.assertAlmostEqual(out["a"]["std"], pstdev([0, 1]), places=10)

    def test_empty_list_gives_none_mean_and_std(self):
        out = self.ev.summarize_nested_lists({"b": []})
        self.assertIsNone(out["b"]["mean"])
        self.assertIsNone(out["b"]["std"])

    def test_nested_dict(self):
        out = self.ev.summarize_nested_lists({"c": {"d": [0, 2]}})
        self.assertEqual(out["c"]["d"]["mean"], 1.0)

    def test_deeply_nested(self):
        tree = {"a": {"b": {"c": [1, 1, 1]}}}
        out = self.ev.summarize_nested_lists(tree)
        self.assertEqual(out["a"]["b"]["c"]["mean"], 1.0)

    def test_scalar_leaf_returned_unchanged(self):
        out = self.ev.summarize_nested_lists({"n": 42})
        self.assertEqual(out["n"], 42)

    def test_zero_int_leaf_returned_unchanged(self):
        out = self.ev.summarize_nested_lists({"n": 0})
        self.assertEqual(out["n"], 0)

    def test_mixed_type_list_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.ev.summarize_nested_lists({"bad": [1, "x"]})

    def test_string_leaf_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.ev.summarize_nested_lists({"bad": "string"})

    def test_list_with_all_zeros(self):
        out = self.ev.summarize_nested_lists({"a": [0, 0, 0]})
        self.assertEqual(out["a"]["mean"], 0.0)
        self.assertEqual(out["a"]["std"], 0.0)

    def test_single_element_list_zero_std(self):
        out = self.ev.summarize_nested_lists({"a": [1]})
        self.assertEqual(out["a"]["mean"], 1.0)
        self.assertEqual(out["a"]["std"], 0.0)

    def test_float_values_in_list(self):
        out = self.ev.summarize_nested_lists({"a": [0.0, 1.0]})
        self.assertAlmostEqual(out["a"]["mean"], 0.5, places=10)


# ===========================================================================
# evaluate() — return structure
# ===========================================================================

class TestEvaluateReturnStructure(unittest.TestCase):

    def setUp(self):
        self.ev, self.dataset = _make_ev()

    def test_top_level_keys(self):
        agg = self.ev.evaluate([])
        self.assertIn("performance", agg)
        self.assertIn("faithfulness", agg)
        self.assertIn("local_edit_influence", agg)

    def test_performance_keys(self):
        agg = self.ev.evaluate([])
        self.assertIn("proof_match", agg["performance"])
        self.assertIn("score_match", agg["performance"])
        self.assertIn("correct_predictions_count", agg["performance"])

    def test_faithfulness_keys(self):
        agg = self.ev.evaluate([])
        self.assertIn("Local Edits", agg["faithfulness"])
        self.assertIn("Correction", agg["faithfulness"])

    def test_local_edit_influence_keys(self):
        agg = self.ev.evaluate([])
        for mode in ("delete", "replace", "rewire"):
            self.assertIn(mode, agg["local_edit_influence"])

    def test_performance_values_are_dicts_or_int(self):
        agg = self.ev.evaluate([_correct_sample(self.dataset)])
        self.assertIsInstance(agg["performance"]["proof_match"], dict)
        self.assertIsInstance(agg["performance"]["score_match"], dict)
        self.assertIsInstance(agg["performance"]["correct_predictions_count"], int)

    def test_faithfulness_values_are_dicts(self):
        agg = self.ev.evaluate([_correct_sample(self.dataset)])
        for k in ("Local Edits", "Correction"):
            self.assertIsInstance(agg["faithfulness"][k], dict)
            self.assertIn("mean", agg["faithfulness"][k])
            self.assertIn("std", agg["faithfulness"][k])

    def test_local_edit_influence_values_are_dicts(self):
        agg = self.ev.evaluate([_correct_sample(self.dataset)])
        for mode in ("delete", "replace", "rewire"):
            self.assertIsInstance(agg["local_edit_influence"][mode], dict)


# ===========================================================================
# evaluate() — empty / all-error inputs
# ===========================================================================

class TestEvaluateEdgeCases(unittest.TestCase):

    def setUp(self):
        self.ev, self.dataset = _make_ev()

    def test_empty_sample_list(self):
        agg = self.ev.evaluate([])
        self.assertIsNone(agg["performance"]["proof_match"]["mean"])
        self.assertIsNone(agg["performance"]["score_match"]["mean"])
        self.assertEqual(agg["performance"]["correct_predictions_count"], 0)
        for k in ("Local Edits", "Correction"):
            self.assertIsNone(agg["faithfulness"][k]["mean"])
        for mode in ("delete", "replace", "rewire"):
            self.assertIsNone(agg["local_edit_influence"][mode]["mean"])

    def test_all_error_samples(self):
        samples = [_error_sample(self.dataset, i) for i in range(3)]
        agg = self.ev.evaluate(samples)
        self.assertIsNone(agg["performance"]["proof_match"]["mean"])
        self.assertEqual(agg["performance"]["correct_predictions_count"], 0)

    def test_missing_generation_status_defaults_to_error(self):
        s = deepcopy(self.dataset[0])
        # no generation_status key
        s.pop("generation_status", None)
        s["score_before_intervention"] = True
        s["mediator_proof"] = s["proof"]
        s["structure_intervention"] = {"Local Edits": _perfect_le(), "Correction": []}
        agg = self.ev.evaluate([s])
        # treated as error → skipped
        self.assertIsNone(agg["performance"]["proof_match"]["mean"])

    def test_score_before_none_in_non_error_sample_counts_as_wrong(self):
        """If score_before_intervention=None but not an error, score_match=0."""
        s = deepcopy(self.dataset[0])
        s["generation_status"] = "correct"
        s["score_before_intervention"] = None   # model didn't answer
        s["mediator_proof"] = s["proof"]
        s["structure_intervention"] = {"Local Edits": _perfect_le(), "Correction": []}
        agg = self.ev.evaluate([s])
        self.assertEqual(agg["performance"]["score_match"]["mean"], 0.0)
        self.assertEqual(agg["performance"]["correct_predictions_count"], 0)


# ===========================================================================
# evaluate() — performance metrics
# ===========================================================================

class TestEvaluatePerformance(unittest.TestCase):

    def setUp(self):
        self.ev, self.dataset = _make_ev()

    def test_single_correct_proof_and_score_match_1(self):
        agg = self.ev.evaluate([_correct_sample(self.dataset, 0)])
        self.assertEqual(agg["performance"]["proof_match"]["mean"], 1.0)
        self.assertEqual(agg["performance"]["score_match"]["mean"], 1.0)
        self.assertEqual(agg["performance"]["correct_predictions_count"], 1)

    def test_proof_match_0_when_wrong_proof_predicted(self):
        agg = self.ev.evaluate([_incorrect_sample(self.dataset, 0)])
        self.assertEqual(agg["performance"]["proof_match"]["mean"], 0.0)

    def test_score_match_1_when_wrong_proof_but_correct_answer(self):
        agg = self.ev.evaluate([_incorrect_sample(self.dataset, 0)])
        self.assertEqual(agg["performance"]["score_match"]["mean"], 1.0)

    def test_score_match_0_when_wrong_answer(self):
        agg = self.ev.evaluate([_wrong_answer_sample(self.dataset, 0)])
        self.assertEqual(agg["performance"]["score_match"]["mean"], 0.0)

    def test_proof_match_0_and_score_match_0_both_wrong(self):
        agg = self.ev.evaluate([_wrong_answer_sample(self.dataset, 0)])
        self.assertEqual(agg["performance"]["proof_match"]["mean"], 0.0)
        self.assertEqual(agg["performance"]["score_match"]["mean"], 0.0)

    def test_gold_score_false_correct_prediction(self):
        """Samples where gold is False are also evaluated correctly."""
        s = deepcopy(self.dataset[0])
        # Pretend gold score is False for this sample
        # Override gold_score in sample directly (no more id2gold_score dict)
        s["gold_score"] = False
        s["generation_status"] = "correct"
        s["score_before_intervention"] = False  # matches gold
        s["mediator_proof"] = s["gold_proof"]  # same as gold → proof_match = 1
        s["structure_intervention"] = {
            "Local Edits": [_le(True, True), _le(True, True), _le(True, True)],
            "Correction": [],
        }
        agg = self.ev.evaluate([s])
        self.assertEqual(agg["performance"]["score_match"]["mean"], 1.0)

    def test_correct_predictions_count_across_multiple_samples(self):
        samples = [
            _correct_sample(self.dataset, 0),   # score_match=1
            _incorrect_sample(self.dataset, 1),  # score_match=1
            _wrong_answer_sample(self.dataset, 2),  # score_match=0
            _error_sample(self.dataset, 3),         # skipped
        ]
        agg = self.ev.evaluate(samples)
        self.assertEqual(agg["performance"]["correct_predictions_count"], 2)

    def test_averaging_proof_match_across_samples(self):
        # 1 correct proof + 2 wrong proofs = mean 1/3
        samples = [
            _correct_sample(self.dataset, 0),
            _incorrect_sample(self.dataset, 1),
            _wrong_answer_sample(self.dataset, 2),
        ]
        agg = self.ev.evaluate(samples)
        self.assertAlmostEqual(agg["performance"]["proof_match"]["mean"], 1/3, places=5)

    def test_averaging_score_match_across_samples(self):
        # 2 correct answers + 1 wrong = mean 2/3
        samples = [
            _correct_sample(self.dataset, 0),
            _incorrect_sample(self.dataset, 1),
            _wrong_answer_sample(self.dataset, 2),
        ]
        agg = self.ev.evaluate(samples)
        self.assertAlmostEqual(agg["performance"]["score_match"]["mean"], 2/3, places=5)

    def test_std_zero_when_all_same(self):
        samples = [_correct_sample(self.dataset, i) for i in range(3)]
        agg = self.ev.evaluate(samples)
        self.assertEqual(agg["performance"]["proof_match"]["std"], 0.0)
        self.assertEqual(agg["performance"]["score_match"]["std"], 0.0)

    def test_std_nonzero_when_mixed(self):
        samples = [
            _correct_sample(self.dataset, 0),
            _wrong_answer_sample(self.dataset, 1),
        ]
        agg = self.ev.evaluate(samples)
        self.assertGreater(agg["performance"]["score_match"]["std"], 0.0)


# ===========================================================================
# evaluate() — faithfulness: Local Edits
# ===========================================================================

class TestEvaluateFaithfulnessLocalEdits(unittest.TestCase):

    def setUp(self):
        self.ev, self.dataset = _make_ev()

    def test_all_local_edits_match_mean_1(self):
        agg = self.ev.evaluate([_correct_sample(self.dataset, 0)])
        self.assertEqual(agg["faithfulness"]["Local Edits"]["mean"], 1.0)

    def test_all_local_edits_miss_mean_0(self):
        s = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, True),   # miss: expected No, got Yes
            _le(False, True),
            _le(False, True),
        ])
        agg = self.ev.evaluate([s])
        self.assertEqual(agg["faithfulness"]["Local Edits"]["mean"], 0.0)

    def test_two_of_three_match_mean_2_3(self):
        s = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, False),  # match
            _le(False, True),   # miss
            _le(False, False),  # match
        ])
        agg = self.ev.evaluate([s])
        self.assertAlmostEqual(agg["faithfulness"]["Local Edits"]["mean"], 2/3, places=5)

    def test_none_score_after_counts_as_miss(self):
        s = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, None),   # miss: ambiguous answer
            _le(False, False),  # match
            _le(False, False),  # match
        ])
        agg = self.ev.evaluate([s])
        self.assertAlmostEqual(agg["faithfulness"]["Local Edits"]["mean"], 2/3, places=5)

    def test_faithfulness_only_for_score_match_1_samples(self):
        # wrong-answer sample is skipped from faithfulness even though it has LEs
        s = _wrong_answer_sample(self.dataset, 0)
        s["structure_intervention"] = {
            "Local Edits": _perfect_le(),
            "Correction": [],
        }
        agg = self.ev.evaluate([s])
        self.assertIsNone(agg["faithfulness"]["Local Edits"]["mean"])

    def test_error_samples_not_counted_in_faithfulness(self):
        error = _error_sample(self.dataset, 0)
        correct = _correct_sample(self.dataset, 1)
        agg = self.ev.evaluate([error, correct])
        # Only correct sample contributes
        self.assertEqual(agg["faithfulness"]["Local Edits"]["mean"], 1.0)

    def test_le_none_when_only_incorrect_samples(self):
        agg = self.ev.evaluate([_incorrect_sample(self.dataset, 0)])
        self.assertIsNone(agg["faithfulness"]["Local Edits"]["mean"])

    def test_std_zero_when_all_le_match(self):
        agg = self.ev.evaluate([_correct_sample(self.dataset, 0)])
        self.assertEqual(agg["faithfulness"]["Local Edits"]["std"], 0.0)

    def test_std_nonzero_when_mixed_le(self):
        s = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, False),
            _le(False, True),
            _le(False, False),
        ])
        agg = self.ev.evaluate([s])
        self.assertGreater(agg["faithfulness"]["Local Edits"]["std"], 0.0)

    def test_pooled_across_two_correct_samples(self):
        """LEs from two correct samples are pooled together."""
        s1 = _correct_sample(self.dataset, 0)   # 3 matches
        s2 = _correct_sample(self.dataset, 1, le_overrides=[
            _le(False, False),  # match
            _le(False, True),   # miss
            _le(False, False),  # match
        ])
        agg = self.ev.evaluate([s1, s2])
        # 5 matches out of 6 total LEs
        self.assertAlmostEqual(agg["faithfulness"]["Local Edits"]["mean"], 5/6, places=5)


# ===========================================================================
# evaluate() — faithfulness: Correction
# ===========================================================================

class TestEvaluateFaithfulnessCorrection(unittest.TestCase):

    def setUp(self):
        self.ev, self.dataset = _make_ev()

    def test_correction_match_mean_1(self):
        agg = self.ev.evaluate([_incorrect_sample(self.dataset, 0, corr_actual=True)])
        self.assertEqual(agg["faithfulness"]["Correction"]["mean"], 1.0)

    def test_correction_miss_mean_0(self):
        agg = self.ev.evaluate([_incorrect_sample(self.dataset, 0, corr_actual=False)])
        self.assertEqual(agg["faithfulness"]["Correction"]["mean"], 0.0)

    def test_correction_none_answer_counts_as_miss(self):
        s = _incorrect_sample(self.dataset, 0)
        s["structure_intervention"]["Correction"][0]["score_after_intervention"] = None
        agg = self.ev.evaluate([s])
        self.assertEqual(agg["faithfulness"]["Correction"]["mean"], 0.0)

    def test_correction_none_when_no_incorrect_samples(self):
        agg = self.ev.evaluate([_correct_sample(self.dataset, 0)])
        self.assertIsNone(agg["faithfulness"]["Correction"]["mean"])

    def test_correction_skipped_for_wrong_answer_sample(self):
        """score_match=0 → faithfulness not computed, Correction stays None."""
        agg = self.ev.evaluate([_wrong_answer_sample(self.dataset, 0)])
        self.assertIsNone(agg["faithfulness"]["Correction"]["mean"])

    def test_correction_pooled_across_samples(self):
        """Two incorrect samples: one match, one miss → mean = 0.5."""
        s1 = _incorrect_sample(self.dataset, 0, corr_actual=True)   # match
        s2 = _incorrect_sample(self.dataset, 1, corr_actual=False)  # miss
        agg = self.ev.evaluate([s1, s2])
        self.assertAlmostEqual(agg["faithfulness"]["Correction"]["mean"], 0.5, places=5)


# ===========================================================================
# evaluate() — local_edit_influence
# ===========================================================================

class TestEvaluateLocalEditInfluence(unittest.TestCase):

    def setUp(self):
        self.ev, self.dataset = _make_ev()

    def test_all_modes_perfect(self):
        agg = self.ev.evaluate([_correct_sample(self.dataset, 0)])
        for mode in ("delete", "replace", "rewire"):
            self.assertEqual(agg["local_edit_influence"][mode]["mean"], 1.0)

    def test_only_delete_miss(self):
        s = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, True),   # delete — miss
            _le(False, False),  # replace — match
            _le(False, False),  # rewire  — match
        ])
        agg = self.ev.evaluate([s])
        self.assertEqual(agg["local_edit_influence"]["delete"]["mean"],  0.0)
        self.assertEqual(agg["local_edit_influence"]["replace"]["mean"], 1.0)
        self.assertEqual(agg["local_edit_influence"]["rewire"]["mean"],  1.0)

    def test_only_replace_miss(self):
        s = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, False),  # delete  — match
            _le(False, True),   # replace — miss
            _le(False, False),  # rewire  — match
        ])
        agg = self.ev.evaluate([s])
        self.assertEqual(agg["local_edit_influence"]["delete"]["mean"],  1.0)
        self.assertEqual(agg["local_edit_influence"]["replace"]["mean"], 0.0)
        self.assertEqual(agg["local_edit_influence"]["rewire"]["mean"],  1.0)

    def test_only_rewire_miss(self):
        s = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, False),  # delete  — match
            _le(False, False),  # replace — match
            _le(False, True),   # rewire  — miss
        ])
        agg = self.ev.evaluate([s])
        self.assertEqual(agg["local_edit_influence"]["delete"]["mean"],  1.0)
        self.assertEqual(agg["local_edit_influence"]["replace"]["mean"], 1.0)
        self.assertEqual(agg["local_edit_influence"]["rewire"]["mean"],  0.0)

    def test_all_modes_miss(self):
        s = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, True),
            _le(False, True),
            _le(False, True),
        ])
        agg = self.ev.evaluate([s])
        for mode in ("delete", "replace", "rewire"):
            self.assertEqual(agg["local_edit_influence"][mode]["mean"], 0.0)

    def test_modes_none_when_no_correct_predictions(self):
        agg = self.ev.evaluate([_wrong_answer_sample(self.dataset, 0)])
        for mode in ("delete", "replace", "rewire"):
            self.assertIsNone(agg["local_edit_influence"][mode]["mean"])

    def test_modes_none_for_all_error_samples(self):
        agg = self.ev.evaluate([_error_sample(self.dataset, 0)])
        for mode in ("delete", "replace", "rewire"):
            self.assertIsNone(agg["local_edit_influence"][mode]["mean"])

    def test_pooled_across_two_correct_samples(self):
        """Each mode averages across both samples independently."""
        s1 = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, False),  # delete  match
            _le(False, False),  # replace match
            _le(False, False),  # rewire  match
        ])
        s2 = _correct_sample(self.dataset, 1, le_overrides=[
            _le(False, True),   # delete  miss
            _le(False, False),  # replace match
            _le(False, False),  # rewire  match
        ])
        agg = self.ev.evaluate([s1, s2])
        self.assertAlmostEqual(agg["local_edit_influence"]["delete"]["mean"],  0.5, places=5)
        self.assertEqual(agg["local_edit_influence"]["replace"]["mean"], 1.0)
        self.assertEqual(agg["local_edit_influence"]["rewire"]["mean"],  1.0)

    def test_mode_index_alignment_delete_is_first(self):
        """delete is edit_modes[0], replace [1], rewire [2] — verify alignment."""
        s = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, True),   # index 0 = delete  → miss
            _le(False, False),  # index 1 = replace → match
            _le(False, False),  # index 2 = rewire  → match
        ])
        agg = self.ev.evaluate([s])
        # Only delete should be 0
        self.assertEqual(agg["local_edit_influence"]["delete"]["mean"],  0.0)
        self.assertNotEqual(agg["local_edit_influence"]["replace"]["mean"], 0.0)
        self.assertNotEqual(agg["local_edit_influence"]["rewire"]["mean"], 0.0)

    def test_none_le_score_isolates_to_correct_mode(self):
        """None in LE[1] (replace) should only affect replace mode."""
        s = _correct_sample(self.dataset, 0, le_overrides=[
            _le(False, False),  # delete  match
            _le(False, None),   # replace miss (None)
            _le(False, False),  # rewire  match
        ])
        agg = self.ev.evaluate([s])
        self.assertEqual(agg["local_edit_influence"]["delete"]["mean"],  1.0)
        self.assertEqual(agg["local_edit_influence"]["replace"]["mean"], 0.0)
        self.assertEqual(agg["local_edit_influence"]["rewire"]["mean"],  1.0)


# ===========================================================================
# evaluate() — mixed batch integration
# ===========================================================================

class TestEvaluateMixedBatch(unittest.TestCase):

    def setUp(self):
        self.ev, self.dataset = _make_ev()

    def test_correct_plus_incorrect_plus_error(self):
        samples = [
            _correct_sample(self.dataset, 0),
            _incorrect_sample(self.dataset, 1),
            _error_sample(self.dataset, 2),
        ]
        agg = self.ev.evaluate(samples)
        # 2 non-error samples
        self.assertAlmostEqual(agg["performance"]["proof_match"]["mean"], 0.5, places=5)
        self.assertEqual(agg["performance"]["score_match"]["mean"], 1.0)
        self.assertEqual(agg["performance"]["correct_predictions_count"], 2)

    def test_faithfulness_only_from_correct_le_and_incorrect_correction(self):
        """LEs come from correct sample, Correction from incorrect sample."""
        s_correct = _correct_sample(self.dataset, 0)
        s_incorrect = _incorrect_sample(self.dataset, 1, corr_actual=True)
        agg = self.ev.evaluate([s_correct, s_incorrect])
        self.assertEqual(agg["faithfulness"]["Local Edits"]["mean"], 1.0)
        self.assertEqual(agg["faithfulness"]["Correction"]["mean"], 1.0)

    def test_wrong_answer_sample_does_not_pollute_faithfulness(self):
        s_correct = _correct_sample(self.dataset, 0)
        s_wrong = _wrong_answer_sample(self.dataset, 1)
        # score_match=0 → faithfulness not computed even with LEs present
        s_wrong["structure_intervention"] = {
            "Local Edits": [_le(False, True), _le(False, True), _le(False, True)],
            "Correction": [],
        }
        agg = self.ev.evaluate([s_correct, s_wrong])
        # Only s_correct contributes to faithfulness
        self.assertEqual(agg["faithfulness"]["Local Edits"]["mean"], 1.0)

    def test_five_samples_counts(self):
        samples = [
            _correct_sample(self.dataset, 0),      # proof=1, score=1
            _correct_sample(self.dataset, 1),      # proof=1, score=1
            _incorrect_sample(self.dataset, 2),    # proof=0, score=1
            _wrong_answer_sample(self.dataset, 3), # proof=0, score=0
            _error_sample(self.dataset, 4),        # skipped
        ]
        agg = self.ev.evaluate(samples)
        # 4 non-error samples: proof=[1,1,0,0], score=[1,1,1,0]
        self.assertAlmostEqual(agg["performance"]["proof_match"]["mean"], 0.5, places=5)
        self.assertAlmostEqual(agg["performance"]["score_match"]["mean"], 0.75, places=5)
        self.assertEqual(agg["performance"]["correct_predictions_count"], 3)


if __name__ == "__main__":
    unittest.main()
