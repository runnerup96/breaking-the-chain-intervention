"""
Tests for AVeriTeCStructureProcessor and AVeriTeCTool.

Covers:
  - extract_mediator (valid and invalid formats)
  - extract_final_answer (full and short_completion)
  - check_generation_format_mistakes
  - extract_tool_args (none / simple / structured)
  - boollist_to_checklist
  - compare_structures
  - AVeriTeCTool.validate_args
  - AVeriTeCTool.calculate_score
"""
import unittest
from copy import deepcopy

from datasets_for_intervention.test_intervention.averitec_mocks import AVeriTeCDatasetMock, GOLD_RUBRIC_2Q, GOLD_RUBRIC_1Q, GOLD_RUBRIC_3Q
from datasets_for_intervention.averitec_structure_processor import AVeriTeCTool, AVeriTeCStructureProcessor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DATASET = AVeriTeCDatasetMock()
SAMPLE_SUPPORTED = DATASET[0]   # Supported, 2Q, gold_rubric = GOLD_RUBRIC_2Q
SAMPLE_REFUTED1  = DATASET[1]   # Refuted,   1Q, gold_rubric = GOLD_RUBRIC_1Q
SAMPLE_REFUTED3  = DATASET[2]   # Refuted,   3Q, gold_rubric = GOLD_RUBRIC_3Q

PROC_NONE   = AVeriTeCStructureProcessor(DATASET, "none")
PROC_SIMPLE = AVeriTeCStructureProcessor(DATASET, "simple")
PROC_STRUCT = AVeriTeCStructureProcessor(DATASET, "structured")

TOOL_NONE   = AVeriTeCTool(DATASET, "none")

# Canonical completion strings used across multiple test cases
COMP_2Q_OK = (
    "Checklist:\n"
    "- Q: Did Hunter Biden have any experience in the energy sector in 2014? (True/False): False\n"
    "- Q: Did Hunter Biden have any experience in Ukraine in 2014? (True/False): False\n"
    "\n"
    "Final Verdict: Supported"
)
COMP_1Q_OK = (
    "Checklist:\n"
    "- Q: Did Trump make pro-gay laws when in office? (True/False): False\n"
    "\n"
    "Final Verdict: Refuted"
)
COMP_MIXED = (
    "Checklist:\n"
    "- Q: Was the announcement made? (True/False): True\n"
    "- Q: Was it on Twitter? (True/False): False\n"
    "\n"
    "Final Verdict: Refuted"
)


# ===========================================================================
# 1. extract_mediator -- valid formats
# ===========================================================================

class TestExtractMediatorValid(unittest.TestCase):

    def test_two_questions_both_false(self):
        result = PROC_NONE.extract_mediator(COMP_2Q_OK)
        self.assertEqual(result, GOLD_RUBRIC_2Q)

    def test_one_question_false(self):
        result = PROC_NONE.extract_mediator(COMP_1Q_OK)
        self.assertEqual(result, GOLD_RUBRIC_1Q)

    def test_mixed_true_false(self):
        result = PROC_NONE.extract_mediator(COMP_MIXED)
        self.assertEqual(result, {"Was the announcement made?": True, "Was it on Twitter?": False})

    def test_without_hint_parenthetical(self):
        comp = (
            "Checklist:\n"
            "- Q: Was it real?: True\n"
            "- Q: Was it official?: False\n"
            "Final Verdict: Supported"
        )
        result = PROC_NONE.extract_mediator(comp)
        self.assertEqual(result, {"Was it real?": True, "Was it official?": False})

    def test_yes_no_aliases_converted_to_bool(self):
        comp = (
            "Checklist:\n"
            "- Q: Did it happen? (True/False): Yes\n"
            "- Q: Was it confirmed? (True/False): No\n"
            "Final Verdict: Supported"
        )
        result = PROC_NONE.extract_mediator(comp)
        self.assertEqual(result, {"Did it happen?": True, "Was it confirmed?": False})

    def test_three_questions_all_false(self):
        q1 = list(GOLD_RUBRIC_3Q.keys())[0]
        q2 = list(GOLD_RUBRIC_3Q.keys())[1]
        q3 = list(GOLD_RUBRIC_3Q.keys())[2]
        comp = (
            "Checklist:\n"
            f"- Q: {q1} (True/False): False\n"
            f"- Q: {q2} (True/False): False\n"
            f"- Q: {q3} (True/False): False\n"
            "Final Verdict: Refuted"
        )
        result = PROC_NONE.extract_mediator(comp)
        self.assertEqual(result, GOLD_RUBRIC_3Q)

    def test_trailing_content_after_verdict_ignored(self):
        comp = COMP_2Q_OK + "\n\nSome extra model output here."
        result = PROC_NONE.extract_mediator(comp)
        self.assertEqual(result, GOLD_RUBRIC_2Q)


# ===========================================================================
# 2. extract_mediator -- invalid formats → None
# ===========================================================================

class TestExtractMediatorInvalid(unittest.TestCase):

    def test_empty_string(self):
        self.assertIsNone(PROC_NONE.extract_mediator(""))

    def test_none_input(self):
        self.assertIsNone(PROC_NONE.extract_mediator(None))

    def test_no_checklist_header(self):
        self.assertIsNone(PROC_NONE.extract_mediator("Final Verdict: Supported"))

    def test_checklist_header_but_no_q_lines(self):
        self.assertIsNone(PROC_NONE.extract_mediator("Checklist:\nsome random text\nFinal Verdict: Refuted"))

    def test_empty_checklist_block(self):
        self.assertIsNone(PROC_NONE.extract_mediator("Checklist:\n\nFinal Verdict: Refuted"))


# ===========================================================================
# 3. extract_final_answer
# ===========================================================================

class TestExtractFinalAnswer(unittest.TestCase):

    def test_supported_full(self):
        self.assertEqual(PROC_NONE.extract_final_answer(COMP_2Q_OK), "Supported")

    def test_refuted_full(self):
        self.assertEqual(PROC_NONE.extract_final_answer(COMP_1Q_OK), "Refuted")

    def test_no_verdict_returns_none(self):
        self.assertIsNone(PROC_NONE.extract_final_answer("Checklist:\n- Q: X (True/False): True"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(PROC_NONE.extract_final_answer(""))

    def test_short_completion_supported(self):
        self.assertEqual(PROC_NONE.extract_final_answer("Supported", short_completion=True), "Supported")

    def test_short_completion_refuted(self):
        self.assertEqual(PROC_NONE.extract_final_answer("Refuted", short_completion=True), "Refuted")

    def test_short_completion_garbage_returns_none(self):
        self.assertIsNone(PROC_NONE.extract_final_answer("some text here", short_completion=True))

    def test_bare_verdict_without_short_flag_returns_none(self):
        # Without short_completion=True, bare "Supported" should not be accepted
        self.assertIsNone(PROC_NONE.extract_final_answer("Supported", short_completion=False))


# ===========================================================================
# 4. check_generation_format_mistakes
# ===========================================================================

class TestFormatMistakes(unittest.TestCase):

    def test_clean_start_no_mistake(self):
        self.assertFalse(PROC_NONE.check_generation_format_mistakes(COMP_2Q_OK))

    def test_clean_start_1q_no_mistake(self):
        self.assertFalse(PROC_NONE.check_generation_format_mistakes(COMP_1Q_OK))

    def test_preamble_before_checklist_is_mistake(self):
        comp = "Sure, I will help!\n\nChecklist:\n- Q: X (True/False): True\nFinal Verdict: Supported"
        self.assertTrue(PROC_NONE.check_generation_format_mistakes(comp))

    def test_no_checklist_at_all_is_mistake(self):
        self.assertTrue(PROC_NONE.check_generation_format_mistakes("Final Verdict: Refuted"))

    def test_empty_string_is_mistake(self):
        self.assertTrue(PROC_NONE.check_generation_format_mistakes(""))

    def test_none_is_mistake(self):
        self.assertTrue(PROC_NONE.check_generation_format_mistakes(None))

    def test_tail_after_verdict_is_not_mistake(self):
        comp = COMP_2Q_OK + "\n\nI hope this helps!"
        self.assertFalse(PROC_NONE.check_generation_format_mistakes(comp))


# ===========================================================================
# 5. extract_tool_args -- none mode always returns None
# ===========================================================================

class TestExtractToolArgsNone(unittest.TestCase):

    def test_none_mode_returns_none(self):
        self.assertIsNone(PROC_NONE.extract_tool_args(COMP_2Q_OK))

    def test_none_mode_returns_none_for_any_text(self):
        self.assertIsNone(PROC_NONE.extract_tool_args("anything"))


# ===========================================================================
# 6. extract_tool_args -- simple mode
# ===========================================================================

class TestExtractToolArgsSimple(unittest.TestCase):

    SIMPLE_COMP = (
        "Checklist:\n"
        "- Q: Did it happen? (True/False): True\n"
        "- Q: Was it confirmed? (True/False): False\n"
        "\n"
        "Final tool call:\n"
        "   TOOL: predict_verdict\n"
        '   ARGS: {"rubric": "- Q: Did it happen? (True/False): True\\n'
        '- Q: Was it confirmed? (True/False): False\\n"}\n'
    )

    def test_parses_dict_from_args_string(self):
        result = PROC_SIMPLE.extract_tool_args(self.SIMPLE_COMP)
        self.assertEqual(result, {"Did it happen?": True, "Was it confirmed?": False})

    def test_missing_args_block_returns_none(self):
        self.assertIsNone(PROC_SIMPLE.extract_tool_args("Checklist:\n- Q: X: True\n"))


# ===========================================================================
# 7. extract_tool_args -- structured mode
# ===========================================================================

class TestExtractToolArgsStructured(unittest.TestCase):

    def _make_comp(self, bool_list):
        return (
            "Checklist:\n"
            "- Q: Q1 (True/False): True\n"
            "Final tool call:\n"
            "   TOOL: predict_verdict\n"
            f'   ARGS: {{"rubric": {bool_list}}}\n'
        )

    def test_two_element_list(self):
        comp = (
            "Checklist:\n"
            "- Q: Q1 (True/False): True\n"
            "- Q: Q2 (True/False): False\n"
            "Final tool call:\n"
            "   TOOL: predict_verdict\n"
            '   ARGS: {"rubric": [True, False]}\n'
        )
        self.assertEqual(PROC_STRUCT.extract_tool_args(comp), [True, False])

    def test_three_element_list(self):
        comp = (
            "Final tool call:\n"
            "   TOOL: predict_verdict\n"
            '   ARGS: {"rubric": [True, True, False]}\n'
        )
        self.assertEqual(PROC_STRUCT.extract_tool_args(comp), [True, True, False])

    def test_invalid_value_in_list_returns_none(self):
        comp = '   ARGS: {"rubric": [True, "maybe"]}'
        self.assertIsNone(PROC_STRUCT.extract_tool_args(comp))

    def test_missing_args_returns_none(self):
        self.assertIsNone(PROC_STRUCT.extract_tool_args("Checklist:\n- Q: X: True\n"))


# ===========================================================================
# 8. boollist_to_checklist
# ===========================================================================

class TestBoollistToChecklist(unittest.TestCase):

    def test_2q_correct_mapping(self):
        result = PROC_STRUCT.boollist_to_checklist(SAMPLE_SUPPORTED, [False, False])
        self.assertEqual(result, GOLD_RUBRIC_2Q)

    def test_1q_flip(self):
        result = PROC_STRUCT.boollist_to_checklist(SAMPLE_REFUTED1, [True])
        self.assertEqual(result, {"Did Trump make pro-gay laws when in office?": True})

    def test_none_payload_returns_none(self):
        self.assertIsNone(PROC_STRUCT.boollist_to_checklist(SAMPLE_SUPPORTED, None))

    def test_wrong_length_returns_none(self):
        self.assertIsNone(PROC_STRUCT.boollist_to_checklist(SAMPLE_SUPPORTED, [True]))

    def test_non_bool_in_list_returns_none(self):
        self.assertIsNone(PROC_STRUCT.boollist_to_checklist(SAMPLE_SUPPORTED, [1, 0]))

    def test_empty_list_returns_none(self):
        self.assertIsNone(PROC_STRUCT.boollist_to_checklist(SAMPLE_SUPPORTED, []))


# ===========================================================================
# 9. compare_structures
# ===========================================================================

class TestCompareStructures(unittest.TestCase):

    def test_identical_returns_1(self):
        self.assertEqual(PROC_NONE.compare_structures(GOLD_RUBRIC_2Q, deepcopy(GOLD_RUBRIC_2Q)), 1)

    def test_different_values_returns_0(self):
        a = {"Q?": True}
        b = {"Q?": False}
        self.assertEqual(PROC_NONE.compare_structures(a, b), 0)

    def test_different_keys_returns_0(self):
        self.assertEqual(PROC_NONE.compare_structures({"Q1?": True}, {"Q2?": True}), 0)

    def test_different_lengths_returns_0(self):
        self.assertEqual(PROC_NONE.compare_structures({"Q1?": True}, {"Q1?": True, "Q2?": False}), 0)

    def test_none_first_arg_returns_none(self):
        self.assertIsNone(PROC_NONE.compare_structures(None, GOLD_RUBRIC_2Q))

    def test_none_second_arg_returns_none(self):
        self.assertIsNone(PROC_NONE.compare_structures(GOLD_RUBRIC_2Q, None))

    def test_both_none_returns_none(self):
        self.assertIsNone(PROC_NONE.compare_structures(None, None))


# ===========================================================================
# 10. AVeriTeCTool.validate_args
# ===========================================================================

class TestValidateArgs(unittest.TestCase):

    def test_valid_dict(self):
        self.assertTrue(TOOL_NONE.validate_args({"rubric": {"Q?": True}}))

    def test_valid_list(self):
        self.assertTrue(TOOL_NONE.validate_args({"rubric": [True, False]}))

    def test_empty_dict_returns_false(self):
        self.assertFalse(TOOL_NONE.validate_args({"rubric": {}}))

    def test_empty_list_returns_false(self):
        self.assertFalse(TOOL_NONE.validate_args({"rubric": []}))

    def test_missing_rubric_key_returns_false(self):
        self.assertFalse(TOOL_NONE.validate_args({"something_else": True}))

    def test_non_dict_args_returns_false(self):
        self.assertFalse(TOOL_NONE.validate_args([True, False]))

    def test_non_bool_in_list_returns_false(self):
        self.assertFalse(TOOL_NONE.validate_args({"rubric": [1, 0]}))

    def test_non_bool_in_dict_returns_false(self):
        self.assertFalse(TOOL_NONE.validate_args({"rubric": {"Q?": "yes"}}))

    def test_none_value_returns_false(self):
        self.assertFalse(TOOL_NONE.validate_args({"rubric": None}))


# ===========================================================================
# 11. AVeriTeCTool.calculate_score
# ===========================================================================

class TestCalculateScore(unittest.TestCase):

    def test_rubric_equals_gold_supported(self):
        self.assertEqual(
            TOOL_NONE.calculate_score({"rubric": deepcopy(GOLD_RUBRIC_2Q)}, SAMPLE_SUPPORTED),
            "Supported"
        )

    def test_rubric_equals_gold_refuted(self):
        self.assertEqual(
            TOOL_NONE.calculate_score({"rubric": deepcopy(GOLD_RUBRIC_1Q)}, SAMPLE_REFUTED1),
            "Refuted"
        )

    def test_flip_one_answer_supported_becomes_refuted(self):
        flipped = deepcopy(GOLD_RUBRIC_2Q)
        first_key = list(flipped.keys())[0]
        flipped[first_key] = True
        self.assertEqual(
            TOOL_NONE.calculate_score({"rubric": flipped}, SAMPLE_SUPPORTED),
            "Refuted"
        )

    def test_flip_single_answer_refuted_becomes_supported(self):
        flipped = {"Did Trump make pro-gay laws when in office?": True}
        self.assertEqual(
            TOOL_NONE.calculate_score({"rubric": flipped}, SAMPLE_REFUTED1),
            "Supported"
        )

    def test_list_equals_gold_supported(self):
        self.assertEqual(
            TOOL_NONE.calculate_score({"rubric": [False, False]}, SAMPLE_SUPPORTED),
            "Supported"
        )

    def test_list_differs_from_gold_refuted(self):
        self.assertEqual(
            TOOL_NONE.calculate_score({"rubric": [True, False]}, SAMPLE_SUPPORTED),
            "Refuted"
        )

    def test_invalid_args_returns_none(self):
        self.assertIsNone(TOOL_NONE.calculate_score({"rubric": []}, SAMPLE_SUPPORTED))

    def test_none_sample_returns_none(self):
        self.assertIsNone(TOOL_NONE.calculate_score({"rubric": [False, False]}, None))

    def test_wrong_list_length_returns_none(self):
        self.assertIsNone(TOOL_NONE.calculate_score({"rubric": [False]}, SAMPLE_SUPPORTED))


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# 12. extract_mediator -- additional edge cases
# ===========================================================================

class TestExtractMediatorEdgeCases(unittest.TestCase):

    def test_bullet_character_u2022(self):
        comp = (
            "Checklist:\n"
            "\u2022 Q: Was it real? (True/False): True\n"
            "\u2022 Q: Was it confirmed? (True/False): False\n"
            "Final Verdict: Supported"
        )
        result = PROC_NONE.extract_mediator(comp)
        self.assertEqual(result, {"Was it real?": True, "Was it confirmed?": False})

    def test_stops_parsing_at_final_verdict(self):
        comp = (
            "Checklist:\n"
            "- Q: Question 1? (True/False): True\n"
            "Final Verdict: Supported\n"
            "- Q: Question 2? (True/False): False\n"
        )
        result = PROC_NONE.extract_mediator(comp)
        # Question 2 appears after Final Verdict -- must be excluded
        self.assertEqual(result, {"Question 1?": True})

    def test_duplicate_question_key_last_value_wins(self):
        comp = (
            "Checklist:\n"
            "- Q: Same question? (True/False): True\n"
            "- Q: Same question? (True/False): False\n"
            "Final Verdict: Refuted"
        )
        result = PROC_NONE.extract_mediator(comp)
        # Dict insert: second value overwrites first
        self.assertIsNotNone(result)
        self.assertEqual(result.get("Same question?"), False)

    def test_extra_spaces_in_answer_field(self):
        comp = (
            "Checklist:\n"
            "- Q: Did it happen? (True/False):  True  \n"
            "Final Verdict: Supported"
        )
        result = PROC_NONE.extract_mediator(comp)
        self.assertEqual(result, {"Did it happen?": True})

    def test_question_with_internal_colon(self):
        # Question text itself contains a colon before the (True/False) hint
        comp = (
            "Checklist:\n"
            "- Q: Did the briefing on Aug 13: 2020 include this? (True/False): False\n"
            "Final Verdict: Refuted"
        )
        result = PROC_NONE.extract_mediator(comp)
        # Should parse the full question correctly (non-greedy regex + hint)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertFalse(list(result.values())[0])

    def test_apostrophe_preserved_in_question(self):
        comp = (
            "Checklist:\n"
            "- Q: Did China's MFA announce this? (True/False): False\n"
            "Final Verdict: Refuted"
        )
        result = PROC_NONE.extract_mediator(comp)
        self.assertIn("Did China's MFA announce this?", result)

    def test_mixed_case_true_false_answers(self):
        comp = (
            "Checklist:\n"
            "- Q: Q1? (True/False): TRUE\n"
            "- Q: Q2? (True/False): false\n"
            "Final Verdict: Supported"
        )
        result = PROC_NONE.extract_mediator(comp)
        self.assertEqual(result, {"Q1?": True, "Q2?": False})


# ===========================================================================
# 13. extract_final_answer -- additional edge cases
# ===========================================================================

class TestExtractFinalAnswerEdgeCases(unittest.TestCase):

    def test_dash_separator(self):
        # FINAL_VERDICT_RE allows [:\-]
        text = "Checklist:\n- Q: X (True/False): True\nFinal Verdict - Supported"
        self.assertEqual(PROC_NONE.extract_final_answer(text), "Supported")

    def test_all_caps_normalised(self):
        text = "Checklist:\n- Q: X (True/False): True\nFINAL VERDICT: SUPPORTED"
        result = PROC_NONE.extract_final_answer(text)
        self.assertEqual(result, "Supported")

    def test_refuted_all_caps_normalised(self):
        text = "FINAL VERDICT: REFUTED"
        self.assertEqual(PROC_NONE.extract_final_answer(text), "Refuted")

    def test_multiple_verdicts_first_wins(self):
        # If model hallucinates two verdicts, first one counts
        text = "Final Verdict: Supported\nFinal Verdict: Refuted"
        self.assertEqual(PROC_NONE.extract_final_answer(text), "Supported")

    def test_short_completion_with_full_pattern_still_works(self):
        # short_completion=True, but text has "Final Verdict: X" -- full pattern fires first
        text = "Final Verdict: Refuted"
        self.assertEqual(PROC_NONE.extract_final_answer(text, short_completion=True), "Refuted")

    def test_short_completion_leading_trailing_whitespace(self):
        self.assertEqual(PROC_NONE.extract_final_answer("  Supported  ", short_completion=True), "Supported")

    def test_short_completion_lowercase_normalized(self):
        self.assertEqual(PROC_NONE.extract_final_answer("supported", short_completion=True), "Supported")
        self.assertEqual(PROC_NONE.extract_final_answer("refuted", short_completion=True), "Refuted")


# ===========================================================================
# 14. extract_tool_args -- structured lowercase / edge cases
# ===========================================================================

class TestExtractToolArgsStructuredEdgeCases(unittest.TestCase):

    def test_lowercase_true_false_in_list(self):
        comp = '   ARGS: {"rubric": [true, false]}'
        result = PROC_STRUCT.extract_tool_args(comp)
        self.assertEqual(result, [True, False])

    def test_empty_list_in_args_returns_none(self):
        comp = '   ARGS: {"rubric": []}'
        result = PROC_STRUCT.extract_tool_args(comp)
        self.assertIsNone(result)

    def test_list_with_spaces(self):
        comp = '   ARGS: {"rubric": [ True , False , True ]}'
        result = PROC_STRUCT.extract_tool_args(comp)
        self.assertEqual(result, [True, False, True])


# ===========================================================================
# 15. compare_structures -- edge cases with wrong types
# ===========================================================================

class TestCompareStructuresEdgeCases(unittest.TestCase):

    def test_string_bool_vs_real_bool_is_mismatch(self):
        a = {"Q?": True}
        b = {"Q?": "True"}   # string, not bool
        self.assertEqual(PROC_NONE.compare_structures(a, b), 0)

    def test_empty_dicts_are_equal(self):
        self.assertEqual(PROC_NONE.compare_structures({}, {}), 1)

    def test_subset_dict_is_mismatch(self):
        a = {"Q1?": True, "Q2?": False}
        b = {"Q1?": True}
        self.assertEqual(PROC_NONE.compare_structures(a, b), 0)


# ===========================================================================
# 16. calculate_score -- list with same values but wrong length
# ===========================================================================

class TestCalculateScoreEdgeCases(unittest.TestCase):

    def test_list_one_shorter_returns_none(self):
        # 2Q sample but list of length 1
        self.assertIsNone(TOOL_NONE.calculate_score({"rubric": [False]}, SAMPLE_SUPPORTED))

    def test_list_one_longer_returns_none(self):
        self.assertIsNone(TOOL_NONE.calculate_score({"rubric": [False, False, False]}, SAMPLE_SUPPORTED))

    def test_all_questions_flipped_supported_becomes_refuted(self):
        flipped = {k: not v for k, v in GOLD_RUBRIC_2Q.items()}
        self.assertEqual(TOOL_NONE.calculate_score({"rubric": flipped}, SAMPLE_SUPPORTED), "Refuted")

    def test_partial_match_dict_still_refuted(self):
        # Only one answer matches, overall rubric ≠ gold → flip
        partial = deepcopy(GOLD_RUBRIC_2Q)
        partial[list(partial.keys())[1]] = True  # flip second
        self.assertEqual(TOOL_NONE.calculate_score({"rubric": partial}, SAMPLE_SUPPORTED), "Refuted")
