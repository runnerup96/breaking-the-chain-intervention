import random
import unittest
from copy import deepcopy

from datasets_for_intervention.ricechem_structure_processor import RiceChemStructureProcessor, RiceChemTool
from datasets_for_intervention.test_intervention.ricechem_mocks import RiceChemDatasetMock


class TestRiceChemProcessorAndTool(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()

        self.proc_none   = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        self.proc_simple = RiceChemStructureProcessor(self.dataset, tool_mode="simple")
        self.proc_struct = RiceChemStructureProcessor(self.dataset, tool_mode="structured")

        self.tool_simple = RiceChemTool(self.dataset, tool_mode="simple")
        self.tool_struct = RiceChemTool(self.dataset, tool_mode="structured")

        self.sample = deepcopy(self.dataset[0])

    # ------------------------------------------------------------------
    # extract_final_answer
    # ------------------------------------------------------------------

    def test_extract_final_answer_full_completion(self):
        s = "Checklist:\nA item (True/False): True\nFinal grade (0-3): 2.5\n"
        self.assertEqual(self.proc_none.extract_final_answer(s, short_completion=False), 2.5)

        s = "Some text\nFinal grade: -1.25 pts\nOther text"
        self.assertEqual(self.proc_none.extract_final_answer(s, short_completion=False), -1.25)

    def test_extract_final_answer_tail_only(self):
        self.assertEqual(self.proc_none.extract_final_answer(" 7.5 ", short_completion=True), 7.5)
        self.assertIsNone(self.proc_none.extract_final_answer("7.5 pts", short_completion=True))
        self.assertEqual(self.proc_none.extract_final_answer("Final grade: 7.5", short_completion=True), 7.5)

    # ------------------------------------------------------------------
    # extract_mediator
    # ------------------------------------------------------------------

    def test_extract_mediator_basic(self):
        completion = (
            "Checklist:\n"
            "A item (True/False): True\n"
            "B item (True/False): False\n"
            "C item (True/False): True\n"
            "Final grade (0-3): 2.0\n"
        )
        m = self.proc_none.extract_mediator(completion)
        self.assertEqual(m, {"A item": True, "B item": False, "C item": True})

    def test_extract_mediator_stops_before_tool_block(self):
        completion = (
            "Checklist:\n"
            "A item (True/False): True\n"
            "B item (True/False): False\n"
            "Final tool call:\n"
            "TOOL: calculate_score\n"
            'ARGS: {"rubric": [True, False]}\n'
        )
        m = self.proc_none.extract_mediator(completion)
        self.assertEqual(m, {"A item": True, "B item": False})

    def test_extract_mediator_returns_none_when_missing(self):
        self.assertIsNone(self.proc_none.extract_mediator("No checklist here."))

    def test_extract_mediator_parses_separator_variants(self):
        completion = (
            "Checklist:\n"
            "A item (True/False) - True\n"
            "B item (True/False): False\n"
            "Final grade: 1\n"
        )
        m = self.proc_none.extract_mediator(completion)
        self.assertEqual(m, {"A item": True, "B item": False})

    def test_extract_mediator_strips_quotes_in_question(self):
        completion = (
            "Checklist:\n"
            '"A item" (True/False): True\n'
            "'B item' (True/False): False\n"
            "Final grade: 1\n"
        )
        m = self.proc_none.extract_mediator(completion)
        self.assertEqual(m, {"A item": True, "B item": False})

    def test_extract_mediator_header_case_insensitive(self):
        completion = (
            "cHeCkLiSt:\n"
            "A item (True/False): True\n"
            "B item (True/False): False\n"
            "Final grade: 1\n"
        )
        m = self.proc_none.extract_mediator(completion)
        self.assertEqual(m, {"A item": True, "B item": False})

    def test_extract_mediator_accepts_missing_options_parentheses(self):
        completion = (
            "Checklist:\n"
            "A item: True\n"
            "B item - False\n"
            "Final grade: 1\n"
        )
        m = self.proc_none.extract_mediator(completion)
        self.assertEqual(m, {"A item": True, "B item": False})

    def test_extract_mediator_accepts_weight_and_weird_options(self):
        completion = (
            "Checklist:\n"
            "A item (weight: 1.0) (TRUE|FALSE): True\n"
            "B item (weight: -2) (True,False): False\n"
            "Final grade: 1\n"
        )
        m = self.proc_none.extract_mediator(completion)
        self.assertEqual(m, {"A item": True, "B item": False})

    def test_extract_mediator_duplicate_keys_last_wins(self):
        completion = (
            "Checklist:\n"
            "A item (True/False): True\n"
            "A item (True/False): False\n"
            "Final grade: 0\n"
        )
        m = self.proc_none.extract_mediator(completion)
        self.assertEqual(m, {"A item": False})

    def test_extract_mediator_ignores_lines_with_trailing_garbage(self):
        completion = (
            "Checklist:\n"
            "A item (True/False): True extra\n"
            "B item (True/False): False # comment\n"
            "Final grade: 0\n"
        )
        m = self.proc_none.extract_mediator(completion)
        self.assertIsNone(m)

        completion2 = (
            "Checklist:\n"
            "A item (True/False): True\n"
            "B item (True/False): False # comment\n"
            "Final grade: 0\n"
        )
        m2 = self.proc_none.extract_mediator(completion2)
        self.assertEqual(m2, {"A item": True})

    # ------------------------------------------------------------------
    # extract_tool_args — simple
    # ------------------------------------------------------------------

    def test_extract_tool_args_simple_from_full_completion(self):
        completion = (
            "Checklist:\n"
            "A item (True/False): True\n"
            "B item (True/False): False\n"
            "Final tool call:\n"
            "TOOL: calculate_score\n"
            'ARGS: {"rubric": "A item (True/False): True\\nB item (True/False): False"}\n'
        )
        args = self.proc_simple.extract_tool_args(completion, short_completion=False)
        self.assertEqual(args, {"A item": True, "B item": False})

    def test_extract_tool_args_simple_handles_double_braces_and_escaped_newlines(self):
        completion = (
            "TOOL: calculate_score\n"
            'ARGS: {{"rubric": "A item (True/False): True\\r\\nB item (True/False): False\\n"}}\n'
        )
        args = self.proc_simple.extract_tool_args(completion, short_completion=False)
        self.assertEqual(args, {"A item": True, "B item": False})

    def test_extract_tool_args_simple_tail_only_block(self):
        completion = '{"rubric": "A item (True/False): True\\nB item (True/False): False"}'
        args = self.proc_simple.extract_tool_args(completion, short_completion=True)
        self.assertEqual(args, {"A item": True, "B item": False})

    def test_extract_tool_args_simple_invalid_missing_rubric(self):
        completion = 'TOOL: calculate_score\nARGS: {"nope": "x"}'
        self.assertIsNone(self.proc_simple.extract_tool_args(completion, short_completion=False))

    def test_extract_tool_args_simple_allows_extra_json_fields(self):
        completion = (
            "TOOL: calculate_score\n"
            'ARGS: {"rubric": "A item (True/False): True\\nB item (True/False): False", "foo": 1}\n'
        )
        args = self.proc_simple.extract_tool_args(completion, short_completion=False)
        self.assertEqual(args, {"A item": True, "B item": False})

    def test_extract_tool_args_simple_handles_double_escaping_and_backslashes(self):
        completion = (
            "TOOL: calculate_score\n"
            'ARGS: {"rubric": "A item (True/False): True\\\\nB item (True/False): False\\\\n"}\n'
        )
        args = self.proc_simple.extract_tool_args(completion, short_completion=False)
        self.assertEqual(args, {"A item": True, "B item": False})

    def test_extract_tool_args_simple_rejects_empty_rubric_string(self):
        completion = 'TOOL: calculate_score\nARGS: {"rubric": ""}\n'
        self.assertIsNone(self.proc_simple.extract_tool_args(completion, short_completion=False))

    # ------------------------------------------------------------------
    # extract_tool_args — structured
    # ------------------------------------------------------------------

    def test_extract_tool_args_structured_parses_case_and_trailing_comma(self):
        completion = (
            "TOOL: calculate_score\n"
            'ARGS: {"rubric": [true, FALSE, True ,false,]}\n'
        )
        args = self.proc_struct.extract_tool_args(completion, short_completion=False)
        self.assertEqual(args, [True, False, True, False])

    def test_extract_tool_args_structured_tail_only_block(self):
        completion = '{"rubric": [True, False, True]}'
        args = self.proc_struct.extract_tool_args(completion, short_completion=True)
        self.assertEqual(args, [True, False, True])

    def test_extract_tool_args_structured_rejects_non_bool_tokens(self):
        completion = 'TOOL: calculate_score\nARGS: {"rubric": [True, null, False]}'
        self.assertIsNone(self.proc_struct.extract_tool_args(completion, short_completion=False))

    def test_extract_tool_args_structured_accepts_linebreaks_spaces_and_trailing_comma(self):
        completion = (
            "TOOL: calculate_score\n"
            'ARGS: {"rubric": [\n  TRUE,\n false,\n True,\n]}\n'
        )
        args = self.proc_struct.extract_tool_args(completion, short_completion=False)
        self.assertEqual(args, [True, False, True])

    def test_extract_tool_args_structured_rejects_empty_list(self):
        completion = 'TOOL: calculate_score\nARGS: {"rubric": []}\n'
        self.assertIsNone(self.proc_struct.extract_tool_args(completion, short_completion=False))

    def test_extract_tool_args_allows_code_fence_block(self):
        completion = (
            "TOOL: calculate_score\n"
            "ARGS: ```json\n"
            '{"rubric": [True, False, True]}\n'
            "```\n"
        )
        args = self.proc_struct.extract_tool_args(completion, short_completion=False)
        self.assertEqual(args, [True, False, True])

    # ------------------------------------------------------------------
    # boollist_to_checklist
    # ------------------------------------------------------------------

    def test_boollist_to_checklist_uses_gold_key_order(self):
        payload = [False, True, False]
        out = self.proc_struct.boollist_to_checklist(self.sample, payload)
        self.assertEqual(out, {"A item": False, "B item": True, "C item": False})

    def test_boollist_to_checklist_rejects_bad_payload(self):
        self.assertIsNone(self.proc_struct.boollist_to_checklist(self.sample, []))
        self.assertIsNone(self.proc_struct.boollist_to_checklist(self.sample, [True, False]))  # wrong length
        self.assertIsNone(self.proc_struct.boollist_to_checklist(self.sample, [True, 0, False]))  # 0 not bool

    def test_boollist_to_checklist_falls_back_to_weights_when_gold_missing(self):
        s = deepcopy(self.sample)
        s["gold_rubric"] = None
        payload = [True, False, True]
        out = self.proc_struct.boollist_to_checklist(s, payload)
        self.assertEqual(out, {"A item": True, "B item": False, "C item": True})

    # ------------------------------------------------------------------
    # compare_structures
    # ------------------------------------------------------------------

    def test_compare_structures_match_mismatch_none(self):
        a = {"A": True,  "B": False}
        b = {"A": True,  "B": False}
        c = {"A": True,  "B": True}
        d = {"A": True}

        self.assertEqual(self.proc_none.compare_structures(a, b), 1)
        self.assertEqual(self.proc_none.compare_structures(a, c), 0)
        self.assertEqual(self.proc_none.compare_structures(a, d), 0)
        self.assertIsNone(self.proc_none.compare_structures(None, b))

    def test_compare_structures_extra_key_is_mismatch(self):
        a = {"A": True}
        b = {"A": True, "B": False}
        self.assertEqual(self.proc_none.compare_structures(a, b), 0)

    # ------------------------------------------------------------------
    # RiceChemTool — validate_args
    # ------------------------------------------------------------------

    def test_tool_validate_args(self):
        self.assertFalse(self.tool_simple.validate_args("nope"))
        self.assertFalse(self.tool_simple.validate_args({}))
        self.assertFalse(self.tool_simple.validate_args({"rubric": {}}))
        self.assertFalse(self.tool_simple.validate_args({"rubric": {"A": "True"}}))
        self.assertTrue(self.tool_simple.validate_args({"rubric": {"A": True}}))

        self.assertFalse(self.tool_struct.validate_args({"rubric": []}))
        self.assertFalse(self.tool_struct.validate_args({"rubric": [True, 0]}))
        self.assertTrue(self.tool_struct.validate_args({"rubric": [True, False]}))

        # Дополнительные проверки типов
        self.assertFalse(self.tool_simple.validate_args({"rubric": []}))
        self.assertTrue(self.tool_struct.validate_args({"rubric": {"A": True}}))
        self.assertFalse(self.tool_struct.validate_args({"rubric": (True, False)}))

    # ------------------------------------------------------------------
    # RiceChemTool — calculate_score
    # ------------------------------------------------------------------

    def test_tool_calculate_score_simple_weighted_and_ignores_unknown_keys(self):
        weights = self.dataset.task2rubric_weights[1]
        rubric = {"A item": True, "B item": False, "C item": True, "UNKNOWN": True}
        expected = float(weights["A item"] + weights["C item"])
        got = self.tool_simple.calculate_score({"rubric": rubric}, {"task_idx": 1})
        self.assertEqual(got, expected)

    def test_tool_calculate_score_structured_uses_gold_key_order(self):
        sample_meta = deepcopy(self.sample)
        sample_meta["gold_rubric"] = {"B item": False, "A item": True, "C item": True}
        payload = [True, False, True]  # B=True, A=False, C=True

        weights = self.dataset.task2rubric_weights[1]
        # B item (weight 1.5) True + C item (weight 0.5) True = 2.0
        expected = float(weights["B item"] + weights["C item"])

        got = self.tool_struct.calculate_score({"rubric": payload}, sample_meta)
        self.assertEqual(got, expected)

    def test_tool_calculate_score_structured_length_mismatch_is_none(self):
        got = self.tool_struct.calculate_score({"rubric": [True, False]}, self.sample)
        self.assertIsNone(got)

    def test_calculate_score_simple_returns_none_on_empty_dict(self):
        self.assertIsNone(self.tool_simple.calculate_score({"rubric": {}}, {"task_idx": 1}))

    def test_calculate_score_structured_returns_none_on_empty_list(self):
        self.assertIsNone(self.tool_struct.calculate_score({"rubric": []}, self.sample))

    def test_calculate_score_structured_uses_weights_order_when_gold_missing(self):
        s = deepcopy(self.sample)
        s["gold_rubric"] = None
        payload = [True, False, True]
        weights = self.dataset.task2rubric_weights[1]
        expected = float(weights["A item"] + weights["C item"])
        got = self.tool_struct.calculate_score({"rubric": payload}, s)
        self.assertEqual(got, expected)

    # ------------------------------------------------------------------
    # Fuzz roundtrip
    # ------------------------------------------------------------------

    def test_roundtrip_parse_checklist_block_fuzz(self):
        random.seed(0)
        keys = ["A item", "B item", "C item"]
        for _ in range(50):
            d = {k: bool(random.getrandbits(1)) for k in keys}
            lines = []
            for k in keys:
                sep = random.choice([":", "-", " : ", " - "])
                bool_txt = random.choice(
                    ["True" if d[k] else "False",
                     "TRUE" if d[k] else "false"]
                )
                if random.getrandbits(1):
                    lines.append(f"{k} (weight: 1.0) (True/False){sep}{bool_txt}")
                else:
                    lines.append(f"{k}{sep}{bool_txt}")
            completion = "Checklist:\n" + "\n".join(lines) + "\nFinal grade: 0\n"
            got = self.proc_none.extract_mediator(completion)
            self.assertEqual(got, d)