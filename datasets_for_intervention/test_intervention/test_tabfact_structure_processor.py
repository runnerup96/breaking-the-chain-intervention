"""
test_tabfact_capture.py
~~~~~~~~~~~~~~~~~~~~~~~
Tests for TabFactStructureProcessor and TabFactTool.

Analogous to test_avritec_capture.py for AVeriTeC:
  - parsing / regex layer (extract_mediator, extract_final_answer, extract_tool_args)
  - validation (check_generation_format_mistakes, compare_structures)
  - column/value extraction and set_match
  - tool validate_args, calculate_score

All tests use real processor and tool instances backed by TabFactEngine.
"""

import unittest
import pandas as pd

from datasets_for_intervention.tabfact_dsl_engine import TabFactEngine
from datasets_for_intervention.tabfact_structure_processor import TabFactStructureProcessor, TabFactTool


# -----------------------------------------------------------------------
# Shared table fixture
# -----------------------------------------------------------------------
_TABLE_CSV = (
    "rank#athlete#nation#gold#silver#bronze\n"
    "1#Usain Bolt#Jamaica#8#0#1\n"
    "2#Shawn Crawford#United States#1#2#0\n"
    "3#Carl Lewis#United States#9#1#0"
)

_TABLE_WITH_TIME = (
    "rank#athlete#nation#gold#silver#bronze#time\n"
    "1#Usain Bolt#Jamaica#8#0#1#9.63\n"
    "2#Shawn Crawford#United States#1#2#0#19.79\n"
    "3#Carl Lewis#United States#9#1#0#8.87"
)

_SAMPLE = {"table_html_csv": _TABLE_CSV}
_SAMPLE_TIME = {"table_html_csv": _TABLE_WITH_TIME}


# -----------------------------------------------------------------------
# Fixtures: full completions (non-tool mode)
# -----------------------------------------------------------------------

COMP_TRUE = (
    "Verifier Query: greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
    "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True\n"
    "Execution Result: True\n"
)

COMP_FALSE = (
    "Verifier Query: eq{hop{argmax{all_rows; gold}; athlete}; Usain Bolt}=True\n"
    "Execution Result: False\n"
)

COMP_NO_QUERY = "Execution Result: True\n"

COMP_PREAMBLE = (
    "Let me think about this...\n"
    "Verifier Query: eq{1; 1}=True\n"
    "Execution Result: True\n"
)

COMP_INVALID_SUFFIX = (
    "Verifier Query: greater{1; 2}\n"  # missing =True/=False
    "Execution Result: True\n"
)

COMP_TOOL = (
    "Verifier Query: eq{count{all_rows}; 3}=True\n"
    "Final tool call:\n"
    "TOOL: check_query\n"
    'ARGS: {"query": "eq{count{all_rows}; 3}=True"}\n'
)

COMP_TOOL_MARKDOWN = (
    "Verifier Query: eq{count{all_rows}; 3}=True\n"
    "Final tool call:\n"
    "TOOL: check_query\n"
    "ARGS: ```json\n"
    '{"query": "eq{count{all_rows}; 3}=True"}\n'
    "```\n"
)


# =======================================================================
# TabFactStructureProcessor — extract_mediator
# =======================================================================

class TestExtractMediator(unittest.TestCase):
    def setUp(self):
        engine = TabFactEngine()
        self.proc = TabFactStructureProcessor(engine)

    def test_extracts_valid_query(self):
        q = self.proc.extract_mediator(COMP_TRUE)
        self.assertEqual(
            q,
            "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
            "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
        )

    def test_extracts_false_suffix(self):
        text = "Verifier Query: eq{1; 2}=False\nExecution Result: False\n"
        q = self.proc.extract_mediator(text)
        self.assertEqual(q, "eq{1; 2}=False")

    def test_returns_none_when_no_verifier_query_line(self):
        self.assertIsNone(self.proc.extract_mediator(COMP_NO_QUERY))

    def test_returns_none_for_invalid_suffix(self):
        self.assertIsNone(self.proc.extract_mediator(COMP_INVALID_SUFFIX))

    def test_returns_none_for_empty_string(self):
        self.assertIsNone(self.proc.extract_mediator(""))
        self.assertIsNone(self.proc.extract_mediator(None))

    def test_case_insensitive_header(self):
        text = "verifier query: eq{1; 1}=True\nExecution Result: True"
        q = self.proc.extract_mediator(text)
        self.assertEqual(q, "eq{1; 1}=True")

    def test_ignores_preamble_when_extracting(self):
        # extract_mediator does not validate preamble — it just finds the first match
        q = self.proc.extract_mediator(COMP_PREAMBLE)
        self.assertEqual(q, "eq{1; 1}=True")

    def test_multiline_completion_returns_first_match(self):
        text = (
            "Verifier Query: eq{count{all_rows}; 3}=True\n"
            "Execution Result: True\n"
            "Verifier Query: eq{count{all_rows}; 99}=False\n"
        )
        q = self.proc.extract_mediator(text)
        self.assertEqual(q, "eq{count{all_rows}; 3}=True")

    def test_complex_nested_query(self):
        query = (
            "and{greater{hop{filter_eq{all_rows; nation; Jamaica}; gold}; 5}; "
            "less{count{filter_eq{all_rows; nation; United States}}; 3}}=True"
        )
        text = f"Verifier Query: {query}\nExecution Result: True"
        self.assertEqual(self.proc.extract_mediator(text), query)


# =======================================================================
# TabFactStructureProcessor — extract_final_answer
# =======================================================================

class TestExtractFinalAnswer(unittest.TestCase):
    def setUp(self):
        engine = TabFactEngine()
        self.proc = TabFactStructureProcessor(engine)

    def test_extracts_true(self):
        self.assertTrue(self.proc.extract_final_answer(COMP_TRUE))

    def test_extracts_false(self):
        self.assertFalse(self.proc.extract_final_answer(COMP_FALSE))

    def test_returns_none_when_missing(self):
        self.assertIsNone(self.proc.extract_final_answer("Verifier Query: eq{1;1}=True"))
        self.assertIsNone(self.proc.extract_final_answer(""))
        self.assertIsNone(self.proc.extract_final_answer(None))

    def test_case_insensitive_execution_result(self):
        self.assertTrue(self.proc.extract_final_answer("execution result: TRUE"))
        self.assertFalse(self.proc.extract_final_answer("execution result: false"))

    def test_short_completion_true(self):
        self.assertTrue(self.proc.extract_final_answer("True", short_completion=True))
        self.assertTrue(self.proc.extract_final_answer("True.", short_completion=True))
        self.assertTrue(self.proc.extract_final_answer("True!", short_completion=True))

    def test_short_completion_false(self):
        self.assertFalse(self.proc.extract_final_answer("False", short_completion=True))
        self.assertFalse(self.proc.extract_final_answer("False.", short_completion=True))
        self.assertFalse(self.proc.extract_final_answer("False!", short_completion=True))

    def test_short_completion_none_on_garbage(self):
        self.assertIsNone(self.proc.extract_final_answer("Maybe", short_completion=True))
        self.assertIsNone(self.proc.extract_final_answer("", short_completion=True))

    def test_full_mode_takes_priority_over_short(self):
        # Even if text looks like a short completion, full mode finds the line
        text = "True\nExecution Result: False\n"
        self.assertFalse(self.proc.extract_final_answer(text, short_completion=False))

    def test_short_completion_ignores_line_format(self):
        # short_completion=True: just bare "True"/"False", NOT "Execution Result: True"
        text = "Execution Result: True"
        self.assertTrue(self.proc.extract_final_answer(text, short_completion=True))  # line found first


# =======================================================================
# TabFactStructureProcessor — extract_tool_args
# =======================================================================

class TestExtractToolArgs(unittest.TestCase):
    def setUp(self):
        engine = TabFactEngine()
        self.proc = TabFactStructureProcessor(engine)

    def test_extracts_valid_args(self):
        result = self.proc.extract_tool_args(COMP_TOOL)
        self.assertEqual(result, {"query": "eq{count{all_rows}; 3}=True"})

    def test_extracts_args_from_markdown_fence(self):
        result = self.proc.extract_tool_args(COMP_TOOL_MARKDOWN)
        self.assertEqual(result, {"query": "eq{count{all_rows}; 3}=True"})

    def test_returns_none_when_no_args_block(self):
        self.assertIsNone(self.proc.extract_tool_args("No tool block here"))
        self.assertIsNone(self.proc.extract_tool_args(""))
        self.assertIsNone(self.proc.extract_tool_args(None))

    def test_returns_none_for_invalid_query_suffix(self):
        bad = 'ARGS: {"query": "eq{1; 2}"}\n'  # no =True/=False
        self.assertIsNone(self.proc.extract_tool_args(bad))

    def test_short_completion_mode(self):
        # In short mode the text IS the ARGS JSON itself
        text = '{"query": "eq{count{all_rows}; 3}=True"}'
        result = self.proc.extract_tool_args(text, short_completion=True)
        self.assertEqual(result, {"query": "eq{count{all_rows}; 3}=True"})

    def test_short_completion_garbage_returns_none(self):
        self.assertIsNone(self.proc.extract_tool_args("not json", short_completion=True))

    def test_complex_query_in_args(self):
        query = (
            "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
            "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
        )
        text = f'ARGS: {{"query": "{query}"}}\n'
        result = self.proc.extract_tool_args(text)
        self.assertEqual(result, {"query": query})


# =======================================================================
# TabFactStructureProcessor — compare_structures
# =======================================================================

class TestCompareStructures(unittest.TestCase):
    def setUp(self):
        engine = TabFactEngine()
        self.proc = TabFactStructureProcessor(engine)

    def test_identical_strings_return_1(self):
        q = "eq{count{all_rows}; 3}=True"
        self.assertEqual(self.proc.compare_structures(q, q), 1)

    def test_different_strings_return_0(self):
        a = "eq{count{all_rows}; 3}=True"
        b = "eq{count{all_rows}; 4}=True"
        self.assertEqual(self.proc.compare_structures(a, b), 0)

    def test_operator_difference_returns_0(self):
        a = "greater{1; 2}=True"
        b = "less{1; 2}=True"
        self.assertEqual(self.proc.compare_structures(a, b), 0)

    def test_suffix_difference_returns_0(self):
        a = "eq{1; 1}=True"
        b = "eq{1; 1}=False"
        self.assertEqual(self.proc.compare_structures(a, b), 0)

    def test_none_first_arg_returns_none(self):
        self.assertIsNone(self.proc.compare_structures(None, "eq{1;1}=True"))

    def test_none_second_arg_returns_none(self):
        self.assertIsNone(self.proc.compare_structures("eq{1;1}=True", None))

    def test_both_none_returns_none(self):
        self.assertIsNone(self.proc.compare_structures(None, None))


# =======================================================================
# TabFactStructureProcessor — check_generation_format_mistakes
# =======================================================================

class TestCheckGenerationFormatMistakes(unittest.TestCase):
    def setUp(self):
        engine = TabFactEngine()
        self.proc = TabFactStructureProcessor(engine)

    def test_clean_completion_returns_false(self):
        # Starts with "Verifier Query:" — no mistake
        self.assertFalse(self.proc.check_generation_format_mistakes(COMP_TRUE))

    def test_preamble_returns_true(self):
        self.assertTrue(self.proc.check_generation_format_mistakes(COMP_PREAMBLE))

    def test_empty_returns_true(self):
        self.assertTrue(self.proc.check_generation_format_mistakes(""))
        self.assertTrue(self.proc.check_generation_format_mistakes(None))

    def test_only_whitespace_returns_true(self):
        self.assertTrue(self.proc.check_generation_format_mistakes("   \n  "))

    def test_execution_result_only_returns_true(self):
        self.assertTrue(self.proc.check_generation_format_mistakes("Execution Result: True"))

    def test_case_insensitive_header_accepted(self):
        text = "verifier query: eq{1;1}=True\nExecution Result: True"
        self.assertFalse(self.proc.check_generation_format_mistakes(text))

    def test_trailing_garbage_after_result_ok(self):
        text = "Verifier Query: eq{1;1}=True\nExecution Result: True\nSome extra text"
        self.assertFalse(self.proc.check_generation_format_mistakes(text))


# =======================================================================
# TabFactStructureProcessor — extract_columns_values
# =======================================================================

class TestExtractColumnsValues(unittest.TestCase):
    def setUp(self):
        engine = TabFactEngine()
        self.proc = TabFactStructureProcessor(engine)

    def test_filter_eq_extracts_column_and_value(self):
        q = "eq{count{filter_eq{all_rows; athlete; Usain Bolt}}; 1}=True"
        cols, vals = self.proc.extract_columns_values(q)
        self.assertIn("athlete", cols)
        self.assertIn("usain bolt", vals)

    def test_hop_extracts_column(self):
        q = "eq{hop{filter_eq{all_rows; nation; Jamaica}; gold}; 8}=True"
        cols, vals = self.proc.extract_columns_values(q)
        self.assertIn("nation", cols)
        self.assertIn("gold", cols)
        self.assertIn("jamaica", vals)

    def test_numeric_values_excluded_from_vals(self):
        q = "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; 5}=True"
        cols, vals = self.proc.extract_columns_values(q)
        # "5" is numeric -> should NOT be in vals
        self.assertNotIn("5", vals)
        self.assertIn("usain bolt", vals)

    def test_argmax_extracts_column(self):
        q = "eq{hop{argmax{all_rows; gold}; athlete}; Carl Lewis}=True"
        cols, vals = self.proc.extract_columns_values(q)
        # gold is argmax field (col), athlete is hop field (col)
        self.assertIn("gold", cols)
        self.assertIn("athlete", cols)
        # "Carl Lewis" is the eq comparison literal, NOT a filter value -> may or may not be extracted
        # The test verifies columns are captured, not that eq args are treated as values

    def test_all_rows_not_in_columns(self):
        q = "eq{count{filter_eq{all_rows; nation; Jamaica}}; 1}=True"
        cols, vals = self.proc.extract_columns_values(q)
        self.assertNotIn("all_rows", cols)
        self.assertNotIn("all_rows", vals)

    def test_multiple_filter_eq(self):
        q = (
            "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
            "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
        )
        cols, vals = self.proc.extract_columns_values(q)
        self.assertIn("athlete", cols)
        self.assertIn("gold", cols)
        self.assertIn("usain bolt", vals)
        self.assertIn("shawn crawford", vals)

    def test_unparseable_query_returns_empty_sets(self):
        cols, vals = self.proc.extract_columns_values("not_a_valid_query")
        self.assertEqual(cols, set())
        self.assertEqual(vals, set())

    def test_empty_query_returns_empty_sets(self):
        cols, vals = self.proc.extract_columns_values("")
        self.assertEqual(cols, set())
        self.assertEqual(vals, set())


# =======================================================================
# TabFactStructureProcessor — set_match
# =======================================================================

class TestSetMatch(unittest.TestCase):
    def setUp(self):
        engine = TabFactEngine()
        self.proc = TabFactStructureProcessor(engine)

    def test_identical_queries_return_1(self):
        q = "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; 5}=True"
        self.assertEqual(self.proc.set_match(q, q), 1)

    def test_operator_only_difference_returns_1(self):
        # greater vs less — same columns (athlete, gold) and values (usain bolt)
        a = "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; 5}=True"
        b = "less{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; 5}=True"
        self.assertEqual(self.proc.set_match(a, b), 1)

    def test_suffix_only_difference_returns_1(self):
        a = "eq{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; 8}=True"
        b = "eq{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; 8}=False"
        self.assertEqual(self.proc.set_match(a, b), 1)

    def test_value_difference_returns_0(self):
        a = "eq{count{filter_eq{all_rows; athlete; Usain Bolt}}; 1}=True"
        b = "eq{count{filter_eq{all_rows; athlete; Carl Lewis}}; 1}=True"
        self.assertEqual(self.proc.set_match(a, b), 0)

    def test_column_difference_returns_0(self):
        a = "eq{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; 8}=True"
        b = "eq{hop{filter_eq{all_rows; athlete; Usain Bolt}; silver}; 8}=True"
        self.assertEqual(self.proc.set_match(a, b), 0)

    def test_none_first_returns_none(self):
        self.assertIsNone(self.proc.set_match(None, "eq{1;1}=True"))

    def test_none_second_returns_none(self):
        self.assertIsNone(self.proc.set_match("eq{1;1}=True", None))

    def test_both_none_returns_none(self):
        self.assertIsNone(self.proc.set_match(None, None))

    def test_empty_sets_both_sides_fallback_to_string_match(self):
        # Both queries have no identifiable columns/values -> fallback to exact string match
        a = "eq{1; 1}=True"
        b = "eq{1; 1}=True"
        self.assertEqual(self.proc.set_match(a, b), 1)

        c = "eq{1; 1}=True"
        d = "eq{1; 2}=True"
        self.assertEqual(self.proc.set_match(c, d), 0)


# =======================================================================
# TabFactTool — validate_args
# =======================================================================

class TestTabFactToolValidateArgs(unittest.TestCase):
    def setUp(self):
        engine = TabFactEngine()
        self.tool = TabFactTool(engine)

    def test_valid_true_suffix(self):
        self.assertTrue(self.tool.validate_args({"query": "eq{1; 1}=True"}))

    def test_valid_false_suffix(self):
        self.assertTrue(self.tool.validate_args({"query": "eq{1; 2}=False"}))

    def test_missing_suffix_returns_false(self):
        self.assertFalse(self.tool.validate_args({"query": "eq{1; 1}"}))

    def test_empty_query_returns_false(self):
        self.assertFalse(self.tool.validate_args({"query": ""}))

    def test_missing_query_key_returns_false(self):
        self.assertFalse(self.tool.validate_args({}))

    def test_not_a_dict_returns_false(self):
        self.assertFalse(self.tool.validate_args("eq{1;1}=True"))
        self.assertFalse(self.tool.validate_args(None))
        self.assertFalse(self.tool.validate_args(42))

    def test_wrong_suffix_returns_false(self):
        self.assertFalse(self.tool.validate_args({"query": "eq{1;1}=Maybe"}))

    def test_complex_valid_query(self):
        q = (
            "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
            "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
        )
        self.assertTrue(self.tool.validate_args({"query": q}))


# =======================================================================
# TabFactTool — calculate_score
# =======================================================================

class TestTabFactToolCalculateScore(unittest.TestCase):
    def setUp(self):
        engine = TabFactEngine()
        self.tool = TabFactTool(engine)

    # ---- happy-path ----

    def test_true_result_when_assertion_holds(self):
        # Usain Bolt gold=8 > Shawn Crawford gold=1 → True, =True → final=True
        args = {
            "query": (
                "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
                "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
            )
        }
        result = self.tool.calculate_score(args, _SAMPLE)
        self.assertTrue(result)

    def test_false_result_when_assertion_fails(self):
        # suffix =False: gold assertion is True so final=False
        args = {
            "query": (
                "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
                "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=False"
            )
        }
        result = self.tool.calculate_score(args, _SAMPLE)
        self.assertFalse(result)

    def test_count_all_rows(self):
        args = {"query": "eq{count{all_rows}; 3}=True"}
        result = self.tool.calculate_score(args, _SAMPLE)
        self.assertTrue(result)

    def test_argmax_gold_is_carl_lewis(self):
        # Carl Lewis has gold=9, highest
        args = {"query": "eq{hop{argmax{all_rows; gold}; athlete}; Carl Lewis}=True"}
        result = self.tool.calculate_score(args, _SAMPLE)
        self.assertTrue(result)

    def test_avg_query_on_time_table(self):
        # avg(19.79, 8.87) = 14.33 < 15
        args = {"query": "less{avg{filter_eq{all_rows; nation; United States}; time}; 15}=True"}
        result = self.tool.calculate_score(args, _SAMPLE_TIME)
        self.assertTrue(result)

    def test_query_evaluates_to_false_correctly(self):
        # Shawn Crawford gold=1, NOT > Usain Bolt gold=8
        args = {
            "query": (
                "greater{hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}; "
                "hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}}=True"
            )
        }
        result = self.tool.calculate_score(args, _SAMPLE)
        self.assertFalse(result)

    def test_flipped_suffix_inverts_result(self):
        q_true = "eq{count{all_rows}; 3}=True"
        q_false = "eq{count{all_rows}; 3}=False"
        self.assertTrue(self.tool.calculate_score({"query": q_true}, _SAMPLE))
        self.assertFalse(self.tool.calculate_score({"query": q_false}, _SAMPLE))

    # ---- error paths ----

    def test_invalid_args_returns_none(self):
        self.assertIsNone(self.tool.calculate_score({}, _SAMPLE))
        self.assertIsNone(self.tool.calculate_score({"query": "no_suffix"}, _SAMPLE))
        self.assertIsNone(self.tool.calculate_score("not_a_dict", _SAMPLE))

    def test_missing_table_returns_none(self):
        args = {"query": "eq{count{all_rows}; 3}=True"}
        self.assertIsNone(self.tool.calculate_score(args, {}))
        self.assertIsNone(self.tool.calculate_score(args, {"table_html_csv": ""}))

    def test_nonexistent_column_returns_none(self):
        # Engine cannot resolve an unknown column name -> not executable -> None
        args = {"query": "eq{count{filter_eq{all_rows; nonexistent_col; x}}; 0}=True"}
        result = self.tool.calculate_score(args, _SAMPLE)
        self.assertIsNone(result)

    def test_unparseable_query_returns_none(self):
        args = {"query": "not_even_valid_dsl=True"}
        result = self.tool.calculate_score(args, _SAMPLE)
        # returns None when not executable
        self.assertIsNone(result)

    def test_filter_returning_no_rows_hop_gives_false_or_none(self):
        # filter_eq returns empty set; hop on empty set is executable but comparison fails
        # Engine may return False (assertion doesn't hold) or None (not executable)
        args = {"query": "eq{hop{filter_eq{all_rows; athlete; Nobody}; gold}; 0}=True"}
        result = self.tool.calculate_score(args, _SAMPLE)
        self.assertIn(result, [None, False])


# =======================================================================
# Integration: processor + tool round-trip
# =======================================================================

class TestProcessorToolRoundTrip(unittest.TestCase):
    def setUp(self):
        engine = TabFactEngine()
        self.proc = TabFactStructureProcessor(engine)
        self.tool = TabFactTool(engine)

    def test_extract_then_score_true(self):
        q = self.proc.extract_mediator(COMP_TRUE)
        self.assertIsNotNone(q)
        result = self.tool.calculate_score({"query": q}, _SAMPLE)
        self.assertTrue(result)

    def test_extract_then_score_false_query(self):
        # eq{argmax(gold).athlete; Usain Bolt}=True -> Carl Lewis != Usain Bolt -> False
        q = self.proc.extract_mediator(COMP_FALSE)
        self.assertIsNotNone(q)
        result = self.tool.calculate_score({"query": q}, _SAMPLE)
        self.assertFalse(result)

    def test_extract_tool_args_then_score(self):
        args = self.proc.extract_tool_args(COMP_TOOL)
        self.assertIsNotNone(args)
        result = self.tool.calculate_score(args, _SAMPLE)
        self.assertTrue(result)

    def test_no_query_gives_none_score(self):
        q = self.proc.extract_mediator(COMP_NO_QUERY)
        self.assertIsNone(q)
        result = self.tool.calculate_score({"query": q}, _SAMPLE) if q else None
        self.assertIsNone(result)

    def test_compare_structures_symmetric(self):
        q = "eq{count{all_rows}; 3}=True"
        self.assertEqual(self.proc.compare_structures(q, q), 1)
        self.assertEqual(self.proc.compare_structures(q, q + "X"), 0)

    def test_set_match_detects_same_content_different_op(self):
        a = "greater{hop{filter_eq{all_rows; nation; Jamaica}; gold}; 5}=True"
        b = "less{hop{filter_eq{all_rows; nation; Jamaica}; gold}; 5}=True"
        # same columns (nation, gold) + values (jamaica) -> set_match=1
        self.assertEqual(self.proc.set_match(a, b), 1)
        # compare_structures returns 0 (strings differ)
        self.assertEqual(self.proc.compare_structures(a, b), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)