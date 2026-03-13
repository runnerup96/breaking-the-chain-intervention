import unittest
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import glob

from datasets_for_intervention.tabfact_dsl_engine import (
    TabFactEngine,
    ProgramParser,
    FunctionRegistry,
    Literal,
    Call,
    Program,
    RowSet,
    ScalarList,
)


class TestParserBasics(unittest.TestCase):
    def setUp(self):
        self.registry = FunctionRegistry()
        self.parser = ProgramParser(self.registry)

    # ---------- helpers ----------

    def assertCall(self, node, name=None, nargs=None):
        self.assertIsInstance(node, Call)
        if name is not None:
            self.assertEqual(node.name, name)
        if nargs is not None:
            self.assertEqual(len(node.args), nargs)

    def assertLit(self, node, raw=None):
        self.assertIsInstance(node, Literal)
        if raw is not None:
            self.assertEqual(node.raw, raw)

    # ---------- suffix parsing ----------

    def test_parse_program_suffix_true(self):
        prog = self.parser.parse_program("eq{1;1}=True")
        self.assertIsInstance(prog, Program)
        self.assertTrue(prog.expected)
        self.assertCall(prog.expr, name="eq", nargs=2)

    def test_parse_program_suffix_false(self):
        prog = self.parser.parse_program("eq{1;1}=False")
        self.assertFalse(prog.expected)
        self.assertCall(prog.expr, name="eq", nargs=2)

    def test_parse_program_missing_suffix_raises(self):
        with self.assertRaises(Exception):
            self.parser.parse_program("eq{1;1}")

    def test_parse_program_bad_suffix_raises(self):
        with self.assertRaises(Exception):
            self.parser.parse_program("eq{1;1}=Maybe")

    def test_last_top_level_equals(self):
        # '=' inside nested expressions must not be treated as the suffix
        # The suffix is only the last '=' at the top level
        prog = self.parser.parse_program("eq{diff{2;1};1}=True")
        self.assertTrue(prog.expected)
        self.assertCall(prog.expr, name="eq", nargs=2)
        self.assertCall(prog.expr.args[0], name="diff", nargs=2)

    # ---------- delimiter rules ----------

    def test_split_by_semicolon_when_present(self):
        node = self.parser.parse_expr("eq{a; b}")
        self.assertCall(node, name="eq", nargs=2)
        self.assertLit(node.args[0], "a")
        self.assertLit(node.args[1], "b")

    def test_split_by_comma_when_no_semicolon(self):
        node = self.parser.parse_expr("eq{a, b}")
        self.assertCall(node, name="eq", nargs=2)
        self.assertLit(node.args[0], "a")
        self.assertLit(node.args[1], "b")

    def test_nested_calls_keep_structure(self):
        node = self.parser.parse_expr("and{eq{1;1}; not{false}; or{true; false}}")
        self.assertCall(node, name="and", nargs=3)
        self.assertCall(node.args[0], name="eq", nargs=2)
        self.assertCall(node.args[1], name="not", nargs=1)
        self.assertCall(node.args[2], name="or", nargs=2)

    # ---------- alias canonicalization ----------

    def test_alias_gt_to_greater(self):
        node = self.parser.parse_expr("gt{2;1}")
        self.assertCall(node, name="greater", nargs=2)

    def test_alias_more_than_to_greater(self):
        node = self.parser.parse_expr("more_than{2;1}")
        self.assertCall(node, name="greater", nargs=2)

    def test_alias_equal_to_eq(self):
        node = self.parser.parse_expr("equal{2;1}")
        self.assertCall(node, name="eq", nargs=2)

    def test_alias_neq_to_not_eq(self):
        node = self.parser.parse_expr("neq{2;1}")
        self.assertCall(node, name="not_eq", nargs=2)

    def test_alias_filter_ne_to_filter_not_eq(self):
        node = self.parser.parse_expr("filter_ne{all_rows; col; x}")
        self.assertCall(node, name="filter_not_eq", nargs=3)

    def test_alias_filter_equal_to_filter_eq(self):
        node = self.parser.parse_expr("filter_equal{all_rows; col; x}")
        self.assertCall(node, name="filter_eq", nargs=3)

    # ---------- HTML quotes & quotes handling ----------

    def test_html_quotes_preprocess(self):
        # &#34; -> "  (important for quote-aware splitting)
        prog = self.parser.parse_program('eq{a; &#34;kuala lumpur;&#34;}=True')
        self.assertCall(prog.expr, name="eq", nargs=2)
        # the second argument must remain a single literal, without splitting on ';'
        self.assertLit(prog.expr.args[1], '"kuala lumpur;"')

    def test_semicolon_inside_quotes_not_a_delimiter(self):
        # the top-level delimiter here is ';', but the second argument contains ';' inside quotes
        node = self.parser.parse_expr('eq{a; "kuala lumpur;"}')
        self.assertCall(node, name="eq", nargs=2)
        self.assertLit(node.args[1], '"kuala lumpur;"')

    def test_commas_inside_quotes_not_split_when_semicolon_delim(self):
        node = self.parser.parse_expr('filter_eq{all_rows; name; "a, b, c"}')
        self.assertCall(node, name="filter_eq", nargs=3)
        self.assertLit(node.args[2], '"a, b, c"')

    # ---------- arity repair ----------

    def test_repair_filter_two_args_field_value_in_one(self):
        # filter_less{C; field, value} -> filter_less{C; field; value}
        node = self.parser.parse_expr("filter_less{all_rows; crowd, 15000}")
        self.assertCall(node, name="filter_less", nargs=3)
        self.assertLit(node.args[1], "crowd")
        self.assertLit(node.args[2], "15000")

    def test_repair_unary_extra_args_trim(self):
        node = self.parser.parse_expr("count{all_rows; bogus}")
        self.assertCall(node, name="count", nargs=1)
        self.assertLit(node.args[0], "all_rows")

        node2 = self.parser.parse_expr("only{all_rows; bogus; junk}")
        self.assertCall(node2, name="only", nargs=1)

    def test_repair_comparisons_too_many_args_join_tail(self):
        node = self.parser.parse_expr("eq{a; b; c}")
        self.assertCall(node, name="eq", nargs=2)
        # the second argument becomes "b, c" via join_tail
        self.assertLit(node.args[1], "b, c")

    def test_repair_filter_too_many_args_join_tail(self):
        node = self.parser.parse_expr("filter_eq{all_rows; name; a; b; c}")
        self.assertCall(node, name="filter_eq", nargs=3)
        self.assertLit(node.args[2], "a, b, c")

    def test_repair_all_eq_more_than_3_join_tail(self):
        node = self.parser.parse_expr("all_eq{all_rows; field; a; b; c}")
        self.assertCall(node, name="all_eq", nargs=3)
        self.assertLit(node.args[2], "a, b, c")

    def test_repair_and_or_drop_empty_args(self):
        node = self.parser.parse_expr("and{true;;false;}")
        self.assertCall(node, name="and")
        # after removing empty items, there should be 2 arguments: true, false
        self.assertEqual(len(node.args), 2)

    # ---------- malformed syntax behavior ----------

    def test_malformed_unbalanced_brace_becomes_literal(self):
        node = self.parser.parse_expr("eq{1;2")
        self.assertLit(node)

    def test_malformed_trailing_garbage_becomes_literal(self):
        node = self.parser.parse_expr("eq{1;2} junk")
        self.assertLit(node)

    def test_unmatched_quote_does_not_crash(self):
        # we do not guarantee correct parsing here, but we test robustness
        node = self.parser.parse_expr('eq{a; "unterminated}')
        self.assertIsNotNone(node)


class TestEngineExecutionSmoke(unittest.TestCase):
    def setUp(self):
        self.engine = TabFactEngine()

    def test_all_rows_literal_executes(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        res = self.engine.execute("eq{count{all_rows}; 3}=True", df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    def test_rank_with_rank_column(self):
        df = pd.DataFrame({
            "rank": ["2", "1", "3"],
            "name": ["B", "A", "C"],
        })
        # rank{all_rows; name} returns the name at the minimal rank, i.e. "A"
        res = self.engine.execute('eq{rank{all_rows; name}; "A"}=True', df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    def test_rank_without_rank_column_fallback_top(self):
        df = pd.DataFrame({
            "name": ["TopName", "Other"],
        })
        res = self.engine.execute('eq{rank{all_rows; name}; "TopName"}=True', df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    def test_aliases_work_end_to_end(self):
        df = pd.DataFrame({"x": ["2", "1"]})
        # gt -> greater (strict numeric comparison)
        res = self.engine.execute("gt{hop{0; x}; hop{1; x}}=True", df)
        # IMPORTANT: hop{0; x} currently does not support row=0 as an int (in the current engine, a row int is not created from Literal)
        # Therefore, this test is expected to be NOT executable.
        # We keep it as a guard test: if you later add int literals, it will become executable.
        self.assertFalse(res.executable)

    def test_greater_non_numeric_is_not_executable(self):
        df = pd.DataFrame({"x": ["abc", "def"]})
        res = self.engine.execute("greater{hop{filter_eq{all_rows; x; abc}; x}; 1}=True", df)
        self.assertFalse(res.executable)

    def test_filter_repair_two_args_executes(self):
        df = pd.DataFrame({
            "crowd": ["100", "200", "150"],
            "name": ["a", "b", "c"],
        })
        # filter_less{all_rows; crowd, 160} -> rows with crowd < 160 : rows 0 and 2
        # count(...) == 2
        res = self.engine.execute("eq{count{filter_less{all_rows; crowd, 160}}; 2}=True", df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)


class TestParserAdvanced(unittest.TestCase):
    def setUp(self):
        self.registry = FunctionRegistry()
        self.parser = ProgramParser(self.registry)

    def assertCall(self, node, name=None, nargs=None):
        self.assertIsInstance(node, Call)
        if name is not None:
            self.assertEqual(node.name, name)
        if nargs is not None:
            self.assertEqual(len(node.args), nargs)

    def assertLit(self, node, raw=None):
        self.assertIsInstance(node, Literal)
        if raw is not None:
            self.assertEqual(node.raw, raw)

    # 1) very deep nesting
    def test_deep_nesting_15_levels(self):
        s = "not{not{not{not{not{not{not{not{not{not{not{not{not{not{not{true}}}}}}}}}}}}}}}=False"
        prog = self.parser.parse_program(s)
        self.assertFalse(prog.expected)
        # expr = not{...}
        self.assertCall(prog.expr, name="not", nargs=1)

    # 2) long string with lots of spaces and newlines
    def test_whitespace_and_newlines(self):
        s = """
            and{
                eq{ 1 ; 1 } ;
                not{ false } ;
                or{ true ; false }
            } = True
        """
        prog = self.parser.parse_program(s)
        self.assertTrue(prog.expected)
        self.assertCall(prog.expr, name="and")

    # 3) '=' inside quotes must not break suffix parsing
    def test_equals_inside_quotes_ignored_for_suffix(self):
        prog = self.parser.parse_program('eq{"a=b"; "a=b"}=True')
        self.assertTrue(prog.expected)
        self.assertCall(prog.expr, name="eq", nargs=2)
        self.assertLit(prog.expr.args[0], '"a=b"')
        self.assertLit(prog.expr.args[1], '"a=b"')

    # 4) value with ';' inside quotes
    def test_semicolon_inside_quotes(self):
        node = self.parser.parse_expr('eq{a; "kuala lumpur;"}')
        self.assertCall(node, name="eq", nargs=2)
        self.assertLit(node.args[1], '"kuala lumpur;"')

    # 5) HTML quotes + ';' inside
    def test_html_quotes_with_semicolon_inside(self):
        prog = self.parser.parse_program('eq{a; &#34;kuala lumpur;&#34;}=True')
        self.assertCall(prog.expr, name="eq", nargs=2)
        self.assertLit(prog.expr.args[1], '"kuala lumpur;"')

    # 6) if ';' exists at the top level, commas inside the value must not split the argument
    def test_semicolon_delim_keeps_commas_in_value(self):
        node = self.parser.parse_expr('filter_eq{all_rows; city; "cape vincent , ny"}')
        self.assertCall(node, name="filter_eq", nargs=3)
        self.assertLit(node.args[2], '"cape vincent , ny"')

    # 7) comma-delimited variant + commas in the value -> repair should join the tail back
    def test_comma_delim_filter_eq_value_with_commas_repair(self):
        # no ';' => delimiter=','
        node = self.parser.parse_expr("filter_eq{all_rows, city, cape vincent , ny}")
        self.assertCall(node, name="filter_eq", nargs=3)
        self.assertLit(node.args[0], "all_rows")
        self.assertLit(node.args[1], "city")
        # the tail is joined back
        self.assertLit(node.args[2], "cape vincent, ny")

    # 8) repair: filter_* with 2 args (field, value) packed into one argument
    def test_repair_filter_two_args_split_field_value(self):
        node = self.parser.parse_expr("filter_less{all_rows; crowd, 15000}")
        self.assertCall(node, name="filter_less", nargs=3)
        self.assertLit(node.args[1], "crowd")
        self.assertLit(node.args[2], "15000")

    # 9) repair: unary call with extra arguments
    def test_repair_unary_trim(self):
        node = self.parser.parse_expr("count{all_rows; x; y}")
        self.assertCall(node, name="count", nargs=1)

        node2 = self.parser.parse_expr("not{true; extra}")
        self.assertCall(node2, name="not", nargs=1)

    # 10) repair: comparison with extra args gets joined
    def test_repair_eq_too_many_args(self):
        node = self.parser.parse_expr("eq{a; b; c; d}")
        self.assertCall(node, name="eq", nargs=2)
        self.assertLit(node.args[1], "b, c, d")

    # 11) and/or removes empty arguments
    def test_repair_and_drops_empty(self):
        node = self.parser.parse_expr("and{true;;false;}")
        self.assertCall(node, name="and")
        self.assertEqual(len(node.args), 2)

    # 12) aliases are canonicalized in the AST
    def test_aliases_canonicalize_in_ast(self):
        self.assertCall(self.parser.parse_expr("gt{2;1}"), name="greater", nargs=2)
        self.assertCall(self.parser.parse_expr("lt{2;1}"), name="less", nargs=2)
        self.assertCall(self.parser.parse_expr("equal{2;1}"), name="eq", nargs=2)
        self.assertCall(self.parser.parse_expr("neq{2;1}"), name="not_eq", nargs=2)
        self.assertCall(self.parser.parse_expr("filter_ne{all_rows; x; 1}"), name="filter_not_eq", nargs=3)
        self.assertCall(self.parser.parse_expr("filter_equal{all_rows; x; 1}"), name="filter_eq", nargs=3)

    # 13) malformed syntax: trailing garbage -> Literal
    def test_trailing_garbage_becomes_literal(self):
        prog = self.parser.parse_program("eq{1;2} junk=True")
        self.assertIsInstance(prog.expr, Literal)

    # 14) unbalanced braces must not crash the parser
    def test_unbalanced_brace_no_crash(self):
        node = self.parser.parse_expr("and{eq{1;1}; not{false}")
        self.assertIsNotNone(node)


class TestEngineExecutionAdvanced(unittest.TestCase):
    def setUp(self):
        self.engine = TabFactEngine()

    # 1) case-insensitive columns + spaces
    def test_case_insensitive_column_resolution(self):
        df = pd.DataFrame({"Call Sign": ["KDSD"], "Format": ["Public Radio"]})
        res = self.engine.execute('eq{hop{filter_eq{all_rows; call sign; kdsd}; format}; "public radio"}=True', df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 2) filter_eq with an empty value
    def test_filter_eq_empty_value(self):
        df = pd.DataFrame({"x": ["", "a", ""]})
        res = self.engine.execute("eq{count{filter_eq{all_rows; x; }}; 2}=True", df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 3) only/any/none on a rowset
    def test_only_any_none_rowset(self):
        df = pd.DataFrame({"x": ["a", "b", "b"]})
        res1 = self.engine.execute("only{filter_eq{all_rows; x; a}}=True", df)
        res2 = self.engine.execute("any{filter_eq{all_rows; x; a}}=True", df)
        res3 = self.engine.execute("none{filter_eq{all_rows; x; z}}=True", df)
        self.assertTrue(res1.executable and res1.final)
        self.assertTrue(res2.executable and res2.final)
        self.assertTrue(res3.executable and res3.final)

    # 4) zero{count{...}}
    def test_zero_count(self):
        df = pd.DataFrame({"x": ["a", "b"]})
        res = self.engine.execute("zero{count{filter_eq{all_rows; x; z}}}=True", df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 5) top/bottom + before/after
    def test_top_bottom_before_after(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        res1 = self.engine.execute("before{top{all_rows}; bottom{all_rows}}=True", df)
        res2 = self.engine.execute("after{bottom{all_rows}; top{all_rows}}=True", df)
        self.assertTrue(res1.executable and res1.final)
        self.assertTrue(res2.executable and res2.final)

    # 6) positional: first/second/third/last
    def test_positional_first_second_third_last(self):
        df = pd.DataFrame({"x": ["a", "b", "c", "d"]})
        # C = all_rows, D = top{filter_eq{...}} -> row
        res1 = self.engine.execute("first{all_rows; top{filter_eq{all_rows; x; a}}}=True", df)
        res2 = self.engine.execute("second{all_rows; top{filter_eq{all_rows; x; b}}}=True", df)
        res3 = self.engine.execute("third{all_rows; top{filter_eq{all_rows; x; c}}}=True", df)
        res4 = self.engine.execute("last{all_rows; top{filter_eq{all_rows; x; d}}}=True", df)
        self.assertTrue(res1.executable and res1.final)
        self.assertTrue(res2.executable and res2.final)
        self.assertTrue(res3.executable and res3.final)
        self.assertTrue(res4.executable and res4.final)

    # 7) argmax/argmin (2-arg)
    def test_argmax_argmin_two_args(self):
        df = pd.DataFrame({"name": ["a", "b", "c"], "score": ["10", "30", "20"]})
        res1 = self.engine.execute('eq{hop{argmax{all_rows; score}; name}; "b"}=True', df)
        res2 = self.engine.execute('eq{hop{argmin{all_rows; score}; name}; "a"}=True', df)
        self.assertTrue(res1.executable and res1.final)
        self.assertTrue(res2.executable and res2.final)

    # 8) 1-arg argmax via hint_field (we obtain a RowSet from filter_greater on score)
    def test_argmax_one_arg_via_hint_field(self):
        df = pd.DataFrame({"name": ["a", "b", "c"], "score": ["10", "30", "20"]})
        res = self.engine.execute('eq{hop{argmax{filter_greater{all_rows; score; 0}}; name}; "b"}=True', df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 9) within / not_within
    def test_within_not_within(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        res1 = self.engine.execute("within{all_rows; x; b}=True", df)
        res2 = self.engine.execute("not_within{all_rows; x; z}=True", df)
        self.assertTrue(res1.executable and res1.final)
        self.assertTrue(res2.executable and res2.final)

    # 10) uniq
    def test_uniq(self):
        df = pd.DataFrame({"x": ["A", "a", "b", "b"]})
        res = self.engine.execute("eq{uniq{all_rows; x}; 2}=True", df)  # A and a are treated as the same value (normalize)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 11) most_freq with tie-break
    def test_most_freq_tie_break(self):
        # a and b both appear 2 times; tie-break by key => "a"
        df = pd.DataFrame({"x": ["b", "a", "b", "a"]})
        res = self.engine.execute('eq{most_freq{all_rows; x}; "a"}=True', df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 12) sum/avg/max/min with “smart” numbers
    def test_smart_numbers_aggregates(self):
        df = pd.DataFrame({"laps": ["40 laps", "10 laps", "2 laps"]})
        res1 = self.engine.execute("eq{sum{all_rows; laps}; 52}=True", df)
        res2 = self.engine.execute("eq{max{all_rows; laps}; 40}=True", df)
        res3 = self.engine.execute("eq{min{all_rows; laps}; 2}=True", df)
        res4 = self.engine.execute("eq{avg{all_rows; laps}; 17.333333333333332}=True", df)
        self.assertTrue(res1.executable and res1.final)
        self.assertTrue(res2.executable and res2.final)
        self.assertTrue(res3.executable and res3.final)
        self.assertTrue(res4.executable and res4.final)

    # 13) diff/add/half
    def test_arithmetic_diff_add_half(self):
        df = pd.DataFrame({"x": ["a", "b", "c", "d"]})  # len=4, half=2
        res1 = self.engine.execute("eq{half{all_rows}; 2}=True", df)
        res2 = self.engine.execute("eq{diff{10; 3}; 7}=True", df)
        res3 = self.engine.execute("eq{add{10; 3}; 13}=True", df)
        self.assertTrue(res1.executable and res1.final)
        self.assertTrue(res2.executable and res2.final)
        self.assertTrue(res3.executable and res3.final)

    # 14) all_eq (3-arg) and all_not_eq
    def test_all_eq_and_all_not_eq_rowset_field(self):
        df = pd.DataFrame({"x": ["a", "a", "b"]})
        res1 = self.engine.execute("all_eq{filter_eq{all_rows; x; a}; x; a}=True", df)
        res2 = self.engine.execute("all_not_eq{filter_eq{all_rows; x; a}; x; b}=True", df)
        self.assertTrue(res1.executable and res1.final)
        self.assertTrue(res2.executable and res2.final)

    # 15) all_eq (2-arg overload on ScalarList)
    def test_all_eq_scalarlist_overload(self):
        df = pd.DataFrame({"x": ["A", "a", "a"]})
        # hop returns a ScalarList (rowset size > 1)
        res = self.engine.execute("all_eq{hop{filter_eq{all_rows; x; a}; x}; a}=True", df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 16) all_greater (2-arg overload on ScalarList)
    def test_all_greater_scalarlist_overload(self):
        df = pd.DataFrame({"n": ["11", "12", "13"]})
        res = self.engine.execute("all_greater{hop{all_rows; n}; 10}=True", df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 17) rank with a rank column
    def test_rank_with_rank_column(self):
        df = pd.DataFrame({"rank": ["2", "1", "3"], "name": ["B", "A", "C"]})
        res = self.engine.execute('eq{rank{all_rows; name}; "A"}=True', df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 18) rank without a rank column (fallback to top)
    def test_rank_without_rank_column(self):
        df = pd.DataFrame({"name": ["TopName", "Other"]})
        res = self.engine.execute('eq{rank{all_rows; name}; "TopName"}=True', df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 19) rank with empty/non-numeric rank values
    def test_rank_with_missing_rank_values(self):
        df = pd.DataFrame({"rank": ["", "x", "5", "2"], "name": ["A", "B", "C", "D"]})
        # minimal numeric rank = 2 -> "D"
        res = self.engine.execute('eq{rank{all_rows; name}; "D"}=True', df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 20) expression returns a non-bool => not executable
    def test_expression_non_bool_not_executable(self):
        df = pd.DataFrame({"x": ["a", "b"]})
        res = self.engine.execute("count{all_rows}=True", df)  # expr_value float
        self.assertFalse(res.executable)

    # 21) long “realistic” string with deep nesting
    def test_long_realistic_nested_program(self):
        df = pd.DataFrame({
            "city": ["cape vincent , ny", "other"],
            "value": ["12", "7"],
            "group": ["A", "B"]
        })
        program = (
            'and{'
            '  within{all_rows; city; "cape vincent , ny"};'
            '  eq{hop{top{filter_eq{all_rows; city; "cape vincent , ny"}}; group}; A};'
            '  greater{hop{top{filter_eq{all_rows; city; "cape vincent , ny"}}; value}; 10}'
            '}=True'
        )
        res = self.engine.execute(program, df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 22) large DataFrame (10k rows) + real DSL expression
    def test_big_table_10k_rows(self):
        n = 10_000
        df = pd.DataFrame({
            "group": ["A" if i % 2 == 0 else "B" for i in range(n)],
            "value": [str(i) for i in range(n)],
        })

        # rows where value > 9000 AND group == A (even numbers in 9001..9999)
        # expected count = 499, and max value among them = 9998
        program = (
            "and{"
            "  eq{count{filter_eq{filter_greater{all_rows; value; 9000}; group; A}}; 499};"
            "  eq{hop{argmax{filter_eq{filter_greater{all_rows; value; 9000}; group; A}; value}; value}; 9998}"
            "}=True"
        )
        res = self.engine.execute(program, df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)


class TestParserMore(unittest.TestCase):
    def setUp(self):
        self.registry = FunctionRegistry()
        self.parser = ProgramParser(self.registry)

    def assertCall(self, node, name=None, nargs=None):
        self.assertIsInstance(node, Call)
        if name is not None:
            self.assertEqual(node.name, name)
        if nargs is not None:
            self.assertEqual(len(node.args), nargs)

    def assertLit(self, node, raw=None):
        self.assertIsInstance(node, Literal)
        if raw is not None:
            self.assertEqual(node.raw, raw)

    # 23) comma-delimited input + nested calls: split must respect brace depth
    def test_comma_delim_nested_calls_split(self):
        node = self.parser.parse_expr("eq{count{all_rows}, 3}")
        self.assertCall(node, name="eq", nargs=2)
        self.assertCall(node.args[0], name="count", nargs=1)
        self.assertLit(node.args[1], "3")

    # 24) comma-delimited input with empty value: the empty argument must be preserved
    def test_comma_delim_empty_value_kept(self):
        node = self.parser.parse_expr("filter_eq{all_rows, x, }")
        self.assertCall(node, name="filter_eq", nargs=3)
        self.assertLit(node.args[0], "all_rows")
        self.assertLit(node.args[1], "x")
        self.assertLit(node.args[2], "")

    # 25) extra closing brace -> trailing garbage => Literal (conservative behavior)
    def test_extra_closing_brace_becomes_literal(self):
        node = self.parser.parse_expr("eq{1;2}}")
        self.assertLit(node)

    # 26) multiple '=' at the top level: the suffix is taken from the last '=',
    #     and in that case expr usually becomes a Literal (because of trailing garbage)
    def test_multiple_equals_last_suffix_wins_expr_literal(self):
        prog = self.parser.parse_program("eq{1;1}=True=False")
        self.assertFalse(prog.expected)  # the last suffix is =False
        self.assertIsInstance(prog.expr, Literal)

    # 27) empty {} in not: the parser must not crash; we get not{""} (after trimming extra args)
    def test_not_with_empty_braces_parses(self):
        node = self.parser.parse_expr("not{}")
        self.assertCall(node, name="not", nargs=1)
        self.assertLit(node.args[0], "")

    # 28) an unknown function still parses as a Call (execution will decide what to do later)
    def test_unknown_function_parses_as_call(self):
        node = self.parser.parse_expr("weirdop{a; b}")
        self.assertCall(node, name="weirdop", nargs=2)
        self.assertLit(node.args[0], "a")
        self.assertLit(node.args[1], "b")


class TestEngineMore(unittest.TestCase):
    def setUp(self):
        self.engine = TabFactEngine()

    # 29) filter_greater_eq / filter_less_eq (and comparison arithmetic)
    def test_filter_greater_eq_and_less_eq(self):
        df = pd.DataFrame({"n": ["1", "2", "3", "4"]})
        # >=2 -> 3 rows, <=2 -> 2 rows
        r1 = self.engine.execute("eq{count{filter_greater_eq{all_rows; n; 2}}; 3}=True", df)
        r2 = self.engine.execute("eq{count{filter_less_eq{all_rows; n; 2}}; 2}=True", df)
        self.assertTrue(r1.executable and r1.final)
        self.assertTrue(r2.executable and r2.final)

    # 30) all_greater_eq / all_less_eq
    def test_all_greater_eq_all_less_eq(self):
        df = pd.DataFrame({"n": ["2", "3", "4"]})
        r1 = self.engine.execute("all_greater_eq{all_rows; n; 2}=True", df)
        r2 = self.engine.execute("all_less_eq{all_rows; n; 4}=True", df)
        self.assertTrue(r1.executable and r1.final)
        self.assertTrue(r2.executable and r2.final)

    # 31) smart-number: mixed fraction "1 - 1 / 16" > "1"
    def test_smart_number_mixed_fraction_compare(self):
        df = pd.DataFrame({"id": ["a", "b"], "v": ["1 - 1 / 16", "1"]})
        program = (
            "greater{"
            "  hop{top{filter_eq{all_rows; id; a}}; v};"
            "  hop{top{filter_eq{all_rows; id; b}}; v}"
            "}=True"
        )
        res = self.engine.execute(program, df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 32) smart-number: time "2:00.00" > "1:59.00"
    def test_smart_number_time_compare(self):
        df = pd.DataFrame({"id": ["a", "b"], "t": ["2:00.00", "1:59.00"]})
        program = (
            "greater{"
            "  hop{top{filter_eq{all_rows; id; a}}; t};"
            "  hop{top{filter_eq{all_rows; id; b}}; t}"
            "}=True"
        )
        res = self.engine.execute(program, df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 33) none{hop{empty_rowset; field}}: hop(empty) -> ScalarList([]), none -> True
    def test_none_on_empty_scalarlist_from_hop(self):
        df = pd.DataFrame({"x": ["a", "b"]})
        res = self.engine.execute("none{hop{filter_eq{all_rows; x; z}; x}}=True", df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)

    # 34) fully comma-delimited program (not a single ';' at the top level)
    def test_comma_delim_program_executes(self):
        df = pd.DataFrame({"x": ["a", "b", "b"], "n": ["0", "2", "3"]})
        program = (
            "and{"
            "  within{all_rows, x, b},"
            "  eq{count{filter_greater{all_rows, n, 1}}, 2}"
            "}=True"
        )
        res = self.engine.execute(program, df)
        self.assertTrue(res.executable)
        self.assertTrue(res.final)


def _norm_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("&#34;", '"').replace("&quot;", '"').strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s.strip().lower()


def _parse_number(x: Any) -> Optional[float]:
    """Independent-ish numeric parse (time, mixed frac, frac, float, first number)."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip().replace(",", "")
    if not s:
        return None

    # time M:SS(.xx)
    import re
    m = re.fullmatch(r"(\d+)\s*:\s*(\d+(?:\.\d+)?)", s)
    if m:
        mm = float(m.group(1))
        ss = float(m.group(2))
        return mm * 60.0 + ss

    # mixed fraction: a - b / c
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)\s*/\s*(\d+)", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        c = float(m.group(3))
        if c == 0:
            return None
        return a + (b / c)

    # fraction: b / c
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", s)
    if m:
        b = float(m.group(1))
        c = float(m.group(2))
        if c == 0:
            return None
        return b / c

    # strict float
    try:
        return float(s)
    except Exception:
        pass

    # first number
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def _read_table_csv(path: str) -> pd.DataFrame:
    """
    TabFact tables often come with sep='#'.
    We do a robust read: first try '#', and if there is only 1 column, try ','.
    """
    df = pd.read_csv(path, sep="#", dtype=str, keep_default_na=False)
    if df.shape[1] <= 1:
        df2 = pd.read_csv(path, sep=",", dtype=str, keep_default_na=False)
        if df2.shape[1] > df.shape[1]:
            df = df2
    df = df.fillna("")
    # normalize column names (at least strip them)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _safe_quoted_literal(v: str) -> str:
    """
    Safe literal for DSL:
    - always wrapped in double quotes
    - if it contains double quotes or curly braces, it is better to reject it (return an empty string)
    """
    if v is None:
        return '""'
    s = str(v)
    if '"' in s:
        return ""
    if "{" in s or "}" in s:
        return ""
    # allow ';' and ',' — the parser is quote-aware
    return f'"{s}"'


def _pick_text_column_and_value(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Choose a column and value such that:
    - the column contains a repeated value
    - the value can be safely quoted (no '"', '{', '}')
    """
    for col in df.columns:
        vals = df[col].astype(str).tolist()
        normed = [_norm_text(x) for x in vals]
        # frequencies
        freq: Dict[str, int] = {}
        for t in normed:
            if t == "":
                continue
            freq[t] = freq.get(t, 0) + 1
        if not freq:
            continue
        # we want a value that appears at least 2 times
        candidates = sorted(freq.items(), key=lambda kv: -kv[1])
        for key, cnt in candidates:
            if cnt < 2:
                break
            # find the original value that normalizes to key
            for raw in vals:
                if _norm_text(raw) == key:
                    q = _safe_quoted_literal(str(raw))
                    if q:
                        return col, q
    return None


def _pick_numeric_column(df: pd.DataFrame, min_numeric: int = 8) -> Optional[str]:
    """
    Choose a column that contains enough numeric values (according to _parse_number).
    """
    nrows = len(df)
    if nrows == 0:
        return None
    for col in df.columns:
        nums = [_parse_number(x) for x in df[col].astype(str).tolist()]
        cnt = sum(1 for x in nums if x is not None)
        if cnt >= max(min_numeric, int(0.4 * nrows)):
            return col
    return None


def _rowset_filter_eq(df: pd.DataFrame, col: str, qval: str) -> List[int]:
    """
    Reference for filter_eq.
    qval is already quoted as "..."
    """
    # remove quotes as in norm_text
    target = _norm_text(qval)
    out = []
    for i, x in enumerate(df[col].astype(str).tolist()):
        if _norm_text(x) == target:
            out.append(i)
    return out


def _rowset_filter_not_eq(df: pd.DataFrame, col: str, qval: str) -> List[int]:
    target = _norm_text(qval)
    out = []
    for i, x in enumerate(df[col].astype(str).tolist()):
        if _norm_text(x) != target:
            out.append(i)
    return out


def _rowset_filter_num(df: pd.DataFrame, col: str, thr: float, mode: str) -> List[int]:
    out = []
    xs = df[col].astype(str).tolist()
    for i, x in enumerate(xs):
        n = _parse_number(x)
        if n is None:
            continue
        if mode == "gt" and n > thr:
            out.append(i)
        elif mode == "lt" and n < thr:
            out.append(i)
        elif mode == "ge" and n >= thr:
            out.append(i)
        elif mode == "le" and n <= thr:
            out.append(i)
    return out


def _argmax_row(df: pd.DataFrame, rows: List[int], col: str) -> Optional[int]:
    best_i = None
    best_v = None
    for r in rows:
        n = _parse_number(df.at[r, col])
        if n is None:
            continue
        if best_i is None or n > best_v:
            best_i = r
            best_v = n
    return best_i


def _argmin_row(df: pd.DataFrame, rows: List[int], col: str) -> Optional[int]:
    best_i = None
    best_v = None
    for r in rows:
        n = _parse_number(df.at[r, col])
        if n is None:
            continue
        if best_i is None or n < best_v:
            best_i = r
            best_v = n
    return best_i


def _mode_value_norm(df: pd.DataFrame, col: str, rows: Optional[List[int]] = None) -> Optional[str]:
    xs = df[col].astype(str).tolist()
    if rows is None:
        rows = list(range(len(df)))
    freq: Dict[str, int] = {}
    for r in rows:
        k = _norm_text(xs[r])
        if k == "":
            continue
        freq[k] = freq.get(k, 0) + 1
    if not freq:
        return None
    # tie-break: lexicographically by key
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _uniq_count_norm(df: pd.DataFrame, col: str, rows: Optional[List[int]] = None) -> int:
    xs = df[col].astype(str).tolist()
    if rows is None:
        rows = list(range(len(df)))
    s = set()
    for r in rows:
        t = _norm_text(xs[r])
        if t != "":
            s.add(t)
    return len(s)


# =========================
# Tests on real tables
# =========================

class TestTabFactEngineOnRealTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = TabFactEngine()

        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        base_dir = os.environ.get(
            "TABFACT_TABLE_DIR",
            f"{PROJECT_ROOT}/statics/datasets/TabFact/data/all_csv",
        )
        max_tables = int(os.environ.get("TABFACT_MAX_TABLES", "12"))

        paths = []
        if os.path.isdir(base_dir):
            paths = sorted(glob.glob(os.path.join(base_dir, "*.csv")))
        if not paths:
            # fallback: what is usually present in the project environment
            paths = sorted(glob.glob("/mnt/data/*.html.csv")) + sorted(glob.glob("/mnt/data/*.csv"))

        # filter real files
        paths = [p for p in paths if os.path.isfile(p)]
        cls.table_paths = paths[:max_tables]

        if not cls.table_paths:
            raise unittest.SkipTest("No TabFact tables found. Set TABFACT_TABLE_DIR or provide CSVs.")

        # DataFrame cache so we do not read the same file 30 times
        cls._df_cache: Dict[str, pd.DataFrame] = {}

    def _df(self, path: str) -> pd.DataFrame:
        if path not in self._df_cache:
            self._df_cache[path] = _read_table_csv(path)
        return self._df_cache[path]

    def _table(self, i: int) -> Tuple[str, pd.DataFrame]:
        path = self.table_paths[i % len(self.table_paths)]
        return path, self._df(path)

    # -------------------------
    # 1) Basic correctness on all_rows
    # -------------------------

    def test_real_01_count_all_rows(self):
        for i, path in enumerate(self.table_paths):
            df = self._df(path)
            n = len(df)
            program = f"eq{{count{{all_rows}}; {n}}}=True"
            res = self.engine.execute(program, df)
            with self.subTest(table=path):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final, f"expr={res.expr_value} error={res.error}")

    # -------------------------
    # 2) filter_eq / within / filter_not_eq
    # -------------------------

    def test_real_02_filter_eq_count_matches_reference(self):
        for path in self.table_paths:
            df = self._df(path)
            pick = _pick_text_column_and_value(df)
            if pick is None:
                continue
            col, qval = pick
            rows = _rowset_filter_eq(df, col, qval)
            expected = len(rows)

            program = f"eq{{count{{filter_eq{{all_rows; {col}; {qval}}}}}; {expected}}}=True"
            res = self.engine.execute(program, df)

            with self.subTest(table=path, col=col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final, f"expr={res.expr_value} expected_count={expected}")

    def test_real_03_within_matches_reference(self):
        for path in self.table_paths:
            df = self._df(path)
            pick = _pick_text_column_and_value(df)
            if pick is None:
                continue
            col, qval = pick
            rows = _rowset_filter_eq(df, col, qval)
            expected = (len(rows) > 0)

            program = f"within{{all_rows; {col}; {qval}}}=True"
            res = self.engine.execute(program, df)

            with self.subTest(table=path, col=col):
                self.assertTrue(res.executable, res.error)
                # expr_value must match
                self.assertEqual(res.expr_value, expected)

    def test_real_04_filter_not_eq_count_matches_reference(self):
        for path in self.table_paths:
            df = self._df(path)
            pick = _pick_text_column_and_value(df)
            if pick is None:
                continue
            col, qval = pick
            rows = _rowset_filter_not_eq(df, col, qval)
            expected = len(rows)

            program = f"eq{{count{{filter_not_eq{{all_rows; {col}; {qval}}}}}; {expected}}}=True"
            res = self.engine.execute(program, df)

            with self.subTest(table=path, col=col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final, f"expr={res.expr_value} expected_count={expected}")

    # -------------------------
    # 3) Numeric filters (greater/less/eq) and argmax/argmin chains
    # -------------------------

    def test_real_05_filter_greater_count_matches_reference(self):
        for path in self.table_paths:
            df = self._df(path)
            num_col = _pick_numeric_column(df)
            if num_col is None:
                continue

            nums = [_parse_number(x) for x in df[num_col].astype(str).tolist()]
            nn = [x for x in nums if x is not None]
            if len(nn) < 8:
                continue
            # threshold = median-ish
            nn_sorted = sorted(nn)
            thr = nn_sorted[len(nn_sorted) // 2]

            rows = _rowset_filter_num(df, num_col, thr, mode="gt")
            expected = len(rows)

            program = f"eq{{count{{filter_greater{{all_rows; {num_col}; {thr}}}}}; {expected}}}=True"
            res = self.engine.execute(program, df)
            with self.subTest(table=path, col=num_col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final, f"expected_count={expected}, expr={res.expr_value}")

    def test_real_06_filter_less_count_matches_reference(self):
        for path in self.table_paths:
            df = self._df(path)
            num_col = _pick_numeric_column(df)
            if num_col is None:
                continue

            nums = [_parse_number(x) for x in df[num_col].astype(str).tolist()]
            nn = [x for x in nums if x is not None]
            if len(nn) < 8:
                continue
            thr = sorted(nn)[len(nn) // 2]

            rows = _rowset_filter_num(df, num_col, thr, mode="lt")
            expected = len(rows)

            # intentionally use the "repair" format: "{col}, {thr}" inside the 2nd arg
            program = f"eq{{count{{filter_less{{all_rows; {num_col}, {thr}}}}}; {expected}}}=True"
            res = self.engine.execute(program, df)
            with self.subTest(table=path, col=num_col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final, f"expected_count={expected}, expr={res.expr_value}")

    def test_real_07_argmax_hop_numeric_matches_reference(self):
        for path in self.table_paths:
            df = self._df(path)
            num_col = _pick_numeric_column(df)
            if num_col is None:
                continue

            rows = list(range(len(df)))
            r = _argmax_row(df, rows, num_col)
            if r is None:
                continue
            max_num = _parse_number(df.at[r, num_col])
            if max_num is None:
                continue

            program = (
                f"eq{{"
                f"  hop{{argmax{{all_rows; {num_col}}}; {num_col}}};"
                f"  {max_num}"
                f"}}=True"
            )
            res = self.engine.execute(program, df)

            with self.subTest(table=path, col=num_col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final, f"expected_max={max_num}, expr={res.expr_value}")

    def test_real_08_argmin_hop_numeric_matches_reference(self):
        for path in self.table_paths:
            df = self._df(path)
            num_col = _pick_numeric_column(df)
            if num_col is None:
                continue

            rows = list(range(len(df)))
            r = _argmin_row(df, rows, num_col)
            if r is None:
                continue
            min_num = _parse_number(df.at[r, num_col])
            if min_num is None:
                continue

            program = (
                f"eq{{"
                f"  hop{{argmin{{all_rows; {num_col}}}; {num_col}}};"
                f"  {min_num}"
                f"}}=True"
            )
            res = self.engine.execute(program, df)

            with self.subTest(table=path, col=num_col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final, f"expected_min={min_num}, expr={res.expr_value}")

    # -------------------------
    # 4) uniq / most_freq on real cols
    # -------------------------

    def test_real_09_uniq_matches_reference(self):
        for path in self.table_paths:
            df = self._df(path)
            if df.shape[1] == 0 or len(df) == 0:
                continue
            col = df.columns[0]
            expected = _uniq_count_norm(df, col)

            program = f"eq{{uniq{{all_rows; {col}}}; {expected}}}=True"
            res = self.engine.execute(program, df)
            with self.subTest(table=path, col=col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final, f"expected={expected}, expr={res.expr_value}")

    def test_real_10_most_freq_matches_reference(self):
        for path in self.table_paths:
            df = self._df(path)
            if df.shape[1] == 0 or len(df) == 0:
                continue
            col = df.columns[0]
            expected = _mode_value_norm(df, col)
            if expected is None:
                continue

            # expected is already a normalized string (lower), while most_freq returns the normalized key
            program = f'eq{{most_freq{{all_rows; {col}}}; "{expected}"}}=True'
            res = self.engine.execute(program, df)

            with self.subTest(table=path, col=col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final, f"expected={expected}, expr={res.expr_value}")

    # -------------------------
    # 5) rank (if there is a rank column) + fallback (if there is not)
    # -------------------------

    def test_real_11_rank_min_rank_row_if_present(self):
        for path in self.table_paths:
            df = self._df(path)
            # search for a rank column case-insensitively
            rank_col = None
            for c in df.columns:
                if str(c).strip().lower() == "rank":
                    rank_col = c
                    break
            if rank_col is None:
                continue
            # and we also need at least one other column to extract from
            out_col = None
            for c in df.columns:
                if c != rank_col:
                    out_col = c
                    break
            if out_col is None:
                continue

            # reference: choose the row with the minimal numeric rank (the first one in case of a tie)
            rows = list(range(len(df)))
            best_r = None
            best_v = None
            for r in rows:
                n = _parse_number(df.at[r, rank_col])
                if n is None:
                    continue
                if best_r is None or n < best_v:
                    best_r = r
                    best_v = n
            if best_r is None:
                continue

            expected_norm = _norm_text(df.at[best_r, out_col])
            qexp = _safe_quoted_literal(str(df.at[best_r, out_col]))
            if not qexp:
                continue

            program = f"eq{{rank{{all_rows; {out_col}}}; {qexp}}}=True"
            res = self.engine.execute(program, df)

            with self.subTest(table=path, out_col=out_col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final, f"expected_norm={expected_norm}, expr={res.expr_value}")

    def test_real_12_rank_fallback_top_when_no_rank(self):
        for path in self.table_paths:
            df = self._df(path)
            has_rank = any(str(c).strip().lower() == "rank" for c in df.columns)
            if has_rank:
                continue
            if len(df) == 0 or df.shape[1] == 0:
                continue
            out_col = df.columns[0]
            qexp = _safe_quoted_literal(str(df.at[0, out_col]))
            if not qexp:
                continue

            program = f"eq{{rank{{all_rows; {out_col}}}; {qexp}}}=True"
            res = self.engine.execute(program, df)

            with self.subTest(table=path, out_col=out_col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final)

    # -------------------------
    # 6) Complex chained queries (practical)
    # -------------------------

    def test_real_13_complex_chain_filter_then_argmax_then_compare(self):
        """
        Build a real complex query:
        - take a text column and a value (filter_eq)
        - take a numeric column
        - perform argmax over the numeric column on the subset rows
        - compare that hop(...) >= median(subset)
        """
        for path in self.table_paths:
            df = self._df(path)
            pick = _pick_text_column_and_value(df)
            num_col = _pick_numeric_column(df)
            if pick is None or num_col is None:
                continue
            text_col, qval = pick

            subset = _rowset_filter_eq(df, text_col, qval)
            if len(subset) < 3:
                continue

            # numbers on subset
            nums = []
            for r in subset:
                n = _parse_number(df.at[r, num_col])
                if n is not None:
                    nums.append(n)
            if len(nums) < 3:
                continue

            nums_sorted = sorted(nums)
            med = nums_sorted[len(nums_sorted) // 2]

            # expected: max(subset) >= med  (this must be True by definition)
            # in DSL this would be: greater_eq{hop{argmax{subset; num_col}; num_col}; med}, but there is no greater_eq,
            # so we use: not{less{...; med}} (equivalent to >=)
            program = (
                f"not{{less{{"
                f"  hop{{argmax{{filter_eq{{all_rows; {text_col}; {qval}}}; {num_col}}}; {num_col}}};"
                f"  {med}"
                f"}}}}=True"
            )
            res = self.engine.execute(program, df)

            with self.subTest(table=path, text_col=text_col, num_col=num_col):
                self.assertTrue(res.executable, res.error)
                self.assertTrue(res.final)

    def test_real_14_complex_chain_two_filters_and_and_or(self):
        """
        Another “practical” template:
        - choose text_col/value (subset A)
        - choose a numeric column and threshold = median(all numeric values)
        - check (within subsetA) AND (count(filter_greater(all_rows)) >= 1) OR (none(filter_less(subsetA)))
        """
        for path in self.table_paths:
            df = self._df(path)
            pick = _pick_text_column_and_value(df)
            num_col = _pick_numeric_column(df)
            if pick is None or num_col is None:
                continue
            text_col, qval = pick

            # all numeric threshold
            nn = [x for x in (_parse_number(v) for v in df[num_col].astype(str).tolist()) if x is not None]
            if len(nn) < 8:
                continue
            thr = sorted(nn)[len(nn) // 2]

            # reference parts
            subset = _rowset_filter_eq(df, text_col, qval)
            within_subset = (len(subset) > 0)

            gt_rows = _rowset_filter_num(df, num_col, thr, "gt")
            part2 = (len(gt_rows) >= 1)

            lt_subset = _rowset_filter_num(df.iloc[subset] if subset else df.head(0), num_col, thr, "lt")
            part3 = (len(lt_subset) == 0)

            expected = (within_subset and part2) or part3

            program = (
                f"or{{"
                f"  and{{"
                f"    within{{filter_eq{{all_rows; {text_col}; {qval}}}; {text_col}; {qval}}};"
                f"    not{{zero{{count{{filter_greater{{all_rows; {num_col}; {thr}}}}}}}}}"
                f"  }};"
                f"  none{{hop{{filter_less{{filter_eq{{all_rows; {text_col}; {qval}}}; {num_col}; {thr}}}; {num_col}}}}}"
                f"}}=True"
            )

            res = self.engine.execute(program, df)

            with self.subTest(table=path, text_col=text_col, num_col=num_col):
                self.assertTrue(res.executable, res.error)
                self.assertEqual(res.expr_value, expected)



if __name__ == "__main__":
    unittest.main(verbosity=2)
