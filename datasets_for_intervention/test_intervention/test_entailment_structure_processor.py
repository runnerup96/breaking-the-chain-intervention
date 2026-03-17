"""
Comprehensive tests for EntailmentTool, module-level parsing utilities,
and EntailmentStructureProcessor.

Coverage map
============
Rule                        — construction, iteration, indexing, repr
parse_step_proof            — standard, single-lhs, zero-lhs, annotations,
                              whitespace variants, invalid chunks, edge cases
serialize_step_proof        — arity 0/1/N, annotations on/off, empty list, roundtrip
build_graph                 — parents/children membership and structure
collect_supporting_rules    — chain, branch, missing target, deep nesting, no int* path
EntailmentTool              — spec, validate_args, calculate_score
extract_mediator            — full ## prefix, short prefix fallbacks, numbered prefix,
                              2)-stripping, reversed order, zero/two-count guards,
                              empty, whitespace, no structure, multiline proof content
extract_final_answer        — Yes/No, ambiguous, missing, empty, short_completion mode,
                              multiple FA sections in input, case sensitivity
extract_tool_args           — tool_mode off, full block, short mode, int-filter,
                              sorting, code-fence stripping, empty nodes, partial JSON
compare_structures          — identical, whitespace-normalised, annotation-agnostic,
                              different, None inputs, unparseable
jaccard_similarity          — identical, disjoint, partial, only-int nodes, None
check_generation_format_mistakes
                            — ## Proof ok, plain Proof ok, preamble, # Proof,
                              numbered prefix, multiple FA, empty, whitespace-only
"""

import json
import unittest

from datasets_for_intervention.entailment_structure_processor import (
    EntailmentTool,
    EntailmentStructureProcessor,
    Rule,
    parse_step_proof,
    serialize_step_proof,
    build_graph,
    collect_supporting_rules,
)
from datasets_for_intervention.entailment_intervention import (
    delete_one_antecedent,
    replace_antecedent_with_distractor,
    rewire_drop_support_creation,
    global_break,
    intervene_step_proof,
    _resolve_target_rhs,
    _pick_distractor,
    _ensure_structural_change,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PROOF_2STEP = "sent1 & sent2 -> int1; int1 & sent3 -> hypothesis; "
PROOF_1STEP = "sent16 & sent24 -> hypothesis; "
PROOF_DEEP  = "sent1 & sent2 -> int1; int1 & sent3 -> int2; int2 & sent4 -> hypothesis; "
PROOF_ANN   = "sent1 & sent2 -> int1: the northern hemisphere is a place; int1 & sent3 -> hypothesis; "
DISTRACTORS = ["sent10", "sent11", "sent12", "sent13"]


# ===========================================================================
# Rule
# ===========================================================================

class TestRule(unittest.TestCase):

    def test_construction(self):
        r = Rule(["sent1", "sent2"], "int1", "some annotation")
        self.assertEqual(r.lhs_ids, ["sent1", "sent2"])
        self.assertEqual(r.rhs_id, "int1")
        self.assertEqual(r.annotation, "some annotation")

    def test_construction_no_annotation(self):
        r = Rule(["sent1"], "hypothesis")
        self.assertIsNone(r.annotation)

    def test_tuple_unpack(self):
        r = Rule(["sent1", "sent2"], "int1")
        lhs, rhs = r
        self.assertEqual(lhs, ["sent1", "sent2"])
        self.assertEqual(rhs, "int1")

    def test_indexing(self):
        r = Rule(["sent1"], "hypothesis")
        self.assertEqual(r[0], ["sent1"])
        self.assertEqual(r[1], "hypothesis")

    def test_indexing_out_of_range(self):
        r = Rule(["sent1"], "hypothesis")
        with self.assertRaises(IndexError):
            _ = r[2]

    def test_repr(self):
        r = Rule(["sent1"], "hyp", "ann")
        self.assertIn("sent1", repr(r))
        self.assertIn("hyp", repr(r))
        self.assertIn("ann", repr(r))


# ===========================================================================
# parse_step_proof
# ===========================================================================

class TestParseStepProof(unittest.TestCase):

    # ---- happy path ----

    def test_two_step_proof(self):
        rules = parse_step_proof(PROOF_2STEP)
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0].lhs_ids, ["sent1", "sent2"])
        self.assertEqual(rules[0].rhs_id, "int1")
        self.assertIsNone(rules[0].annotation)
        self.assertEqual(rules[1].lhs_ids, ["int1", "sent3"])
        self.assertEqual(rules[1].rhs_id, "hypothesis")

    def test_single_antecedent(self):
        rules = parse_step_proof("sent1 -> hypothesis; ")
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].lhs_ids, ["sent1"])
        self.assertEqual(rules[0].rhs_id, "hypothesis")

    def test_high_arity_lhs(self):
        rules = parse_step_proof("sent1 & sent2 & sent3 & sent4 -> int1; ")
        self.assertEqual(rules[0].lhs_ids, ["sent1", "sent2", "sent3", "sent4"])

    def test_zero_lhs(self):
        # "-> hypothesis" — no antecedents
        rules = parse_step_proof("-> hypothesis; ")
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].lhs_ids, [])
        self.assertEqual(rules[0].rhs_id, "hypothesis")

    def test_annotation_preserved(self):
        rules = parse_step_proof(PROOF_ANN)
        self.assertEqual(rules[0].annotation, "the northern hemisphere is a place")
        self.assertIsNone(rules[1].annotation)

    def test_annotation_with_colon_in_text(self):
        # annotation itself contains a colon — only first colon splits
        rules = parse_step_proof("sent1 -> int1: note: sub-note; ")
        self.assertEqual(rules[0].annotation, "note: sub-note")

    def test_empty_annotation_becomes_none(self):
        # "sent1 -> int1: " — annotation part is empty string
        rules = parse_step_proof("sent1 -> int1: ; ")
        self.assertIsNone(rules[0].annotation)

    def test_no_trailing_semicolon(self):
        rules = parse_step_proof("sent1 -> hypothesis")
        self.assertEqual(len(rules), 1)

    def test_deep_proof(self):
        rules = parse_step_proof(PROOF_DEEP)
        self.assertEqual(len(rules), 3)
        self.assertEqual(rules[2].rhs_id, "hypothesis")

    # ---- whitespace tolerance ----

    def test_extra_spaces_around_arrow(self):
        rules = parse_step_proof("sent1  &  sent2  ->  int1 ; ")
        self.assertEqual(rules[0].lhs_ids, ["sent1", "sent2"])
        self.assertEqual(rules[0].rhs_id, "int1")

    def test_leading_trailing_whitespace_chunks(self):
        rules = parse_step_proof("  sent1 -> int1 ;  sent2 -> hypothesis ; ")
        self.assertEqual(len(rules), 2)

    # ---- invalid / degenerate inputs ----

    def test_empty_string(self):
        self.assertEqual(parse_step_proof(""), [])

    def test_only_whitespace(self):
        self.assertEqual(parse_step_proof("   \n\t  "), [])

    def test_only_semicolons(self):
        self.assertEqual(parse_step_proof(";;; ;"), [])

    def test_chunk_without_arrow_skipped(self):
        # no -> in this chunk
        rules = parse_step_proof("sent1 sent2 int1; sent3 -> hypothesis; ")
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].rhs_id, "hypothesis")

    def test_invalid_rhs_multi_word_skipped(self):
        # rhs has a space → not \w+ → skipped
        rules = parse_step_proof("sent1 -> int 1; sent2 -> hypothesis; ")
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].rhs_id, "hypothesis")

    def test_mixed_valid_invalid(self):
        proof = "sent1 & sent2 -> int1; bad chunk; int1 -> hypothesis; "
        rules = parse_step_proof(proof)
        self.assertEqual(len(rules), 2)

    def test_returns_list_of_rules(self):
        rules = parse_step_proof(PROOF_2STEP)
        for r in rules:
            self.assertIsInstance(r, Rule)


# ===========================================================================
# serialize_step_proof
# ===========================================================================

class TestSerializeStepProof(unittest.TestCase):

    def test_empty_list(self):
        # Empty list → just trailing "; "
        self.assertEqual(serialize_step_proof([]), "; ")

    def test_single_rule_single_lhs(self):
        rules = [Rule(["sent1"], "hypothesis")]
        s = serialize_step_proof(rules)
        self.assertIn("sent1 -> hypothesis", s)
        self.assertTrue(s.endswith("; "))

    def test_single_rule_multi_lhs(self):
        rules = [Rule(["sent1", "sent2"], "int1")]
        s = serialize_step_proof(rules)
        self.assertIn("sent1 & sent2 -> int1", s)

    def test_single_rule_zero_lhs(self):
        rules = [Rule([], "hypothesis")]
        s = serialize_step_proof(rules)
        self.assertIn("-> hypothesis", s)

    def test_annotation_included_by_default(self):
        rules = [Rule(["sent1"], "int1", "the note")]
        s = serialize_step_proof(rules)
        self.assertIn(": the note", s)

    def test_annotation_excluded_when_flag_false(self):
        rules = [Rule(["sent1"], "int1", "the note")]
        s = serialize_step_proof(rules, include_annotations=False)
        self.assertNotIn("the note", s)

    def test_multiple_rules_joined_by_semicolon(self):
        rules = parse_step_proof(PROOF_2STEP)
        s = serialize_step_proof(rules)
        self.assertEqual(s.count(";"), 2)  # one per rule + trailing

    def test_roundtrip_fields(self):
        rules = parse_step_proof(PROOF_2STEP)
        rt = parse_step_proof(serialize_step_proof(rules))
        self.assertEqual(len(rt), len(rules))
        for orig, back in zip(rules, rt):
            self.assertEqual(orig.lhs_ids, back.lhs_ids)
            self.assertEqual(orig.rhs_id, back.rhs_id)

    def test_roundtrip_with_annotation(self):
        rules = parse_step_proof(PROOF_ANN)
        rt = parse_step_proof(serialize_step_proof(rules))
        self.assertEqual(rt[0].annotation, rules[0].annotation)

    def test_trailing_semicolon_space(self):
        rules = parse_step_proof(PROOF_1STEP)
        s = serialize_step_proof(rules)
        self.assertTrue(s.endswith("; "))


# ===========================================================================
# build_graph
# ===========================================================================

class TestBuildGraph(unittest.TestCase):

    def test_parents_and_children_basic(self):
        rules = parse_step_proof(PROOF_2STEP)
        parents, children = build_graph(rules)
        # int1 is produced by sent1 & sent2
        self.assertIn("int1", parents)
        # sent1 has a child int1
        self.assertIn("int1", children["sent1"])
        # hypothesis is produced by int1 & sent3
        self.assertIn("hypothesis", parents)

    def test_children_of_int_node(self):
        rules = parse_step_proof(PROOF_2STEP)
        _, children = build_graph(rules)
        self.assertIn("hypothesis", children["int1"])

    def test_leaf_sent_nodes_have_children(self):
        rules = parse_step_proof(PROOF_2STEP)
        _, children = build_graph(rules)
        for sent in ["sent1", "sent2", "sent3"]:
            self.assertIn(sent, children)
            self.assertTrue(len(children[sent]) > 0)

    def test_empty_rules(self):
        parents, children = build_graph([])
        self.assertEqual(len(parents), 0)
        self.assertEqual(len(children), 0)


# ===========================================================================
# collect_supporting_rules
# ===========================================================================

class TestCollectSupportingRules(unittest.TestCase):

    def test_all_rules_support_hypothesis(self):
        rules = parse_step_proof(PROOF_2STEP)
        idxs = collect_supporting_rules(rules, "hypothesis")
        self.assertEqual(sorted(idxs), [0, 1])

    def test_single_step_proof(self):
        rules = parse_step_proof(PROOF_1STEP)
        idxs = collect_supporting_rules(rules, "hypothesis")
        self.assertEqual(idxs, [0])

    def test_deep_chain_all_rules(self):
        rules = parse_step_proof(PROOF_DEEP)
        idxs = collect_supporting_rules(rules, "hypothesis")
        self.assertEqual(sorted(idxs), [0, 1, 2])

    def test_target_not_in_proof_returns_empty(self):
        rules = parse_step_proof(PROOF_2STEP)
        self.assertEqual(collect_supporting_rules(rules, "nonexistent"), [])

    def test_target_is_leaf_sent_returns_empty(self):
        # sent1 appears only as LHS — not produced by any rule
        rules = parse_step_proof(PROOF_2STEP)
        self.assertEqual(collect_supporting_rules(rules, "sent1"), [])

    def test_target_is_int1_only_first_rule(self):
        rules = parse_step_proof(PROOF_2STEP)
        idxs = collect_supporting_rules(rules, "int1")
        self.assertEqual(idxs, [0])  # only the rule that produces int1

    def test_branching_proof_only_relevant_branch(self):
        # proof with two independent chains
        proof = "sent1 -> int1; int1 -> conclusion_a; sent2 -> int2; int2 -> conclusion_b; "
        rules = parse_step_proof(proof)
        idxs_a = collect_supporting_rules(rules, "conclusion_a")
        idxs_b = collect_supporting_rules(rules, "conclusion_b")
        self.assertEqual(sorted(idxs_a), [0, 1])
        self.assertEqual(sorted(idxs_b), [2, 3])
        # No overlap
        self.assertEqual(set(idxs_a) & set(idxs_b), set())

    def test_returns_sorted_list(self):
        rules = parse_step_proof(PROOF_DEEP)
        idxs = collect_supporting_rules(rules, "hypothesis")
        self.assertEqual(idxs, sorted(idxs))

    def test_empty_rules_list(self):
        self.assertEqual(collect_supporting_rules([], "hypothesis"), [])


# ===========================================================================
# EntailmentTool
# ===========================================================================

class TestEntailmentTool(unittest.TestCase):

    def setUp(self):
        self.tool = EntailmentTool()

    def test_name(self):
        self.assertEqual(self.tool.name, "verify_proof")

    def test_spec_is_dict_with_required_keys(self):
        spec = self.tool.spec
        self.assertIn("properties", spec)
        self.assertIn("proof_nodes", spec["properties"])
        self.assertIn("required", spec)

    def test_spec_json_is_valid_json(self):
        parsed = json.loads(self.tool.spec_json())
        self.assertIn("proof_nodes", parsed["properties"])

    def test_validate_args_valid(self):
        self.assertTrue(self.tool.validate_args({"proof_nodes": ["sent1", "sent3"]}))

    def test_validate_args_single_node(self):
        self.assertTrue(self.tool.validate_args({"proof_nodes": ["sent99"]}))

    def test_validate_args_rejects_empty_list(self):
        self.assertFalse(self.tool.validate_args({"proof_nodes": []}))

    def test_validate_args_rejects_missing_key(self):
        self.assertFalse(self.tool.validate_args({}))

    def test_validate_args_rejects_int_ids(self):
        self.assertFalse(self.tool.validate_args({"proof_nodes": ["int1"]}))

    def test_validate_args_rejects_mixed_valid_invalid(self):
        # int1 is invalid
        self.assertFalse(self.tool.validate_args({"proof_nodes": ["sent1", "int1"]}))

    def test_validate_args_rejects_id_with_space(self):
        self.assertFalse(self.tool.validate_args({"proof_nodes": ["sent 1"]}))

    def test_validate_args_rejects_non_dict(self):
        self.assertFalse(self.tool.validate_args(None))
        self.assertFalse(self.tool.validate_args(["sent1"]))
        self.assertFalse(self.tool.validate_args("sent1"))

    def test_calculate_score_valid(self):
        self.assertIs(self.tool.calculate_score({"proof_nodes": ["sent1"]}, {}), True)

    def test_calculate_score_invalid_returns_none(self):
        self.assertIsNone(self.tool.calculate_score({}, {}))
        self.assertIsNone(self.tool.calculate_score({"proof_nodes": []}, {}))


# ===========================================================================
# extract_mediator
# ===========================================================================

class TestExtractMediator(unittest.TestCase):

    def setUp(self):
        self.proc = EntailmentStructureProcessor()

    # ---- full ## prefix ----

    def test_full_prefix_yes(self):
        c = f"## Proof\n{PROOF_2STEP}## Final Answer\nIs the hypothesis correct? Yes"
        self.assertEqual(self.proc.extract_mediator(c), PROOF_2STEP.strip())

    def test_full_prefix_no(self):
        c = f"## Proof\n{PROOF_2STEP}## Final Answer\nIs the hypothesis correct? No"
        self.assertEqual(self.proc.extract_mediator(c), PROOF_2STEP.strip())

    def test_full_prefix_multiline_proof(self):
        multiline = "sent1 & sent2 -> int1;\nint1 & sent3 -> hypothesis;\n"
        c = f"## Proof\n{multiline}## Final Answer\nIs the hypothesis correct? Yes"
        result = self.proc.extract_mediator(c)
        self.assertIsNotNone(result)
        self.assertIn("sent1", result)

    def test_full_prefix_preserves_annotation(self):
        c = f"## Proof\n{PROOF_ANN}## Final Answer\nIs the hypothesis correct? Yes"
        result = self.proc.extract_mediator(c)
        self.assertIsNotNone(result)
        self.assertIn("the northern hemisphere is a place", result)

    # ---- short prefix fallback ----

    def test_short_prefix_hash_proof(self):
        c = f"# Proof\n{PROOF_2STEP} # Final Answer\nYes"
        result = self.proc.extract_mediator(c)
        self.assertIsNotNone(result)
        self.assertIn("sent1", result)

    def test_short_prefix_bare_proof(self):
        c = f"Proof\n{PROOF_2STEP}Final Answer\nYes"
        result = self.proc.extract_mediator(c)
        self.assertIsNotNone(result)
        self.assertIn("sent1", result)

    def test_numbered_prefix_strips_2_paren(self):
        c = f"1) Proof:\n{PROOF_2STEP} 2) Final Answer:\nYes"
        result = self.proc.extract_mediator(c)
        self.assertIsNotNone(result)
        self.assertNotIn("2)", result)

    def test_numbered_prefix_with_blank_line(self):
        c = f"1) Proof:\n{PROOF_2STEP} \n\n 2) Final Answer:\nYes"
        result = self.proc.extract_mediator(c)
        self.assertIsNotNone(result)

    # ---- must return None ----

    def test_empty_string(self):
        self.assertIsNone(self.proc.extract_mediator(""))

    def test_whitespace_only(self):
        self.assertIsNone(self.proc.extract_mediator("   \n\t  "))

    def test_no_structure(self):
        self.assertIsNone(self.proc.extract_mediator("This is some random text."))

    def test_missing_proof_section(self):
        # "Final Answer" present, "Proof" absent
        self.assertIsNone(self.proc.extract_mediator("## Final Answer\nIs the hypothesis correct? Yes"))

    def test_missing_final_answer_section(self):
        self.assertIsNone(self.proc.extract_mediator(f"## Proof\n{PROOF_2STEP}"))

    def test_reversed_order(self):
        c = f"## Final Answer\nIs the hypothesis correct? Yes\n## Proof\n{PROOF_2STEP}"
        self.assertIsNone(self.proc.extract_mediator(c))

    def test_two_proof_sections(self):
        c = (
            f"## Proof\n{PROOF_2STEP}## Final Answer\nYes\n"
            f"## Proof\nsent5 -> int3; ## Final Answer\nNo"
        )
        self.assertIsNone(self.proc.extract_mediator(c))

    def test_two_final_answer_sections(self):
        c = f"## Proof\n{PROOF_2STEP}## Final Answer\nYes\n## Final Answer\nNo"
        self.assertIsNone(self.proc.extract_mediator(c))

    def test_proof_word_in_proof_text_causes_double_count(self):
        # If the word "Proof" appears in the annotation, count becomes 2 → None
        c = "## Proof\nsent1 -> int1: Proof of concept; ## Final Answer\nYes"
        self.assertIsNone(self.proc.extract_mediator(c))

    def test_zero_proof_occurrences(self):
        c = "## Evidence\nsent1 -> hypothesis; ## Final Answer\nYes"
        self.assertIsNone(self.proc.extract_mediator(c))


# ===========================================================================
# extract_final_answer
# ===========================================================================

class TestExtractFinalAnswer(unittest.TestCase):

    def setUp(self):
        self.proc = EntailmentStructureProcessor()

    # ---- True cases ----

    def test_full_prefix_yes(self):
        self.assertIs(
            self.proc.extract_final_answer("## Final Answer\nIs the hypothesis correct? Yes"),
            True
        )

    def test_short_prefix_yes(self):
        self.assertIs(self.proc.extract_final_answer("Final Answer\nYes"), True)

    def test_hash_prefix_yes(self):
        self.assertIs(self.proc.extract_final_answer("# Final Answer\nYes"), True)

    def test_inline_yes(self):
        self.assertIs(
            self.proc.extract_final_answer("## Final Answer Is the hypothesis correct? Yes"),
            True
        )

    def test_yes_with_leading_colon(self):
        self.assertIs(self.proc.extract_final_answer("Final Answer:\nYes"), True)

    def test_yes_in_longer_completion(self):
        c = f"## Proof\n{PROOF_2STEP}## Final Answer\nIs the hypothesis correct? Yes"
        self.assertIs(self.proc.extract_final_answer(c), True)

    # ---- False cases ----

    def test_full_prefix_no(self):
        self.assertIs(self.proc.extract_final_answer("## Final Answer\nNo"), False)

    def test_short_prefix_no(self):
        self.assertIs(self.proc.extract_final_answer("Final Answer\nNo"), False)

    def test_inline_no(self):
        self.assertIs(
            self.proc.extract_final_answer("## Final Answer Is the hypothesis correct? No"),
            False
        )

    # ---- None cases ----

    def test_empty_string(self):
        self.assertIsNone(self.proc.extract_final_answer(""))

    def test_ambiguous_yes_and_no_in_tail(self):
        self.assertIsNone(
            self.proc.extract_final_answer("## Final Answer\nYes and No")
        )

    def test_neither_yes_nor_no(self):
        self.assertIsNone(self.proc.extract_final_answer("## Final Answer\nmaybe"))

    def test_no_final_answer_section_no_short(self):
        # No "Final Answer" prefix, not short_completion → searches whole text
        # "Yes" is not present → None
        self.assertIsNone(self.proc.extract_final_answer("## Proof\nsent1 -> hypothesis; "))

    def test_no_section_text_contains_both(self):
        # No FA section, whole text has both Yes and No
        self.assertIsNone(self.proc.extract_final_answer("Yes No"))

    # ---- short_completion=True ----

    def test_short_yes(self):
        self.assertIs(self.proc.extract_final_answer("Yes", short_completion=True), True)

    def test_short_no(self):
        self.assertIs(self.proc.extract_final_answer("No", short_completion=True), False)

    def test_short_ambiguous(self):
        self.assertIsNone(self.proc.extract_final_answer("Yes and No", short_completion=True))

    def test_short_empty(self):
        self.assertIsNone(self.proc.extract_final_answer("", short_completion=True))

    def test_short_random_text(self):
        self.assertIsNone(self.proc.extract_final_answer("I think so", short_completion=True))

    def test_short_takes_whole_text_when_no_fa_prefix(self):
        # short_completion uses whole text as tail when no FA prefix
        self.assertIs(self.proc.extract_final_answer("Yes!", short_completion=True), True)

    # ---- FA prefix still wins over short_completion ----

    def test_fa_prefix_takes_priority_in_short_mode(self):
        # Has FA section — always uses tail after FA prefix
        c = "## Final Answer\nYes"
        self.assertIs(self.proc.extract_final_answer(c, short_completion=True), True)


# ===========================================================================
# extract_tool_args
# ===========================================================================

class TestExtractToolArgs(unittest.TestCase):

    def setUp(self):
        self.proc      = EntailmentStructureProcessor(tool_mode="simple")
        self.proc_none = EntailmentStructureProcessor(tool_mode="none")

    def test_returns_none_when_tool_mode_off(self):
        self.assertIsNone(self.proc_none.extract_tool_args('ARGS: {"proof_nodes": ["sent1"]}'))

    def test_parses_standard_args_block(self):
        r = self.proc.extract_tool_args('ARGS: {"proof_nodes": ["sent1", "sent3"]}')
        self.assertIsNotNone(r)
        self.assertEqual(r["proof_nodes"], ["sent1", "sent3"])

    def test_filters_int_ids(self):
        r = self.proc.extract_tool_args('ARGS: {"proof_nodes": ["sent1", "int1", "sent3"]}')
        self.assertNotIn("int1", r["proof_nodes"])
        self.assertIn("sent1", r["proof_nodes"])
        self.assertIn("sent3", r["proof_nodes"])

    def test_only_int_ids_returns_none(self):
        self.assertIsNone(self.proc.extract_tool_args('ARGS: {"proof_nodes": ["int1", "int2"]}'))

    def test_returns_sorted(self):
        r = self.proc.extract_tool_args('ARGS: {"proof_nodes": ["sent3", "sent1", "sent20"]}')
        self.assertEqual(r["proof_nodes"], sorted(r["proof_nodes"]))

    def test_empty_proof_nodes_returns_none(self):
        self.assertIsNone(self.proc.extract_tool_args('ARGS: {"proof_nodes": []}'))

    def test_missing_args_block_returns_none(self):
        self.assertIsNone(self.proc.extract_tool_args("No args here at all"))

    def test_short_completion_mode(self):
        r = self.proc.extract_tool_args('{"proof_nodes": ["sent2", "sent5"]}', short_completion=True)
        self.assertIsNotNone(r)
        self.assertIn("sent2", r["proof_nodes"])
        self.assertIn("sent5", r["proof_nodes"])

    def test_code_fence_stripped(self):
        text = 'ARGS: ```json\n{"proof_nodes": ["sent1"]}\n```'
        r = self.proc.extract_tool_args(text)
        self.assertIsNotNone(r)
        self.assertIn("sent1", r["proof_nodes"])

    def test_empty_text_returns_none(self):
        self.assertIsNone(self.proc.extract_tool_args(""))
        self.assertIsNone(self.proc.extract_tool_args(None))


# ===========================================================================
# compare_structures
# ===========================================================================

class TestCompareStructures(unittest.TestCase):

    def setUp(self):
        self.proc = EntailmentStructureProcessor()

    def test_identical_returns_1(self):
        self.assertEqual(self.proc.compare_structures(PROOF_2STEP, PROOF_2STEP), 1)

    def test_whitespace_variant_normalises_to_1(self):
        spaced = PROOF_2STEP.replace(";", " ;").replace("->", " -> ").replace("&", " & ")
        self.assertEqual(self.proc.compare_structures(PROOF_2STEP, spaced), 1)

    def test_different_proofs_returns_0(self):
        self.assertEqual(
            self.proc.compare_structures(PROOF_2STEP, PROOF_1STEP), 0
        )

    def test_modified_rhs_returns_0(self):
        rules = parse_step_proof(PROOF_2STEP)
        rules[-1].rhs_id = "bogus"
        modified = serialize_step_proof(rules)
        self.assertEqual(self.proc.compare_structures(PROOF_2STEP, modified), 0)

    def test_none_a_returns_none(self):
        self.assertIsNone(self.proc.compare_structures(None, PROOF_2STEP))

    def test_none_b_returns_none(self):
        self.assertIsNone(self.proc.compare_structures(PROOF_2STEP, None))

    def test_both_none_returns_none(self):
        self.assertIsNone(self.proc.compare_structures(None, None))

    def test_unparseable_a_returns_none(self):
        self.assertIsNone(self.proc.compare_structures("complete garbage", PROOF_2STEP))

    def test_unparseable_b_returns_none(self):
        self.assertIsNone(self.proc.compare_structures(PROOF_2STEP, "complete garbage"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.proc.compare_structures("", PROOF_2STEP))

    def test_annotation_difference_returns_0(self):
        # Same structure, different annotation → normalized differ
        no_ann = serialize_step_proof(parse_step_proof(PROOF_ANN), include_annotations=False)
        # PROOF_ANN has annotation, serialize without keeps no annotation
        # But _normalize_proof always uses include_annotations=True
        # So proofs with different annotations → different serialized → 0
        rules_no_ann = parse_step_proof(PROOF_ANN)
        rules_no_ann[0].annotation = None
        other = serialize_step_proof(rules_no_ann)
        self.assertEqual(self.proc.compare_structures(PROOF_ANN, other), 0)

    def test_same_structure_without_annotation_matches(self):
        # Two proofs with same nodes, no annotation → 1
        p1 = "sent1 & sent2 -> int1; int1 & sent3 -> hypothesis; "
        p2 = "sent1 & sent2 -> int1; int1 & sent3 -> hypothesis; "
        self.assertEqual(self.proc.compare_structures(p1, p2), 1)


# ===========================================================================
# jaccard_similarity
# ===========================================================================

class TestJaccardSimilarity(unittest.TestCase):

    def setUp(self):
        self.proc = EntailmentStructureProcessor()

    def test_identical(self):
        self.assertEqual(self.proc.jaccard_similarity(PROOF_2STEP, PROOF_2STEP), 1.0)

    def test_disjoint(self):
        a = "sent1 -> hypothesis; "
        b = "sent2 -> hypothesis; "
        self.assertEqual(self.proc.jaccard_similarity(a, b), 0.0)

    def test_partial_overlap(self):
        a = "sent1 & sent2 -> hypothesis; "
        b = "sent1 & sent3 -> hypothesis; "
        # intersection={sent1}, union={sent1,sent2,sent3} → 1/3
        self.assertAlmostEqual(self.proc.jaccard_similarity(a, b), 1/3, places=5)

    def test_full_overlap_different_proof_structure(self):
        # Same sentences but different chain structure
        a = "sent1 & sent2 -> int1; int1 -> hypothesis; "
        b = "sent1 & sent2 -> hypothesis; "
        self.assertEqual(self.proc.jaccard_similarity(a, b), 1.0)

    def test_int_nodes_not_counted(self):
        # int1 should not appear in sent* set
        a = "sent1 & sent2 -> int1; int1 -> hypothesis; "
        b = "sent1 & sent2 -> hypothesis; "
        # Both have sent1, sent2 → jaccard = 1
        self.assertEqual(self.proc.jaccard_similarity(a, b), 1.0)

    def test_empty_both_returns_1(self):
        # No sent* IDs in either
        a = "int1 -> int2; "
        b = "int3 -> int4; "
        self.assertEqual(self.proc.jaccard_similarity(a, b), 1.0)

    def test_none_a_returns_none(self):
        self.assertIsNone(self.proc.jaccard_similarity(None, PROOF_2STEP))

    def test_none_b_returns_none(self):
        self.assertIsNone(self.proc.jaccard_similarity(PROOF_2STEP, None))


# ===========================================================================
# check_generation_format_mistakes
# ===========================================================================

class TestCheckGenerationFormatMistakes(unittest.TestCase):

    def setUp(self):
        self.proc = EntailmentStructureProcessor()

    # ---- clean completions (False) ----

    def test_full_prefix_clean(self):
        c = f"## Proof\n{PROOF_2STEP}## Final Answer\nYes"
        self.assertFalse(self.proc.check_generation_format_mistakes(c))

    def test_bare_proof_prefix_clean(self):
        # Starts with "Proof" (no ##) — valid
        c = f"Proof\n{PROOF_2STEP}Final Answer\nYes"
        self.assertFalse(self.proc.check_generation_format_mistakes(c))

    def test_proof_word_at_start_clean(self):
        self.assertFalse(self.proc.check_generation_format_mistakes("Proof: something"))

    def test_proof_with_leading_newline_is_error(self):
        # Leading whitespace is stripped, so \n## Proof → ## Proof after strip
        c = f"\n## Proof\n{PROOF_2STEP}## Final Answer\nYes"
        self.assertFalse(self.proc.check_generation_format_mistakes(c))

    # ---- format mistakes (True) ----

    def test_preamble_before_proof(self):
        c = f"Sure! Here is my answer:\n## Proof\n{PROOF_2STEP}## Final Answer\nYes"
        self.assertTrue(self.proc.check_generation_format_mistakes(c))

    def test_hash_proof_not_double_hash_is_error(self):
        # "# Proof" — single hash — does NOT match (?:##\s*)?Proof → error
        c = f"# Proof\n{PROOF_2STEP}# Final Answer\nYes"
        self.assertTrue(self.proc.check_generation_format_mistakes(c))

    def test_numbered_prefix_is_error(self):
        c = f"1) Proof:\n{PROOF_2STEP} 2) Final Answer:\nYes"
        self.assertTrue(self.proc.check_generation_format_mistakes(c))

    def test_empty_string_is_error(self):
        self.assertTrue(self.proc.check_generation_format_mistakes(""))

    def test_whitespace_only_is_error(self):
        self.assertTrue(self.proc.check_generation_format_mistakes("   \n\t "))

    def test_no_proof_section_is_error(self):
        self.assertTrue(self.proc.check_generation_format_mistakes("## Final Answer\nYes"))

    def test_multiple_final_answer_sections_is_error(self):
        c = f"## Proof\n{PROOF_2STEP}## Final Answer\nYes\n## Final Answer\nNo"
        self.assertTrue(self.proc.check_generation_format_mistakes(c))

    def test_three_final_answer_sections_is_error(self):
        c = "## Proof\nx; Final Answer\nYes\nFinal Answer\nNo\nFinal Answer\nYes"
        self.assertTrue(self.proc.check_generation_format_mistakes(c))


if __name__ == "__main__":
    unittest.main()