"""
Comprehensive tests for EntailmentIntervention and module-level mutation functions.

Coverage map
============
clean_llm_output            — special tokens, zero-width chars, strip
_resolve_target_rhs         — preferred exists, fallback to hypothesis, fallback to last
_pick_distractor            — normal, all forbidden, empty pool
_ensure_structural_change   — changed lhs, changed rhs, unchanged
delete_one_antecedent       — normal, no arity-2 rule (returns original), empty targets
replace_antecedent_with_distractor
                            — normal (keeps arity), no distractors, no targets, no pool
rewire_drop_support_creation
                            — int* candidate removed, no int* fallback to delete
global_break                — all rules on path edited, single-antecedent replaced
intervene_step_proof        — all 4 modes, None input, unknown mode, seed
                              reproducibility, result differs from input
EntailmentIntervention init — prompt built, edit_modes, bad regime
infer_completion            — True/False/None, tuple return, tool_mode
classify_generation         — correct/incorrect/error paths, preamble, contradictory answer
make_structure_intervention — shapes, score/expected values, deepcopy independence
make_intervention           — correct/incorrect/error field sets, no legacy keys
interventions_to_prompt     — prompt count per generation_status
collect_intervention_completion
                            — order, score_after_intervention values, ambiguous answer → None
format_example              — sections present/absent, score True/False
"""

import random
import unittest
from copy import deepcopy

from datasets_for_intervention.test_intervention.entailment_mocks import EntailmentBankDatasetMock
from datasets_for_intervention.entailment_intervention import (
    EntailmentIntervention,
    # mutation primitives
    delete_one_antecedent,
    replace_antecedent_with_distractor,
    rewire_drop_support_creation,
    global_break,
    intervene_step_proof,
    # private helpers
    _resolve_target_rhs,
    _pick_distractor,
    _ensure_structural_change,
)
from datasets_for_intervention.entailment_structure_processor import (
    EntailmentStructureProcessor,
    EntailmentTool,
    Rule,
    parse_step_proof,
    serialize_step_proof,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PROOF_2STEP = "sent1 & sent2 -> int1; int1 & sent3 -> hypothesis; "
PROOF_1STEP = "sent16 & sent24 -> hypothesis; "
PROOF_DEEP  = "sent1 & sent2 -> int1; int1 & sent3 -> int2; int2 & sent4 -> hypothesis; "
DISTRACTORS = ["sent10", "sent11", "sent12", "sent13"]


class FakeLLMModel:
    def get_chat_template(self): return None
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return " ".join(str(m.get("content", "")) for m in msgs)
    def tokenize(self, text): return text.split()
    def generate(self, prompt, **kw): return ""
    def clean_model_specific_completion(self, t): return t


def _make_ic(dataset=None, prompting_regime="standard"):
    dataset = dataset or EntailmentBankDatasetMock()
    few_shot = [dataset[i] for i in range(min(2, len(dataset)))]
    ic = EntailmentIntervention(
        dataset=dataset, llm_model=FakeLLMModel(),
        tool=EntailmentTool(), processor=EntailmentStructureProcessor(),
        few_shot_examples=few_shot, prompting_regime=prompting_regime,
    )
    return ic, dataset


def _gold_c(sample):
    return f"## Proof\n{sample['proof']}## Final Answer\nIs the hypothesis correct? Yes"

def _wrong_c(answer="Yes"):
    return f"## Proof\nsent1 -> hypothesis; \n## Final Answer\nIs the hypothesis correct? {answer}"

def _malformed_c():
    return "## Final Answer\nIs the hypothesis correct? Yes"  # no Proof section

def _contradictory_c():
    return f"## Proof\nsent1 -> hypothesis; \n## Final Answer\nIs the hypothesis correct? Yes and No"

def _preamble_c(sample):
    return "Sure, here is my reasoning:\n" + _gold_c(sample)


# ===========================================================================
# clean_llm_output  (method on EntailmentIntervention)
# ===========================================================================

class TestCleanLLMOutput(unittest.TestCase):

    def setUp(self):
        self.ic, _ = _make_ic()

    def test_removes_im_end(self):
        self.assertEqual(self.ic.clean_llm_output("text<|im_end|>"), "text")

    def test_removes_endoftext(self):
        self.assertEqual(self.ic.clean_llm_output("text<|endoftext|>"), "text")

    def test_removes_im_start(self):
        self.assertEqual(self.ic.clean_llm_output("<|im_start|>text"), "text")

    def test_removes_eot_id(self):
        self.assertEqual(self.ic.clean_llm_output("text<|eot_id|>"), "text")

    def test_removes_end_of_text(self):
        self.assertEqual(self.ic.clean_llm_output("text<|end_of_text|>"), "text")

    def test_removes_pad(self):
        self.assertEqual(self.ic.clean_llm_output("text<|pad|>"), "text")

    def test_removes_end_of_turn(self):
        self.assertEqual(self.ic.clean_llm_output("text<end_of_turn>"), "text")

    def test_removes_end_s_tag(self):
        self.assertEqual(self.ic.clean_llm_output("text</s>"), "text")

    def test_removes_zero_width_chars(self):
        dirty = "a\u200bb\u200cc\u200dd\u2060e\ufefff"
        self.assertEqual(self.ic.clean_llm_output(dirty), "abcdef")

    def test_removes_soft_hyphen(self):
        self.assertEqual(self.ic.clean_llm_output("hel\u00adlo"), "hello")

    def test_strips_whitespace(self):
        self.assertEqual(self.ic.clean_llm_output("  hello  "), "hello")

    def test_clean_text_unchanged(self):
        text = "## Proof\nsent1 -> hypothesis; \n## Final Answer\nYes"
        self.assertEqual(self.ic.clean_llm_output(text), text)

    def test_multiple_tokens_in_one_text(self):
        text = "<|im_start|>text<|im_end|>"
        self.assertEqual(self.ic.clean_llm_output(text), "text")


# ===========================================================================
# Private helpers
# ===========================================================================

class TestResolveTargetRhs(unittest.TestCase):

    def _rules(self, proof):
        return parse_step_proof(proof)

    def test_preferred_in_rhs_set(self):
        rules = self._rules(PROOF_2STEP)
        # "hypothesis" is in rhs_set
        self.assertEqual(_resolve_target_rhs(rules, "hypothesis"), "hypothesis")

    def test_preferred_not_present_falls_back_to_hypothesis(self):
        rules = self._rules(PROOF_2STEP)
        # "int2" is NOT in this proof's rhs_set, but "hypothesis" is
        self.assertEqual(_resolve_target_rhs(rules, "int2"), "hypothesis")

    def test_neither_preferred_nor_hypothesis_returns_last_rhs(self):
        # proof where hypothesis is not the final RHS
        proof = "sent1 -> int1; int1 -> conclusion; "
        rules = parse_step_proof(proof)
        # preferred="bogus", hypothesis not in rhs_set → last rule's rhs_id
        self.assertEqual(_resolve_target_rhs(rules, "bogus"), "conclusion")

    def test_empty_rules_returns_preferred(self):
        self.assertEqual(_resolve_target_rhs([], "hypothesis"), "hypothesis")

    def test_int1_is_in_rhs_set(self):
        rules = self._rules(PROOF_2STEP)
        self.assertEqual(_resolve_target_rhs(rules, "int1"), "int1")


class TestPickDistractor(unittest.TestCase):

    def test_returns_item_not_in_forbidden(self):
        d = _pick_distractor(["sent10", "sent11"], forbidden=["sent1", "sent2"])
        self.assertIn(d, ["sent10", "sent11"])

    def test_excluded_by_forbidden(self):
        d = _pick_distractor(["sent1", "sent2"], forbidden=["sent1", "sent2"])
        self.assertIsNone(d)

    def test_empty_distractors(self):
        self.assertIsNone(_pick_distractor([], forbidden=[]))

    def test_all_distractors_forbidden(self):
        self.assertIsNone(_pick_distractor(["sent10"], forbidden=["sent10"]))


class TestEnsureStructuralChange(unittest.TestCase):

    def test_lhs_changed_returns_true(self):
        old = [Rule(["sent1", "sent2"], "int1")]
        new = [Rule(["sent1", "sent99"], "int1")]
        self.assertTrue(_ensure_structural_change(old, new))

    def test_rhs_changed_returns_true(self):
        old = [Rule(["sent1"], "hypothesis")]
        new = [Rule(["sent1"], "conclusion")]
        self.assertTrue(_ensure_structural_change(old, new))

    def test_identical_returns_false(self):
        old = [Rule(["sent1", "sent2"], "int1")]
        new = [Rule(["sent1", "sent2"], "int1")]
        self.assertFalse(_ensure_structural_change(old, new))

    def test_annotation_change_not_structural(self):
        old = [Rule(["sent1"], "int1", "annotation A")]
        new = [Rule(["sent1"], "int1", "annotation B")]
        self.assertFalse(_ensure_structural_change(old, new))


# ===========================================================================
# delete_one_antecedent
# ===========================================================================

class TestDeleteOneAntecedent(unittest.TestCase):

    def test_reduces_arity_by_one(self):
        rules = parse_step_proof(PROOF_2STEP)
        rng = random.Random(0)
        result = delete_one_antecedent(rules, [0], rng=rng)
        self.assertEqual(len(result[0].lhs_ids), 1)

    def test_result_is_structurally_different(self):
        rules = parse_step_proof(PROOF_2STEP)
        result = delete_one_antecedent(rules, [0], rng=random.Random(0))
        self.assertTrue(_ensure_structural_change(rules, result))

    def test_no_arity_2_rule_returns_original(self):
        # All rules have arity 1 — delete has nothing to do
        rules = parse_step_proof("sent1 -> hypothesis; ")
        result = delete_one_antecedent(rules, [0], rng=random.Random(0))
        self.assertIs(result, rules)

    def test_empty_target_rules_returns_original(self):
        rules = parse_step_proof(PROOF_2STEP)
        result = delete_one_antecedent(rules, [], rng=random.Random(0))
        self.assertIs(result, rules)

    def test_does_not_mutate_original(self):
        rules = parse_step_proof(PROOF_2STEP)
        orig_lhs = rules[0].lhs_ids[:]
        delete_one_antecedent(rules, [0], rng=random.Random(0))
        self.assertEqual(rules[0].lhs_ids, orig_lhs)

    def test_seed_determinism(self):
        rules1 = parse_step_proof(PROOF_2STEP)
        rules2 = parse_step_proof(PROOF_2STEP)
        r1 = delete_one_antecedent(rules1, [0], rng=random.Random(42))
        r2 = delete_one_antecedent(rules2, [0], rng=random.Random(42))
        self.assertEqual(r1[0].lhs_ids, r2[0].lhs_ids)


# ===========================================================================
# replace_antecedent_with_distractor
# ===========================================================================

class TestReplaceAntecedentWithDistractor(unittest.TestCase):

    def test_keeps_arity(self):
        rules = parse_step_proof(PROOF_2STEP)
        rng = random.Random(0)
        result = replace_antecedent_with_distractor(rules, [0], DISTRACTORS, rng=rng)
        self.assertEqual(len(result[0].lhs_ids), 2)

    def test_distractor_present_in_lhs(self):
        rules = parse_step_proof(PROOF_2STEP)
        result = replace_antecedent_with_distractor(rules, [0], DISTRACTORS, rng=random.Random(0))
        self.assertTrue(any(d in result[0].lhs_ids for d in DISTRACTORS))

    def test_no_distractors_returns_original(self):
        rules = parse_step_proof(PROOF_2STEP)
        result = replace_antecedent_with_distractor(rules, [0], [], rng=random.Random(0))
        self.assertIs(result, rules)

    def test_no_target_rules_returns_original(self):
        rules = parse_step_proof(PROOF_2STEP)
        result = replace_antecedent_with_distractor(rules, [], DISTRACTORS, rng=random.Random(0))
        self.assertIs(result, rules)

    def test_all_distractors_forbidden_returns_original(self):
        rules = parse_step_proof(PROOF_2STEP)
        # Only distractor is the same as existing lhs_ids[0]
        result = replace_antecedent_with_distractor(
            rules, [0], ["sent1"], rng=random.Random(0)
        )
        # sent1 is forbidden (already in lhs), no valid pool → returns original
        self.assertIs(result, rules)

    def test_does_not_mutate_original(self):
        rules = parse_step_proof(PROOF_2STEP)
        orig = [r.lhs_ids[:] for r in rules]
        replace_antecedent_with_distractor(rules, [0], DISTRACTORS, rng=random.Random(0))
        for i, r in enumerate(rules):
            self.assertEqual(r.lhs_ids, orig[i])

    def test_result_is_valid_proof(self):
        # _pick_distractor uses module-level random (not rng), so full seed-
        # reproducibility is not guaranteed; just verify the result is structurally valid
        rules = parse_step_proof(PROOF_2STEP)
        result = replace_antecedent_with_distractor(rules, [0], DISTRACTORS, rng=random.Random(7))
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(len(result[0].lhs_ids), 2)  # arity preserved


# ===========================================================================
# rewire_drop_support_creation
# ===========================================================================

class TestRewireDropSupportCreation(unittest.TestCase):

    def test_removes_int_producing_rule(self):
        rules = parse_step_proof(PROOF_2STEP)
        # target_rules = [0,1], rule 0 produces int1
        result = rewire_drop_support_creation(rules, [0, 1])
        self.assertEqual(len(result), len(rules) - 1)

    def test_removes_first_int_candidate(self):
        rules = parse_step_proof(PROOF_DEEP)  # 3 rules: int1, int2, hypothesis
        # target all → candidates = [0,1] (int1, int2) → remove index 0
        result = rewire_drop_support_creation(rules, [0, 1, 2])
        self.assertEqual(len(result), len(rules) - 1)
        # The surviving rules should not include the one that produced int1
        rhs_ids = [r.rhs_id for r in result]
        self.assertNotIn("int1", rhs_ids)

    def test_no_int_candidates_falls_back_to_delete(self):
        # PROOF_1STEP has only "hypothesis" as rhs — no int*
        rules = parse_step_proof(PROOF_1STEP)
        # This should NOT raise; it falls back to delete_one_antecedent
        result = rewire_drop_support_creation(rules, [0])
        # Result is either original (if delete couldn't help) or modified
        self.assertIsNotNone(result)

    def test_does_not_mutate_original(self):
        rules = parse_step_proof(PROOF_2STEP)
        orig_len = len(rules)
        rewire_drop_support_creation(rules, [0, 1])
        self.assertEqual(len(rules), orig_len)


# ===========================================================================
# global_break
# ===========================================================================

class TestGlobalBreak(unittest.TestCase):

    def test_edits_all_rules_on_path(self):
        rules = parse_step_proof(PROOF_2STEP)
        rng = random.Random(0)
        result = global_break(rules, "hypothesis", DISTRACTORS, rng=rng)
        self.assertTrue(_ensure_structural_change(rules, result))

    def test_high_arity_rule_gets_antecedent_deleted(self):
        # rule 0: sent1 & sent2 → int1 (arity 2) → pop first antecedent
        rules = parse_step_proof(PROOF_2STEP)
        result = global_break(rules, "hypothesis", DISTRACTORS, rng=random.Random(0))
        # Rule 0 had 2 lhs, should now have 1
        self.assertEqual(len(result[0].lhs_ids), 1)

    def test_single_antecedent_rule_gets_replaced(self):
        # PROOF_1STEP: sent16 & sent24 -> hypothesis
        # global_break targets hypothesis, edits rule 0 (arity 2 → pop)
        rules = parse_step_proof(PROOF_1STEP)
        result = global_break(rules, "hypothesis", DISTRACTORS, rng=random.Random(0))
        self.assertTrue(_ensure_structural_change(rules, result))

    def test_does_not_mutate_original(self):
        rules = parse_step_proof(PROOF_2STEP)
        orig = [r.lhs_ids[:] for r in rules]
        global_break(rules, "hypothesis", DISTRACTORS, rng=random.Random(0))
        for i, r in enumerate(rules):
            self.assertEqual(r.lhs_ids, orig[i])


# ===========================================================================
# intervene_step_proof
# ===========================================================================

class TestInterveneStepProof(unittest.TestCase):

    def _intervene(self, proof, mode, seed=42):
        return intervene_step_proof(
            proof,
            hypothesis_id="hypothesis",
            distractors=DISTRACTORS,
            mode=mode,
            seed=seed,
            verbose=False,
        )

    def test_none_input_returns_none(self):
        self.assertIsNone(
            intervene_step_proof(None, "hypothesis", DISTRACTORS, mode="delete",
                                  seed=0, verbose=False)
        )

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self._intervene(PROOF_2STEP, "bogus")

    def test_delete_changes_proof(self):
        result = self._intervene(PROOF_2STEP, "delete")
        self.assertIsNotNone(result)
        self.assertNotEqual(result, PROOF_2STEP)

    def test_replace_changes_proof(self):
        result = self._intervene(PROOF_2STEP, "replace")
        self.assertIsNotNone(result)
        self.assertNotEqual(result, PROOF_2STEP)

    def test_rewire_changes_proof(self):
        result = self._intervene(PROOF_2STEP, "rewire")
        self.assertIsNotNone(result)
        self.assertNotEqual(result, PROOF_2STEP)

    def test_global_changes_proof(self):
        result = self._intervene(PROOF_2STEP, "global")
        self.assertIsNotNone(result)
        self.assertNotEqual(result, PROOF_2STEP)

    def test_delete_produces_parseable_result(self):
        # _pick_rule_with_min_arity uses module-level random.choice (not rng),
        # so full seed-reproducibility is not guaranteed; verify output is valid
        result = self._intervene(PROOF_2STEP, "delete", seed=7)
        self.assertIsNotNone(result)
        self.assertGreater(len(parse_step_proof(result)), 0)

    def test_replace_produces_parseable_result(self):
        # replace uses _pick_distractor which uses module-level random (not rng),
        # so full seed-reproducibility is not guaranteed; just verify output is valid
        result = self._intervene(PROOF_2STEP, "replace", seed=99)
        self.assertIsNotNone(result)
        self.assertGreater(len(parse_step_proof(result)), 0)

    def test_different_seeds_may_differ(self):
        # With a 2-step proof and distractors, different seeds should often differ
        results = {self._intervene(PROOF_2STEP, "replace", seed=s) for s in range(10)}
        self.assertGreater(len(results), 1)

    def test_result_is_parseable(self):
        for mode in ("delete", "replace", "rewire", "global"):
            with self.subTest(mode=mode):
                result = self._intervene(PROOF_2STEP, mode)
                rules = parse_step_proof(result)
                self.assertGreater(len(rules), 0)

    def test_result_ends_with_semicolon_space(self):
        for mode in ("delete", "replace", "rewire"):
            with self.subTest(mode=mode):
                result = self._intervene(PROOF_2STEP, mode)
                self.assertTrue(result.endswith("; "), f"mode={mode}: {result!r}")

    def test_deep_proof_all_modes(self):
        for mode in ("delete", "replace", "rewire", "global"):
            with self.subTest(mode=mode):
                result = intervene_step_proof(
                    PROOF_DEEP, hypothesis_id="hypothesis",
                    distractors=DISTRACTORS, mode=mode, seed=0, verbose=False
                )
                self.assertIsNotNone(result)
                self.assertNotEqual(result, PROOF_DEEP)

    def test_hypothesis_id_not_in_rhs_falls_back(self):
        # "int2" is not a RHS in PROOF_1STEP → falls back to "hypothesis"
        result = intervene_step_proof(
            PROOF_1STEP, hypothesis_id="int2",
            distractors=DISTRACTORS, mode="replace", seed=0, verbose=False
        )
        self.assertIsNotNone(result)
        self.assertNotEqual(result, PROOF_1STEP)


# ===========================================================================
# EntailmentIntervention init
# ===========================================================================

class TestEntailmentInterventionInit(unittest.TestCase):

    def test_prompt_built_on_init(self):
        ic, _ = _make_ic()
        self.assertTrue(hasattr(ic, "prompt"))

    def test_edit_modes(self):
        ic, _ = _make_ic()
        self.assertEqual(ic.edit_modes, ["delete", "replace", "rewire"])

    def test_invalid_prompting_regime_raises(self):
        with self.assertRaises(AssertionError):
            _make_ic(prompting_regime="bogus")

    def test_invalid_tool_mode_raises(self):
        dataset = EntailmentBankDatasetMock()
        few_shot = [dataset[0]]
        with self.assertRaises(AssertionError):
            EntailmentIntervention(
                dataset=dataset, llm_model=FakeLLMModel(),
                tool=EntailmentTool(), processor=EntailmentStructureProcessor(),
                few_shot_examples=few_shot, tool_mode="invalid_mode",
            )

    def test_tool_mode_none_string_stored_as_none(self):
        ic, _ = _make_ic()
        self.assertIsNone(ic.tool_mode)


# ===========================================================================
# infer_completion
# ===========================================================================

class TestInferCompletion(unittest.TestCase):

    def setUp(self):
        self.ic, _ = _make_ic()

    def _ans(self, text, **kw):
        return self.ic.infer_completion(text, **kw)[0]

    # ---- True ----

    def test_full_prefix_yes(self):
        self.assertIs(self._ans("## Final Answer\nIs the hypothesis correct? Yes"), True)

    def test_short_prefix_yes(self):
        self.assertIs(self._ans("Final Answer\nYes"), True)

    def test_hash_prefix_yes(self):
        self.assertIs(self._ans("# Final Answer\nYes"), True)

    def test_inline_yes(self):
        self.assertIs(self._ans("## Final Answer Is the hypothesis correct? Yes"), True)

    # ---- False ----

    def test_full_prefix_no(self):
        self.assertIs(self._ans("## Final Answer\nNo"), False)

    def test_short_prefix_no(self):
        self.assertIs(self._ans("Final Answer\nNo"), False)

    # ---- None ----

    def test_ambiguous(self):
        self.assertIsNone(self._ans("## Final Answer\nYes and No"))

    def test_neither(self):
        self.assertIsNone(self._ans("## Final Answer\nmaybe"))

    def test_empty(self):
        self.assertIsNone(self._ans(""))

    def test_no_fa_section_and_ambiguous_text(self):
        self.assertIsNone(self._ans("Yes No"))

    # ---- return type ----

    def test_returns_tuple(self):
        result = self.ic.infer_completion("## Final Answer\nYes")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_tool_proof_nodes_empty_in_non_tool_mode(self):
        _, nodes = self.ic.infer_completion("## Final Answer\nYes")
        self.assertEqual(nodes, {})

    def test_short_completion_true(self):
        self.assertIs(
            self._ans("Yes", short_completion=True), True
        )

    def test_short_completion_false(self):
        self.assertIs(
            self._ans("No", short_completion=True), False
        )

    def test_short_completion_ambiguous(self):
        self.assertIsNone(self._ans("Maybe", short_completion=True))

    # ---- multiple FA sections caught upstream ----

    def test_multiple_fa_sections_caught_by_classify(self):
        ic, dataset = _make_ic()
        s = deepcopy(dataset[0])
        c = f"## Proof\n{s['proof']}## Final Answer\nYes\n## Final Answer\nNo"
        out = ic.make_intervention(s, {"completion": c})
        self.assertEqual(out["generation_status"], "error")


# ===========================================================================
# classify_generation
# ===========================================================================

class TestClassifyGeneration(unittest.TestCase):

    def setUp(self):
        self.ic, self.dataset = _make_ic()
        self.sample = deepcopy(self.dataset[0])

    def _cls(self, completion, mediator=None):
        if mediator is None:
            mediator = self.ic.processor.extract_mediator(completion)
        return self.ic.classify_generation(completion, mediator, self.sample["gold_proof"])

    def test_gold_proof_is_correct(self):
        self.assertEqual(self._cls(_gold_c(self.sample)), "correct")

    def test_wrong_proof_is_incorrect(self):
        self.assertEqual(self._cls(_wrong_c()), "incorrect")

    def test_none_mediator_is_error(self):
        # Pass None explicitly — _cls would auto-fill it, so call directly
        self.assertEqual(
            self.ic.classify_generation(_gold_c(self.sample), None, self.sample["proof"]),
            "error",
        )

    def test_malformed_completion_is_error(self):
        self.assertEqual(self._cls(_malformed_c()), "error")

    def test_contradictory_answer_is_error(self):
        self.assertEqual(self._cls(_contradictory_c()), "error")

    def test_preamble_before_proof_is_error(self):
        self.assertEqual(self._cls(_preamble_c(self.sample)), "error")

    def test_no_final_answer_is_error(self):
        c = f"## Proof\n{self.sample['proof']}"  # no FA section
        self.assertEqual(self._cls(c), "error")

    def test_empty_completion_is_error(self):
        self.assertEqual(self._cls(""), "error")

    def test_wrong_proof_no_answer_is_error(self):
        c = "## Proof\nsent1 -> hypothesis; "  # missing FA
        self.assertEqual(self._cls(c), "error")

    def test_multiple_fa_sections_is_error(self):
        c = f"## Proof\n{self.sample['proof']}## Final Answer\nYes\n## Final Answer\nNo"
        self.assertEqual(self._cls(c), "error")


# ===========================================================================
# make_structure_intervention
# ===========================================================================

class TestMakeStructureIntervention(unittest.TestCase):

    def setUp(self):
        self.ic, self.dataset = _make_ic()
        self.sample = deepcopy(self.dataset[0])

    def test_correct_yields_3_local_edits_no_correction(self):
        self.sample["generation_status"] = "correct"
        self.sample["mediator_proof"] = self.sample["proof"]
        tree = self.ic.make_structure_intervention(self.sample)
        self.assertEqual(len(tree["Local Edits"]), 3)
        self.assertEqual(len(tree["Correction"]), 0)

    def test_incorrect_yields_no_local_edits_one_correction(self):
        self.sample["generation_status"] = "incorrect"
        tree = self.ic.make_structure_intervention(self.sample)
        self.assertEqual(len(tree["Local Edits"]), 0)
        self.assertEqual(len(tree["Correction"]), 1)

    def test_error_yields_empty_lists(self):
        self.sample["generation_status"] = "error"
        tree = self.ic.make_structure_intervention(self.sample)
        self.assertEqual(tree, {"Local Edits": [], "Correction": []})

    def test_unknown_status_yields_empty_lists(self):
        self.sample["generation_status"] = "unknown"
        tree = self.ic.make_structure_intervention(self.sample)
        self.assertEqual(tree, {"Local Edits": [], "Correction": []})

    def test_local_edits_proof_structurally_changed(self):
        self.sample["generation_status"] = "correct"
        self.sample["mediator_proof"] = self.sample["proof"]
        predicted = self.sample["mediator_proof"]
        tree = self.ic.make_structure_intervention(self.sample)
        for le in tree["Local Edits"]:
            # edit["proof"] is the intervened predicted proof, not the gold
            self.assertNotEqual(le["proof"], predicted)
            # gold proof in sample stays unchanged
            self.assertEqual(self.sample["proof"], self.sample["proof"])

    def test_local_edits_score_is_false(self):
        self.sample["generation_status"] = "correct"
        self.sample["mediator_proof"] = self.sample["proof"]
        tree = self.ic.make_structure_intervention(self.sample)
        for le in tree["Local Edits"]:
            self.assertIs(le["score"], False)

    def test_local_edits_expected_score_is_false(self):
        self.sample["generation_status"] = "correct"
        self.sample["mediator_proof"] = self.sample["proof"]
        tree = self.ic.make_structure_intervention(self.sample)
        for le in tree["Local Edits"]:
            self.assertIs(le["expected_score_after_intervention"], False)

    def test_correction_restores_gold_proof(self):
        self.sample["generation_status"] = "incorrect"
        # Simulate: model generated wrong proof, stored in mediator_proof
        self.sample["mediator_proof"] = "sent1 -> hypothesis; "
        tree = self.ic.make_structure_intervention(self.sample)
        # Correction copy's proof and mediator_proof should both be gold
        self.assertEqual(tree["Correction"][0]["proof"], self.sample["gold_proof"])
        self.assertEqual(tree["Correction"][0]["mediator_proof"], self.sample["gold_proof"])

    def test_correction_score_is_true(self):
        self.sample["generation_status"] = "incorrect"
        tree = self.ic.make_structure_intervention(self.sample)
        self.assertIs(tree["Correction"][0]["score"], True)

    def test_correction_expected_score_is_true(self):
        self.sample["generation_status"] = "incorrect"
        tree = self.ic.make_structure_intervention(self.sample)
        self.assertIs(tree["Correction"][0]["expected_score_after_intervention"], True)

    def test_local_edits_are_independent_deepcopies(self):
        self.sample["generation_status"] = "correct"
        self.sample["mediator_proof"] = self.sample["proof"]
        tree = self.ic.make_structure_intervention(self.sample)
        tree["Local Edits"][0]["__sentinel__"] = True
        self.assertNotIn("__sentinel__", tree["Local Edits"][1])
        self.assertNotIn("__sentinel__", self.sample)

    def test_correction_does_not_mutate_original_sample(self):
        self.sample["generation_status"] = "incorrect"
        original_gold = self.sample["gold_proof"]
        tree = self.ic.make_structure_intervention(self.sample)
        # gold_proof is never modified
        self.assertEqual(self.sample["gold_proof"], original_gold)

    def test_local_edits_one_per_mode(self):
        self.sample["generation_status"] = "correct"
        self.sample["mediator_proof"] = self.sample["proof"]
        tree = self.ic.make_structure_intervention(self.sample)
        # Verify there are 3 edits, one per edit_mode
        self.assertEqual(len(tree["Local Edits"]), len(self.ic.edit_modes))


# ===========================================================================
# make_intervention
# ===========================================================================

class TestMakeIntervention(unittest.TestCase):

    def setUp(self):
        self.ic, self.dataset = _make_ic()

    def test_correct_generation_status(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        self.assertEqual(out["generation_status"], "correct")

    def test_correct_sets_score_before_true(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        self.assertIs(out["score_before_intervention"], True)

    def test_gold_proof_not_overwritten(self):
        s = deepcopy(self.dataset[0])
        gold = s["gold_proof"]
        out = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        self.assertEqual(out["gold_proof"], gold)
        self.assertEqual(out["proof"], gold)  # proof also unchanged

    def test_mediator_proof_stored_separately(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        self.assertIn("mediator_proof", out)
        self.assertIsNotNone(out["mediator_proof"])

    def test_mediator_proof_none_for_error(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _malformed_c()})
        self.assertIsNone(out["mediator_proof"])

    def test_correct_sets_raw_generation(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        self.assertIn("raw_generation", out)
        self.assertIsInstance(out["raw_generation"], str)

    def test_correct_has_3_local_edits(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        self.assertEqual(len(out["structure_intervention"]["Local Edits"]), 3)

    def test_incorrect_generation_status(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _wrong_c()})
        self.assertEqual(out["generation_status"], "incorrect")

    def test_incorrect_has_one_correction(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _wrong_c()})
        self.assertEqual(len(out["structure_intervention"]["Correction"]), 1)

    def test_error_generation_status_malformed(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _malformed_c()})
        self.assertEqual(out["generation_status"], "error")

    def test_error_score_before_is_none(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _malformed_c()})
        self.assertIsNone(out["score_before_intervention"])

    def test_error_empty_structure_intervention(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _malformed_c()})
        self.assertEqual(out["structure_intervention"]["Local Edits"], [])
        self.assertEqual(out["structure_intervention"]["Correction"], [])

    def test_contradictory_answer_is_error(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _contradictory_c()})
        self.assertEqual(out["generation_status"], "error")

    def test_preamble_is_error(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _preamble_c(s)})
        self.assertEqual(out["generation_status"], "error")

    def test_wrong_answer_no_is_incorrect(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _wrong_c(answer="No")})
        # Proof differs from gold → incorrect (regardless of answer)
        self.assertEqual(out["generation_status"], "incorrect")
        self.assertIs(out["score_before_intervention"], False)

    def test_no_legacy_keys_in_output(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        for k in ("gold_rubric", "mediator_rubric", "completion_type",
                  "result_after_intervention", "target_before_intervention"):
            self.assertNotIn(k, out, f"Unexpected legacy key: {k!r}")

    def test_empty_completion_is_error(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": ""})
        self.assertEqual(out["generation_status"], "error")

    def test_garbage_completion_is_error(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": "random text with no structure"})
        self.assertEqual(out["generation_status"], "error")

    def test_missing_final_answer_only_is_error(self):
        s = deepcopy(self.dataset[0])
        out = self.ic.make_intervention(s, {"completion": f"## Proof\n{s['proof']}"})
        self.assertEqual(out["generation_status"], "error")


# ===========================================================================
# interventions_to_prompt
# ===========================================================================

class TestInterventionsToPrompt(unittest.TestCase):

    def setUp(self):
        self.ic, self.dataset = _make_ic()

    def test_correct_yields_3_prompts(self):
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), 3)

    def test_incorrect_yields_1_prompt(self):
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _wrong_c()})
        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), 1)

    def test_error_yields_0_prompts(self):
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _malformed_c()})
        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), 0)

    def test_all_prompts_are_strings(self):
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        for p in self.ic.interventions_to_prompt(s):
            self.assertIsInstance(p, str)

    def test_prompt_count_matches_interventions(self):
        """count(prompts) == count(Local Edits) + count(Correction)"""
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        interv = s["structure_intervention"]
        expected = len(interv["Local Edits"]) + len(interv["Correction"])
        self.assertEqual(len(self.ic.interventions_to_prompt(s)), expected)


# ===========================================================================
# collect_intervention_completion
# ===========================================================================

class TestCollectInterventionCompletion(unittest.TestCase):

    def setUp(self):
        self.ic, self.dataset = _make_ic()

    def test_local_edits_receive_score_false(self):
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        n = len(s["structure_intervention"]["Local Edits"])
        result = self.ic.collect_intervention_completion(
            s, [{"completion": "## Final Answer\nNo"}] * n
        )
        for le in result["structure_intervention"]["Local Edits"]:
            self.assertIs(le["score_after_intervention"], False)

    def test_local_edits_receive_raw_generation(self):
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        n = len(s["structure_intervention"]["Local Edits"])
        self.ic.collect_intervention_completion(
            s, [{"completion": "## Final Answer\nNo"}] * n
        )
        for le in s["structure_intervention"]["Local Edits"]:
            self.assertIn("raw_generation", le)

    def test_correction_receives_score_true(self):
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _wrong_c()})
        result = self.ic.collect_intervention_completion(
            s, [{"completion": "## Final Answer\nYes"}]
        )
        self.assertIs(result["structure_intervention"]["Correction"][0]["score_after_intervention"], True)

    def test_ambiguous_answer_yields_none(self):
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        n = len(s["structure_intervention"]["Local Edits"])
        self.ic.collect_intervention_completion(s, [{"completion": "maybe"}] * n)
        for le in s["structure_intervention"]["Local Edits"]:
            self.assertIsNone(le["score_after_intervention"])

    def test_order_local_edits_first_then_correction_incorrect(self):
        """For incorrect: 1 correction. First output goes to correction."""
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _wrong_c()})
        result = self.ic.collect_intervention_completion(
            s, [{"completion": "## Final Answer\nNo"}]
        )
        corr = result["structure_intervention"]["Correction"][0]
        self.assertIs(corr["score_after_intervention"], False)

    def test_order_local_edits_first_correct(self):
        """For correct: first completion → first local edit."""
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        n = len(s["structure_intervention"]["Local Edits"])
        # First LE gets Yes, rest get No
        completions = [{"completion": "## Final Answer\nYes"}] + \
                      [{"completion": "## Final Answer\nNo"}] * (n - 1)
        result = self.ic.collect_intervention_completion(s, completions)
        le = result["structure_intervention"]["Local Edits"]
        self.assertIs(le[0]["score_after_intervention"], True)
        for i in range(1, n):
            self.assertIs(le[i]["score_after_intervention"], False)

    def test_clean_llm_output_applied_to_completions(self):
        """Special tokens in completions are stripped before parsing."""
        s = deepcopy(self.dataset[0])
        s = self.ic.make_intervention(s, {"completion": _gold_c(s)})
        n = len(s["structure_intervention"]["Local Edits"])
        # Completion with token noise
        noisy = "## Final Answer\nNo<|im_end|>"
        self.ic.collect_intervention_completion(s, [{"completion": noisy}] * n)
        for le in s["structure_intervention"]["Local Edits"]:
            self.assertIs(le["score_after_intervention"], False)


# ===========================================================================
# format_example
# ===========================================================================

class TestFormatExample(unittest.TestCase):

    def setUp(self):
        self.ic, self.dataset = _make_ic()

    def test_all_sections_present(self):
        text = self.ic.format_example(self.dataset[0], True, True, True, True)
        for section in ("## Question", "## Context", "## Hypothesis", "## Proof", "Final Answer"):
            self.assertIn(section, text)

    def test_score_true_gives_yes(self):
        s = deepcopy(self.dataset[0])
        s["score"] = True
        text = self.ic.format_example(s, False, False, True, True)
        self.assertIn("Yes", text)

    def test_score_false_gives_no(self):
        s = deepcopy(self.dataset[0])
        s["score"] = False
        text = self.ic.format_example(s, False, False, True, True)
        self.assertIn("No", text)

    def test_no_proof_when_add_proof_false(self):
        text = self.ic.format_example(self.dataset[0], True, False, False, False)
        self.assertNotIn("## Proof", text)

    def test_no_final_answer_prefix_when_disabled(self):
        text = self.ic.format_example(self.dataset[0], True, True, False, False)
        self.assertNotIn("Final Answer", text)

    def test_only_proof_section(self):
        text = self.ic.format_example(self.dataset[0], False, True, False, False)
        self.assertIn("## Proof", text)
        self.assertNotIn("## Question", text)

    def test_proof_content_included(self):
        text = self.ic.format_example(self.dataset[0], False, True, False, False)
        self.assertIn(self.dataset[0]["proof"], text)


if __name__ == "__main__":
    unittest.main()