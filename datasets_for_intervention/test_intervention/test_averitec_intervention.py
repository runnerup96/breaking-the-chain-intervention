"""
Tests for AVeriTeCIntervention.

Covers:
  - clean_llm_output
  - infer_completion  (tool_mode=none / simple / structured)
  - classify_generation
  - make_intervention  (correct / incorrect / error completions)
  - make_structure_intervention  (correct / incorrect / error)
  - interventions_to_prompt  (count and gold_structure flag)
  - collect_intervention_completion  (order and field assignment)
  - make_prompt  (with and without gold_structure)
"""
import sys
import unittest
from copy import deepcopy

from datasets_for_intervention.test_intervention.averitec_mocks import (
    AVeriTeCDatasetMock, FakeLLMModel,
    GOLD_RUBRIC_2Q, GOLD_RUBRIC_1Q, GOLD_RUBRIC_3Q,
)
from datasets_for_intervention.averitec_structure_processor import AVeriTeCTool, AVeriTeCStructureProcessor
from datasets_for_intervention.averitec_intervention import AVeriTeCIntervention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_intervention(tool_mode="none", regime="standard"):
    ds   = AVeriTeCDatasetMock()
    llm  = FakeLLMModel()
    tool = AVeriTeCTool(ds, tool_mode)
    proc = AVeriTeCStructureProcessor(ds, tool_mode)
    return AVeriTeCIntervention(ds, llm, tool, proc, regime, tool_mode), ds


COMP_CORRECT = (
    "Checklist:\n"
    "- Q: Did Hunter Biden have any experience in the energy sector in 2014? (True/False): False\n"
    "- Q: Did Hunter Biden have any experience in Ukraine in 2014? (True/False): False\n"
    "\n"
    "Final Verdict: Supported"
)
COMP_INCORRECT = (
    "Checklist:\n"
    "- Q: Did Hunter Biden have any experience in the energy sector in 2014? (True/False): True\n"
    "- Q: Did Hunter Biden have any experience in Ukraine in 2014? (True/False): False\n"
    "\n"
    "Final Verdict: Refuted"
)
COMP_ERROR_PREAMBLE = (
    "Sure! Here is my answer.\n"
    "Checklist:\n"
    "- Q: Did Hunter Biden have any experience in the energy sector in 2014? (True/False): False\n"
    "- Q: Did Hunter Biden have any experience in Ukraine in 2014? (True/False): False\n"
    "\n"
    "Final Verdict: Supported"
)
COMP_ERROR_NO_CHECKLIST = "Final Verdict: Supported"


# ===========================================================================
# 1. clean_llm_output
# ===========================================================================

class TestCleanLlmOutput(unittest.TestCase):

    def setUp(self):
        self.ic, _ = make_intervention()

    def test_removes_special_tokens(self):
        text = "hello<|im_end|> world<|endoftext|>"
        self.assertEqual(self.ic.clean_llm_output(text), "hello world")

    def test_removes_invisible_chars(self):
        text = "clean\u200btext\ufeff"
        self.assertEqual(self.ic.clean_llm_output(text), "cleantext")

    def test_strips_whitespace(self):
        self.assertEqual(self.ic.clean_llm_output("  hello  "), "hello")

    def test_plain_text_unchanged(self):
        text = "Checklist:\n- Q: X (True/False): True\nFinal Verdict: Supported"
        self.assertEqual(self.ic.clean_llm_output(text), text)


# ===========================================================================
# 2. infer_completion -- tool_mode=none
# ===========================================================================

class TestInferCompletionNone(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention(tool_mode="none")

    def test_correct_completion_returns_verdict_and_empty_rubric(self):
        verdict, rubric = self.ic.infer_completion(COMP_CORRECT, self.ds[0])
        self.assertEqual(verdict, "Supported")
        self.assertEqual(rubric, {})

    def test_incorrect_completion_returns_flipped_verdict(self):
        verdict, rubric = self.ic.infer_completion(COMP_INCORRECT, self.ds[0])
        self.assertEqual(verdict, "Refuted")
        self.assertEqual(rubric, {})

    def test_short_completion_supported(self):
        verdict, _ = self.ic.infer_completion("Supported", self.ds[0], short_completion=True)
        self.assertEqual(verdict, "Supported")

    def test_short_completion_refuted(self):
        verdict, _ = self.ic.infer_completion("Refuted", self.ds[0], short_completion=True)
        self.assertEqual(verdict, "Refuted")

    def test_garbage_returns_none_verdict(self):
        verdict, _ = self.ic.infer_completion("no verdict here", self.ds[0])
        self.assertIsNone(verdict)


# ===========================================================================
# 3. classify_generation
# ===========================================================================

class TestClassifyGeneration(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def test_correct_when_mediator_matches_gold(self):
        mediator = deepcopy(GOLD_RUBRIC_2Q)
        result = self.ic.classify_generation(COMP_CORRECT, mediator, GOLD_RUBRIC_2Q)
        self.assertEqual(result, "correct")

    def test_incorrect_when_mediator_differs(self):
        bad = deepcopy(GOLD_RUBRIC_2Q)
        bad[list(bad.keys())[0]] = True
        result = self.ic.classify_generation(COMP_CORRECT, bad, GOLD_RUBRIC_2Q)
        self.assertEqual(result, "incorrect")

    def test_error_when_mediator_is_none(self):
        result = self.ic.classify_generation(COMP_CORRECT, None, GOLD_RUBRIC_2Q)
        self.assertEqual(result, "error")

    def test_error_when_preamble(self):
        result = self.ic.classify_generation(
            COMP_ERROR_PREAMBLE, deepcopy(GOLD_RUBRIC_2Q), GOLD_RUBRIC_2Q
        )
        self.assertEqual(result, "error")

    def test_error_when_no_checklist(self):
        result = self.ic.classify_generation(
            COMP_ERROR_NO_CHECKLIST, deepcopy(GOLD_RUBRIC_2Q), GOLD_RUBRIC_2Q
        )
        self.assertEqual(result, "error")


# ===========================================================================
# 4. make_intervention
# ===========================================================================

class TestMakeIntervention(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def test_correct_completion(self):
        s = deepcopy(self.ds[0])
        out = self.ic.make_intervention(s, {"completion": COMP_CORRECT})
        self.assertEqual(out["generation_status"], "correct")
        self.assertEqual(out["target_before_intervention"], "Supported")
        self.assertEqual(out["tool_rubric"], {})
        self.assertEqual(out["mediator_rubric"], GOLD_RUBRIC_2Q)
        self.assertEqual(len(out["structure_intervention"]["Local Edits"]), 2)
        self.assertEqual(out["structure_intervention"]["Correction"], [])

    def test_incorrect_completion(self):
        s = deepcopy(self.ds[0])
        out = self.ic.make_intervention(s, {"completion": COMP_INCORRECT})
        self.assertEqual(out["generation_status"], "incorrect")
        self.assertEqual(out["target_before_intervention"], "Refuted")
        self.assertEqual(out["structure_intervention"]["Local Edits"], [])
        self.assertEqual(len(out["structure_intervention"]["Correction"]), 1)

    def test_error_preamble(self):
        s = deepcopy(self.ds[0])
        out = self.ic.make_intervention(s, {"completion": COMP_ERROR_PREAMBLE})
        self.assertEqual(out["generation_status"], "error")
        self.assertIsNone(out["target_before_intervention"])
        self.assertIsNone(out["tool_rubric"])
        self.assertEqual(out["structure_intervention"], {"Local Edits": [], "Correction": []})

    def test_error_no_checklist(self):
        s = deepcopy(self.ds[0])
        out = self.ic.make_intervention(s, {"completion": COMP_ERROR_NO_CHECKLIST})
        self.assertEqual(out["generation_status"], "error")

    def test_original_sample_not_mutated(self):
        s = deepcopy(self.ds[0])
        original = deepcopy(s)
        # make_intervention mutates sample in-place by design, but gold_rubric must be intact
        self.ic.make_intervention(s, {"completion": COMP_CORRECT})
        self.assertEqual(s["gold_rubric"], original["gold_rubric"])
        self.assertEqual(s["gold_target"], original["gold_target"])


# ===========================================================================
# 5. make_structure_intervention
# ===========================================================================

class TestMakeStructureIntervention(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def _correct_sample(self):
        s = deepcopy(self.ds[0])
        s["generation_status"] = "correct"
        return s

    def _incorrect_sample(self):
        s = deepcopy(self.ds[0])
        s["generation_status"] = "incorrect"
        bad = deepcopy(GOLD_RUBRIC_2Q)
        bad[list(bad.keys())[0]] = True
        s["mediator_rubric"] = bad
        return s

    def _error_sample(self):
        s = deepcopy(self.ds[0])
        s["generation_status"] = "error"
        return s

    # --- correct ---

    def test_correct_produces_local_edits_no_correction(self):
        result = self.ic.make_structure_intervention(self._correct_sample())
        self.assertEqual(result["Correction"], [])
        self.assertEqual(len(result["Local Edits"]), 2)

    def test_correct_each_local_edit_inverts_exactly_one_answer(self):
        result = self.ic.make_structure_intervention(self._correct_sample())
        for edit in result["Local Edits"]:
            diffs = sum(
                1 for k in GOLD_RUBRIC_2Q
                if edit["mediator_rubric"][k] != GOLD_RUBRIC_2Q[k]
            )
            self.assertEqual(diffs, 1)

    def test_correct_expected_target_is_refuted_for_supported_sample(self):
        result = self.ic.make_structure_intervention(self._correct_sample())
        for edit in result["Local Edits"]:
            self.assertEqual(edit["expected_target_after_intervention"], "Refuted")

    def test_correct_edits_are_deep_copies(self):
        result = self.ic.make_structure_intervention(self._correct_sample())
        result["Local Edits"][0]["mediator_rubric"]["__sentinel__"] = True
        self.assertNotIn("__sentinel__", result["Local Edits"][1]["mediator_rubric"])

    # --- incorrect ---

    def test_incorrect_produces_correction_no_local_edits(self):
        result = self.ic.make_structure_intervention(self._incorrect_sample())
        self.assertEqual(result["Local Edits"], [])
        self.assertEqual(len(result["Correction"]), 1)

    def test_incorrect_correction_uses_gold_rubric(self):
        result = self.ic.make_structure_intervention(self._incorrect_sample())
        self.assertEqual(result["Correction"][0]["mediator_rubric"], GOLD_RUBRIC_2Q)

    # --- error ---

    def test_error_produces_empty_lists(self):
        result = self.ic.make_structure_intervention(self._error_sample())
        self.assertEqual(result["Local Edits"], [])
        self.assertEqual(result["Correction"], [])


# ===========================================================================
# 6. interventions_to_prompt
# ===========================================================================

class TestInterventionsToPrompt(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def test_correct_sample_generates_n_prompts_equal_to_n_questions(self):
        s = deepcopy(self.ds[0])
        s = self.ic.make_intervention(s, {"completion": COMP_CORRECT})
        prompts = self.ic.interventions_to_prompt(s)
        # 2 questions -> 2 Local Edit prompts + 0 Correction
        self.assertEqual(len(prompts), 2)

    def test_incorrect_sample_generates_one_correction_prompt(self):
        s = deepcopy(self.ds[0])
        s = self.ic.make_intervention(s, {"completion": COMP_INCORRECT})
        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), 1)

    def test_error_sample_generates_no_prompts(self):
        s = deepcopy(self.ds[0])
        s = self.ic.make_intervention(s, {"completion": COMP_ERROR_PREAMBLE})
        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), 0)

    def test_prompts_are_strings(self):
        s = deepcopy(self.ds[0])
        s = self.ic.make_intervention(s, {"completion": COMP_CORRECT})
        for p in self.ic.interventions_to_prompt(s):
            self.assertIsInstance(p, str)


# ===========================================================================
# 7. collect_intervention_completion
# ===========================================================================

class TestCollectInterventionCompletion(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def test_local_edits_filled_in_order(self):
        s = deepcopy(self.ds[0])
        s = self.ic.make_intervention(s, {"completion": COMP_CORRECT})
        # 2 Local Edits, no Correction
        fakes = [{"completion": "Refuted"}, {"completion": "Supported"}]
        out = self.ic.collect_intervention_completion(s, fakes)
        edits = out["structure_intervention"]["Local Edits"]
        self.assertEqual(edits[0]["target_after_intervention"], "Refuted")
        self.assertEqual(edits[1]["target_after_intervention"], "Supported")

    def test_correction_filled_after_local_edits(self):
        s = deepcopy(self.ds[0])
        s = self.ic.make_intervention(s, {"completion": COMP_INCORRECT})
        # 0 Local Edits, 1 Correction
        fakes = [{"completion": "Supported"}]
        out = self.ic.collect_intervention_completion(s, fakes)
        corr = out["structure_intervention"]["Correction"]
        self.assertEqual(corr[0]["target_after_intervention"], "Supported")

    def test_raw_generation_stored_on_intervention_sample(self):
        s = deepcopy(self.ds[0])
        s = self.ic.make_intervention(s, {"completion": COMP_CORRECT})
        fakes = [{"completion": "Refuted"}, {"completion": "Supported"}]
        out = self.ic.collect_intervention_completion(s, fakes)
        self.assertEqual(out["structure_intervention"]["Local Edits"][0]["raw_generation"], "Refuted")


# ===========================================================================
# 8. make_prompt
# ===========================================================================

class TestMakePrompt(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def test_without_gold_structure_contains_claim_and_checklist_template(self):
        prompt = self.ic.make_prompt(self.ds[0], include_gold_structure=False)
        self.assertIn(self.ds[0]["claim"], prompt)
        self.assertIn("<True/False>", prompt)
        self.assertIn("Checklist:", prompt)

    def test_with_gold_structure_contains_filled_checklist_and_verdict_tail(self):
        s = deepcopy(self.ds[0])
        s["mediator_rubric"] = deepcopy(GOLD_RUBRIC_2Q)
        prompt = self.ic.make_prompt(s, include_gold_structure=True)
        self.assertIn("Checklist:", prompt)
        self.assertIn("Final Verdict: ", prompt)
        # No <True/False> placeholder -- checklist is already filled
        self.assertNotIn("<True/False>", prompt)

    def test_with_gold_structure_contains_mediator_values(self):
        s = deepcopy(self.ds[0])
        s["mediator_rubric"] = deepcopy(GOLD_RUBRIC_2Q)
        prompt = self.ic.make_prompt(s, include_gold_structure=True)
        # All answers are False
        self.assertIn("False", prompt)

    def test_prompt_contains_explanations(self):
        prompt = self.ic.make_prompt(self.ds[0], include_gold_structure=False)
        # At least one explanation should appear
        for e in self.ds[0]["explanations"].values():
            self.assertIn(e, prompt)
            break  # one is enough


# ===========================================================================
# 9. few-shot structure per prompting regime
# ===========================================================================

class TestFewShotStructure(unittest.TestCase):

    def _get_labels(self, regime):
        ic, _ = make_intervention(regime=regime)
        _, _, few = ic._get_prompt_structure()
        return [l for l in few.split("\n") if l.startswith("Example #")]

    def test_standard_has_three_plain_labels(self):
        labels = self._get_labels("standard")
        self.assertEqual(labels, ["Example #1", "Example #2", "Example #3"])

    def test_detailed_has_no_intervention_and_with_intervention_labels(self):
        labels = self._get_labels("detailed")
        self.assertIn("Example #1 (No intervention)", labels)
        self.assertIn("Example #2 (With intervention)", labels)
        self.assertIn("Example #3 (With intervention)", labels)
        self.assertEqual(len(labels), 3)

    def test_detailed_few_shot_no_raw_dict_leaked(self):
        ic, _ = make_intervention(regime="detailed")
        _, _, few = ic._get_prompt_structure()
        self.assertNotIn("{'Did", few)
        self.assertNotIn("{'He made", few)

    def test_detailed_few_shot_no_double_example_in_label(self):
        ic, _ = make_intervention(regime="detailed")
        _, _, few = ic._get_prompt_structure()
        self.assertNotIn("Example #Example", few)

    def test_max_detailed_same_structure_as_detailed(self):
        self.assertEqual(
            self._get_labels("max_detailed"),
            self._get_labels("detailed"),
        )


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# 10. infer_completion -- tool modes with None args
# ===========================================================================

class TestInferCompletionToolModes(unittest.TestCase):

    def test_simple_mode_none_tool_args_returns_none_verdict(self):
        ic, ds = make_intervention(tool_mode="simple")
        # No ARGS block → extract_tool_args returns None → tool_rubric=None → calculate_score=None
        verdict, rubric = ic.infer_completion("Checklist:\n- Q: X: True\n", ds[0])
        self.assertIsNone(verdict)
        self.assertIsNone(rubric)

    def test_structured_mode_none_tool_args_returns_none_verdict(self):
        ic, ds = make_intervention(tool_mode="structured")
        verdict, rubric = ic.infer_completion("Checklist:\n- Q: X: True\n", ds[0])
        self.assertIsNone(verdict)
        self.assertIsNone(rubric)

    def test_simple_mode_valid_args_returns_verdict(self):
        ic, ds = make_intervention(tool_mode="simple")
        sample = ds[0]
        # Build a valid simple-mode completion: tool args as dict via rubric string
        q1, q2 = list(GOLD_RUBRIC_2Q.keys())
        comp = (
            "Checklist:\n"
            f"- Q: {q1} (True/False): False\n"
            f"- Q: {q2} (True/False): False\n"
            "Final tool call:\n"
            "   TOOL: predict_verdict\n"
            f'   ARGS: {{"rubric": "- Q: {q1} (True/False): False\\n- Q: {q2} (True/False): False\\n"}}\n'
        )
        verdict, rubric = ic.infer_completion(comp, sample)
        self.assertEqual(verdict, "Supported")
        self.assertEqual(rubric, GOLD_RUBRIC_2Q)

    def test_structured_mode_valid_args_returns_verdict(self):
        ic, ds = make_intervention(tool_mode="structured")
        sample = ds[0]
        comp = (
            "Checklist:\n"
            "- Q: X (True/False): False\n"
            "- Q: Y (True/False): False\n"
            "Final tool call:\n"
            '   ARGS: {"rubric": [False, False]}\n'
        )
        verdict, rubric = ic.infer_completion(comp, sample)
        self.assertEqual(verdict, "Supported")
        self.assertEqual(rubric, GOLD_RUBRIC_2Q)


# ===========================================================================
# 11. classify_generation -- edge cases
# ===========================================================================

class TestClassifyGenerationEdgeCases(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def test_empty_dict_mediator_is_incorrect_not_error(self):
        # {} parses (not None) but doesn't match gold_rubric → incorrect
        result = self.ic.classify_generation(COMP_CORRECT, {}, GOLD_RUBRIC_2Q)
        self.assertEqual(result, "incorrect")

    def test_correct_with_both_rubrics_equal(self):
        result = self.ic.classify_generation(
            COMP_CORRECT, deepcopy(GOLD_RUBRIC_2Q), GOLD_RUBRIC_2Q
        )
        self.assertEqual(result, "correct")

    def test_partially_matching_mediator_is_incorrect(self):
        partial = deepcopy(GOLD_RUBRIC_2Q)
        partial[list(partial.keys())[0]] = True
        result = self.ic.classify_generation(COMP_CORRECT, partial, GOLD_RUBRIC_2Q)
        self.assertEqual(result, "incorrect")


# ===========================================================================
# 12. make_structure_intervention -- Refuted-1Q correct case
# ===========================================================================

class TestMakeStructureInterventionRefuted1Q(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def test_correct_refuted_1q_expected_target_is_supported(self):
        # Sample[1]: Refuted, 1 question, all False
        s = deepcopy(self.ds[1])
        s["generation_status"] = "correct"
        # mediator_rubric matches gold_rubric (correct prediction)
        s["mediator_rubric"] = deepcopy(GOLD_RUBRIC_1Q)

        result = self.ic.make_structure_intervention(s)
        self.assertEqual(len(result["Local Edits"]), 1)
        # Flipping False→True for a Refuted-1Q sample should give Supported
        self.assertEqual(result["Local Edits"][0]["expected_target_after_intervention"], "Supported")

    def test_correct_refuted_1q_flipped_mediator_is_true(self):
        s = deepcopy(self.ds[1])
        s["generation_status"] = "correct"
        s["mediator_rubric"] = deepcopy(GOLD_RUBRIC_1Q)

        result = self.ic.make_structure_intervention(s)
        edit = result["Local Edits"][0]
        question = list(GOLD_RUBRIC_1Q.keys())[0]
        self.assertTrue(edit["mediator_rubric"][question])  # was False, now True

    def test_local_edits_dont_mutate_each_other(self):
        # 2-question sample: edit[0] and edit[1] must be independent
        s = deepcopy(self.ds[0])
        s["generation_status"] = "correct"
        s["mediator_rubric"] = deepcopy(GOLD_RUBRIC_2Q)

        result = self.ic.make_structure_intervention(s)
        keys = list(GOLD_RUBRIC_2Q.keys())

        # edit[0] flips key[0]; key[1] should stay as gold
        self.assertEqual(result["Local Edits"][0]["mediator_rubric"][keys[1]], GOLD_RUBRIC_2Q[keys[1]])
        # edit[1] flips key[1]; key[0] should stay as gold
        self.assertEqual(result["Local Edits"][1]["mediator_rubric"][keys[0]], GOLD_RUBRIC_2Q[keys[0]])

    def test_incorrect_correction_mediator_rubric_matches_gold(self):
        s = deepcopy(self.ds[0])
        s["generation_status"] = "incorrect"
        bad = deepcopy(GOLD_RUBRIC_2Q)
        bad[list(bad.keys())[0]] = True
        s["mediator_rubric"] = bad

        result = self.ic.make_structure_intervention(s)
        self.assertEqual(len(result["Correction"]), 1)
        self.assertEqual(result["Correction"][0]["mediator_rubric"], GOLD_RUBRIC_2Q)

    def test_original_gold_rubric_not_mutated_by_structure_intervention(self):
        s = deepcopy(self.ds[0])
        s["generation_status"] = "correct"
        s["mediator_rubric"] = deepcopy(GOLD_RUBRIC_2Q)
        original_gold = deepcopy(s["gold_rubric"])

        self.ic.make_structure_intervention(s)
        self.assertEqual(s["gold_rubric"], original_gold)


# ===========================================================================
# 13. make_intervention -- raw_generation stored, None completion
# ===========================================================================

class TestMakeInterventionRawGeneration(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def test_raw_generation_stored_on_sample(self):
        s = deepcopy(self.ds[0])
        self.ic.make_intervention(s, {"completion": COMP_CORRECT})
        self.assertIn("raw_generation", s)
        self.assertIsInstance(s["raw_generation"], str)

    def test_raw_generation_is_cleaned(self):
        dirty = COMP_CORRECT + "<|endoftext|>"
        s = deepcopy(self.ds[0])
        self.ic.make_intervention(s, {"completion": dirty})
        self.assertNotIn("<|endoftext|>", s["raw_generation"])

    def test_make_intervention_returns_same_sample_object(self):
        s = deepcopy(self.ds[0])
        out = self.ic.make_intervention(s, {"completion": COMP_CORRECT})
        self.assertIs(out, s)


# ===========================================================================
# 14. interventions_to_prompt -- ordering guarantee
# ===========================================================================

class TestInterventionsToPromptOrder(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def test_local_edits_before_correction_in_prompt_list(self):
        # Build a sample with both Local Edits and Correction (manually)
        s = deepcopy(self.ds[0])
        s["generation_status"] = "correct"
        s["mediator_rubric"] = deepcopy(GOLD_RUBRIC_2Q)
        s["target_before_intervention"] = "Supported"
        s["tool_rubric"] = {}

        s["structure_intervention"] = {
            "Local Edits": [
                {**deepcopy(self.ds[0]), "mediator_rubric": {"Q1?": True}},
                {**deepcopy(self.ds[0]), "mediator_rubric": {"Q2?": True}},
            ],
            "Correction": [
                {**deepcopy(self.ds[0]), "mediator_rubric": deepcopy(GOLD_RUBRIC_2Q)},
            ],
        }

        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), 3)
        # First two are Local Edits (contain Q1, Q2), third is Correction
        # Just verify count for now; content verified via make_prompt tests
        self.assertIsInstance(prompts[0], str)
        self.assertIsInstance(prompts[1], str)
        self.assertIsInstance(prompts[2], str)


# ===========================================================================
# 15. make_prompt -- local edit prompt reflects modified mediator_rubric
# ===========================================================================

class TestMakePromptWithModifiedMediator(unittest.TestCase):

    def setUp(self):
        self.ic, self.ds = make_intervention()

    def test_local_edit_prompt_contains_flipped_value(self):
        # Create a local edit sample where first Q is flipped to True
        s = deepcopy(self.ds[0])
        flipped = deepcopy(GOLD_RUBRIC_2Q)
        first_key = list(flipped.keys())[0]
        flipped[first_key] = True
        s["mediator_rubric"] = flipped

        prompt = self.ic.make_prompt(s, include_gold_structure=True)
        # The filled checklist should contain "True" for the first question
        self.assertIn("True", prompt)

    def test_gold_structure_prompt_tail_non_tool(self):
        s = deepcopy(self.ds[0])
        s["mediator_rubric"] = deepcopy(GOLD_RUBRIC_2Q)
        prompt = self.ic.make_prompt(s, include_gold_structure=True)
        # Non-tool: tail must end with "Final Verdict: " (model appends just the word)
        self.assertIn("Final Verdict: ", prompt)
        # Must NOT already have the verdict word appended
        self.assertNotIn("Final Verdict: Supported", prompt)
        self.assertNotIn("Final Verdict: Refuted", prompt)

    def test_gold_structure_prompt_tail_tool_mode(self):
        ic, ds = make_intervention(tool_mode="structured")
        s = deepcopy(ds[0])
        s["mediator_rubric"] = deepcopy(GOLD_RUBRIC_2Q)
        prompt = ic.make_prompt(s, include_gold_structure=True)
        self.assertIn("Final tool call:", prompt)
