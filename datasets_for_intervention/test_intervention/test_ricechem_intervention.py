import unittest
from copy import deepcopy
from math import isclose

from datasets_for_intervention.test_intervention.llm_mocks import FakeLLMModel
from datasets_for_intervention.test_intervention.ricechem_mocks import RiceChemDatasetMock
from datasets_for_intervention.ricechem_structure_processor import RiceChemTool, RiceChemStructureProcessor
from datasets_for_intervention.ricechem_intervention import RiceChemIntervention


def make_ic(dataset, llm, tool_mode="none", prompting_regime="standard"):
    tool = RiceChemTool(dataset, tool_mode=tool_mode)
    proc = RiceChemStructureProcessor(dataset, tool_mode=tool_mode)
    return RiceChemIntervention(
        dataset=dataset,
        llm_model=llm,
        tool=tool,
        processor=proc,
        prompting_regime=prompting_regime,
        tool_mode=tool_mode,
    )


# ---------------------------------------------------------------------------
# Хелперы — строить completions для sample[0]
# gold_rubric = {A:True, B:False, C:True}
# ---------------------------------------------------------------------------

def correct_completion():
    return (
        "Checklist:\n"
        "A item (True/False): True\n"
        "B item (True/False): False\n"
        "C item (True/False): True\n"
        "Final grade (0-3): 1.5"
    )

def incorrect_completion():
    return (
        "Checklist:\n"
        "A item (True/False): False\n"
        "B item (True/False): True\n"
        "C item (True/False): False\n"
        "Final grade (0-3): 1.5"
    )

def garbage_preamble_completion():
    return (
        "Sure, let me grade this:\n"
        "Checklist:\n"
        "A item (True/False): True\n"
        "Final grade (0-3): 1.0"
    )

def garbage_trailer_completion():
    return (
        "Checklist:\n"
        "A item (True/False): True\n"
        "Final grade (0-3): 1.0\n"
        "Hope this helps!"
    )


# ===========================================================================
# 1. _has_garbage
# ===========================================================================

class TestHasGarbage(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()

    def _ic(self, tool_mode="none"):
        return make_ic(self.dataset, self.llm, tool_mode=tool_mode)

    # --- none mode ----------------------------------------------------------

    def test_clean_completion_no_garbage(self):
        ic = self._ic()
        self.assertFalse(ic._has_garbage(correct_completion()))

    def test_preamble_detected_as_garbage(self):
        ic = self._ic()
        self.assertTrue(ic._has_garbage(garbage_preamble_completion()))

    def test_trailing_text_detected_as_garbage(self):
        ic = self._ic()
        self.assertTrue(ic._has_garbage(garbage_trailer_completion()))

    def test_missing_final_grade_is_garbage(self):
        ic = self._ic()
        completion = "Checklist:\nA item (True/False): True\n"
        self.assertTrue(ic._has_garbage(completion))

    def test_empty_string_is_garbage(self):
        ic = self._ic()
        self.assertTrue(ic._has_garbage(""))

    def test_whitespace_only_is_garbage(self):
        ic = self._ic()
        self.assertTrue(ic._has_garbage("   \n  "))

    # --- tool mode (simple / structured) ------------------------------------

    def test_tool_mode_clean_completion_no_garbage(self):
        ic = self._ic(tool_mode="simple")
        completion = (
            "Checklist:\n"
            "A item (True/False): True\n"
            "Final tool call:\n"
            "   TOOL: calculate_score\n"
            '   ARGS: {"rubric": "A item (True/False): True"}'
        )
        self.assertFalse(ic._has_garbage(completion))

    def test_tool_mode_missing_args_is_garbage(self):
        ic = self._ic(tool_mode="simple")
        completion = (
            "Checklist:\n"
            "A item (True/False): True\n"
            "Final tool call:\n"
            "   TOOL: calculate_score\n"
        )
        self.assertTrue(ic._has_garbage(completion))

    def test_tool_mode_trailing_text_is_garbage(self):
        ic = self._ic(tool_mode="structured")
        completion = (
            "Checklist:\n"
            "A item (True/False): True\n"
            '   ARGS: {"rubric": [True]}\n'
            "Extra line here."
        )
        self.assertTrue(ic._has_garbage(completion))


# ===========================================================================
# 2. _classify_generation
# ===========================================================================

class TestClassifyGeneration(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()
        self.ic = make_ic(self.dataset, self.llm)
        self.gold = deepcopy(self.dataset[0]["gold_rubric"])  # A:True, B:False, C:True

    def test_error_when_mediator_none(self):
        self.assertEqual(self.ic._classify_generation("anything", None, self.gold), "error")

    def test_error_when_completion_has_garbage(self):
        # mediator распарсен, но completion содержит мусор
        mediator = deepcopy(self.gold)
        self.assertEqual(
            self.ic._classify_generation(garbage_preamble_completion(), mediator, self.gold),
            "error",
        )

    def test_correct_when_mediator_matches_gold(self):
        mediator = deepcopy(self.gold)
        self.assertEqual(
            self.ic._classify_generation(correct_completion(), mediator, self.gold),
            "correct",
        )

    def test_incorrect_when_mediator_differs_from_gold(self):
        mediator = {"A item": False, "B item": True, "C item": False}
        self.assertEqual(
            self.ic._classify_generation(incorrect_completion(), mediator, self.gold),
            "incorrect",
        )


# ===========================================================================
# 3. make_intervention — классификация и структура интервенций
# ===========================================================================

class TestMakeIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()
        self.ic = make_ic(self.dataset, self.llm)
        self.ic.make_prompt = lambda _s, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"
        self.sample = deepcopy(self.dataset[0])

    # --- generation_status --------------------------------------------------

    def test_correct_status_set(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": correct_completion()})
        self.assertEqual(out["generation_status"], "correct")

    def test_incorrect_status_set(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": incorrect_completion()})
        self.assertEqual(out["generation_status"], "incorrect")

    def test_error_status_preamble(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": garbage_preamble_completion()})
        self.assertEqual(out["generation_status"], "error")

    def test_error_status_trailing_text(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": garbage_trailer_completion()})
        self.assertEqual(out["generation_status"], "error")

    def test_error_status_empty(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": ""})
        self.assertEqual(out["generation_status"], "error")

    # --- mediator_rubric + score_before_intervention ------------------------

    def test_mediator_and_score_parsed_for_correct(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": correct_completion()})
        self.assertEqual(out["mediator_rubric"], {"A item": True, "B item": False, "C item": True})
        self.assertEqual(out["score_before_intervention"], 1.5)
        self.assertEqual(out["tool_rubric"], {})

    def test_mediator_empty_and_score_none_for_error(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": ""})
        self.assertEqual(out["mediator_rubric"], {})
        # Для error score и tool_rubric всегда явно None
        self.assertIsNone(out["score_before_intervention"])
        self.assertIsNone(out["tool_rubric"])

    def test_raw_generation_stored(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": correct_completion()})
        self.assertIn("raw_generation", out)

    # --- structure_intervention shape ---------------------------------------

    def test_correct_has_local_edits_only(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": correct_completion()})
        interv = out["structure_intervention"]
        # Нет HSVT
        self.assertNotIn("HSVT", interv)
        self.assertEqual(len(interv["Local Edits"]), len(self.sample["gold_rubric"]))
        self.assertEqual(len(interv["Correction"]), 0)

    def test_incorrect_has_correction_only(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": incorrect_completion()})
        interv = out["structure_intervention"]
        self.assertNotIn("HSVT", interv)
        self.assertEqual(len(interv["Local Edits"]), 0)
        self.assertEqual(len(interv["Correction"]), 1)

    def test_error_has_no_interventions(self):
        out = self.ic.make_intervention(deepcopy(self.sample), {"completion": garbage_preamble_completion()})
        interv = out["structure_intervention"]
        self.assertEqual(len(interv["Local Edits"]), 0)
        self.assertEqual(len(interv["Correction"]), 0)


# ===========================================================================
# 4. make_structure_intervention — детали содержимого
# ===========================================================================

class TestMakeStructureIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()
        self.ic = make_ic(self.dataset, self.llm)
        self.sample = deepcopy(self.dataset[0])

    def test_local_edits_flip_exactly_one_item_each(self):
        s = deepcopy(self.sample)
        s["generation_status"] = "correct"
        tree = self.ic.make_structure_intervention(s)
        for edit in tree["Local Edits"]:
            diffs = [k for k in s["mediator_rubric"] if s["mediator_rubric"][k] != edit["mediator_rubric"][k]]
            self.assertEqual(len(diffs), 1)

    def test_local_edits_cover_all_items(self):
        s = deepcopy(self.sample)
        s["generation_status"] = "correct"
        tree = self.ic.make_structure_intervention(s)
        flipped_items = {
            next(k for k in s["mediator_rubric"] if s["mediator_rubric"][k] != edit["mediator_rubric"][k])
            for edit in tree["Local Edits"]
        }
        self.assertEqual(flipped_items, set(s["mediator_rubric"].keys()))

    def test_local_edits_expected_scores_correct(self):
        s = deepcopy(self.sample)
        s["generation_status"] = "correct"
        tree = self.ic.make_structure_intervention(s)
        weights = self.dataset.task2rubric_weights[1]
        for edit in tree["Local Edits"]:
            expected = float(sum(weights[k] for k, v in edit["mediator_rubric"].items() if v))
            self.assertTrue(isclose(edit["expected_score_after_intervention"], expected, abs_tol=1e-9))

    def test_correction_uses_gold_rubric_not_bad_rubric(self):
        """Correction подставляет gold_rubric, а не какой-то bad_rubric."""
        s = deepcopy(self.sample)
        s["generation_status"] = "incorrect"
        # mediator_rubric намеренно отличается от gold
        s["mediator_rubric"] = {"A item": False, "B item": True, "C item": False}
        tree = self.ic.make_structure_intervention(s)
        self.assertEqual(len(tree["Correction"]), 1)
        self.assertEqual(tree["Correction"][0]["mediator_rubric"], s["gold_rubric"])

    def test_error_returns_empty_dicts(self):
        s = deepcopy(self.sample)
        s["generation_status"] = "error"
        tree = self.ic.make_structure_intervention(s)
        self.assertEqual(tree["Local Edits"], [])
        self.assertEqual(tree["Correction"], [])

    def test_local_edit_does_not_mutate_original_mediator(self):
        s = deepcopy(self.sample)
        s["generation_status"] = "correct"
        original_mediator = deepcopy(s["mediator_rubric"])
        self.ic.make_structure_intervention(s)
        self.assertEqual(s["mediator_rubric"], original_mediator)


# ===========================================================================
# 5. interventions_to_prompt
# ===========================================================================

class TestInterventionsToPrompt(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()
        self.ic = make_ic(self.dataset, self.llm)
        self.ic.make_prompt = lambda _s, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"
        self.sample = deepcopy(self.dataset[0])

    def _intervene(self, completion):
        s = deepcopy(self.sample)
        return self.ic.make_intervention(s, {"completion": completion})

    def test_correct_prompt_count_equals_n_local_edits(self):
        s = self._intervene(correct_completion())
        prompts = self.ic.interventions_to_prompt(s)
        n_local = len(s["structure_intervention"]["Local Edits"])
        self.assertEqual(len(prompts), n_local)

    def test_incorrect_prompt_count_equals_one(self):
        s = self._intervene(incorrect_completion())
        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), 1)

    def test_error_prompt_count_is_zero(self):
        s = self._intervene(garbage_preamble_completion())
        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), 0)

    def test_all_prompts_have_gold_structure_true(self):
        for comp in [correct_completion(), incorrect_completion()]:
            s = self._intervene(comp)
            for p in self.ic.interventions_to_prompt(s):
                self.assertEqual(p, "PROMPT(gold=True)")

    def test_no_hsvt_prompts_generated(self):
        """Убеждаемся, что HSVT полностью отсутствует в любом варианте."""
        for comp in [correct_completion(), incorrect_completion(), garbage_preamble_completion()]:
            s = self._intervene(comp)
            self.assertNotIn("HSVT", s["structure_intervention"])


# ===========================================================================
# 6. collect_intervention_completion
# ===========================================================================

class TestCollectInterventionCompletion(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()
        self.ic = make_ic(self.dataset, self.llm)
        self.ic.make_prompt = lambda _s, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"
        self.sample = deepcopy(self.dataset[0])

    def _intervene(self, completion):
        s = deepcopy(self.sample)
        return self.ic.make_intervention(s, {"completion": completion})

    def test_local_edits_scores_written_in_order(self):
        s = self._intervene(correct_completion())
        n = len(s["structure_intervention"]["Local Edits"])
        completions = [{"completion": str(float(i))} for i in range(n)]
        out = self.ic.collect_intervention_completion(s, completions)
        for i, local in enumerate(out["structure_intervention"]["Local Edits"]):
            self.assertEqual(local["score_after_intervention"], float(i))

    def test_correction_score_written(self):
        s = self._intervene(incorrect_completion())
        completions = [{"completion": "2.5"}]
        out = self.ic.collect_intervention_completion(s, completions)
        self.assertEqual(out["structure_intervention"]["Correction"][0]["score_after_intervention"], 2.5)

    def test_local_edits_then_correction_order(self):
        """Порядок: сначала Local Edits, потом Correction."""
        # Используем сэмпл 0, делаем его incorrect вручную
        s = deepcopy(self.sample)
        s["generation_status"] = "incorrect"
        s["mediator_rubric"] = {"A item": False, "B item": True, "C item": False}
        s["structure_intervention"] = self.ic.make_structure_intervention(s)
        # У incorrect: Local Edits пусты, Correction одна запись
        completions = [{"completion": "99.0"}]
        out = self.ic.collect_intervention_completion(s, completions)
        self.assertEqual(out["structure_intervention"]["Correction"][0]["score_after_intervention"], 99.0)


# ===========================================================================
# 7. Tool mode — infer_completion
# ===========================================================================

class TestInferCompletionToolModes(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()

    def test_simple_populates_tool_rubric(self):
        ic = make_ic(self.dataset, self.llm, tool_mode="simple")
        sample = deepcopy(self.dataset[0])
        completion = (
            "TOOL: calculate_score\n"
            'ARGS: {"rubric": "A item (True/False): True\\nB item (True/False): False\\nC item (True/False): True"}\n'
        )
        score, tool_rubric = ic.infer_completion(completion, sample, short_completion=False)
        weights = self.dataset.task2rubric_weights[1]
        expected = float(weights["A item"] + weights["C item"])
        self.assertEqual(score, expected)
        self.assertEqual(tool_rubric, {"A item": True, "B item": False, "C item": True})

    def test_structured_canonicalizes_boollist(self):
        ic = make_ic(self.dataset, self.llm, tool_mode="structured")
        sample = deepcopy(self.dataset[0])
        completion = 'TOOL: calculate_score\nARGS: {"rubric": [True, False, True]}\n'
        score, tool_rubric = ic.infer_completion(completion, sample, short_completion=False)
        weights = self.dataset.task2rubric_weights[1]
        expected = float(weights["A item"] + weights["C item"])
        self.assertEqual(score, expected)
        self.assertEqual(tool_rubric, {"A item": True, "B item": False, "C item": True})

    def test_simple_returns_none_when_unparseable(self):
        ic = make_ic(self.dataset, self.llm, tool_mode="simple")
        s = deepcopy(self.dataset[0])
        score, tool_rubric = ic.infer_completion('TOOL: calculate_score\nARGS: {"rubric": 123}\n', s)
        self.assertIsNone(score)
        self.assertIsNone(tool_rubric)

    def test_structured_returns_none_on_invalid_tokens(self):
        ic = make_ic(self.dataset, self.llm, tool_mode="structured")
        s = deepcopy(self.dataset[0])
        score, tool_rubric = ic.infer_completion('TOOL: calculate_score\nARGS: {"rubric": [True, null, False]}\n', s)
        self.assertIsNone(score)
        self.assertIsNone(tool_rubric)


# ===========================================================================
# 8. Tool mode — collect_intervention_completion записывает tool_rubric_after
# ===========================================================================

class TestCollectToolRubricAfter(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()

    def test_local_edits_tool_rubric_after_intervention_populated(self):
        ic = make_ic(self.dataset, self.llm, tool_mode="structured")
        ic.make_prompt = lambda _s, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"

        s = deepcopy(self.dataset[0])
        s["generation_status"] = "correct"
        s["mediator_rubric"] = deepcopy(s["gold_rubric"])
        s["score_before_intervention"], s["tool_rubric"] = ic.infer_completion(
            'TOOL: calculate_score\nARGS: {"rubric": [True, False, True]}\n', s, False
        )
        s["structure_intervention"] = ic.make_structure_intervention(s)

        n = len(s["structure_intervention"]["Local Edits"])
        outs = [
            {"completion": 'TOOL: calculate_score\nARGS: {"rubric": [True, True, True]}\n'}
            for _ in range(n)
        ]
        out = ic.collect_intervention_completion(s, outs)
        for local in out["structure_intervention"]["Local Edits"]:
            self.assertIn("tool_rubric_after_intervention", local)
            self.assertEqual(
                local["tool_rubric_after_intervention"],
                {"A item": True, "B item": True, "C item": True},
            )

    def test_correction_tool_rubric_after_intervention_populated(self):
        ic = make_ic(self.dataset, self.llm, tool_mode="structured")
        ic.make_prompt = lambda _s, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"

        s = deepcopy(self.dataset[0])
        s["generation_status"] = "incorrect"
        s["mediator_rubric"] = {"A item": False, "B item": True, "C item": False}
        s["score_before_intervention"] = 1.5
        s["tool_rubric"] = {}
        s["structure_intervention"] = ic.make_structure_intervention(s)

        outs = [{"completion": 'TOOL: calculate_score\nARGS: {"rubric": [True, False, True]}\n'}]
        out = ic.collect_intervention_completion(s, outs)
        corr = out["structure_intervention"]["Correction"][0]
        self.assertIn("tool_rubric_after_intervention", corr)
        self.assertEqual(
            corr["tool_rubric_after_intervention"],
            {"A item": True, "B item": False, "C item": True},
        )


# ===========================================================================
# 9. edge cases
# ===========================================================================

class TestMiscEdgeCases(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()

    def test_init_accepts_all_prompting_regimes(self):
        for regime in ["standard", "detailed", "max_detailed"]:
            make_ic(self.dataset, self.llm, prompting_regime=regime)

    def test_init_accepts_all_tool_modes(self):
        for tm in ["none", "simple", "structured"]:
            make_ic(self.dataset, self.llm, tool_mode=tm)

    def test_clean_llm_output_strips_special_tokens(self):
        ic = make_ic(self.dataset, self.llm)
        raw = "<|im_start|>text\u200b content<|im_end|>"
        self.assertEqual(ic.clean_llm_output(raw), "text content")

    def test_make_intervention_does_not_mutate_original_sample(self):
        ic = make_ic(self.dataset, self.llm)
        original = deepcopy(self.dataset[0])
        sample_copy = deepcopy(original)
        ic.make_intervention(sample_copy, {"completion": correct_completion()})
        # оригинал в датасете не тронут
        self.assertEqual(self.dataset[0]["mediator_rubric"], original["mediator_rubric"])

    def test_make_prompt_returns_string(self):
        ic = make_ic(self.dataset, self.llm)
        s = deepcopy(self.dataset[0])
        prompt = ic.make_prompt(s, include_gold_structure=False)
        self.assertIsInstance(prompt, str)

    def test_make_prompt_with_gold_structure_returns_string(self):
        ic = make_ic(self.dataset, self.llm)
        s = deepcopy(self.dataset[0])
        prompt = ic.make_prompt(s, include_gold_structure=True)
        self.assertIsInstance(prompt, str)