import unittest
from copy import deepcopy
from math import isclose

from datasets_for_intervention.test_intervention.llm_mocks import FakeLLMModel
from datasets_for_intervention.test_intervention.ricechem_mocks import RiceChemDatasetMock
from datasets_for_intervention.ricechem_structure_processor import RiceChemTool, RiceChemStructureProcessor
from datasets_for_intervention.ricechem_intervention import RiceChemIntervention


class TestRiceChemInterventionNonTool(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()

        self.tool = RiceChemTool(self.dataset, tool_mode="simple")
        self.processor = RiceChemStructureProcessor(self.dataset, tool_mode='none')

        self.ic = RiceChemIntervention(
            dataset=self.dataset,
            llm_model=self.llm,
            tool=self.tool,
            processor=self.processor,
            prompting_regime="standard",
            tool_mode='none'
        )

        self.ic.make_prompt = lambda _s, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"
        self.sample = deepcopy(self.dataset[0])

    def test_make_structure_intervention_shapes_and_expected_scores(self):
        s = deepcopy(self.sample)
        s["completion_type"] = "gold_structure"

        tree = self.ic.make_structure_intervention(s)

        self.assertEqual(set(tree.keys()), {"HSVT", "Local Edits", "Correction"})
        self.assertEqual(len(tree["HSVT"]), 1)
        self.assertEqual(len(tree["Local Edits"]), len(s["mediator_rubric"]))
        self.assertEqual(len(tree["Correction"]), 1)

        hsvt = tree["HSVT"][0]
        self.assertNotEqual(hsvt["student_answer"], s["student_answer"])
        self.assertEqual(hsvt["mediator_rubric"], s["mediator_rubric"])

        weights = self.dataset.task2rubric_weights[s["task_idx"]]
        for local in tree["Local Edits"]:
            diffs = [k for k in s["mediator_rubric"] if s["mediator_rubric"][k] != local["mediator_rubric"][k]]
            self.assertEqual(len(diffs), 1)
            expected = float(sum(weights[k] for k, v in local["mediator_rubric"].items() if v))
            self.assertTrue(isclose(local["expected_score_after_intervention"], expected, rel_tol=0, abs_tol=1e-9))

        corr = tree["Correction"][0]
        self.assertEqual(corr["mediator_rubric"], s["bad_rubric"])

    def test_make_intervention_structure_prediction_parses_mediator_and_score(self):
        s = deepcopy(self.sample)
        s["completion_type"] = "structure_prediction"

        completion = (
            "Checklist:\n"
            "A item (True/False): True\n"
            "B item (True/False): False\n"
            "C item (True/False): True\n"
            "Final grade (0-3): 2.0\n"
        )
        out = self.ic.make_intervention(s, {"completion": completion})

        self.assertIn("raw_generation", out)
        self.assertEqual(out["mediator_rubric"], {"A item": True, "B item": False, "C item": True})
        self.assertEqual(out["score_before_intervention"], 2.0)
        self.assertEqual(out["tool_rubric"], {})
        self.assertIn("structure_intervention", out)

    def test_make_intervention_gold_structure_tail_only_score(self):
        s = deepcopy(self.sample)
        s["completion_type"] = "gold_structure"

        out = self.ic.make_intervention(s, {"completion": " 1.75 "})

        self.assertEqual(out["mediator_rubric"], self.sample["mediator_rubric"])
        self.assertEqual(out["score_before_intervention"], 1.75)
        self.assertEqual(out["tool_rubric"], {})

    def test_collect_intervention_completion_order_and_mapping(self):
        s = deepcopy(self.sample)
        s["completion_type"] = "gold_structure"
        s["structure_intervention"] = self.ic.make_structure_intervention(s)

        M = len(s["structure_intervention"]["Local Edits"])
        completions = [{"completion": "1.0"}] + [{"completion": str(2.0 + i)} for i in range(M)] + [{"completion": "9.0"}]
        out = self.ic.collect_intervention_completion(s, completions)

        self.assertEqual(out["structure_intervention"]["HSVT"][0]["score_after_intervention"], 1.0)
        for i in range(M):
            self.assertEqual(out["structure_intervention"]["Local Edits"][i]["score_after_intervention"], 2.0 + i)
        self.assertEqual(out["structure_intervention"]["Correction"][0]["score_after_intervention"], 9.0)

    def test_interventions_to_prompt_counts_and_flag(self):
        s = deepcopy(self.sample)
        s["completion_type"] = "structure_prediction"
        s["structure_intervention"] = self.ic.make_structure_intervention(s)

        prompts = self.ic.interventions_to_prompt(s)

        self.assertEqual(len(prompts), 1 + len(s["structure_intervention"]["Local Edits"]))
        self.assertTrue(all(p.startswith("PROMPT(gold=True)") for p in prompts))


class TestRiceChemInterventionToolModes(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm = FakeLLMModel()

    def test_infer_completion_tool_simple_populates_tool_rubric(self):
        tool = RiceChemTool(self.dataset, tool_mode="simple")
        proc = RiceChemStructureProcessor(self.dataset, tool_mode="simple")
        ic = RiceChemIntervention(self.dataset, self.llm, tool, proc, prompting_regime="standard", tool_mode="simple")

        sample = deepcopy(self.dataset[0])
        completion = (
            "TOOL: calculate_score\n"
            "ARGS: {\"rubric\": \"A item (True/False): True\\nB item (True/False): False\\nC item (True/False): True\"}\n"
        )
        score, tool_rubric = ic.infer_completion(completion, sample, short_completion=False)

        weights = self.dataset.task2rubric_weights[1]
        expected = float(weights["A item"] + weights["C item"])
        self.assertEqual(score, expected)
        self.assertEqual(tool_rubric, {"A item": True, "B item": False, "C item": True})

    def test_infer_completion_tool_structured_boollist_is_canonicalized(self):
        tool = RiceChemTool(self.dataset, tool_mode="structured")
        proc = RiceChemStructureProcessor(self.dataset, tool_mode="structured")
        ic = RiceChemIntervention(self.dataset, self.llm, tool, proc, prompting_regime="standard", tool_mode="structured")

        sample = deepcopy(self.dataset[0])
        completion = "TOOL: calculate_score\nARGS: {\"rubric\": [True, False, True]}\n"
        score, tool_rubric = ic.infer_completion(completion, sample, short_completion=False)

        weights = self.dataset.task2rubric_weights[1]
        expected = float(weights["A item"] + weights["C item"])
        self.assertEqual(score, expected)
        self.assertEqual(tool_rubric, {"A item": True, "B item": False, "C item": True})

    def test_intervention_init_accepts_max_detailed(self):
        tool = RiceChemTool(self.dataset, tool_mode="simple")
        proc = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        RiceChemIntervention(self.dataset, self.llm, tool, proc, prompting_regime="max_detailed", tool_mode='none')

    def test_make_structure_intervention_no_bad_rubric_means_no_correction(self):
        tool = RiceChemTool(self.dataset, tool_mode="simple")
        proc = RiceChemStructureProcessor(self.dataset, tool_mode='none')
        ic = RiceChemIntervention(self.dataset, self.llm, tool, proc, prompting_regime="standard", tool_mode='none')

        s = deepcopy(self.dataset[0])
        s["completion_type"] = "gold_structure"
        s.pop("bad_rubric", None)
        s.pop("bad_score", None)

        tree = ic.make_structure_intervention(s)
        self.assertEqual(tree["Correction"], [])

    def test_infer_completion_tool_simple_returns_none_when_args_unparseable(self):
        tool = RiceChemTool(self.dataset, tool_mode="simple")
        proc = RiceChemStructureProcessor(self.dataset, tool_mode="simple")
        ic = RiceChemIntervention(self.dataset, self.llm, tool, proc, prompting_regime="standard", tool_mode="simple")

        s = deepcopy(self.dataset[0])
        completion = "TOOL: calculate_score\nARGS: {\"rubric\": 123}\n"
        score, tool_rubric = ic.infer_completion(completion, s, short_completion=False)
        self.assertIsNone(score)
        self.assertIsNone(tool_rubric)

    def test_infer_completion_tool_structured_returns_none_on_invalid_tokens(self):
        tool = RiceChemTool(self.dataset, tool_mode="structured")
        proc = RiceChemStructureProcessor(self.dataset, tool_mode="structured")
        ic = RiceChemIntervention(self.dataset, self.llm, tool, proc, prompting_regime="standard", tool_mode="structured")

        s = deepcopy(self.dataset[0])
        completion = "TOOL: calculate_score\nARGS: {\"rubric\": [True, null, False]}\n"
        score, tool_rubric = ic.infer_completion(completion, s, short_completion=False)
        self.assertIsNone(score)
        self.assertIsNone(tool_rubric)

    def test_collect_intervention_completion_tool_mode_populates_tool_rubric_after(self):
        tool = RiceChemTool(self.dataset, tool_mode="structured")
        proc = RiceChemStructureProcessor(self.dataset, tool_mode="structured")
        ic = RiceChemIntervention(self.dataset, self.llm, tool, proc, prompting_regime="standard", tool_mode="structured")

        s = deepcopy(self.dataset[0])
        s["completion_type"] = "gold_structure"
        s["score_before_intervention"], s["tool_rubric"] = ic.infer_completion("TOOL: calculate_score\nARGS: {\"rubric\": [True, False, True]}\n", s, False)
        s["structure_intervention"] = ic.make_structure_intervention(s)

        M = len(s["structure_intervention"]["Local Edits"])
        outs = []
        outs.append({"completion": "TOOL: calculate_score\nARGS: {\"rubric\": [False, False, False]}\n"})
        for i in range(M):
            outs.append({"completion": "TOOL: calculate_score\nARGS: {\"rubric\": [True, True, True]}\n"})
        if len(s["structure_intervention"]["Correction"]) > 0:
            outs.append({"completion": "TOOL: calculate_score\nARGS: {\"rubric\": [True, False, True]}\n"})

        out = ic.collect_intervention_completion(s, outs)

        hsvt = out["structure_intervention"]["HSVT"][0]
        self.assertIn("tool_rubric_after_intervention", hsvt)
        self.assertEqual(hsvt["tool_rubric_after_intervention"], {"A item": False, "B item": False, "C item": False})

        for local in out["structure_intervention"]["Local Edits"]:
            self.assertEqual(local["tool_rubric_after_intervention"], {"A item": True, "B item": True, "C item": True})
