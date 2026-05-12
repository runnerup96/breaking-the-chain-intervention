"""End-to-end smoke tests for CRUXEvalIntervention (no LLM, no HF)."""

import unittest

from datasets_for_intervention.cruxeval_intervention import CRUXEvalIntervention
from datasets_for_intervention.cruxeval_structure_processor import (
    CRUXEvalStructureProcessor,
    CRUXEvalTool,
)
from datasets_for_intervention.cruxeval_trace import (
    LOCAL_EDIT_LEVELS,
    trace_to_text,
)
from datasets_for_intervention.test_intervention.cruxeval_mocks import (
    CRUXEvalDatasetMock,
    FakeLLMModel,
)


def make_intervention(tool_mode="none", regime="standard"):
    ds   = CRUXEvalDatasetMock()
    llm  = FakeLLMModel()
    tool = CRUXEvalTool(ds, tool_mode)
    proc = CRUXEvalStructureProcessor(ds, tool_mode)
    ic   = CRUXEvalIntervention(ds, llm, tool, proc, regime, tool_mode)
    return ic, ds


def correct_completion(sample):
    return (
        "Trace:\n"
        f"{trace_to_text(sample['gold_trace'])}\n"
        f"Final Answer: {sample['gold_target']}\n"
    )


def incorrect_completion(sample):
    # Right trace, wrong answer.
    return (
        "Trace:\n"
        f"{trace_to_text(sample['gold_trace'])}\n"
        "Final Answer: 'definitely_wrong'\n"
    )


class TestPipeline(unittest.TestCase):
    def test_correct_path_creates_local_edits(self):
        ic, ds = make_intervention()
        sample = dict(ds[0])
        result = ic.make_intervention(sample, {"completion": correct_completion(sample)})
        self.assertEqual(result["generation_status"], "correct")
        local_edits = result["structure_intervention"]["Local Edits"]
        self.assertGreater(len(local_edits), 0)
        # Each Local Edit must have an expected target.
        for edit in local_edits:
            self.assertIn("expected_target_after_intervention", edit)
            self.assertIn("perturbation_level", edit)
            self.assertIn(edit["perturbation_level"], LOCAL_EDIT_LEVELS)

    def test_incorrect_path_creates_correction(self):
        ic, ds = make_intervention()
        sample = dict(ds[0])
        result = ic.make_intervention(sample, {"completion": incorrect_completion(sample)})
        self.assertEqual(result["generation_status"], "incorrect")
        self.assertEqual(result["structure_intervention"]["Local Edits"], [])
        self.assertEqual(len(result["structure_intervention"]["Correction"]), 1)

    def test_error_path_on_garbage(self):
        ic, ds = make_intervention()
        sample = dict(ds[0])
        result = ic.make_intervention(
            sample, {"completion": "Sure! Let me think... Final Answer: 8"}
        )
        self.assertEqual(result["generation_status"], "error")
        self.assertEqual(result["structure_intervention"]["Local Edits"], [])
        self.assertEqual(result["structure_intervention"]["Correction"], [])

    def test_prompts_can_be_built(self):
        ic, ds = make_intervention()
        sample = dict(ds[0])
        # Pass 1: no gold structure
        p1 = ic.make_prompt(sample, include_gold_structure=False)
        self.assertIn("def f(x)", p1)
        self.assertIn("Trace format", p1)
        # Pass 2: assistant prefix carries the (possibly perturbed) trace.
        p2 = ic.make_prompt(sample, include_gold_structure=True)
        self.assertIn("Trace:", p2)
        self.assertIn("Final Answer:", p2)


class TestLocalEditSampling(unittest.TestCase):
    def test_sampling_all_uses_every_applicable_level(self):
        ic, ds = make_intervention()
        sample = dict(ds[0])
        result = ic.make_intervention(sample, {"completion": correct_completion(sample)})
        levels = [e["perturbation_level"]
                  for e in result["structure_intervention"]["Local Edits"]]
        self.assertEqual(sorted(levels), sorted(set(levels)),
                         "Each level should appear at most once.")
        # With sampling='all', we expect every level in LOCAL_EDIT_LEVELS that
        # is applicable to show up. f(x)=x+1; *2 has int locals + >=2 steps,
        # so 1..6 are all applicable.
        self.assertEqual(set(levels), set(LOCAL_EDIT_LEVELS))

    def test_sampling_one_produces_single_edit(self):
        ds   = CRUXEvalDatasetMock()
        llm  = FakeLLMModel()
        tool = CRUXEvalTool(ds, "none")
        proc = CRUXEvalStructureProcessor(ds, "none")
        ic = CRUXEvalIntervention(
            ds, llm, tool, proc,
            prompting_regime="standard", tool_mode="none",
            intervention_levels=[1, 3, 5],
            local_edit_sampling="one",
        )
        sample = dict(ds[0])
        result = ic.make_intervention(sample, {"completion": correct_completion(sample)})
        edits = result["structure_intervention"]["Local Edits"]
        self.assertEqual(len(edits), 1)
        self.assertIn(edits[0]["perturbation_level"], [1, 3, 5])

    def test_sampling_one_is_deterministic_per_idx(self):
        ds = CRUXEvalDatasetMock()
        llm = FakeLLMModel()
        tool = CRUXEvalTool(ds, "none")
        proc = CRUXEvalStructureProcessor(ds, "none")

        def run():
            ic = CRUXEvalIntervention(
                ds, llm, tool, proc,
                intervention_levels=[1, 2, 3, 4, 5, 6],
                local_edit_sampling="one",
                perturb_seed=42,
            )
            s = dict(ds[0])
            r = ic.make_intervention(s, {"completion": correct_completion(s)})
            return r["structure_intervention"]["Local Edits"][0]["perturbation_level"]

        self.assertEqual(run(), run())


class TestToolMode(unittest.TestCase):
    def test_structured_tool_simulates_answer(self):
        ic, ds = make_intervention(tool_mode="structured")
        sample = dict(ds[0])
        # Compose a tool-style completion.
        completion = (
            "Trace:\n"
            f"{trace_to_text(sample['gold_trace'])}\n"
            "Final tool call:\n"
            "TOOL: simulate_output\n"
            'ARGS: {"trace": [{"line": 2, "locals": {"x": "3"}}, '
            '{"line": 3, "locals": {"x": "3", "y": "4"}}]}\n'
        )
        result = ic.make_intervention(sample, {"completion": completion})
        self.assertEqual(result["generation_status"], "correct")


if __name__ == "__main__":
    unittest.main()
