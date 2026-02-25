import unittest
import sys
from pathlib import Path

# Add parent directories to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets_for_intervention.prompt import Prompt
from datasets_for_intervention.test_intervention.llm_mocks import FakeLLMModel


class TestPrompt(unittest.TestCase):
    def setUp(self):
        self.llm_model = FakeLLMModel()

    def test_make_prompt_exact_output_with_gold_structure(self):
        """Test exact prompt output with all components."""
        prompt = Prompt(
            prompting_regime="baseline_structure_faithfulness",
            use_tool_call=True,
            tool_call_instruction="TOOL CALL INSTRUCTION",
            instruction="INSTRUCTION",
            few_shot="FEW_SHOT",
            llm_model=self.llm_model,
        )
        result = prompt.make_prompt(
            current_sample="SAMPLE",
            include_gold_structure=True,
            gold_structure="GOLD",
        )
        # Expected exact output from FakeTokenizer
        expected = "User: INSTRUCTION\n\nTOOL CALL INSTRUCTION\n\nFEW_SHOT\n\nSAMPLE\nAssistant: GOLD\n"
        self.assertEqual(result, expected)

    def test_baseline_regime_string_empty(self):
        prompt = Prompt(
            prompting_regime="baseline_structure_faithfulness",
            use_tool_call=False,
            tool_call_instruction="",
            instruction="Domain instruction",
            few_shot="Few-shot examples",
            llm_model=self.llm_model,
        )
        self.assertEqual(prompt.regime_string, "")

    def test_detailed_regime_string_not_empty(self):
        prompt = Prompt(
            prompting_regime="detailed_instruction",
            use_tool_call=False,
            tool_call_instruction="",
            instruction="Domain instruction",
            few_shot="Few-shot examples",
            llm_model=self.llm_model,
        )
        self.assertIn("intervention", prompt.regime_string.lower())
        self.assertIn("structured reasoning block", prompt.regime_string.lower())

    def test_detailed_regime_string_uses_new_wording(self):
        prompt = Prompt(
            prompting_regime="detailed_instruction",
            use_tool_call=False,
            tool_call_instruction="",
            instruction="Domain instruction",
            few_shot="Few-shot examples",
            llm_model=self.llm_model,
        )
        self.assertIn("structured reasoning block", prompt.regime_string)
        self.assertNotIn("intermediate structure", prompt.regime_string.lower())

    def test_maximum_regime_string_contains_priority_rules(self):
        prompt = Prompt(
            prompting_regime="maximum_mediator_faithfulness",
            use_tool_call=False,
            tool_call_instruction="",
            instruction="Domain instruction",
            few_shot="Few-shot examples",
            llm_model=self.llm_model,
        )
        regime = prompt.regime_string
        self.assertIn("structured reasoning block", regime.lower())
        self.assertIn("ULTIMATE TRUTH", regime)
        self.assertIn("SOLELY on your compliance", regime)

    def test_invalid_regime_raises_error(self):
        with self.assertRaises(ValueError) as context:
            Prompt(
                prompting_regime="invalid_regime",
                use_tool_call=False,
                tool_call_instruction="",
                instruction="Domain instruction",
                few_shot="Few-shot examples",
                llm_model=self.llm_model,
            )
        self.assertIn("Unknown prompting regime", str(context.exception))

    def test_build_zeroshot_instruction_baseline_without_tool_call(self):
        prompt = Prompt(
            prompting_regime="baseline_structure_faithfulness",
            use_tool_call=False,
            tool_call_instruction="",
            instruction="Domain instruction",
            few_shot="Few-shot examples",
            llm_model=self.llm_model,
        )
        result = prompt.build_zeroshot_instruction()
        self.assertEqual(result, "Domain instruction")

    def test_build_zeroshot_instruction_baseline_with_tool_call(self):
        prompt = Prompt(
            prompting_regime="baseline_structure_faithfulness",
            use_tool_call=True,
            tool_call_instruction="Use tools for X",
            instruction="Domain instruction",
            few_shot="Few-shot examples",
            llm_model=self.llm_model,
        )
        result = prompt.build_zeroshot_instruction()
        self.assertIn("Domain instruction", result)
        self.assertIn("Use tools for X", result)
        self.assertEqual(result.count("\n\n"), 1)  # One separator

    def test_build_zeroshot_instruction_detailed_with_tool_call(self):
        prompt = Prompt(
            prompting_regime="detailed_instruction",
            use_tool_call=True,
            tool_call_instruction="Use tools for X",
            instruction="Domain instruction",
            few_shot="Few-shot examples",
            llm_model=self.llm_model,
        )
        result = prompt.build_zeroshot_instruction()
        self.assertIn("Domain instruction", result)
        self.assertIn("Use tools for X", result)
        self.assertIn("intervention", result.lower())
        # Order: instruction, tool_call, regime
        domain_idx = result.find("Domain instruction")
        tool_idx = result.find("Use tools for X")
        intervention_idx = result.find("intervention")
        self.assertLess(domain_idx, tool_idx)
        self.assertLess(tool_idx, intervention_idx)

    def test_build_zeroshot_instruction_maximum_with_tool_call(self):
        prompt = Prompt(
            prompting_regime="maximum_mediator_faithfulness",
            use_tool_call=True,
            tool_call_instruction="Use tools for X",
            instruction="Domain instruction",
            few_shot="Few-shot examples",
            llm_model=self.llm_model,
        )
        result = prompt.build_zeroshot_instruction()
        self.assertIn("Domain instruction", result)
        self.assertIn("Use tools for X", result)
        self.assertIn("structured reasoning block", result.lower())
        self.assertIn("ULTIMATE TRUTH", result)
        self.assertLess(result.find("Domain instruction"), result.find("Use tools for X"))
        self.assertLess(result.find("Use tools for X"), result.lower().find("intervention possibility"))

    def test_make_prompt_without_gold_structure(self):
        prompt = Prompt(
            prompting_regime="baseline_structure_faithfulness",
            use_tool_call=False,
            tool_call_instruction="",
            instruction="Instruction",
            few_shot="Few-shot block",
            llm_model=self.llm_model,
        )
        result = prompt.make_prompt(
            current_sample="Sample text",
            include_gold_structure=False,
        )
        self.assertIsInstance(result, str)
        self.assertIn("Instruction", result)
        self.assertIn("Few-shot block", result)
        self.assertIn("Sample text", result)
        self.assertIn("Assistant:", result)  # generation prompt added

    def test_make_prompt_with_gold_structure(self):
        prompt = Prompt(
            prompting_regime="baseline_structure_faithfulness",
            use_tool_call=False,
            tool_call_instruction="",
            instruction="Instruction",
            few_shot="Few-shot block",
            llm_model=self.llm_model,
        )
        result = prompt.make_prompt(
            current_sample="Sample text",
            include_gold_structure=True,
            gold_structure="Gold structure text",
        )
        self.assertIsInstance(result, str)
        self.assertIn("Instruction", result)
        self.assertIn("Few-shot block", result)
        self.assertIn("Sample text", result)
        self.assertIn("Gold structure text", result)
        # Should NOT have generation prompt when gold structure is used
        self.assertFalse(result.strip().endswith("Assistant: "))

    def test_make_prompt_with_empty_few_shot(self):
        prompt = Prompt(
            prompting_regime="baseline_structure_faithfulness",
            use_tool_call=False,
            tool_call_instruction="",
            instruction="Instruction",
            few_shot="",
            llm_model=self.llm_model,
        )
        result = prompt.make_prompt(
            current_sample="Sample text",
            include_gold_structure=False,
        )
        self.assertIsInstance(result, str)
        self.assertIn("Instruction", result)
        self.assertIn("Sample text", result)

    def test_assertion_empty_instruction(self):
        with self.assertRaises(AssertionError):
            Prompt(
                prompting_regime="baseline_structure_faithfulness",
                use_tool_call=False,
                tool_call_instruction="",
                instruction="",  # Empty instruction
                few_shot="Few-shot examples",
                llm_model=self.llm_model,
            )

    def test_assertion_tool_call_required(self):
        with self.assertRaises(AssertionError):
            Prompt(
                prompting_regime="baseline_structure_faithfulness",
                use_tool_call=True,
                tool_call_instruction="",  # Empty when required
                instruction="Instruction",
                few_shot="Few-shot examples",
                llm_model=self.llm_model,
            )

    def test_assertion_gold_structure_required(self):
        prompt = Prompt(
            prompting_regime="baseline_structure_faithfulness",
            use_tool_call=False,
            tool_call_instruction="",
            instruction="Instruction",
            few_shot="Few-shot examples",
            llm_model=self.llm_model,
        )
        with self.assertRaises(AssertionError):
            prompt.make_prompt(
                current_sample="Sample",
                include_gold_structure=True,
                gold_structure="",  # Empty when required
            )

    def test_assertion_empty_current_sample(self):
        prompt = Prompt(
            prompting_regime="baseline_structure_faithfulness",
            use_tool_call=False,
            tool_call_instruction="",
            instruction="Instruction",
            few_shot="Few-shot examples",
            llm_model=self.llm_model,
        )
        with self.assertRaises(AssertionError):
            prompt.make_prompt(
                current_sample="",  # Empty
                include_gold_structure=False,
            )

    def test_ordering_anatomy(self):
        """Verify prompt anatomy: instruction -> tool_call -> regime -> few_shot -> current_sample -> gold_structure"""
        prompt = Prompt(
            prompting_regime="detailed_instruction",
            use_tool_call=True,
            tool_call_instruction="TOOL",
            instruction="INSTR",
            few_shot="FEWSHOT",
            llm_model=self.llm_model,
        )
        result = prompt.make_prompt(
            current_sample="SAMPLE",
            include_gold_structure=True,
            gold_structure="GOLD",
        )
        # Find positions
        instr_pos = result.find("INSTR")
        tool_pos = result.find("TOOL")
        regime_pos = result.find("intervention")
        fewshot_pos = result.find("FEWSHOT")
        sample_pos = result.find("SAMPLE")
        gold_pos = result.find("GOLD")

        self.assertGreater(instr_pos, -1)
        self.assertGreater(tool_pos, -1)
        self.assertGreater(regime_pos, -1)
        self.assertGreater(fewshot_pos, -1)
        self.assertGreater(sample_pos, -1)
        self.assertGreater(gold_pos, -1)

        # Verify ordering
        self.assertLess(instr_pos, tool_pos)
        self.assertLess(tool_pos, regime_pos)
        self.assertLess(regime_pos, fewshot_pos)
        self.assertLess(fewshot_pos, sample_pos)
        self.assertLess(sample_pos, gold_pos)

    def test_ordering_anatomy_maximum_regime(self):
        """Verify prompt anatomy also for maximum mediator faithfulness."""
        prompt = Prompt(
            prompting_regime="maximum_mediator_faithfulness",
            use_tool_call=True,
            tool_call_instruction="TOOL",
            instruction="INSTR",
            few_shot="FEWSHOT",
            llm_model=self.llm_model,
        )
        result = prompt.make_prompt(
            current_sample="SAMPLE",
            include_gold_structure=True,
            gold_structure="GOLD",
        )

        instr_pos = result.find("INSTR")
        tool_pos = result.find("TOOL")
        regime_pos = result.lower().find("intervention possibility")
        fewshot_pos = result.find("FEWSHOT")
        sample_pos = result.find("SAMPLE")
        gold_pos = result.find("GOLD")

        self.assertGreater(instr_pos, -1)
        self.assertGreater(tool_pos, -1)
        self.assertGreater(regime_pos, -1)
        self.assertGreater(fewshot_pos, -1)
        self.assertGreater(sample_pos, -1)
        self.assertGreater(gold_pos, -1)

        self.assertLess(instr_pos, tool_pos)
        self.assertLess(tool_pos, regime_pos)
        self.assertLess(regime_pos, fewshot_pos)
        self.assertLess(fewshot_pos, sample_pos)
        self.assertLess(sample_pos, gold_pos)


if __name__ == "__main__":
    unittest.main()
