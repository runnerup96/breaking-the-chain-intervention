import unittest
from copy import deepcopy

from llm_mocks import FakeLLMModel
from datasets_for_intervention.entailment_intervention import EntailmentIntervention
from entailment_mocks import EntailmentBankDatasetMock

class TestEntailmentIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = EntailmentBankDatasetMock()
        assert all(example["question_paraphrases"] is not None for example in self.dataset), "DEBUG1"
        # few-shot: use first 2 examples from the same small dataset
        self.few_shot = [self.dataset[i] for i in range(min(2, len(self.dataset)))]
        self.ic = EntailmentIntervention(self.dataset, FakeLLMModel(), few_shot_examples=self.few_shot, hsvt_mode="paraphrase")

        self.sample = deepcopy(self.dataset[0])
        # Default to gold structure unless overridden per-test
        self.sample["completion_type"] = "gold_structure"

        # Deterministic prompt header for assertions
        self.ic.system_prompt = "SYSTEM_PROMPT"

    # Shapes and semantics of structure interventions
    def test_make_structure_intervention_shapes_and_effects(self):
        tree = self.ic.make_structure_intervention(self.sample)

        self.assertEqual(set(tree.keys()), {"HSVT", "Local Edits", "Global"})
        self.assertIsInstance(tree["HSVT"], list)
        self.assertEqual(len(tree["HSVT"]), 1)
        self.assertIsInstance(tree["Local Edits"], list)
        self.assertEqual(len(tree["Local Edits"]), len(self.ic.edit_modes))
        self.assertIsInstance(tree["Global"], list)
        self.assertEqual(len(tree["Global"]), 1)

        # HSVT: question changed (lowercased), proof preserved
        hsvt = tree["HSVT"][0]
        if self.ic.hsvt_mode == "paraphrase":
            self.assertEqual(hsvt["question"] in self.sample["question_paraphrases"], True)
        else:
            self.assertEqual(hsvt["question"], self.sample["question"].lower())
        self.assertEqual(hsvt["proof"], self.sample["proof"])

        # Local edits: proof structurally changed and score flipped
        for local in tree["Local Edits"]:
            self.assertNotEqual(local["proof"], self.sample["proof"])  # edited
            self.assertEqual(local["score"], (not self.sample["score"]))

        # Global: proof edited and score flipped
        glob = tree["Global"][0]
        self.assertNotEqual(glob["proof"], self.sample["proof"])  # edited
        self.assertEqual(glob["score"], (not self.sample["score"]))

        # Independence: mutate one variant; others & original unaffected
        glob["__sentinel__"] = True
        self.assertNotIn("__sentinel__", tree["HSVT"][0])
        self.assertNotIn("__sentinel__", tree["Local Edits"][0])
        self.assertNotIn("__sentinel__", self.sample)

    # Completion type routing and field updates
    def test_make_intervention_paths_update_expected_fields(self):
        for completion_type, completion_text in [
            ("structure_prediction", "## Proof\nsent1 -> int1; ## Final Answer\nYes"),
            ("gold_structure", "## Final Answer\nYes"),
        ]:
            with self.subTest(completion_type=completion_type):
                s = deepcopy(self.sample)
                s["completion_type"] = completion_type
                out = self.ic.make_intervention(s, {"completion": completion_text})

                self.assertIn("structure_intervention", out)

                if completion_type == "structure_prediction":
                    # predicted fields are populated from completion
                    self.assertIn("proof", out)
                    self.assertIn("score", out)
                else:
                    # gold structure keeps proof; score parsed
                    self.assertEqual(out["proof"], s["proof"])
                    self.assertIn("score", out)

    # Generated completions ordering to intervention mapping
    def test_collect_intervention_completion_order_and_mapping(self):
        tree = self.ic.make_structure_intervention(self.sample)
        s = deepcopy(self.sample)
        s["structure_intervention"] = tree

        M = len(tree["Local Edits"])
        values = [1] + [i + 2 for i in range(M)] + [M + 2]  # HSVT, locals..., global
        generated = [{"completion": ("## Final Answer\nYes" if v == 1 else "## Final Answer\nNo")} for v in values]

        out = self.ic.collect_intervention_completion(s, generated)
        self.assertIn("result_after_intervention", out["structure_intervention"]["HSVT"][0])
        self.assertIn("result_after_intervention", out["structure_intervention"]["Local Edits"][0])
        self.assertIn("result_after_intervention", out["structure_intervention"]["Global"][0])

    # Prompt construction counts and include_gold_structure flag
    def test_interventions_to_prompt_counts_and_flag(self):
        tree = self.ic.make_structure_intervention(self.sample)
        s = deepcopy(self.sample)
        s["structure_intervention"] = tree
        prompts = self.ic.interventions_to_prompt(s)

        self.assertEqual(len(prompts), 1 + len(tree["Local Edits"]) + 1)
        # All prompts include gold structure by current implementation
        self.assertTrue(all("SYSTEM_PROMPT" in p for p in prompts))

    # Completion parsing for final Yes/No
    def test_infer_completion_parses_answers(self):
        self.assertEqual(self.ic.infer_completion("## Final Answer\nYes"), 1)
        self.assertEqual(self.ic.infer_completion("## Final Answer\nNo"), 0)
        self.assertEqual(self.ic.infer_completion("## Final Answer\nmaybe"), -1)
        self.assertEqual(self.ic.infer_completion("Final Answer\nYes"), 1)
        self.assertEqual(self.ic.infer_completion("## Final Answer Is the hypothesis correct? Yes"), 1)
        self.assertEqual(self.ic.infer_completion("# Final Answer\nYes"), 1)
        self.assertEqual(self.ic.infer_completion("Final Answer\nNo"), 0)
        self.assertEqual(self.ic.infer_completion("## Final Answer Is the hypothesis correct? No"), 0)
        self.assertEqual(self.ic.infer_completion("# Final Answer\nNo"), 0)
        self.assertEqual(self.ic.infer_completion("Yes and No"), -1)

    def test_extract_entailment_proof(self):
        template_proof = "sent1 & sent2 -> int1; int1 & sent3 -> hypothesis"
        self.assertEqual(self.ic._extract_entailment_proof(f"## Proof\n{template_proof}## Final Answer\nYes"), template_proof)
        self.assertEqual(self.ic._extract_entailment_proof(f"## Proof\n{template_proof}## Final Answer\nNo"), template_proof)
        self.assertEqual(self.ic._extract_entailment_proof(f"## Proof\n{template_proof}## Final Answer\nmaybe"), template_proof)
        self.assertEqual(self.ic._extract_entailment_proof("Final Answer\nYes"), None)
        self.assertEqual(self.ic._extract_entailment_proof("## Final Answer Is the hypothesis correct? Yes"), None)
        self.assertEqual(self.ic._extract_entailment_proof("# Final Answer\nYes"), None)
        # Handle formatting flexibly
        self.assertEqual(self.ic._extract_entailment_proof(f"# Proof\n{template_proof} # Final Answer\nYes"), template_proof)
        self.assertEqual(self.ic._extract_entailment_proof(f"Proof{template_proof}Final Answer\nYes"), template_proof)
        self.assertEqual(self.ic._extract_entailment_proof(f"1) Proof:\n{template_proof} 2) Final Answer:\nYes"), template_proof)
        self.assertEqual(self.ic._extract_entailment_proof(f"1) Proof:\n{template_proof} \n\n 2) Final Answer:\nYes"), template_proof)
        
        

    # Test parsing functions with incorrect/malformed completion formats
    def test_parsing_functions_handle_incorrect_formats(self):
        """Test that parsing functions gracefully handle malformed completion text."""
        
        # Test _extract_entailment_proof with various incorrect formats
        test_cases_proof = [
            # Missing "## Proof" prefix
            "sent1 -> int1; ## Final Answer\nYes",
            # Missing "## Final Answer" section  
            "## Proof\nsent1 -> int1;",
            # Empty completion
            "",
            # Only whitespace
            "   \n\t  ",
            # No structured sections at all
            "This is just random text without proper formatting",
            # Reversed order
            "## Final Answer\nYes\n## Proof\nsent1 -> int1;",
            # Several proofs
            "## Proof\nsent1 -> int1; ## Final Answer\nYes\n## Proof\nsent2 -> int2; ## Final Answer\nNo",
        ]
        
        for completion in test_cases_proof:
            with self.subTest(completion=completion):
                result = self.ic._extract_entailment_proof(completion)
                # Should return None for malformed input
                self.assertIsNone(result, f"Expected None for malformed proof: {completion!r}")
        
        # Test _extract_entailment_answer with various incorrect formats
        test_cases_answer = [
            # Missing exact prefix
            # "## Final\nYes",
            # # Wrong formatting
            # "## final answer\nYes",  # Case sensitive
            # Empty or whitespace only
            "",
            "   ",
            # No final answer section
            "## Proof\nsent1 -> int1;",
        ]
        
        for completion in test_cases_answer:
            with self.subTest(completion=completion):
                result = self.ic.infer_completion(completion)
                # Should return None for malformed input
                self.assertEqual(result, -1, f"Expected -1 for malformed answer: {completion!r}")
        
        # Test make_intervention with malformed completions for structure_prediction
        s = deepcopy(self.sample)
        s["completion_type"] = "structure_prediction"
        
        malformed_completions = [
            {"completion": "## Proof\nsent1 -> int1;"},  # Missing final answer
            {"completion": "## Final Answer\nYes"},  # Missing proof
            {"completion": "Random text"},  # No structure at all
            {"completion": ""},  # Empty completion
        ]
        
        for malformed_output in malformed_completions:
            with self.subTest(completion=malformed_output["completion"]):
                # Should not crash and should set fields to extracted values (None for malformed)
                result = self.ic.make_intervention(s, malformed_output)
                self.assertIn("structure_intervention", result)
                # The proof and score should be set to whatever was extracted (could be None)
                self.assertIn("proof", result)
                self.assertIn("score", result)


if __name__ == "__main__":
    unittest.main()


