import unittest
import re
from copy import deepcopy
from llm_mocks import FakeLLMModel
from tabfact_mocks import TabFactDatasetMock
from datasets_for_intervention.tabfact_intervention import TabFactIntervention


class TestTabFactIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.llm_model = FakeLLMModel()
        self.ic = TabFactIntervention(self.dataset, self.llm_model)

        # Fix prompt for deterministic tests
        self.ic.make_prompt = lambda sample, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"

        self.sample = deepcopy(self.dataset[0])

    def test_make_structure_intervention_shapes(self):
        """Ensures structure intervention tree has correct keys and sizes."""
        tree = self.ic.make_structure_intervention(self.sample)
        self.assertEqual(set(tree.keys()), {"HSVT", "Local Edits", "Global"})
        self.assertEqual(len(tree["HSVT"]), 1)
        self.assertEqual(len(tree["Local Edits"]), 3)
        self.assertEqual(len(tree["Global"]), 1)

    def test_make_structure_intervention_hsvt(self):
        """Checks that HSVT modifies the statement but keeps expression unchanged."""
        tree = self.ic.make_structure_intervention(self.sample)
        hsvt = tree["HSVT"][0]
        self.assertNotEqual(hsvt["statement"], self.sample["statement"])
        self.assertEqual(hsvt["verifier_query_gt"], self.sample["verifier_query_gt"])

    def test_make_structure_intervention_global(self):
        """Checks that Global Edit produces an alternative valid expression."""
        tree = self.ic.make_structure_intervention(self.sample)
        global_intervention = tree["Global"][0]
        self.assertIn(global_intervention["verifier_query_gt"],
                      self.dataset.table_id2alt_programs["table1.html.csv"])

    def test_make_intervention_updates_sample(self):
        """Checks that make_intervention updates sample with new expression."""
        s = deepcopy(self.sample)
        s["completion_type"] = "structure_prediction"
        mock_completion = "Verifier Query: eq{Jamaica; hop{filter_eq{all_rows; athlete; Usain Bolt}; nation}}=True\nFinal Verdict: True"
        out = self.ic.make_intervention(s, {"completion": mock_completion})
        self.assertEqual(out["verifier_query_gt"],
                         "eq{Jamaica; hop{filter_eq{all_rows; athlete; Usain Bolt}; nation}}=True")
        self.assertIn("structure_intervention", out)

    def test_collect_intervention_completion_order(self):
        """Checks that intervention completions are mapped in correct order."""
        tree = self.ic.make_structure_intervention(self.sample)
        s = deepcopy(self.sample)
        s["structure_intervention"] = tree

        generated = [
            {"completion": "Final Verdict: True"},
            {"completion": "Final Verdict: False"},
            {"completion": "Final Verdict: True"},
            {"completion": "Final Verdict: False"},
            {"completion": "Final Verdict: False"},
        ]
        out = self.ic.collect_intervention_completion(s, generated)
        self.assertEqual(out["structure_intervention"]["HSVT"][0]["result_after_intervention"], True)
        self.assertEqual(out["structure_intervention"]["Local Edits"][0]["result_after_intervention"], False)
        self.assertEqual(out["structure_intervention"]["Local Edits"][1]["result_after_intervention"], True)
        self.assertEqual(out["structure_intervention"]["Local Edits"][2]["result_after_intervention"], False)
        self.assertEqual(out["structure_intervention"]["Global"][0]["result_after_intervention"], False)

    def test_interventions_to_prompt_count(self):
        """Checks that interventions_to_prompt creates correct number of prompts."""
        tree = self.ic.make_structure_intervention(self.sample)
        s = deepcopy(self.sample)
        s["structure_intervention"] = tree
        prompts = self.ic.interventions_to_prompt(s)
        expected_count = 1 + 3 + 1
        self.assertEqual(len(prompts), expected_count)
        self.assertTrue(all(p.startswith("PROMPT(gold=True)") for p in prompts))

    def test_infer_completion_parses_boolean(self):
        """Checks that infer_completion correctly parses True/False verdicts."""
        self.assertEqual(self.ic.infer_completion("Final verdict: True"), True)
        self.assertEqual(self.ic.infer_completion("Final verdict: False"), False)

    # -------- Local Edits tests --------

    def test_local_edits_change_expression(self):
        """Ensures Local Edits always modify the expression compared to original."""
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            tree = self.ic.make_structure_intervention(sample)
            local_edits = tree["Local Edits"]
            original_expression = sample["verifier_query_gt"]
            for edit in local_edits:
                self.assertNotEqual(edit["verifier_query_gt"], original_expression)

    def test_local_edits_diversity(self):
        """Ensures that Local Edits are diverse and not identical copies."""
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            tree = self.ic.make_structure_intervention(sample)
            local_edits = [edit["verifier_query_gt"] for edit in tree["Local Edits"]]
            self.assertEqual(len(local_edits), 3)
            unique = set(local_edits)
            self.assertGreaterEqual(len(unique), 3)

    def test_local_edits_diversity(self):
        """Checks that Local Edits create diverse changes."""
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            tree = self.ic.make_structure_intervention(sample)
            local_edits = tree["Local Edits"]
            
            # Should have 3 different Local Edits
            self.assertEqual(len(local_edits), 3)
            
            original_expression = sample["verifier_query_gt"]
            expressions = [edit["verifier_query_gt"] for edit in local_edits]
            
            # All expressions should differ from the original
            for expr in expressions:
                self.assertNotEqual(expr, original_expression, 
                                "Local Edit should change the expression")
            
            # Expressions should differ from each other
            unique_expressions = set(expressions)
            self.assertGreaterEqual(len(unique_expressions), 2,
                                "At least 2 out of 3 Local Edits should be different")

    def test_local_edit_inverts_operator_correctly(self):
        """Ensures that operator inversion produces the opposite operator."""
        sample = deepcopy(self.sample)
        sample["verifier_query_gt"] = "greater{5;3}=True"
        tree = self.ic.make_structure_intervention(sample)
        edits = [e["verifier_query_gt"] for e in tree["Local Edits"]]
        inverted = [e for e in edits if "less{" in e]
        self.assertTrue(inverted)

    def test_local_edits_syntax_validity(self):
        """Validates that all Local Edits remain syntactically correct DSL expressions."""
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            tree = self.ic.make_structure_intervention(sample)
            for edit in tree["Local Edits"]:
                expr = edit["verifier_query_gt"]
                self.assertTrue(expr.endswith("=True") or expr.endswith("=False"))
                brace_count = 0
                for c in expr:
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        self.assertGreaterEqual(brace_count, 0)
                self.assertEqual(brace_count, 0)

    def test_no_change_fallback(self):
        """Checks fallback behavior when no intervention points exist."""
        sample = deepcopy(self.sample)
        sample["verifier_query_gt"] = "eq{5; 5}=True"
        sample["distractors"] = {"columns": [], "values": {}, "entity_swaps": []}
        tree = self.ic.make_structure_intervention(sample)
        local_edits = tree["Local Edits"]
        self.assertEqual(len(local_edits), 3)
        for edit in local_edits:
            self.assertTrue(edit["verifier_query_gt"].endswith("=True") or edit["verifier_query_gt"].endswith("=False"))

    def test_multiple_runs_diversity(self):
        """Ensures multiple runs with same sample produce diverse edits."""
        sample = deepcopy(self.sample)
        all_edits = []
        for _ in range(5):
            tree = self.ic.make_structure_intervention(sample)
            all_edits.extend(e["verifier_query_gt"] for e in tree["Local Edits"])
        unique = set(all_edits)
        self.assertGreater(len(unique), 3)


if __name__ == '__main__':
    unittest.main()
