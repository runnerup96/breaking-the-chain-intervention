import unittest
import re
from copy import deepcopy
from llm_mocks import FakeTokenizer
from tabfact_mocks import TabFactDatasetMock
from datasets_for_intervention.tabfact_intervention import TabFactIntervention


class TestTabFactIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.tokenizer = FakeTokenizer()
        self.ic = TabFactIntervention(self.dataset, self.tokenizer)

        # Fix prompt for deterministic tests
        self.ic.make_prompt = lambda sample, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"

        self.sample = deepcopy(self.dataset[0])

    def test_make_structure_intervention_shapes(self):
        """Checks that make_structure_intervention creates the correct structure."""
        tree = self.ic.make_structure_intervention(self.sample)

        # Check presence of all keys
        self.assertEqual(set(tree.keys()), {"HSVT", "Local Edits", "Global"})
        self.assertIsInstance(tree["HSVT"], list)
        self.assertEqual(len(tree["HSVT"]), 1)
        self.assertIsInstance(tree["Local Edits"], list)
        self.assertEqual(len(tree["Local Edits"]), 3)  # Generate 3 local interventions
        self.assertIsInstance(tree["Global"], list)
        self.assertEqual(len(tree["Global"]), 1)  # 1 global intervention

    def test_make_structure_intervention_hsvt(self):
        """Checks that HSVT changes the statement but leaves the expression unchanged."""
        tree = self.ic.make_structure_intervention(self.sample)
        hsvt = tree["HSVT"][0]

        # Statement must change
        self.assertNotEqual(hsvt["statement"], self.sample["statement"])
        # Expression must remain the same
        self.assertEqual(hsvt["verifier_query_gt"], self.sample["verifier_query_gt"])

    def test_make_structure_intervention_global(self):
        """Checks that Global Edit changes the expression."""
        tree = self.ic.make_structure_intervention(self.sample)
        global_intervention = tree["Global"][0]  # Alternative program

        self.assertIn(global_intervention["verifier_query_gt"], self.dataset.table_id2alt_programs["table1.html.csv"])

    def test_make_intervention_updates_sample(self):
        """Checks that make_intervention updates the sample during structure_prediction."""
        s = deepcopy(self.sample)
        s["completion_type"] = "structure_prediction"
        # Mock completion that contains a new expression
        mock_completion = "Verifier Query: eq{Jamaica; hop{filter_eq{all_rows; athlete; Usain Bolt}; nation}}=True\nFinal Verdict: True"
        out = self.ic.make_intervention(s, {"completion": mock_completion})
        # Check that the expression has been updated
        self.assertEqual(out["verifier_query_gt"], "eq{Jamaica; hop{filter_eq{all_rows; athlete; Usain Bolt}; nation}}=True")
        # Check that the intervention structure has been created
        self.assertIn("structure_intervention", out)

    def test_collect_intervention_completion_order(self):
        """Checks that collect_intervention_completion correctly maps results."""
        tree = self.ic.make_structure_intervention(self.sample)
        s = deepcopy(self.sample)
        s["structure_intervention"] = tree

        # Create mock generation results
        generated = [
            {"completion": "Final Verdict: True"},   # HSVT
            {"completion": "Final Verdict: False"},  # Local Edit 1
            {"completion": "Final Verdict: True"},   # Local Edit 2
            {"completion": "Final Verdict: False"},  # Local Edit 3
            {"completion": "Final Verdict: False"},  # Global
        ]

        out = self.ic.collect_intervention_completion(s, generated)

        # Check that results are stored in the correct places
        self.assertEqual(out["structure_intervention"]["HSVT"][0]["result_after_intervention"], True)
        self.assertEqual(out["structure_intervention"]["Local Edits"][0]["result_after_intervention"], False)
        self.assertEqual(out["structure_intervention"]["Local Edits"][1]["result_after_intervention"], True)
        self.assertEqual(out["structure_intervention"]["Local Edits"][2]["result_after_intervention"], False)
        self.assertEqual(out["structure_intervention"]["Global"][0]["result_after_intervention"], False)

    def test_interventions_to_prompt_count(self):
        """Checks that interventions_to_prompt creates the correct number of prompts."""
        tree = self.ic.make_structure_intervention(self.sample)
        s = deepcopy(self.sample)
        s["structure_intervention"] = tree
        prompts = self.ic.interventions_to_prompt(s)

        expected_count = 1 + 3 + 1  # HSVT + Local Edits + Global
        self.assertEqual(len(prompts), expected_count)
        self.assertTrue(all(p.startswith("PROMPT(gold=True)") for p in prompts))

    def test_infer_completion_parses_boolean(self):
        """Checks that infer_completion correctly parses True/False."""
        self.assertEqual(self.ic.infer_completion("Final verdict: True"), True)
        self.assertEqual(self.ic.infer_completion("Final verdict: False"), False)

    def test_make_structure_intervention_local_edits_use_parser(self):
        """
        Integration test: checks that Local Edits use the AST parser and intervention functions.
        """
        # Create a sample with an expression that can be modified
        sample_with_simple_expr = {
            "idx": "test_local_edit",
            "table_id": "table1.html.csv",
            "table_html_csv": "player#team#goals\nMessi#PSG#30\nRonaldo#AlNassr#25",
            "statement": "Ronaldo scored more goals than Messi.",
            "verifier_query_gt": "greater{hop{filter_eq{all_rows; player; Ronaldo}; goals}; hop{filter_eq{all_rows; player; Messi}; goals}}=True",
            "label_gt": False,
            "distractors": {
                "columns": ["player", "team", "goals"],
                "values": {
                    "player": ["Messi", "Ronaldo", "Neymar"],
                    "team": ["PSG", "AlNassr", "Barcelona"],
                    "goals": ["30", "25", "20"]
                },
                "entity_swaps": ["Messi", "Ronaldo", "Neymar", "PSG", "AlNassr", "30", "25"]
            }
        }

        tree = self.ic.make_structure_intervention(sample_with_simple_expr)
        local_edits = tree["Local Edits"]

        original_expression = sample_with_simple_expr["verifier_query_gt"]

        for i, local_edit in enumerate(local_edits):
            with self.subTest(i=i):
                new_expression = local_edit["verifier_query_gt"]
                # Check that the expression has changed
                self.assertNotEqual(new_expression, original_expression)

                # Check that the change is meaningful (e.g., player or field has changed)
                # This is a heuristic check, but better than nothing
                if "Ronaldo" in original_expression and "Ronaldo" not in new_expression:
                    # Ronaldo was replaced — good sign
                    self.assertTrue(any(name in new_expression for name in ["Messi", "Neymar"]))
                elif "goals" in original_expression:
                    # Check that goals could have been replaced with another field
                    self.assertTrue(any(field in new_expression for field in ["team", "player"]))

    def test_local_edits_diversity(self):
        """Checks that Local Edits create diverse changes."""
        tree = self.ic.make_structure_intervention(self.sample)
        local_edits = tree["Local Edits"]
        
        # Should have 3 different Local Edits
        self.assertEqual(len(local_edits), 3)
        
        original_expression = self.sample["verifier_query_gt"]
        expressions = [edit["verifier_query_gt"] for edit in local_edits]
        
        # All expressions should differ from the original
        for expr in expressions:
            self.assertNotEqual(expr, original_expression, 
                               "Local Edit should change the expression")
        
        # Expressions should differ from each other
        unique_expressions = set(expressions)
        self.assertGreaterEqual(len(unique_expressions), 2,
                               "At least 2 out of 3 Local Edits should be different")
        
        # Check that changes are meaningful (not just random characters)
        for expr in expressions:
            self._validate_expression_structure(expr)

    def test_local_edits_different_types(self):
        """Checks that Local Edits include different types of changes."""
        # Use an expression with multiple possible intervention points
        complex_sample = deepcopy(self.sample)
        complex_sample["verifier_query_gt"] = "and{filter_eq{all_rows; athlete; Usain Bolt}; greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; 1}}=True"
        
        tree = self.ic.make_structure_intervention(complex_sample)
        local_edits = tree["Local Edits"]
        
        intervention_types = []
        original_expr = complex_sample["verifier_query_gt"]
        
        for edit in local_edits:
            intervention_type = self._classify_intervention(original_expr, edit["verifier_query_gt"])
            intervention_types.append(intervention_type)
        
        # Should have different intervention types
        unique_types = set(intervention_types)
        self.assertGreaterEqual(len(unique_types), 2,
                               "Should have at least 2 different intervention types")

    def test_intervention_on_various_functions(self):
        """Checks intervention behavior on various functions."""
        test_cases = [
            # (original_expression, expected_changes_min)
            ("greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; 1}=True", 1),
            ("and{filter_eq{all_rows; country; Jamaica}; eq{hop{filter_eq{all_rows; country; Jamaica}; athlete}; Usain Bolt}}=True", 2),
            ("sum{filter_eq{all_rows; event; 100m}; time}=True", 1),
            ("avg{filter_eq{all_rows; medal; gold}; time}=True", 1),
        ]
        
        for original_expr, min_expected_changes in test_cases:
            with self.subTest(expr=original_expr):
                sample = deepcopy(self.sample)
                sample["verifier_query_gt"] = original_expr
                
                tree = self.ic.make_structure_intervention(sample)
                local_edits = tree["Local Edits"]
                
                # Check that changes were made
                changed_count = 0
                for edit in local_edits:
                    if edit["verifier_query_gt"] != original_expr:
                        changed_count += 1
                
                self.assertGreaterEqual(changed_count, min_expected_changes,
                                      f"Expected at least {min_expected_changes} changes for {original_expr}")

    def test_no_change_fallback(self):
        """Checks handling of cases where changes are impossible."""
        # Create an expression with no possible intervention points
        sample = deepcopy(self.sample)
        sample["verifier_query_gt"] = "eq{5; 5}=True"
        sample["distractors"] = {"columns": [], "values": {}, "entity_swaps": []}
        
        tree = self.ic.make_structure_intervention(sample)
        local_edits = tree["Local Edits"]
        
        # Should still get 3 Local Edits even if changes are impossible
        self.assertEqual(len(local_edits), 3)
        
        # But expressions should remain unchanged
        for edit in local_edits:
            self.assertEqual(edit["verifier_query_gt"], sample["verifier_query_gt"])

    def _validate_expression_structure(self, expr: str):
        """Checks that the expression has a correct structure."""
        # Should end with =True or =False
        self.assertTrue(expr.endswith("=True") or expr.endswith("=False"),
                       f"Expression should end with =True or =False: {expr}")
        
        # Should have balanced braces
        brace_count = 0
        for char in expr:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            if brace_count < 0:
                self.fail(f"Unbalanced braces in expression: {expr}")
        self.assertEqual(brace_count, 0, f"Unbalanced braces in expression: {expr}")
        
        # Should contain at least one function
        self.assertTrue(re.search(r'[a-zA-Z_]+{', expr), 
                       f"Expression should contain at least one function: {expr}")

    def _classify_intervention(self, original: str, modified: str) -> str:
        """Classifies the type of intervention."""
        if original == modified:
            return "no_change"
        
        # Check changes in filter functions
        if "filter_eq" in original and "filter_eq" in modified:
            original_parts = original.split(';')
            modified_parts = modified.split(';')
            
            if len(original_parts) >= 3 and len(modified_parts) >= 3:
                # Check value change
                if original_parts[2].strip() != modified_parts[2].strip():
                    return "filter_value_change"
                # Check column change
                if original_parts[1].strip() != modified_parts[1].strip():
                    return "filter_column_change"
        
        # Check changes in hop
        if "hop" in original and "hop" in modified:
            original_match = re.search(r'hop\{[^}]+;([^}]+)\}', original)
            modified_match = re.search(r'hop\{[^}]+;([^}]+)\}', modified)
            if original_match and modified_match and original_match.group(1) != modified_match.group(1):
                return "hop_target_change"
        
        # Check changes in comparisons
        for op in ["eq", "greater", "less", "greater_eq", "less_eq"]:
            if op in original and op in modified:
                original_match = re.search(fr'{op}\{{[^}}]+;([^}}]+)\}}', original)
                modified_match = re.search(fr'{op}\{{[^}}]+;([^}}]+)\}}', modified)
                if original_match and modified_match and original_match.group(1) != modified_match.group(1):
                    return f"{op}_constant_change"
        
        return "other_change"

    def test_multiple_intervention_runs_produce_different_results(self):
        """Checks that multiple runs produce different interventions."""
        sample = deepcopy(self.sample)
        
        # Run multiple times and collect results
        all_interventions = []
        for run in range(5):
            tree = self.ic.make_structure_intervention(sample)
            interventions = [edit["verifier_query_gt"] for edit in tree["Local Edits"]]
            all_interventions.extend(interventions)
        
        # Should have different interventions across runs
        unique_interventions = set(all_interventions)
        self.assertGreater(len(unique_interventions), 3,
                          "Multiple runs should produce different interventions")


    def test_diverse_local_edits_across_samples(self):
        """Checks diversity of Local Edits across different sample types."""
        for sample_idx in range(len(self.dataset)):
            with self.subTest(sample_idx=sample_idx):
                sample = deepcopy(self.dataset[sample_idx])
                tree = self.ic.make_structure_intervention(sample)
                local_edits = tree["Local Edits"]
                
                self.assertEqual(len(local_edits), 3, f"Sample {sample_idx}: Should have 3 Local Edits")
                
                original_expr = sample["verifier_query_gt"]
                expressions = [edit["verifier_query_gt"] for edit in local_edits]
                
                # Check that changes were made
                changed_count = sum(1 for expr in expressions if expr != original_expr)
                self.assertGreaterEqual(changed_count, 2, f"Sample {sample_idx}: At least 2 should be changed")
                
                # Check diversity
                unique_exprs = set(expressions)
                self.assertGreaterEqual(len(unique_exprs), 2, f"Sample {sample_idx}: At least 2 unique expressions")

    def test_intervention_types_coverage(self):
        """Checks that interventions cover different function types."""
        sample = deepcopy(self.dataset[0])  # Base sample with filter_eq and hop
        
        tree = self.ic.make_structure_intervention(sample)
        local_edits = tree["Local Edits"]
        
        intervention_types = set()
        original_expr = sample["verifier_query_gt"]
        
        for edit in local_edits:
            modified_expr = edit["verifier_query_gt"]
            if modified_expr != original_expr:
                intervention_type = self._detect_intervention_type(original_expr, modified_expr)
                intervention_types.add(intervention_type)
        
        # Should have different intervention types
        self.assertGreaterEqual(len(intervention_types), 2, f"Found types: {intervention_types}")

    def _detect_intervention_type(self, original: str, modified: str) -> str:
        """Determines the intervention type based on what changed."""
        if "filter_eq" in original:
            # Analyze changes in filter_eq
            orig_filter = re.search(r'filter_eq\{[^}]+;([^}]+);([^}]+)\}', original)
            mod_filter = re.search(r'filter_eq\{[^}]+;([^}]+);([^}]+)\}', modified)
            
            if orig_filter and mod_filter:
                orig_col, orig_val = orig_filter.groups()
                mod_col, mod_val = mod_filter.groups()
                
                if orig_col.strip() != mod_col.strip():
                    return "filter_column_change"
                if orig_val.strip() != mod_val.strip():
                    return "filter_value_change"
        
        if "hop" in original:
            orig_hop = re.search(r'hop\{[^}]+;([^}]+)\}', original)
            mod_hop = re.search(r'hop\{[^}]+;([^}]+)\}', modified)
            
            if orig_hop and mod_hop and orig_hop.group(1) != mod_hop.group(1):
                return "hop_target_change"
        
        # Detect other types of changes
        if re.search(r'eq\{[^}]+;([^}]+)\}', original) and re.search(r'eq\{[^}]+;([^}]+)\}', modified):
            return "eq_constant_change"
        
        return "other_change"

    def test_complex_expression_interventions(self):
        """Tests interventions on complex expressions with multiple functions."""
        complex_sample = deepcopy(self.dataset[2])  # Sample with aggregation functions
        original_expr = complex_sample["verifier_query_gt"]
        
        tree = self.ic.make_structure_intervention(complex_sample)
        local_edits = tree["Local Edits"]
        
        # Check that interventions work with complex expressions
        for i, edit in enumerate(local_edits):
            modified_expr = edit["verifier_query_gt"]
            self._validate_expression(modified_expr, f"Local Edit {i}")
            
            # Check that changes are meaningful
            if modified_expr != original_expr:
                self.assertIn("avg{", modified_expr)
                self.assertIn("filter_eq{", modified_expr)
                self.assertIn("less{", modified_expr)

    def _validate_expression(self, expr: str, context: str = ""):
        """Validates expression correctness."""
        self.assertTrue(expr.endswith("=True") or expr.endswith("=False"), 
                       f"{context}: Should end with =True/=False: {expr}")
        
        # Check balanced braces
        brace_count = 0
        for char in expr:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count < 0:
                    self.fail(f"{context}: Unbalanced braces: {expr}")
        self.assertEqual(brace_count, 0, f"{context}: Unbalanced braces: {expr}")

    def test_intervention_determinism_with_seed(self):
        """Checks that identical seeds produce identical results."""
        sample = deepcopy(self.dataset[0])
        
        # First run
        tree1 = self.ic.make_structure_intervention(sample)
        exprs1 = [edit["verifier_query_gt"] for edit in tree1["Local Edits"]]
        
        # Second run with same data
        tree2 = self.ic.make_structure_intervention(sample)
        exprs2 = [edit["verifier_query_gt"] for edit in tree2["Local Edits"]]
        
        # Results should be the same (deterministic)
        self.assertEqual(exprs1, exprs2, "Results should be deterministic for same input")


if __name__ == '__main__':
    unittest.main()