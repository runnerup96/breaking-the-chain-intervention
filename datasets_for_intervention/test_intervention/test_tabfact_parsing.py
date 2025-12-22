import unittest
from datasets_for_intervention.tabfact_intervention_helper import (parse_program, _parse_expr, _split_top_args, to_str, 
                                         serialize, visit, replace_first, intervene_filter_column, intervene_filter_value, intervene_eq_constant,
                                         intervene_hop_target, intervene_global_break, intervene_random_semantic_flip, Call, Lit)

class TestTabFactParsingAndIntervention(unittest.TestCase):

    def test_parse_program_basic(self):
        """Tests basic expression parsing."""
        prog = "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
        node, tail = parse_program(prog)

        # Check that the tail (=True) is extracted correctly
        self.assertEqual(tail, "=True")

        # Check AST structure
        self.assertIsInstance(node, Call)
        self.assertEqual(node.name, "greater")
        self.assertEqual(len(node.args), 2)

        # Check the first argument (hop{filter_eq{...}; gold})
        arg1 = node.args[0]
        self.assertIsInstance(arg1, Call)
        self.assertEqual(arg1.name, "hop")
        self.assertEqual(len(arg1.args), 2)
        self.assertIsInstance(arg1.args[0], Call)
        self.assertEqual(arg1.args[0].name, "filter_eq")
        self.assertIsInstance(arg1.args[1], Lit)
        self.assertEqual(arg1.args[1].text, "gold")

        # Check that serialization works
        serialized = serialize(node, tail)
        self.assertEqual(serialized, prog)

    def test_intervene_filter_value(self):
        """Tests value replacement in filter_eq."""
        prog = "eq{hop{filter_eq{all_rows; athlete; Usain Bolt}; nation}; Jamaica}=True"
        new_prog = intervene_filter_value(prog, "Usain Bolt", "Shawn Crawford")

        # Expect "Usain Bolt" to be replaced with "Shawn Crawford"
        expected = "eq{hop{filter_eq{all_rows; athlete; Shawn Crawford}; nation}; Jamaica}=True"
        self.assertEqual(new_prog, expected)

        # Check that replacement doesn't occur if old value is not found
        new_prog2 = intervene_filter_value(prog, "Nonexistent", "Replacement")
        self.assertEqual(new_prog2, prog)  # Expression should not change

    def test_intervene_hop_target(self):
        """Tests target replacement in hop."""
        prog = "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; 1}=True"
        new_prog = intervene_hop_target(prog, "gold", "silver")

        # Expect "gold" to be replaced with "silver"
        expected = "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; silver}; 1}=True"
        self.assertEqual(new_prog, expected)

    def test_intervene_global_break(self):
        """Tests global value replacement in filter_eq."""
        # Use an expression where values are in filter_eq
        prog = "and{filter_eq{all_rows; nation; Jamaica}; filter_eq{all_rows; nation; United States}}=True"
        value_map = {"Jamaica": "Canada", "United States": "Mexico"}
        new_prog = intervene_global_break(prog, value_map)

        # Check that both values were replaced
        self.assertIn("Canada", new_prog)
        self.assertIn("Mexico", new_prog)
        self.assertNotIn("Jamaica", new_prog)
        self.assertNotIn("United States", new_prog)

    def test_intervene_random_semantic_flip(self):
        """Tests the random intervention function."""
        prog = "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"

        # Create distractor dictionaries
        col_distractors = {
            'filter_eq': ['athlete', 'nation', 'gold'],
            'hop': ['gold', 'silver', 'bronze']
        }
        value_distractors = {
            'athlete': ['Usain Bolt', 'Shawn Crawford', 'Carl Lewis'],
            'nation': ['Jamaica', 'United States', 'Canada']
        }
        entity_swaps = {'entity': ['Usain Bolt', 'Shawn Crawford', 'Jamaica', 'United States']}

        # Apply intervention multiple times
        for seed in range(5):
            new_prog = intervene_random_semantic_flip(
                prog,
                col_distractors,
                value_distractors,
                entity_swaps,
                seed=seed
            )
            # Check that the expression changed (or at least didn't cause an error)
            self.assertIsInstance(new_prog, str)


if __name__ == '__main__':
    unittest.main()