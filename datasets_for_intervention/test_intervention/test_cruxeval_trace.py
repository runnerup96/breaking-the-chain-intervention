"""Tests for CRUXEval trace utilities (recording, parsing, perturbation, simulation)."""

import unittest

from datasets_for_intervention.cruxeval_trace import (
    LOCAL_EDIT_LEVELS,
    applicable_levels,
    canonicalize_trace,
    make_trace,
    parse_trace_text,
    perturb_universal,
    simulate_from_trace,
    trace_to_text,
)


def f_add_one_double(x):
    y = x + 1
    return y * 2


def f_concat(s):
    out = s.upper()
    return out + "!"


class TestMakeTrace(unittest.TestCase):
    def test_basic_trace_records_locals(self):
        trace, result = make_trace(f_add_one_double, 3)
        self.assertEqual(result, 8)
        self.assertGreater(len(trace), 0)
        # The last step must have y bound to 4.
        self.assertEqual(trace[-1]["locals"].get("y"), 4)

    def test_deepcopy_isolation(self):
        # Mutating the input after make_trace must not affect the trace.
        lst = [1, 2]
        def f(xs):
            xs.append(3)
            return xs
        trace, _ = make_trace(f, lst)
        lst.append(99)
        # The first step's locals.xs must NOT contain 99.
        self.assertNotIn(99, trace[0]["locals"]["xs"])


class TestSerialization(unittest.TestCase):
    def test_roundtrip(self):
        trace, _ = make_trace(f_add_one_double, 3)
        text = trace_to_text(trace)
        parsed = parse_trace_text(text)
        self.assertEqual(canonicalize_trace(trace), canonicalize_trace(parsed))

    def test_parse_empty(self):
        self.assertIsNone(parse_trace_text(""))
        self.assertIsNone(parse_trace_text("no header here"))


class TestSimulate(unittest.TestCase):
    def test_simulate_matches_real_output(self):
        trace, real = make_trace(f_add_one_double, 3)
        sim = simulate_from_trace(trace, "def f(x):\n    y = x + 1\n    return y * 2\n")
        self.assertEqual(sim, real)

    def test_simulate_follows_perturbation(self):
        trace, _ = make_trace(f_add_one_double, 3)
        # Force y -> 10 in the last step, expect simulated answer = 20.
        trace[-1]["locals"]["y"] = 10
        sim = simulate_from_trace(trace, "def f(x):\n    y = x + 1\n    return y * 2\n")
        self.assertEqual(sim, 20)

    def test_simulate_string_func(self):
        trace, real = make_trace(f_concat, "hi")
        sim = simulate_from_trace(
            trace, "def f(s):\n    out = s.upper()\n    return out + '!'\n"
        )
        self.assertEqual(sim, real)


class TestPerturb(unittest.TestCase):
    def test_clean_is_equivalent(self):
        trace, _ = make_trace(f_add_one_double, 3)
        clean = perturb_universal(trace, level=0)
        self.assertEqual(canonicalize_trace(clean), canonicalize_trace(trace))

    def test_levels_change_trace(self):
        trace, _ = make_trace(f_add_one_double, 3)
        base_canon = canonicalize_trace(trace)
        for lvl in LOCAL_EDIT_LEVELS:
            perturbed = perturb_universal(trace, level=lvl, seed=42 + lvl)
            self.assertNotEqual(
                canonicalize_trace(perturbed),
                base_canon,
                f"Perturbation level {lvl} produced an identical trace.",
            )

    def test_applicable_levels(self):
        trace, _ = make_trace(f_add_one_double, 3)
        levels = applicable_levels(trace)
        for required in (0, 1, 2, 7, 8):
            self.assertIn(required, levels)


if __name__ == "__main__":
    unittest.main()
