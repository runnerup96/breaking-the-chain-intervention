"""
test_tabfact_intervention.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for TabFactIntervention under the new architecture.

Architecture recap:
  - M (mediator) = DSL query string
  - gold_query / mediator_query are strings; gold_target = True always
  - make_structure_intervention:
      correct  -> {"Local Edits": [...], "Correction": []}
      incorrect -> {"Local Edits": [], "Correction": [corr]}
      error    -> {"Local Edits": [], "Correction": []}
  - Local Edits come from dataset.get_local_edits(sample)
    each LE gets mediator_query=edit["query"], expected_target_after_intervention=edit["expected_target"]
  - Correction: corr["mediator_query"] = gold_query
  - collect_intervention_completion: order = Local Edits, then Correction
  - infer_completion returns (bool|None, tool_query)
"""

import re
import unittest
from copy import deepcopy

from datasets_for_intervention.test_intervention.tabfact_mocks import TabFactDatasetMock
from datasets_for_intervention.tabfact_intervention import TabFactIntervention


class FakeLLMModel:
    """Minimal LLM model stub sufficient for TabFactIntervention init."""
    def get_chat_template(self):
        return None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(str(m.get("content", "")) for m in messages)

    def tokenize(self, text):
        return text.split()

    def generate(self, prompt, **kwargs):
        return ""


# -----------------------------------------------------------------------
# Minimal fakes for processor and tool
# -----------------------------------------------------------------------

class _FakeProcessor:
    """Minimal TabFactStructureProcessor stand-in."""

    _QRE = re.compile(r"(?im)^verifier\s+query\s*:\s*(.+)$")
    _ERE = re.compile(r"(?im)^execution\s+result\s*:\s*(true|false)\s*$")

    def extract_mediator(self, text, short=False):
        m = self._QRE.search(text or "")
        if not m:
            return None
        q = m.group(1).strip()
        return q if (q.endswith("=True") or q.endswith("=False")) else None

    def extract_final_answer(self, text, short_completion=False):
        m = self._ERE.search(text or "")
        if m:
            return m.group(1).lower() == "true"
        if short_completion:
            t = (text or "").strip().lower()
            if t in ("true", "true.", "true!"):
                return True
            if t in ("false", "false.", "false!"):
                return False
        return None

    def extract_tool_args(self, text, short_completion=False):
        return None

    def compare_structures(self, a, b):
        if a is None or b is None:
            return None
        return 1 if a == b else 0

    def check_generation_format_mistakes(self, text):
        s = (text or "").strip()
        if not s:
            return True
        return not bool(re.match(r"(?i)verifier\s+query\s*:", s))

    def extract_columns_values(self, q):
        return set(), set()


class _FakeTool:
    """Minimal TabFactTool stand-in."""
    name = "check_query"

    def spec_json(self):
        return '{"title": "check_query", "type": "object"}'

    def validate_args(self, args):
        return isinstance(args, dict) and "query" in args

    def calculate_score(self, args, sample):
        return True if self.validate_args(args) else None


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_ic(dataset=None, tool_mode="none"):
    ds = dataset or TabFactDatasetMock()
    lm = FakeLLMModel()
    proc = _FakeProcessor()
    tool = _FakeTool()
    return TabFactIntervention(ds, lm, tool, proc, prompting_regime="standard", tool_mode=tool_mode)


def _correct_sample(dataset, idx=0):
    s = deepcopy(dataset[idx])
    s["generation_status"] = "correct"
    return s


def _incorrect_sample(dataset, idx=0):
    s = deepcopy(dataset[idx])
    s["generation_status"] = "incorrect"
    s["mediator_query"] = "eq{1; 2}=True"  # does not match gold_query
    return s


def _error_sample(dataset, idx=0):
    s = deepcopy(dataset[idx])
    s["generation_status"] = "error"
    return s


# =======================================================================
# make_structure_intervention
# =======================================================================

class TestMakeStructureIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ic = _make_ic(self.dataset)

    # ---- tree shape ----

    def test_correct_tree_keys(self):
        s = _correct_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        self.assertEqual(set(tree.keys()), {"Local Edits", "Correction"})

    def test_correct_has_local_edits_and_empty_correction(self):
        s = _correct_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        self.assertGreater(len(tree["Local Edits"]), 0)
        self.assertEqual(len(tree["Correction"]), 0)

    def test_incorrect_has_correction_and_empty_local_edits(self):
        s = _incorrect_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        self.assertEqual(len(tree["Local Edits"]), 0)
        self.assertEqual(len(tree["Correction"]), 1)

    def test_error_both_empty(self):
        s = _error_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        self.assertEqual(len(tree["Local Edits"]), 0)
        self.assertEqual(len(tree["Correction"]), 0)

    # ---- Local Edits content ----

    def test_local_edits_mediator_query_set_from_edit(self):
        s = _correct_sample(self.dataset)
        pool = self.dataset.get_local_edits(s)
        tree = self.ic.make_structure_intervention(s)
        for i, le in enumerate(tree["Local Edits"]):
            self.assertEqual(le["mediator_query"], pool[i]["query"])

    def test_local_edits_expected_target_set_from_edit(self):
        s = _correct_sample(self.dataset)
        pool = self.dataset.get_local_edits(s)
        tree = self.ic.make_structure_intervention(s)
        for i, le in enumerate(tree["Local Edits"]):
            self.assertEqual(le["expected_target_after_intervention"], pool[i]["expected_target"])

    def test_local_edits_count_matches_pool(self):
        for i in range(len(self.dataset)):
            s = _correct_sample(self.dataset, i)
            pool = self.dataset.get_local_edits(s)
            tree = self.ic.make_structure_intervention(s)
            self.assertEqual(len(tree["Local Edits"]), len(pool))

    def test_local_edits_all_have_false_expected_target(self):
        # All mock edits are verified to produce expected_target=False
        for i in range(len(self.dataset)):
            s = _correct_sample(self.dataset, i)
            tree = self.ic.make_structure_intervention(s)
            for le in tree["Local Edits"]:
                self.assertFalse(le["expected_target_after_intervention"])

    def test_local_edits_mediator_differs_from_gold(self):
        for i in range(len(self.dataset)):
            s = _correct_sample(self.dataset, i)
            tree = self.ic.make_structure_intervention(s)
            for le in tree["Local Edits"]:
                self.assertNotEqual(le["mediator_query"], s["gold_query"])

    # ---- Correction content ----

    def test_correction_mediator_is_gold_query(self):
        s = _incorrect_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        corr = tree["Correction"][0]
        self.assertEqual(corr["mediator_query"], s["gold_query"])

    def test_correction_preserves_other_fields(self):
        s = _incorrect_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        corr = tree["Correction"][0]
        self.assertEqual(corr["idx"], s["idx"])
        self.assertEqual(corr["gold_target"], s["gold_target"])
        self.assertEqual(corr["table_html_csv"], s["table_html_csv"])

    # ---- deepcopy isolation ----

    def test_local_edit_mutation_does_not_affect_original(self):
        s = _correct_sample(self.dataset)
        original_mq = s["mediator_query"]
        tree = self.ic.make_structure_intervention(s)
        # Mutate first LE
        tree["Local Edits"][0]["mediator_query"] = "MUTATED"
        self.assertEqual(s["mediator_query"], original_mq)

    def test_local_edit_mutation_does_not_affect_other_les(self):
        s = _correct_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        if len(tree["Local Edits"]) < 2:
            return  # skip if only 1 edit
        mq1 = tree["Local Edits"][1]["mediator_query"]
        tree["Local Edits"][0]["mediator_query"] = "MUTATED"
        self.assertEqual(tree["Local Edits"][1]["mediator_query"], mq1)

    def test_correction_mutation_does_not_affect_sample(self):
        s = _incorrect_sample(self.dataset)
        original_mq = s["mediator_query"]
        tree = self.ic.make_structure_intervention(s)
        tree["Correction"][0]["mediator_query"] = "MUTATED"
        self.assertEqual(s["mediator_query"], original_mq)

    def test_make_structure_intervention_does_not_mutate_sample(self):
        s = _correct_sample(self.dataset)
        original = deepcopy(s)
        self.ic.make_structure_intervention(s)
        self.assertEqual(s, original)

    # ---- missing generation_status treated as error ----

    def test_missing_status_returns_empty_tree(self):
        s = deepcopy(self.dataset[0])
        # no generation_status key
        tree = self.ic.make_structure_intervention(s)
        self.assertEqual(len(tree["Local Edits"]), 0)
        self.assertEqual(len(tree["Correction"]), 0)


# =======================================================================
# interventions_to_prompt
# =======================================================================

class TestInterventionsToPrompt(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ic = _make_ic(self.dataset)
        self.ic.make_prompt = lambda s, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"

    def test_correct_prompt_count_equals_local_edits(self):
        s = _correct_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), len(tree["Local Edits"]))

    def test_incorrect_prompt_count_is_one(self):
        s = _incorrect_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), 1)

    def test_error_prompt_count_is_zero(self):
        s = _error_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        prompts = self.ic.interventions_to_prompt(s)
        self.assertEqual(len(prompts), 0)

    def test_all_prompts_use_include_gold_structure_true(self):
        s = _correct_sample(self.dataset)
        s["structure_intervention"] = self.ic.make_structure_intervention(s)
        prompts = self.ic.interventions_to_prompt(s)
        self.assertTrue(all("gold=True" in p for p in prompts))

    def test_order_local_edits_before_correction_for_correct(self):
        # Correct status -> only Local Edits, no Correction
        s = _correct_sample(self.dataset)
        s["structure_intervention"] = self.ic.make_structure_intervention(s)
        prompts = self.ic.interventions_to_prompt(s)
        n_le = len(s["structure_intervention"]["Local Edits"])
        self.assertEqual(len(prompts), n_le)


# =======================================================================
# collect_intervention_completion
# =======================================================================

class TestCollectInterventionCompletion(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ic = _make_ic(self.dataset)

    def _generated(self, results):
        """Turn list of True/False into fake generated outputs."""
        return [
            {"completion": f"Execution Result: {'True' if r else 'False'}"}
            for r in results
        ]

    def test_correct_local_edits_mapped_in_order(self):
        s = _correct_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        n = len(tree["Local Edits"])
        results = [i % 2 == 0 for i in range(n)]
        out = self.ic.collect_intervention_completion(s, self._generated(results))
        for i, (r, le) in enumerate(zip(results, out["structure_intervention"]["Local Edits"])):
            self.assertEqual(le["target_after_intervention"], r, f"LE[{i}] mismatch")

    def test_incorrect_correction_mapped(self):
        s = _incorrect_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        out = self.ic.collect_intervention_completion(s, self._generated([True]))
        corr = out["structure_intervention"]["Correction"][0]
        self.assertTrue(corr["target_after_intervention"])

    def test_raw_generation_stored(self):
        s = _correct_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        n = len(tree["Local Edits"])
        gens = self._generated([True] * n)
        out = self.ic.collect_intervention_completion(s, gens)
        for le in out["structure_intervention"]["Local Edits"]:
            self.assertIn("raw_generation", le)

    def test_none_result_on_unparseable_completion(self):
        s = _correct_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        n = len(tree["Local Edits"])
        gens = [{"completion": "garbled output"} for _ in range(n)]
        out = self.ic.collect_intervention_completion(s, gens)
        for le in out["structure_intervention"]["Local Edits"]:
            self.assertIsNone(le["target_after_intervention"])

    def test_short_completion_true(self):
        s = _correct_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        n = len(tree["Local Edits"])
        gens = [{"completion": "True"} for _ in range(n)]
        out = self.ic.collect_intervention_completion(s, gens)
        for le in out["structure_intervention"]["Local Edits"]:
            self.assertTrue(le["target_after_intervention"])

    def test_short_completion_false(self):
        s = _correct_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        n = len(tree["Local Edits"])
        gens = [{"completion": "False"} for _ in range(n)]
        out = self.ic.collect_intervention_completion(s, gens)
        for le in out["structure_intervention"]["Local Edits"]:
            self.assertFalse(le["target_after_intervention"])

    def test_tool_mode_stores_tool_query_after_intervention(self):
        ic = _make_ic(self.dataset, tool_mode="simple")
        s = _correct_sample(self.dataset)
        tree = ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        n = len(tree["Local Edits"])
        # FakeTool doesn't parse tool args, so tool_query will be None
        gens = [{"completion": "Execution Result: True"} for _ in range(n)]
        out = ic.collect_intervention_completion(s, gens)
        for le in out["structure_intervention"]["Local Edits"]:
            self.assertIn("tool_query_after_intervention", le)

    def test_non_tool_mode_does_not_store_tool_query(self):
        s = _correct_sample(self.dataset)
        tree = self.ic.make_structure_intervention(s)
        s["structure_intervention"] = tree
        n = len(tree["Local Edits"])
        gens = self._generated([True] * n)
        out = self.ic.collect_intervention_completion(s, gens)
        for le in out["structure_intervention"]["Local Edits"]:
            self.assertNotIn("tool_query_after_intervention", le)


# =======================================================================
# infer_completion
# =======================================================================

class TestInferCompletion(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ic = _make_ic(self.dataset)

    def _s(self):
        return deepcopy(self.dataset[0])

    def test_parses_execution_result_true(self):
        target, tq = self.ic.infer_completion("Verifier Query: eq{1;1}=True\nExecution Result: True")
        self.assertTrue(target)
        self.assertEqual(tq, {})

    def test_parses_execution_result_false(self):
        target, tq = self.ic.infer_completion("Verifier Query: eq{1;2}=True\nExecution Result: False")
        self.assertFalse(target)
        self.assertEqual(tq, {})

    def test_returns_none_on_garbage(self):
        target, tq = self.ic.infer_completion("garbled output with no verdict")
        self.assertIsNone(target)
        self.assertEqual(tq, {})

    def test_short_completion_true(self):
        target, tq = self.ic.infer_completion("True", short_completion=True)
        self.assertTrue(target)
        self.assertEqual(tq, {})

    def test_short_completion_false(self):
        target, tq = self.ic.infer_completion("False", short_completion=True)
        self.assertFalse(target)
        self.assertEqual(tq, {})

    def test_short_completion_none_on_garbage(self):
        target, tq = self.ic.infer_completion("Maybe", short_completion=True)
        self.assertIsNone(target)

    def test_non_tool_mode_tool_query_is_empty_dict(self):
        _, tq = self.ic.infer_completion("Execution Result: True")
        self.assertEqual(tq, {})

    def test_returns_tuple(self):
        result = self.ic.infer_completion("Execution Result: True")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# =======================================================================
# classify_generation
# =======================================================================

class TestClassifyGeneration(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ic = _make_ic(self.dataset)

    def _gold(self, idx=0):
        return self.dataset[idx]["gold_query"]

    def test_none_mediator_is_error(self):
        self.assertEqual(self.ic.classify_generation("Verifier Query: eq{1;1}=True\nExecution Result: True", None, self._gold()), "error")

    def test_preamble_is_error(self):
        completion = "Let me think...\nVerifier Query: eq{1;1}=True\nExecution Result: True"
        mediator = "eq{1;1}=True"
        self.assertEqual(self.ic.classify_generation(completion, mediator, self._gold()), "error")

    def test_matching_mediator_is_correct(self):
        gold = self._gold()
        completion = f"Verifier Query: {gold}\nExecution Result: True"
        self.assertEqual(self.ic.classify_generation(completion, gold, gold), "correct")

    def test_different_mediator_is_incorrect(self):
        gold = self._gold()
        predicted = "eq{1; 2}=True"
        completion = f"Verifier Query: {predicted}\nExecution Result: True"
        self.assertEqual(self.ic.classify_generation(completion, predicted, gold), "incorrect")


# =======================================================================
# make_intervention (end-to-end)
# =======================================================================

class TestMakeIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = TabFactDatasetMock()
        self.ic = _make_ic(self.dataset)

    def _sample(self, idx=0):
        return deepcopy(self.dataset[idx])

    def _completion(self, query, verdict="True"):
        return f"Verifier Query: {query}\nExecution Result: {verdict}"

    def test_correct_flow_sets_status_and_tree(self):
        s = self._sample()
        gold = s["gold_query"]
        out = self.ic.make_intervention(s, {"completion": self._completion(gold, "True")})
        self.assertEqual(out["generation_status"], "correct")
        self.assertIn("structure_intervention", out)
        self.assertGreater(len(out["structure_intervention"]["Local Edits"]), 0)
        self.assertEqual(len(out["structure_intervention"]["Correction"]), 0)

    def test_incorrect_flow_sets_status_and_tree(self):
        s = self._sample()
        out = self.ic.make_intervention(s, {"completion": self._completion("eq{1; 2}=True", "True")})
        self.assertEqual(out["generation_status"], "incorrect")
        self.assertEqual(len(out["structure_intervention"]["Local Edits"]), 0)
        self.assertEqual(len(out["structure_intervention"]["Correction"]), 1)

    def test_error_flow_sets_empty_tree(self):
        s = self._sample()
        out = self.ic.make_intervention(s, {"completion": "garbled completion with no structure"})
        self.assertEqual(out["generation_status"], "error")
        self.assertEqual(out["structure_intervention"], {"Local Edits": [], "Correction": []})

    def test_mediator_query_stored(self):
        s = self._sample()
        gold = s["gold_query"]
        out = self.ic.make_intervention(s, {"completion": self._completion(gold)})
        self.assertEqual(out["mediator_query"], gold)

    def test_target_before_intervention_stored(self):
        s = self._sample()
        gold = s["gold_query"]
        out = self.ic.make_intervention(s, {"completion": self._completion(gold, "True")})
        self.assertTrue(out["target_before_intervention"])

    def test_error_sets_target_none(self):
        s = self._sample()
        out = self.ic.make_intervention(s, {"completion": "garbage"})
        self.assertIsNone(out["target_before_intervention"])

    def test_raw_generation_stored(self):
        s = self._sample()
        completion_text = f"Verifier Query: {s['gold_query']}\nExecution Result: True"
        out = self.ic.make_intervention(s, {"completion": completion_text})
        self.assertIn("raw_generation", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)