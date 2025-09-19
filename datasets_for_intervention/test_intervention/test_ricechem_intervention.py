import unittest
from copy import deepcopy
from llm_mocks import FakeLLMModel
from ricechem_mocks import RiceChemDatasetMock, FakeCapture
from datasets_for_intervention.ricechem_intervention import RiceChemIntervention
import math


# ---------- Tests ----------
class TestRiceChemIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.llm_model = FakeLLMModel()
        self.ic = RiceChemIntervention(self.dataset, self.llm_model)

        # Monkeypatch capture_ricechem_checklist in the module where RiceChemIntervention is defined
        module_name = self.ic.__class__.__module__
        mod = __import__(module_name, fromlist=["capture_ricechem_checklist"])
        setattr(mod, "capture_ricechem_checklist", FakeCapture())

        # Deterministic prompt
        self.ic.make_prompt = lambda edit, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"

        self.sample = deepcopy(self.dataset[0])

    # --- mirrors test_make_structure_intervention_shapes_and_scores ---
    def test_make_structure_intervention_shapes_and_scores(self):
        tree = self.ic.make_structure_intervention(self.sample)

        self.assertEqual(set(tree.keys()), {"HSVT", "Local Edits", "Global"})
        self.assertIsInstance(tree["HSVT"], list)
        self.assertEqual(len(tree["HSVT"]), 1)
        self.assertIsInstance(tree["Local Edits"], list)
        self.assertEqual(len(tree["Local Edits"]), len(self.sample["filled_rubric"]))
        self.assertIsInstance(tree["Global"], list)
        self.assertEqual(len(tree["Global"]), 1)

        # HSVT: changed answer, same rubric
        hsvt = tree["HSVT"][0]
        self.assertNotEqual(hsvt["student_answer"], self.sample["student_answer"])
        self.assertEqual(hsvt["filled_rubric"], self.sample["filled_rubric"])

        # Local edits: exactly one key flipped + correct score recompute
        weights = self.ic.dataset.task2rubric_weights[self.sample["task_idx"]]
        for local in tree["Local Edits"]:
            diffs = [k for k in self.sample["filled_rubric"]
                     if self.sample["filled_rubric"][k] != local["filled_rubric"][k]]
            self.assertEqual(len(diffs), 1)
            expected_local = sum(weights[k] for k, v in local["filled_rubric"].items() if v)
            self.assertTrue(math.isclose(local["score"], expected_local, rel_tol=1e-9, abs_tol=1e-9))

        # Global: all flipped + correct score recompute
        glob = tree["Global"][0]
        self.assertTrue(all(glob["filled_rubric"][k] == (not v) for k, v in self.sample["filled_rubric"].items()))
        expected_global = sum(weights[k] for k, v in glob["filled_rubric"].items() if v)
        self.assertTrue(math.isclose(glob["score"], expected_global, rel_tol=1e-9, abs_tol=1e-9))

        # Independence: mutate one variant; others & original unaffected
        glob["filled_rubric"]["__sentinel__"] = True
        self.assertNotIn("__sentinel__", tree["HSVT"][0]["filled_rubric"])
        self.assertNotIn("__sentinel__", tree["Local Edits"][0]["filled_rubric"])
        self.assertNotIn("__sentinel__", self.sample["filled_rubric"])

    # --- mirrors parametrized predicted/gold path test using subTest ---
    def test_make_intervention_paths_update_expected_fields(self):
        for completion_type, completion_text in [
            ("predicted_structure", "model completion 2.0"),
            ("gold_structure", "gold completion 3.5"),
        ]:
            with self.subTest(completion_type=completion_type):
                s = deepcopy(self.sample)
                s["completion_type"] = completion_type
                out = self.ic.make_intervention(s, {"completion": completion_text})

                self.assertIn("structure_intervention", out)

                if completion_type == "predicted_structure":
                    self.assertIn("filled_rubric", out)
                    self.assertIn("score", out)
                else:
                    self.assertEqual(out["filled_rubric"], s["filled_rubric"])
                    self.assertIsInstance(out["score"], float)

    # --- mirrors order/mapping test ---
    def test_collect_intervention_completion_order_and_mapping(self):
        tree = self.ic.make_structure_intervention(self.sample)
        s = deepcopy(self.sample)
        s["structure_intervention"] = tree

        M = len(tree["Local Edits"])
        values = [1.0] + [i + 2.0 for i in range(M)] + [M + 2.0]  # HSVT, locals..., global
        generated = [{"completion": str(v)} for v in values]

        out = self.ic.collect_intervention_completion(s, generated)
        self.assertEqual(out["structure_intervention"]["HSVT"][0]["score_after_intervention"], 1.0)
        for i in range(M):
            self.assertEqual(out["structure_intervention"]["Local Edits"][i]["score_after_intervention"], i + 2.0)
        self.assertEqual(out["structure_intervention"]["Global"][0]["score_after_intervention"], M + 2.0)

    # --- mirrors prompt test ---
    def test_interventions_to_prompt_counts_and_flag(self):
        tree = self.ic.make_structure_intervention(self.sample)
        s = deepcopy(self.sample)
        s["structure_intervention"] = tree
        prompts = self.ic.interventions_to_prompt(s)

        self.assertEqual(len(prompts), 1 + len(tree["Local Edits"]) + 1)
        self.assertTrue(all(p.startswith("PROMPT(gold=True)") for p in prompts))

    # --- mirrors infer_completion test ---
    def test_infer_completion_parses_first_number(self):
        self.assertEqual(self.ic.infer_completion("abc 3.5 def 7"), 3.5)
        self.assertIsNone(self.ic.infer_completion("no numbers"))
        self.assertEqual(self.ic.infer_completion("42"), 42.0)
        val = self.ic.infer_completion("1.25 and 0.1")
        self.assertEqual(val, 1.25)