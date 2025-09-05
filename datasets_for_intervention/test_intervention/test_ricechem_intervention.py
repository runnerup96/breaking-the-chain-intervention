import unittest
from copy import deepcopy
from fake_tokenizer_mock import FakeTokenizer
from ricechem_mocks import RiceChemDatasetMock, FakeCapture
from datasets_for_intervention.ricechem_intervention import RiceChemIntervention
import math


# ---------- Tests ----------
class TestRiceChemIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = RiceChemDatasetMock()
        self.tokenizer = FakeTokenizer()
        self.ic = RiceChemIntervention(self.dataset, self.tokenizer)

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
        self.assertEqual(out["structure_intervention"]["HSVT"][0]["result_after_intervention"], 1.0)
        for i in range(M):
            self.assertEqual(out["structure_intervention"]["Local Edits"][i]["result_after_intervention"], i + 2.0)
        self.assertEqual(out["structure_intervention"]["Global"][0]["result_after_intervention"], M + 2.0)

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







# class TestRiceChemIntervention(unittest.TestCase):

#     def setUp(self):
#         dataset = RiceChemDatasetMock()
#         self.intervention_logic = ricechem_intervention.RiceChemIntervention(dataset, "<|im_end|>")

#         self.generated_output = {"prompt":"blablabla", "completion": "Checklist:\ncorrectly cites decreased electron electron repulsion (weight: 1) (True/False): True\n relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): True\n 3rd and 4th electrons ionized feel same core charge (weight: 1) (True/False): True\n 3rd and 4th electrons ionized from n=3 shell and have same radius (weight: 1) (True/False): True\n 5th electron ionized from n=2 shell and feels higher core charge (weight: 1) (True/False): True\n 5th electron ionized from n=2 shell and has smaller radius (weight: 1) (True/False): True\n correctly explains relationship of potential energy to ionization energy (weight: 1.5) (True/False): True\n partially explains relationship between potential energy and ionization energy (weight: 0.5) (True/False): False\n Final grade (0-8): 7.5"}

#     def test_capture_ricechem_checklist(self):
#         """
#         Test that the find_question_in_prompt method is working correctly.
#         """
#         # Test finding a checklist item that exists
#         prompt = """Checklist:
#         correctly cites decreased electron electron repulsion (weight: 1) (True/False): True
#         relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): False
#         relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): <True/False>
#         Final grade (0-8): 7.5"""

#         entries = capture_ricechem_checklist.extract_checklist_entries(prompt)

#         self.assertEqual(len(entries), 2)
#         self.assertEqual(entries[0]["question"], "correctly cites decreased electron electron repulsion")
#         self.assertEqual(entries[0]["weight"], '1')
#         self.assertEqual(entries[0]["answer"], True)

#         self.assertEqual(entries[1]["question"], "relates decreased electron electron repulsion to decreased potential energy")
#         self.assertEqual(entries[1]["weight"], '1')
#         self.assertEqual(entries[1]["answer"], False)

#         # Test with empty prompt
#         self.assertEqual(capture_ricechem_checklist.extract_checklist_entries(""), [])

#         # Test with malformed checklist
#         malformed = "Some text without proper checklist format"
#         self.assertEqual(capture_ricechem_checklist.extract_checklist_entries(malformed), [])

#     def test_make_intervention(self):
#         """
#         Test that the intervention logic is working correctly -- we get the same prompt when we revert intervened mediator to the original prompt.
#         """
#         # Make interventions on the model completion
#         intervention_outputs = self.intervention_logic.make_intervention(self.generated_output)

#         # Validate that all interventions can be reverted back to original prompt
#         validation_result = self.intervention_logic.validate_all_interventions(
#             self.generated_output,
#             intervention_outputs
#         )

#         self.assertTrue(validation_result)

#     def test_extract_target_from_prompt(self):
#         # Test prompt with target 7.5
#         prompt_with_score = {"prompt":"blablabla", "completion": "Checklist:\ncorrectly cites decreased electron "
#                                                                  "electron repulsion (weight: 1.0) (True/False): "
#                                                                  "True\nrelates decreased electron electron repulsion "
#                                                                  "to decreased potential energy (weight: 1.0) ("
#                                                                  "True/False): False\n3rd and 4th electrons ionized "
#                                                                  "feel same core charge (weight: 1.0) (True/False): "
#                                                                  "True\n3rd and 4th electrons ionized from n=3 shell "
#                                                                  "and have same radius (weight: 1.0) (True/False): "
#                                                                  "False\n5th electron ionized from n=2 shell and "
#                                                                  "feels higher core charge (weight: 1.0) ("
#                                                                  "True/False): True\n5th electron ionized from n=2 "
#                                                                  "shell and has smaller radius (weight: 1.0) ("
#                                                                  "True/False): True\ncorrectly explains relationship "
#                                                                  "of potential energy to ionization energy (weight: "
#                                                                  "1.5) (True/False): False\npartially explains "
#                                                                  "relationship between potential energy and "
#                                                                  "ionization energy (weight: 0.5) (True/False): "
#                                                                  "True\nFinal grade (0-8): 4.5<|im_end|><|endoftext|>"}
#         self.assertEqual(self.intervention_logic.extract_target_from_prompt(prompt_with_score), 4.5)

#         # Test prompt with target 0
#         prompt_with_zero = {"prompt":"blablabla", "completion": """Checklist:
#         correctly cites decreased electron electron repulsion (weight: 1) (True/False): False
#         relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): False
#         Final grade (0-8): 0"""}
#         self.assertEqual(self.intervention_logic.extract_target_from_prompt(prompt_with_zero), 0)

#         # Test prompt with no target
#         prompt_no_target = {"prompt":"blablabla", "completion": """Checklist:
#         correctly cites decreased electron electron repulsion (weight: 1) (True/False): True
#         relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): True"""}
#         self.assertIsNone(self.intervention_logic.extract_target_from_prompt(prompt_no_target))

#         # Test prompt with invalid target (but still parseable)
#         prompt_invalid_target = {"prompt":"blablabla", "completion": """Checklist: correctly cites decreased electron
#         electron repulsion (weight: 1) (True/False): True relates decreased electron electron repulsion to decreased
#         potential energy (weight: 1) (True/False): True\nFinal grade (0-8): 9<|im_end|>"""}
#         self.assertEqual(self.intervention_logic.extract_target_from_prompt(prompt_invalid_target), 9.0)


#         # New test
#         prompt_with_score = {"prompt": "blablabla", "completion": """Checklist:\ncorrectly cites decreased electron
#         electron repulsion (weight: 1) (True/False): True\nrelates decreased electron electron repulsion to decreased
#         potential energy (weight: 1) (True/False): True\n3rd and 4th electrons ionized feel same core charge (weight:
#         1) (True/False): True\n3rd and 4th electrons ionized from n=3 shell and have same radius (weight: 1) (
#         True/False): True\n5th electron ionized from n=2 shell and feels higher core charge (weight: 1) (True/False):
#         True\n5th electron ionized from n=2 shell and has smaller radius (weight: 1) (True/False): True\ncorrectly
#         explains relationship of potential energy to ionization energy (weight: 1.5) (True/False): True\npartially
#         explains relationship between potential energy and ionization energy (weight: 0.5) (True/False): False
#         \nFinal grade (0-8): 7.0<|im_end|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext
#         |><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"""}
#         self.assertEqual(self.intervention_logic.extract_target_from_prompt(prompt_with_score), 7.0)


#     def test_infer_completion(self):
#         # Test completion with score
#         completion_with_score = {"prompt":"blablabla", "completion": """: 7.5"""}
#         self.assertEqual(self.intervention_logic.infer_completion(completion_with_score), 7.5)

#         # Test completion with zero
#         completion_with_zero = {"prompt":"blablabla", "completion": """ Final grade : 0"""}
#         self.assertEqual(self.intervention_logic.infer_completion(completion_with_zero), 0)

#         # Test completion with no valid target
#         completion_no_target = {"prompt":"blablabla", "completion": """ \n\n**Final Grade:** """}
#         self.assertIsNone(self.intervention_logic.infer_completion(completion_no_target))

#         # Test completion with invalid score
#         completion_invalid = {"prompt":"blablabla", "completion": """ 3.5\n\nThe final评分 is """}
#         self.assertEqual(self.intervention_logic.infer_completion(completion_invalid), 3.5)

#         # Test completion
#         completion_invalid = {"prompt": "blablabla", "completion": """ 4.0\n\n**Final Grade**: """}
#         self.assertEqual(self.intervention_logic.infer_completion(completion_invalid), 4.0)

#         # Test completion
#         completion_invalid = {"prompt": "blablabla", "completion": """ 3.5\n\nThe final评分 is """}
#         self.assertEqual(self.intervention_logic.infer_completion(completion_invalid), 3.5)


