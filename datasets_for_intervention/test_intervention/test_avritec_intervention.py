import unittest
from copy import deepcopy

from datasets_for_intervention.test_intervention.llm_mocks import FakeLLMModel
from datasets_for_intervention.test_intervention.averitec_mocks import AVeriTeCDatasetMock, FakeCapture
from datasets_for_intervention.averitec_intervention import AVeriTeCIntervention


class TestAVeriTeCIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = AVeriTeCDatasetMock()
        self.llm_model = FakeLLMModel()
        self.ic = AVeriTeCIntervention(self.dataset, self.llm_model)

        module_name = self.ic.__class__.__module__
        mod = __import__(module_name, fromlist=["capture_averitec_checklist"])
        setattr(mod, "capture_averitec_checklist", FakeCapture())

        # Don't mock make_prompt for prompt tests

        self.sample = deepcopy(self.dataset[0])

    def test_make_structure_intervention_shapes_and_labels(self):
        tree = self.ic.make_structure_intervention(self.sample)

        self.assertEqual(set(tree.keys()), {"HSVT", "Local Edits", "Global"})
        self.assertIsInstance(tree["HSVT"], list)
        self.assertEqual(len(tree["HSVT"]), 1)
        self.assertIsInstance(tree["Local Edits"], list)
        self.assertEqual(len(tree["Local Edits"]), len(self.sample["supporting_questions"]))
        self.assertIsInstance(tree["Global"], list)
        self.assertEqual(len(tree["Global"]), 1)

        hsvt = tree["HSVT"][0]
        self.assertNotEqual(hsvt["claim"], self.sample["claim"])
        self.assertEqual(hsvt["supporting_questions"], self.sample["supporting_questions"])
        self.assertEqual(hsvt["label"], self.sample["label"])

        for i, local in enumerate(tree["Local Edits"]):
            original_questions = self.sample["supporting_questions"]
            question_keys = list(original_questions.keys())
            expected_flipped_key = question_keys[i]
            
            for q, a in local["supporting_questions"].items():
                if q == expected_flipped_key:
                    self.assertNotEqual(a, original_questions[q])
                else:
                    self.assertEqual(a, original_questions[q])
            
            self.assertNotEqual(local["label"], self.sample["label"])

        glob = tree["Global"][0]
        for q, a in glob["supporting_questions"].items():
            self.assertNotEqual(a, self.sample["supporting_questions"][q])
        self.assertNotEqual(glob["label"], self.sample["label"])

        glob["supporting_questions"]["__sentinel__"] = "Yes"
        self.assertNotIn("__sentinel__", tree["HSVT"][0]["supporting_questions"])
        self.assertNotIn("__sentinel__", tree["Local Edits"][0]["supporting_questions"])
        self.assertNotIn("__sentinel__", self.sample["supporting_questions"])

    def test_make_intervention_paths_update_expected_fields(self):
        for completion_type, completion_text in [
            ("structure_prediction", "model completion Supported"),
            ("gold_structure", "gold completion Refuted"),
        ]:
            with self.subTest(completion_type=completion_type):
                s = deepcopy(self.sample)
                s["completion_type"] = completion_type
                out = self.ic.make_intervention(s, {"completion": completion_text})

                self.assertIn("structure_intervention", out)

                if completion_type == "structure_prediction":
                    self.assertIn("supporting_questions", out)
                    self.assertIn("label", out)
                else:
                    self.assertEqual(out["supporting_questions"], s["supporting_questions"])
                    self.assertIsInstance(out["label"], str)

    def test_collect_intervention_completion_order_and_mapping(self):
        tree = self.ic.make_structure_intervention(self.sample)
        s = deepcopy(self.sample)
        s["structure_intervention"] = tree

        M = len(tree["Local Edits"])
        values = ["Supported"] + ["Refuted"] * M + ["Supported"]
        generated = [{"completion": v} for v in values]

        out = self.ic.collect_intervention_completion(s, generated)
        self.assertEqual(out["structure_intervention"]["HSVT"][0]["label_after_intervention"], "Supported")
        for i in range(M):
            self.assertEqual(out["structure_intervention"]["Local Edits"][i]["label_after_intervention"], "Refuted")
        self.assertEqual(out["structure_intervention"]["Global"][0]["label_after_intervention"], "Supported")

    def test_interventions_to_prompt_counts_and_flag(self):
        # Mock make_prompt for this test
        original_make_prompt = self.ic.make_prompt
        self.ic.make_prompt = lambda sample, include_gold_structure=True: f"PROMPT(gold={include_gold_structure})"
        
        tree = self.ic.make_structure_intervention(self.sample)
        s = deepcopy(self.sample)
        s["structure_intervention"] = tree
        prompts = self.ic.interventions_to_prompt(s)

        self.assertEqual(len(prompts), 1 + len(tree["Local Edits"]) + 1)
        self.assertTrue(all(p.startswith("PROMPT(gold=True)") for p in prompts))
        
        # Restore original method
        self.ic.make_prompt = original_make_prompt

    def test_infer_completion_parses_verdict(self):
        self.assertEqual(self.ic.infer_completion("abc Supported def"), "Supported")
        self.assertEqual(self.ic.infer_completion("Refuted"), "Refuted")
        self.assertEqual(self.ic.infer_completion("Some text Supported more text"), "Supported")
        self.assertIsNone(self.ic.infer_completion("no verdict here"))
        self.assertIsNone(self.ic.infer_completion(""))

    def test_flip_answer(self):
        self.assertEqual(self.ic.flip_answer("Yes"), "No")
        self.assertEqual(self.ic.flip_answer("No"), "Yes")
        self.assertIsNone(self.ic.flip_answer("Maybe"))
        self.assertIsNone(self.ic.flip_answer(""))

    def test_flip_label(self):
        self.assertEqual(self.ic.flip_label("Supported"), "Refuted")
        self.assertEqual(self.ic.flip_label("Refuted"), "Supported")
        self.assertIsNone(self.ic.flip_label("Maybe"))
        self.assertIsNone(self.ic.flip_label(""))

    def test_make_prompt_includes_gold_structure(self):
        prompt = self.ic.make_prompt(self.sample, include_gold_structure=True)
        self.assertIn("Structured Reasoning Block:", prompt)
        self.assertIn("Final Verdict:", prompt)
        self.assertIn(self.sample["claim"], prompt)

    def test_make_prompt_without_gold_structure(self):
        prompt = self.ic.make_prompt(self.sample, include_gold_structure=False)
        self.assertIn(self.sample["claim"], prompt)
        # The prompt contains "Intermediate Structure:" and "Final Verdict:" in the instructions
        self.assertIn("Structured Reasoning Block:", prompt)  # This is in the instructions
        self.assertIn("Final Verdict:", prompt)  # This is also in the instructions

    def test_make_structure_intervention_hsvt_uses_paraphrase(self):
        original_claim = self.sample["claim"]
        tree = self.ic.make_structure_intervention(self.sample)
        hsvt_claim = tree["HSVT"][0]["claim"]
        
        self.assertNotEqual(hsvt_claim, original_claim)
        self.assertIn(hsvt_claim, self.dataset.idx2paraphrases[self.sample["idx"]])

    def test_make_structure_intervention_local_edits_flip_correct_questions(self):
        tree = self.ic.make_structure_intervention(self.sample)
        original_questions = self.sample["supporting_questions"]
        question_keys = list(original_questions.keys())
        
        for i, local in enumerate(tree["Local Edits"]):
            flipped_count = 0
            for q, a in local["supporting_questions"].items():
                if a != original_questions[q]:
                    flipped_count += 1
                    self.assertEqual(q, question_keys[i])
            self.assertEqual(flipped_count, 1)

    def test_make_structure_intervention_global_flips_all_questions(self):
        tree = self.ic.make_structure_intervention(self.sample)
        global_sample = tree["Global"][0]
        original_questions = self.sample["supporting_questions"]
        
        for q, a in global_sample["supporting_questions"].items():
            self.assertNotEqual(a, original_questions[q])

    def test_make_structure_intervention_preserves_original_sample(self):
        original_sample = deepcopy(self.sample)
        self.ic.make_structure_intervention(self.sample)
        self.assertEqual(self.sample, original_sample)
