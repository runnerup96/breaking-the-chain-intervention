import unittest
from copy import deepcopy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_mocks import FakeLLMModel
from pauq_mocks import PAUQDatasetMock
from datasets_for_intervention.pauq_intervention import PAUQIntervention


class TestPAUQIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = PAUQDatasetMock()
        self.llm_model = FakeLLMModel()
        self.ic = PAUQIntervention(self.dataset, self.llm_model)
        self.sample = deepcopy(self.dataset[0])

    def test_make_structure_intervention(self):
        tree = self.ic.make_structure_intervention(self.sample)

        self.assertEqual(set(tree.keys()), {"HSVT", "Local Edits", "Global"})
        
        self.assertTrue(isinstance(tree["HSVT"], list))
        self.assertTrue(isinstance(tree["Local Edits"], list))
        self.assertTrue(isinstance(tree["Global"], list))
        
        self.assertEqual(len(tree["HSVT"]), 1)
        self.assertEqual(len(tree["Local Edits"]), 2)
        self.assertEqual(len(tree["Global"]), 1)
        
        self.assertNotEqual(tree["HSVT"][0]["question"], self.sample["question"])
        self.assertEqual(tree["HSVT"][0]["question"], self.sample["paraphrase"])

        for local_edit in tree["Local Edits"]:
            self.assertEqual(set(local_edit.keys()), set(self.sample.keys()) | set(["local_intervention"]))
            self.assertTrue(isinstance(local_edit["local_intervention"], dict))
            self.assertEqual(set(local_edit["local_intervention"].keys()), {"type", "before", "after"})
        self.assertEqual(tree["Local Edits"][0]["local_intervention"]["type"], "column")
        self.assertEqual(tree["Local Edits"][1]["local_intervention"]["type"], "table")

        global_intervention = tree["Global"][0]
        self.assertEqual(set(global_intervention.keys()), set(self.sample.keys()) | set(["global_intervention"]))
        self.assertTrue(isinstance(global_intervention["global_intervention"], list))
        self.assertTrue(len(global_intervention["global_intervention"]) > 1)

    def test_remove_special_tokens(self):
        self.assertEqual(self.ic.remove_special_tokens("TEXT<|im_end|><|endoftext|>"), "TEXT")
        self.assertEqual(self.ic.remove_special_tokens("TEXT<|im_end|><|endoftext|><|pad|>"), "TEXT")
        self.assertEqual(self.ic.remove_special_tokens("TEXT</s></s></s></s>"), "TEXT")

    def test_interventions_to_prompt(self):
        self.sample["completion_type"] = "structure_prediction"
        self.ic.make_intervention(self.sample, {"completion": self.sample["generated_output"]})
        prompts = self.ic.interventions_to_prompt(self.sample)

        self.assertTrue(isinstance(prompts, list))
        self.assertEqual(len(prompts), 4)
        for prompt in prompts:
            self.assertTrue("===SCHEMA_LINKS===" in prompt)
            self.assertTrue(prompt.endswith("===SQL===\n"))

    def test_collect_intervention_completion(self):
        generated_output = [{"completion": self.sample["generated_output_gold_structure"]} for _ in range(4)]
        self.sample["completion_type"] = "structure_prediction"
        self.ic.make_intervention(self.sample, {"completion": self.sample["generated_output"]})
        self.ic.collect_intervention_completion(self.sample, generated_output)

        self.assertTrue(self.sample['structure_intervention']['HSVT'][0]['generated_sql'].startswith("SELECT"))
        self.assertTrue(self.sample['structure_intervention']['HSVT'][0]['generated_sql'].endswith(";"))

        self.assertTrue(self.sample['structure_intervention']['Local Edits'][0]['generated_sql'].startswith("SELECT"))
        self.assertTrue(self.sample['structure_intervention']['Local Edits'][1]['generated_sql'].startswith("SELECT"))
        self.assertTrue(self.sample['structure_intervention']['Local Edits'][0]['generated_sql'].endswith(";"))
        self.assertTrue(self.sample['structure_intervention']['Local Edits'][1]['generated_sql'].endswith(";"))

        self.assertTrue(self.sample['structure_intervention']['Global'][0]['generated_sql'].startswith("SELECT"))
        self.assertTrue(self.sample['structure_intervention']['Global'][0]['generated_sql'].endswith(";"))
        
        
        

    
