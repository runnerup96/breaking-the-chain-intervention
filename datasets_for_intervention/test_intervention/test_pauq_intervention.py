import unittest
from copy import deepcopy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_mocks import FakeLLMModel
from datasets_for_intervention.pauq_dataset import PAUQDataset
from datasets_for_intervention.pauq_intervention import PAUQIntervention


class TestPAUQIntervention(unittest.TestCase):
    def setUp(self):
        self.dataset = PAUQDataset("./pauq")
        self.llm_model = FakeLLMModel()
        self.ic = PAUQIntervention(self.dataset, self.llm_model)

        # Don't mock make_prompt for prompt tests

        self.sample = deepcopy(self.dataset[0])

    def test_make_local_intervention(self):
        local_intervention_result = self.ic.make_local_intervention(self.sample)
        self.assertIsInstance(local_intervention_result, dict)
        self.assertEqual(set(local_intervention_result.keys()), {"intervened_schema", "intervention"})
        local_intervention = local_intervention_result["intervention"]
        self.assertIsInstance(local_intervention, dict)
        self.assertIsInstance(local_intervention["type"], str)
        self.assertIsInstance(local_intervention["before"], str)
        self.assertIsInstance(local_intervention["after"], str)
        self.assertTrue(local_intervention["type"] in {"table", "column"})
        self.assertNotEqual(local_intervention["before"], local_intervention["after"])

    def test_make_global_intervention(self):
        global_intervention_result = self.ic.make_global_intervention(self.sample)
        self.assertIsInstance(global_intervention_result, dict)
        self.assertEqual(set(global_intervention_result.keys()), {"intervened_schema", "intervention"})
        global_intervention = global_intervention_result["intervention"]
        self.assertIsInstance(global_intervention, list)
        for intervention in global_intervention:
            self.assertIsInstance(intervention, dict)
            self.assertIsInstance(intervention["type"], str)
            self.assertIsInstance(intervention["before"], str)
            self.assertIsInstance(intervention["after"], str)
            self.assertTrue(intervention["type"] in {"table", "column"})
            self.assertNotEqual(intervention["before"], intervention["after"])

    def test_make_hsvt_intervention(self):
        hsvt_intervention = self.ic.make_hsvt_intervention(self.sample)

