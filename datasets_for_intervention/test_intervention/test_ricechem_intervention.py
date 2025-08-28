import unittest
from datasets_for_intervention import ricechem_intervention
from datasets_for_intervention import capture_ricechem_checklist


class RiceChemDatasetMock():
    def __init__(self):
        # we take the rubrics from the data
        # we look at invalid_items items in GitHub repo to manually filter out incorrect rubrics
        q1_rubric = {
            'correctly cites decreased electron electron repulsion': 1,
            'relates decreased electron electron repulsion to decreased potential energy': 1,
            '3rd and 4th electrons ionized feel same core charge': 1,
            '3rd and 4th electrons ionized from n=3 shell and have same radius': 1,
            '5th electron ionized from n=2 shell and feels higher core charge': 1,
            '5th electron ionized from n=2 shell and has smaller radius': 1,
            'correctly explains relationship of potential energy to ionization energy': 1.5,
            'partially explains relationship between potential energy and ionization energy': 0.5
        }
        q1_score_range = "0-8"

        q2_rubric = {
            'Correctly states that frequency is proportional to energy of light': 1,
            'Explaining sentence 1: energy levels of an electron in an atom are quantized': 1.5,
            'Explaining sentence 1: FULLY explains energy/frequency absorbed must equal the difference in energy levels in an electron': 2,
            'Explaining sentence 1: PARTIALLY explains energy/frequency absorbed must equal the difference in energy levels in an electron': 1,
            'Explaining sentence 2: a minimum amount of energy is needed to eject an electron': 1.5,
            'Explaining sentence 2: any additional energy becomes kinetic energy': 1
        }
        q2_score_range = "0-8"

        q3_rubric = {
            'Sentence 1 is correct. Valence bond theory describes that atomic orbitals must be half-filled to participate in covalent bonding.': 1,
            'Sentence 2: Correct number of hybrid orbitals. In this molecule, carbon must form three hybrid orbitals to form three electron domains.': 1,
            'Sentence 2: Correct type of hybrid orbitals. Carbon must form sp2 hybrid orbitals (from using a 2s and two 2p orbitals)': 1.5,
            'Sentence 3: Correctly states that nitrogen is hybridized': 1,
            'Sentence 3: Correct type of hybridization. Nitrogen is sp2 hybridized to form 3 electron domains': 1.5,
            'Sentence 3: Correct description of hybrid orbital bonds in nitrogen. Two sp2 orbitals form two sigma bonds.': 1.5,
            'Sentence 3: Correct description of unhybridized orbital bonds in nitrogen. Unhybridized p orbital forms pi bond': 1.5
        }
        q3_score_range = "0-9"

        q4_rubric = {
            'Fixed mass of one element': 1,
            'Mass data in LoMP': 1,
            'Combine to form compounds': 1.5,
            'Integer/whole number ratio': 1.5,
            'Whole numbers mean indivisible/discrete': 1.5,
            'Indivisible unit of mass = atom': 1.5
        }
        q4_score_range = "0-8"

        self.graded_rubric_list = [[None, q1_rubric, q1_score_range], [None, q2_rubric, q2_score_range],
                                   [None, q3_rubric, q3_score_range], [None, q4_rubric, q4_score_range]]

        question_1_task = "When studying the emission sources within the Milky Way, a satellite detected interplanetary clouds containing silicon atoms that have lost five electrons.\nb) The ionization energies corresponding to the removal of the third, fourth, and fifth electrons in silicon are 3231, 4356, and 16091 kJ/mol, respectively. \nUsing core charge calculations and your understanding of Coulomb's Law, briefly explain 1) why the removal of each additional electron requires more energy than the removal of the previous one, and 2) the relative magnitude of the values observed.\nThis question can be answered reasonably in around 150 words or fewer."

        question_2_task = "In each statement below (a-c), two observations are given which seem to contrast with each other. Using your knowledge of electron configurations, orbitals, Coulomb’s law, and/or atomic and molecular structures, briefly explain why both of these observations are true, and how the two observations can be reconciled in each case.\n\nb) If light is used to excite an electron to a higher energy level in an atom, only certain frequencies of light can be absorbed. However, if it is used to eject an electron from the atom, any value above a minimum threshold frequency can be absorbed. What’s up with that?! ¯\ (°-°) /¯\n\nThis question can be answered reasonably in around 150 words or fewer."

        question_3_task = "A CHEM 121 student was asked what hybrid orbitals must be present to form methanimine (CH2NH), for which a correct Lewis structure is shown below:\n\nThe student responded:\nAccording to valence bond theory, Carbon cannot form four bonds because it only has two unpaired valence electrons. So, it has to form four sp3 hybrid orbitals to create the four bonds. Nitrogen doesn’t need to hybridize because it already has three unpaired 2p valence electrons to form the three bonds with Carbon and Hydrogen.\nAssess the accuracy and logic of the student’s response: briefly explain whether the reasoning presented is logical, noting what information is correct or incorrect and providing correct logical reasoning and explanation where needed.\nThis question can be reasonably answered in 150 words or fewer."

        question_4_task = "How did the Law of Multiple Proportions lead to the conclusion that matter is made of atoms?\nThis question can be reasonably answered in around 75 words or fewer.\n"

        self.student_answers_list = [[None, question_1_task], [None, question_2_task],
                                     [None, question_3_task], [None, question_4_task]]


class TestRiceChemIntervention(unittest.TestCase):

    def setUp(self):
        dataset = RiceChemDatasetMock()
        self.intervention_logic = ricechem_intervention.RiceChemIntervention(dataset, "<|im_end|>")

        self.generated_output = {"prompt":"blablabla", "completion": "Checklist:\ncorrectly cites decreased electron electron repulsion (weight: 1) (True/False): True\n relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): True\n 3rd and 4th electrons ionized feel same core charge (weight: 1) (True/False): True\n 3rd and 4th electrons ionized from n=3 shell and have same radius (weight: 1) (True/False): True\n 5th electron ionized from n=2 shell and feels higher core charge (weight: 1) (True/False): True\n 5th electron ionized from n=2 shell and has smaller radius (weight: 1) (True/False): True\n correctly explains relationship of potential energy to ionization energy (weight: 1.5) (True/False): True\n partially explains relationship between potential energy and ionization energy (weight: 0.5) (True/False): False\n Final grade (0-8): 7.5"}

    def test_capture_ricechem_checklist(self):
        """
        Test that the find_question_in_prompt method is working correctly.
        """
        # Test finding a checklist item that exists
        prompt = """Checklist:
        correctly cites decreased electron electron repulsion (weight: 1) (True/False): True
        relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): False
        relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): <True/False>
        Final grade (0-8): 7.5"""
        
        entries = capture_ricechem_checklist.extract_checklist_entries(prompt)
        
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["question"], "correctly cites decreased electron electron repulsion")
        self.assertEqual(entries[0]["weight"], '1')
        self.assertEqual(entries[0]["answer"], True)
        
        self.assertEqual(entries[1]["question"], "relates decreased electron electron repulsion to decreased potential energy") 
        self.assertEqual(entries[1]["weight"], '1')
        self.assertEqual(entries[1]["answer"], False)

        # Test with empty prompt
        self.assertEqual(capture_ricechem_checklist.extract_checklist_entries(""), [])

        # Test with malformed checklist
        malformed = "Some text without proper checklist format"
        self.assertEqual(capture_ricechem_checklist.extract_checklist_entries(malformed), [])

    def test_make_intervention(self):
        """
        Test that the intervention logic is working correctly -- we get the same prompt when we revert intervened mediator to the original prompt.
        """
        # Make interventions on the model completion
        intervention_outputs = self.intervention_logic.make_intervention(self.generated_output)

        # Validate that all interventions can be reverted back to original prompt
        validation_result = self.intervention_logic.validate_all_interventions(
            self.generated_output,
            intervention_outputs
        )

        self.assertTrue(validation_result)

    def test_extract_target_from_prompt(self):
        # Test prompt with target 7.5
        prompt_with_score = {"prompt":"blablabla", "completion": "Checklist:\ncorrectly cites decreased electron "
                                                                 "electron repulsion (weight: 1.0) (True/False): "
                                                                 "True\nrelates decreased electron electron repulsion "
                                                                 "to decreased potential energy (weight: 1.0) ("
                                                                 "True/False): False\n3rd and 4th electrons ionized "
                                                                 "feel same core charge (weight: 1.0) (True/False): "
                                                                 "True\n3rd and 4th electrons ionized from n=3 shell "
                                                                 "and have same radius (weight: 1.0) (True/False): "
                                                                 "False\n5th electron ionized from n=2 shell and "
                                                                 "feels higher core charge (weight: 1.0) ("
                                                                 "True/False): True\n5th electron ionized from n=2 "
                                                                 "shell and has smaller radius (weight: 1.0) ("
                                                                 "True/False): True\ncorrectly explains relationship "
                                                                 "of potential energy to ionization energy (weight: "
                                                                 "1.5) (True/False): False\npartially explains "
                                                                 "relationship between potential energy and "
                                                                 "ionization energy (weight: 0.5) (True/False): "
                                                                 "True\nFinal grade (0-8): 4.5<|im_end|><|endoftext|>"}
        self.assertEqual(self.intervention_logic.extract_target_from_prompt(prompt_with_score), 4.5)

        # Test prompt with target 0
        prompt_with_zero = {"prompt":"blablabla", "completion": """Checklist:
        correctly cites decreased electron electron repulsion (weight: 1) (True/False): False
        relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): False
        Final grade (0-8): 0"""}
        self.assertEqual(self.intervention_logic.extract_target_from_prompt(prompt_with_zero), 0)

        # Test prompt with no target
        prompt_no_target = {"prompt":"blablabla", "completion": """Checklist:
        correctly cites decreased electron electron repulsion (weight: 1) (True/False): True
        relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): True"""}
        self.assertIsNone(self.intervention_logic.extract_target_from_prompt(prompt_no_target))

        # Test prompt with invalid target (but still parseable)
        prompt_invalid_target = {"prompt":"blablabla", "completion": """Checklist: correctly cites decreased electron 
        electron repulsion (weight: 1) (True/False): True relates decreased electron electron repulsion to decreased 
        potential energy (weight: 1) (True/False): True\nFinal grade (0-8): 9<|im_end|>"""}
        self.assertEqual(self.intervention_logic.extract_target_from_prompt(prompt_invalid_target), 9.0)


        # New test
        prompt_with_score = {"prompt": "blablabla", "completion": """Checklist:\ncorrectly cites decreased electron
        electron repulsion (weight: 1) (True/False): True\nrelates decreased electron electron repulsion to decreased
        potential energy (weight: 1) (True/False): True\n3rd and 4th electrons ionized feel same core charge (weight:
        1) (True/False): True\n3rd and 4th electrons ionized from n=3 shell and have same radius (weight: 1) (
        True/False): True\n5th electron ionized from n=2 shell and feels higher core charge (weight: 1) (True/False):
        True\n5th electron ionized from n=2 shell and has smaller radius (weight: 1) (True/False): True\ncorrectly
        explains relationship of potential energy to ionization energy (weight: 1.5) (True/False): True\npartially
        explains relationship between potential energy and ionization energy (weight: 0.5) (True/False): False
        \nFinal grade (0-8): 7.0<|im_end|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext
        |><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"""}
        self.assertEqual(self.intervention_logic.extract_target_from_prompt(prompt_with_score), 7.0)


    def test_infer_completion(self):
        # Test completion with score
        completion_with_score = {"prompt":"blablabla", "completion": """: 7.5"""}
        self.assertEqual(self.intervention_logic.infer_completion(completion_with_score), 7.5)

        # Test completion with zero
        completion_with_zero = {"prompt":"blablabla", "completion": """ Final grade : 0"""}
        self.assertEqual(self.intervention_logic.infer_completion(completion_with_zero), 0)

        # Test completion with no valid target
        completion_no_target = {"prompt":"blablabla", "completion": """ \n\n**Final Grade:** """}
        self.assertIsNone(self.intervention_logic.infer_completion(completion_no_target))

        # Test completion with invalid score
        completion_invalid = {"prompt":"blablabla", "completion": """ 3.5\n\nThe final评分 is """}
        self.assertEqual(self.intervention_logic.infer_completion(completion_invalid), 3.5)

        # Test completion
        completion_invalid = {"prompt": "blablabla", "completion": """ 4.0\n\n**Final Grade**: """}
        self.assertEqual(self.intervention_logic.infer_completion(completion_invalid), 4.0)

        # Test completion
        completion_invalid = {"prompt": "blablabla", "completion": """ 3.5\n\nThe final评分 is """}
        self.assertEqual(self.intervention_logic.infer_completion(completion_invalid), 3.5)


