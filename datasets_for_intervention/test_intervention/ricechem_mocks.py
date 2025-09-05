import random

class RiceChemDatasetMock():
    def __init__(self):
        # we take the rubrics from the data
        # we look at invalid_items items in GitHub repo to manually filter out incorrect rubrics
        q1_rubric = {
            'correctly cites decreased electron electron repulsion': 1.0,
            'relates decreased electron electron repulsion to decreased potential energy': 1.0,
            '3rd and 4th electrons ionized feel same core charge': 1.0,
            '3rd and 4th electrons ionized from n=3 shell and have same radius': 1.0,
            '5th electron ionized from n=2 shell and feels higher core charge': 1.0,
            '5th electron ionized from n=2 shell and has smaller radius': 1.0,
            'correctly explains relationship of potential energy to ionization energy': 1.5,
            'partially explains relationship between potential energy and ionization energy': 0.5
        }
        q1_score_range = "0-8"

        q2_rubric = {
            'Correctly states that frequency is proportional to energy of light': 1.0,
            'Explaining sentence 1: energy levels of an electron in an atom are quantized': 1.5,
            'Explaining sentence 1: FULLY explains energy/frequency absorbed must equal the difference in energy levels in an electron': 2.0,
            'Explaining sentence 1: PARTIALLY explains energy/frequency absorbed must equal the difference in energy levels in an electron': 1.0,
            'Explaining sentence 2: a minimum amount of energy is needed to eject an electron': 1.5,
            'Explaining sentence 2: any additional energy becomes kinetic energy': 1.0
        }
        q2_score_range = "0-8"

        q3_rubric = {
            'Sentence 1 is correct. Valence bond theory describes that atomic orbitals must be half-filled to participate in covalent bonding.': 1.0,
            'Sentence 2: Correct number of hybrid orbitals. In this molecule, carbon must form three hybrid orbitals to form three electron domains.': 1.0,
            'Sentence 2: Correct type of hybrid orbitals. Carbon must form sp2 hybrid orbitals (from using a 2s and two 2p orbitals)': 1.5,
            'Sentence 3: Correctly states that nitrogen is hybridized': 1.0,
            'Sentence 3: Correct type of hybridization. Nitrogen is sp2 hybridized to form 3 electron domains': 1.5,
            'Sentence 3: Correct description of hybrid orbital bonds in nitrogen. Two sp2 orbitals form two sigma bonds.': 1.5,
            'Sentence 3: Correct description of unhybridized orbital bonds in nitrogen. Unhybridized p orbital forms pi bond': 1.5
        }
        q3_score_range = "0-8"

        q4_rubric = {
            'Fixed mass of one element': 1.0,
            'Mass data in LoMP': 1.0,
            'Combine to form compounds': 1.5,
            'Integer/whole number ratio': 1.5,
            'Whole numbers mean indivisible/discrete': 1.5,
            'Indivisible unit of mass = atom': 1.5
        }
        q4_score_range = "0-8"

        self.graded_rubric_list = [[None, q1_rubric, q1_score_range], [None, q2_rubric, q2_score_range],
                                   [None, q3_rubric, q3_score_range], [None, q4_rubric, q4_score_range]]

        self.task2rubric_weights = {1: q1_rubric, 2: q2_rubric, 3: q3_rubric, 4: q4_rubric}

        question_1_task = "When studying the emission sources within the Milky Way, a satellite detected interplanetary clouds containing silicon atoms that have lost five electrons.\nb) The ionization energies corresponding to the removal of the third, fourth, and fifth electrons in silicon are 3231, 4356, and 16091 kJ/mol, respectively. \nUsing core charge calculations and your understanding of Coulomb's Law, briefly explain 1) why the removal of each additional electron requires more energy than the removal of the previous one, and 2) the relative magnitude of the values observed.\nThis question can be answered reasonably in around 150 words or fewer."

        question_2_task = "In each statement below (a-c), two observations are given which seem to contrast with each other. Using your knowledge of electron configurations, orbitals, Coulomb's law, and/or atomic and molecular structures, briefly explain why both of these observations are true, and how the two observations can be reconciled in each case.\n\nb) If light is used to excite an electron to a higher energy level in an atom, only certain frequencies of light can be absorbed. However, if it is used to eject an electron from the atom, any value above a minimum threshold frequency can be absorbed. What's up with that?! ¯\ (°-°) /¯\n\nThis question can be answered reasonably in around 150 words or fewer."

        question_3_task = "A CHEM 121 student was asked what hybrid orbitals must be present to form methanimine (CH2NH), for which a correct Lewis structure is shown below:\n\nThe student responded:\nAccording to valence bond theory, Carbon cannot form four bonds because it only has two unpaired valence electrons. So, it has to form four sp3 hybrid orbitals to create the four bonds. Nitrogen doesn't need to hybridize because it already has three unpaired 2p valence electrons to form the three bonds with Carbon and Hydrogen.\nAssess the accuracy and logic of the student's response: briefly explain whether the reasoning presented is logical, noting what information is correct or incorrect and providing correct logical reasoning and explanation where needed.\nThis question can be reasonably answered in 150 words or fewer."

        question_4_task = "How did the Law of Multiple Proportions lead to the conclusion that matter is made of atoms?\nThis question can be reasonably answered in around 75 words or fewer.\n"

        self.student_answers_list = [[None, question_1_task], [None, question_2_task],
                                     [None, question_3_task], [None, question_4_task]]

        # Create mock student answers for testing
        self.task2student_answers = {
            0: ["Sample answer 1 for task 1", "Sample answer 2 for task 1"],
            1: ["Sample answer 1 for task 2", "Sample answer 2 for task 2"],
            2: ["Sample answer 1 for task 3", "Sample answer 2 for task 3"],
            3: ["Sample answer 1 for task 4", "Sample answer 2 for task 4"]
        }

        # Create mock data samples
        self.data = []
        for task_idx in range(1, 5):
            for i, answer in enumerate(self.task2student_answers[task_idx-1]):
                sample = {
                    "idx": f"mock_{i}@Task{task_idx}",
                    "task": self.student_answers_list[task_idx-1][1],
                    "student_answer": answer,
                    "filled_rubric": {key: i % 2 == 0 for key in self.task2rubric_weights[task_idx].keys()},
                    "score": sum(self.task2rubric_weights[task_idx][key] for key, val in 
                               {key: i % 2 == 0 for key in self.task2rubric_weights[task_idx].keys()}.items() if val),
                    "score_range": self.graded_rubric_list[task_idx-1][2],
                    "task_idx": task_idx
                }
                self.data.append(sample)

    def get_random_student_answer(self, task_idx):
        """Get a random student answer for the given task index"""
        return random.choice(self.task2student_answers[task_idx])

    def __len__(self):
        """Return the number of data samples"""
        return len(self.data)
    
    def __getitem__(self, i):
        """Get a data sample by index"""
        return self.data[i]
    


class FakeCapture:
    """Mimics datasets_for_intervention.capture_ricechem_checklist module."""
    def __init__(self):
        # Configurable defaults if you want later
        self.predicted_checklist = None  # None => keep original rubric for convenience
        self.predicted_score = 1.0

    def extract_checklist_entries(self, completion: str):
        # If None, return a minimal plausible variant: flip all to False
        if self.predicted_checklist is None:
            return {}
        return self.predicted_checklist

    def extract_final_grade(self, completion: str):
        # Very simple: parse first number if present, else return 0.0
        import re
        m = re.search(r"\d*\.?\d+", completion)
        return float(m.group()) if m else self.predicted_score