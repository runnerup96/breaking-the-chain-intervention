import pandas as pd
import os

pd.options.mode.chained_assignment = None

class RiceChemDataset:
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the folder with csv files
        """
        self.data_path = data_path
        self.data = None
        
        gr_q1 = pd.read_csv(os.path.join(data_path, "Graded Rubric Q1.csv"))
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
        gr_q1 = gr_q1.dropna(subset=list(q1_rubric.keys()) + ['Score'])
        q1_score_range = "0-8"
        
        gr_q2 = pd.read_csv(os.path.join(data_path, "Graded Rubric Q2.csv"))
        q2_rubric = {
            'Correctly states that frequency is proportional to energy of light': 1.0,
            'Explaining sentence 1: energy levels of an electron in an atom are quantized': 1.5,
            'Explaining sentence 1: FULLY explains energy/frequency absorbed must equal the difference in energy levels in an electron': 2.0,
            'Explaining sentence 1: PARTIALLY explains energy/frequency absorbed must equal the difference in energy levels in an electron': 1.0,
            'Explaining sentence 2: a minimum amount of energy is needed to eject an electron': 1.5,
            'Explaining sentence 2: any additional energy becomes kinetic energy': 1.0
        }
        # we also drop the rows with None values in the rubric or score
        gr_q2 = gr_q2.dropna(subset=list(q2_rubric.keys()) + ['Score'])
        q2_score_range = "0-8"

        gr_q3 = pd.read_csv(os.path.join(data_path, "Graded Rubric Q3.csv"))
        q3_rubric = {
            'Sentence 1 is correct. Valence bond theory describes that atomic orbitals must be half-filled to participate in covalent bonding.': 1.0,
            'Sentence 2: Correct number of hybrid orbitals. In this molecule, carbon must form three hybrid orbitals to form three electron domains.': 1.0,
            'Sentence 2: Correct type of hybrid orbitals. Carbon must form sp2 hybrid orbitals (from using a 2s and two 2p orbitals)': 1.0,
            'Sentence 3: Correctly states that nitrogen is hybridized': 1.0,
            'Sentence 3: Correct type of hybridization. Nitrogen is sp2 hybridized to form 3 electron domains': 1.0,
            'Sentence 3: Correct description of hybrid orbital bonds in nitrogen. Two sp2 orbitals form two sigma bonds.': 1.5,
            'Sentence 3: Correct description of unhybridized orbital bonds in nitrogen. Unhybridized p orbital forms pi bond': 1.5
        }
        gr_q3 = gr_q3.dropna(subset=list(q3_rubric.keys()) + ['Score'])
        q3_score_range = "0-8"

        gr_q4 = pd.read_csv(os.path.join(data_path, "Graded Rubric Q4.csv"))
        q4_rubric = {
            'Fixed mass of one element': 1.0,
            'Mass data in LoMP': 1.0,
            'Combine to form compounds': 1.5,
            'Integer/whole number ratio': 1.5,
            'Whole numbers mean indivisible/discrete': 1.5,
            'Indivisible unit of mass = atom': 1.5
        }
        gr_q4 = gr_q4.dropna(subset=list(q4_rubric.keys()) + ['Score'])
        q4_score_range = "0-8"

        self.graded_rubric_list = [[gr_q1, q1_rubric, q1_score_range], [gr_q2, q2_rubric, q2_score_range], 
                                   [gr_q3, q3_rubric, q3_score_range], [gr_q4, q4_rubric, q4_score_range]]
        
        def filter_empty_answers(df):
            df.columns = ["SID", "Answer", "Rubric"]
            df = df.dropna(subset=["Answer"])
            df['Answer_length'] = df['Answer'].apply(lambda x: len(x))
            df = df[df['Answer_length'] != 0]
            return df

        sa_q1 = pd.read_csv(os.path.join(data_path, "Student Answers Q1.csv"))
        print("Original length of sa_q1: ", len(sa_q1))
        sa_q1 = filter_empty_answers(sa_q1)
        print("Length of sa_q1 after dropping empty answers: ", len(sa_q1))
        question_1_task = "When studying the emission sources within the Milky Way, a satellite detected interplanetary clouds containing silicon atoms that have lost five electrons.\nb) The ionization energies corresponding to the removal of the third, fourth, and fifth electrons in silicon are 3231, 4356, and 16091 kJ/mol, respectively. \nUsing core charge calculations and your understanding of Coulomb's Law, briefly explain 1) why the removal of each additional electron requires more energy than the removal of the previous one, and 2) the relative magnitude of the values observed.\nThis question can be answered reasonably in around 150 words or fewer."

        # we drop the rows with None values in the answer
        sa_q2 = pd.read_csv(os.path.join(data_path, "Student Answers Q2.csv"))
        print("Original length of sa_q2: ", len(sa_q2))
        sa_q2 = filter_empty_answers(sa_q2)
        print("Length of sa_q2 after dropping empty answers: ", len(sa_q2))
        question_2_task = "In each statement below (a-c), two observations are given which seem to contrast with each other. Using your knowledge of electron configurations, orbitals, Coulomb’s law, and/or atomic and molecular structures, briefly explain why both of these observations are true, and how the two observations can be reconciled in each case.\n\nb) If light is used to excite an electron to a higher energy level in an atom, only certain frequencies of light can be absorbed. However, if it is used to eject an electron from the atom, any value above a minimum threshold frequency can be absorbed. What’s up with that?! ¯\ (°-°) /¯\n\nThis question can be answered reasonably in around 150 words or fewer."

        sa_q3 = pd.read_csv(os.path.join(data_path, "Student Answers Q3.csv"))
        print("Original length of sa_q3: ", len(sa_q3))
        sa_q3 = filter_empty_answers(sa_q3)
        print("Length of sa_q3 after dropping empty answers: ", len(sa_q3))
        question_3_task = "A CHEM 121 student was asked what hybrid orbitals must be present to form methanimine (CH2NH), for which a correct Lewis structure is shown below:\n\nThe student responded:\nAccording to valence bond theory, Carbon cannot form four bonds because it only has two unpaired valence electrons. So, it has to form four sp3 hybrid orbitals to create the four bonds. Nitrogen doesn’t need to hybridize because it already has three unpaired 2p valence electrons to form the three bonds with Carbon and Hydrogen.\nAssess the accuracy and logic of the student’s response: briefly explain whether the reasoning presented is logical, noting what information is correct or incorrect and providing correct logical reasoning and explanation where needed.\nThis question can be reasonably answered in 150 words or fewer."

        sa_q4 = pd.read_csv(os.path.join(data_path, "Student Answers Q4.csv"))
        print("Original length of sa_q4: ", len(sa_q4))
        sa_q4 = filter_empty_answers(sa_q4)
        print("Length of sa_q4 after dropping empty answers: ", len(sa_q4))

        question_4_task = "How did the Law of Multiple Proportions lead to the conclusion that matter is made of atoms?\nThis question can be reasonably answered in around 75 words or fewer.\n"

        self.student_answers_list = [[sa_q1, question_1_task], [sa_q2, question_2_task], 
                                     [sa_q3, question_3_task], [sa_q4, question_4_task]]
        self.process_data()


    def process_data(self):

        def preprocess_answer(answer_str: str):
            answer_str = answer_str.replace('Â', '').replace('\n', " ").replace('â€™', "")
            return answer_str

        rubric_data = {i: dict() for i in range(1, len(self.student_answers_list) + 1)}
        for task_idx, (answer_df, task_str) in enumerate(self.student_answers_list):
            task_idx = task_idx + 1
            for response_id, answer in zip(answer_df.iloc[1:, 0], answer_df.iloc[1:, 1]):
                rubric_data[task_idx][response_id] = {"student_answer": preprocess_answer(answer),
                                                        "task": task_str}
        
        for task_idx, task in enumerate(self.graded_rubric_list):
            task_idx = task_idx + 1
            task_df, task_rubric, task_score_range = task
            response_ids = task_df.iloc[1:, 0]
            for response_id in response_ids:
                rubric_answer_dict = dict()
                for rubric_item, rubric_weight in task_rubric.items():
                    rubric_answer = bool(task_df[task_df["SID"] == response_id][rubric_item].item())
                    rubric_answer_dict[rubric_item] = {"answer": rubric_answer,
                                                                "weight": rubric_weight}

                if response_id in rubric_data[task_idx]:
                    rubric_data[task_idx][response_id]["filled_rubric"] = rubric_answer_dict
                    # rubric_data[task_idx][response_id]["score"] = float(task_df[task_df["SID"] == response_id]["Score"].item())
                    # Calculate score by summing up weights for each rubric item that was answered correctly
                    # currently this version, since i do not know original weights
                    score = sum(item["weight"] for item in rubric_answer_dict.values() if item["answer"])
                    rubric_data[task_idx][response_id]["score"] = float(score)
                    rubric_data[task_idx][response_id]["score_range"] = task_score_range

        self.data = []
        for task_idx in rubric_data:
            for response_id, data in rubric_data[task_idx].items():
                if set(data.keys()) == set(["task", "student_answer", "filled_rubric", "score", "score_range"]):
                    sample = {
                        "idx": f"{response_id}@Task{task_idx}",
                        "task": data["task"],
                        "student_answer": data["student_answer"],
                        "filled_rubric": data["filled_rubric"],
                        "score": data["score"],
                        "score_range": data["score_range"],
                        "task_idx": task_idx
                    }
                    self.data.append(sample)

        print('Total samples =', len(self.data))
   
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    

if __name__ == "__main__":
    import json
    data_path = "/Users/somov-od/Documents/phd/projects/frontdoor_llm_causality/statics/result_splits/RiceChem"
    dataset = RiceChemDataset(data_path=data_path)
    for i in range(3):
        print("Student answer: ", dataset[i])
        print('--------------------------------')

    json.dump(dataset.data, open("sample.json", 'w'), ensure_ascii=False, indent=4)