import json
import os
import random


class AVeriTeCDataset:
    def __init__(self, data_path: str):
        self.data_path = data_path

        self.filtered_samples = json.load(open(os.path.join(self.data_path, "onlyboolean_samples.json"), 'r', encoding='utf-8'))
        self.paraphrases = json.load(open(os.path.join(self.data_path, "onlyboolean_paraphrases.json"), 'r', encoding='utf-8'))

        self.idx2paraphrases = dict()
        for sample in self.paraphrases:
            self.idx2paraphrases[sample['sample_idx']] = sample['response']['response'].split('|')

        self.excluded_few_shot_examples = [
            "Hunter Biden had no experience in Ukraine or in the energy sector when he joined the board of Burisma.",
            "President Trump is the most pro-gay president in American history.",
            "Beijing government announced that Chinese people should not travel to the United States or buy American-made products."
        ]
        self.data = []
        self.process_data()


    def get_random_paraphrase(self, idx):
        return random.choice(self.idx2paraphrases[idx])

    def process_data(self):

        for idx, sample in enumerate(self.filtered_samples):
            question_answer_dict,question_explanation_dict = {}, {}
            for question_data in sample['questions']:
                question = question_data['question']
                if question_data.get('answers') and len(question_data['answers']) > 0:
                    answer = question_data['answers'][0]['answer']# just get the first answer
                    explanation = question_data['answers'][0]['boolean_explanation']
                    question_answer_dict[question] = answer
                    question_explanation_dict[question] = explanation


            # filtering of appropriate samples
            support_label_check = sample['label'] == 'Supported'
            correct_refuted_check = True if sample['label'] == 'Refuted' and len(question_answer_dict) == 1 else False
            few_shot_exclusion = True if sample['claim'] not in self.excluded_few_shot_examples else False
            
            if (support_label_check or correct_refuted_check) and len(question_answer_dict) > 0 and few_shot_exclusion:
                processed_sample = {
                    "idx": idx,
                    "claim": sample['claim'],
                    "label": sample['label'],
                    "supporting_questions": question_answer_dict,
                    "explanations": question_explanation_dict
                }
            
                self.data.append(processed_sample)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

