import re
import random
from copy import deepcopy
from datasets_for_intervention import capture_ricechem_checklist



class RiceChemIntervention:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.stop_token = tokenizer.eos_token


    def interventions_to_prompt(self, sample:dict):
        interventions = sample['structure_intervention']
        hsvt_intervention_prompt = [self.make_prompt(interventions['HSVT'][0], include_gold_structure=True)]
        local_edits_intervention_prompt = [ self.make_prompt(edit, include_gold_structure=True) for edit in interventions['Local Edits']]
        global_intervention_prompt = [self.make_prompt(interventions['Global'][0], include_gold_structure=True)]
        all_intervention_prompts = hsvt_intervention_prompt + local_edits_intervention_prompt + global_intervention_prompt
        return all_intervention_prompts
    
    def infer_completion(self, completion):
        "extract only the completion after the interventiuon, when we test model ability to make a correct decsion"
        match = re.search(r'\d*\.?\d+', completion)
        return float(match.group()) if match else None
    
    def collect_intervention_completion(self, sample:dict, generated_output:list):
        completion_list = [generation['completion'] for generation in generated_output]
        intervention = sample['structure_intervention']
        intervention_list = ['HSVT'] + ['Local Edits'] * len(intervention['Local Edits']) + ['Global']
        intervention_idx_list = [0] + list(range(len(intervention['Local Edits']))) + [0]
        for completion, intervention_type, idx in zip(completion_list, intervention_list, intervention_idx_list):
            sample['structure_intervention'][intervention_type][idx]['score_after_intervention'] = self.infer_completion(completion)
        return sample

    def make_intervention(self, sample:dict, generated_output:dict):
        # i get the sample, make the intervention
        # here i have gold stucture, predicted strcutre and make intervention on both of them.

        completion = generated_output['completion']
        # here we update the sample with the predicted structure, we have gold result in dataset
        if sample['completion_type'] == "structure_prediction":
            predicted_checklist = capture_ricechem_checklist.extract_checklist_entries(completion)
            predicted_answer = capture_ricechem_checklist.extract_final_grade(completion)
            sample['filled_rubric'] = predicted_checklist
            sample['score'] = predicted_answer
        elif sample['completion_type'] == "gold_structure":
            gold_answer = self.infer_completion(completion)
            sample['score'] = gold_answer

        # TODO: Here i need some kind of validation that the parsing is correct
        # По сути, надо сделать task2rubric weights check

        interventions = self.make_structure_intervention(sample)
        sample['structure_intervention'] = interventions
        # hsvt_intervention_prompt = [self.make_prompt(interventions['HSVT'])]
        # local_edits_intervention_prompt = [ self.make_prompt(edit) for edit in interventions['Local Edits']]
        # global_intervention_prompt = [self.make_prompt(interventions['Global'])]
        # all_interventions = hsvt_intervention_prompt + local_edits_intervention_prompt + global_intervention_prompt
        return sample

        # gold_checklist = sample['filled_rubric']
        # gold_intervened_checklist_list = self.make_structure_intervention(sample, gold_checklist)

        # gold_hsvt_intervention_prompt = [self.make_prompt(gold_intervened_checklist_list['HSVT'])]
        # gold_local_edits_intervention_prompt = [ self.make_prompt(edit) for edit in gold_intervened_checklist_list['Local Edits']]
        # gold_global_intervention_prompt = [self.make_prompt(gold_intervened_checklist_list['Global'])]

        # predicted_checklist = self.extract_checklist_stucture_from_completion(generated_output['completion'])
        # predicted_intervened_checklist_list = self.make_structure_intervention(sample, predicted_checklist)#dict of intervention types

        # hsvt_intervention_prompt = [self.make_prompt(gold_intervened_checklist_list['HSVT'])]
        # local_edits_intervention_prompt = [ self.make_prompt(edit) for edit in predicted_intervened_checklist_list['Local Edits']]
        # global_intervention_prompt = [self.make_prompt(predicted_intervened_checklist_list['Global'])]

        # # i will need to decode back to original for the evaluation
        # all_prediction_interventions = hsvt_intervention_prompt + local_edits_intervention_prompt + global_intervention_prompt
        # all_gold_interventions = hsvt_intervention_prompt + local_edits_intervention_prompt + global_intervention_prompt

        # return all_prediction_interventions, all_gold_interventions


        # # i need to make a prompt for each of the intervented checklists
        # for gold_intervened_checklist, predicted_intervened_checklist in zip(gold_intervened_checklist_list, predicted_intervened_checklist_list):
        #     gold_prompt = self.make_prompt(gold_intervened_checklist)
        #     predicted_prompt = self.make_prompt(predicted_intervened_checklist)

    def make_structure_intervention(self, ricechem_sample:dict):
        # i get a ricechem original sample and make a structure intervention
        # i do 3 types of interventions -- VRHS, local edits and global
        # I get a list 3 types of intervented samples -- 1 + M + 1 size, where M is size of local edits
        # then these samples are used to make a prompt
        # i will also return a list of intervention types for each of the intervented samples
        # this will be tested in tests

        
        # HSVT intervention -- only change student answe
        hsvt_sample = deepcopy(ricechem_sample)
        sample_task_idx = ricechem_sample['task_idx']
        arbitrary_student_answer = self.dataset.get_random_student_answer(sample_task_idx)
        hsvt_sample['student_answer'] = arbitrary_student_answer

        def calculate_new_expected_score(task_idx, checklist):
            # вот тут и происходит ошибка так как мы сгенерированным ключом лезем в истинный
            # Почему то иногда вылазят ковычки странные, из за этого ломается, но таких всего 26 элементов
            return sum(self.dataset.task2rubric_weights[task_idx][item] 
                      for item, value in checklist.items() if value)

        # Local edits intervention -- change filled rubric and score
        local_edits = []
        for checklist_item, checklist_answer in ricechem_sample['filled_rubric'].items():
            local_edits_sample = deepcopy(ricechem_sample)
            local_intervention = not checklist_answer
            local_edits_sample['filled_rubric'][checklist_item] = local_intervention
            local_edits_sample['score'] = calculate_new_expected_score(sample_task_idx, local_edits_sample['filled_rubric'])
            local_edits.append(local_edits_sample)

        # Global intervention -- change filled rubric and score
        global_sample = deepcopy(ricechem_sample)
        global_sample['filled_rubric'] = {item: not value for item, value in ricechem_sample['filled_rubric'].items()}
        global_sample['score'] = calculate_new_expected_score(sample_task_idx, global_sample['filled_rubric'])

        return {"HSVT": [hsvt_sample], "Local Edits": local_edits, "Global": [global_sample]}


        # so basically -- i do need to extract checklist from filled rubric, but then this checklist goes into make structure intervention
        # when i do make prompt -- i just pass the strcture which i modify.

    
    # def extract_checklist_stucture_from_completion(self, completion:str):
    #     # here i extract the checklist structure from completion when the strcture is predicted
    #     # i just return the updated sample with a new checklist
    #     pass

    def make_prompt(self, ricechem_sample:dict, include_gold_structure:bool=False) -> str:

        checklist = []
        item2weight = self.dataset.task2rubric_weights[ricechem_sample['task_idx']]
        for rubric_item in ricechem_sample['filled_rubric']:
            checklist_item = f"{rubric_item} (weight: {item2weight[rubric_item]}) (True/False): <True/False>\n"
            checklist.append(checklist_item)
        checklist_string = "".join(checklist)

        user_prompt = (
        "You are an automated grader for a college-level chemistry class. Your task is to evaluate the answer of "
        "college student to the question by completing a structured checklist of reasoning steps. "
        "Each checklist item has a weight, and the final score is the sum of the weights of the correctly answered items. "
        "After completing the checklist, you have to make the final grading only based on the filled checklist, "
        "according to checklist weights (sum of the weights of the correctly answered items).\n\n"
        "The final answer is in provided in FLOAT (0.0/0.5/1.0/1.5/2.0) range from 0 to 8"
        "Return filled checklist in exactly same format, but replace the final <True/False> with True or False."
        "You have to keep to expected format as presented in EXAMPLE. "
        "Question:\n"
        f"\"\"\"{ricechem_sample['task']}\"\"\"\n\n"
        "Answer:\n"
        f"\"\"\"{ricechem_sample['student_answer']}\"\"\"\n\n"
        f"OUTPUT FORMAT ({len(checklist)} lines + final grade):\n"
        f"Checklist:\n"
        f"\"\"\"{checklist_string}\"\"\"\n\n"
        f"Final grade (0-8): <0-8>\n"
        

        # IMPORTANT: Always make the final grading decision based on the filled checklist, without looking back at the question or answer.
        
        "EXAMPLE of expected output for the arbitrary Question:\n"
        "Checklist:\n"
        "correctly cites decreased electron electron repulsion (weight: 1.0) (True/False): True\n"
        "relates decreased electron electron repulsion to decreased potential energy (weight: 1.0) (True/False): False\n"
        "3rd and 4th electrons ionized feel same core charge (weight: 1.0) (True/False): True\n"
        "3rd and 4th electrons ionized from n=3 shell and have same radius (weight: 1.0) (True/False): True\n"
        "5th electron ionized from n=2 shell and feels higher core charge (weight: 1.0) (True/False): False\n"
        "5th electron ionized from n=2 shell and has smaller radius (weight: 1.0) (True/False): False\n"
        "correctly explains relationship of potential energy to ionization energy (weight: 1.5) (True/False): False\n"
        "partially explains relationship between potential energy and ionization energy (weight: 0.5) (True/False): True\n"
        "Final grade (0-8): 3.5\n"
        )
        messages = [{"role": "user", "content": user_prompt}]
        add_generation_prompt_status = True
        if include_gold_structure:
            checklist_string = "Checklist:\n"
            for rubric_item, answer in ricechem_sample['filled_rubric'].items():
                checklist_item = f"{rubric_item} (weight: {item2weight[rubric_item]}) (True/False): {answer}\n"
                checklist_string += checklist_item
            checklist_string += "Final grade (0-8): "
            messages.append({"role": "assistant", "content": checklist_string})

            add_generation_prompt_status = False


        clean_end_token = lambda text, eos_token: text[:-len(f"{eos_token}\n")] if text.endswith(f"{eos_token}\n") else text
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt_status,
            enable_thinking=False
        )

        # remove the end token if it is present since we need to continue the generation
        if add_generation_prompt_status == False:
            prompt = clean_end_token(prompt, self.stop_token)

        return prompt