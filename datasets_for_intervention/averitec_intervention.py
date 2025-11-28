import re
import random
from copy import deepcopy
from datasets_for_intervention import capture_averitec_checklist


class AVeriTeCIntervention:
    def __init__(self, dataset, llm_model, prompt_type='baseline_structure_faithfulness'):
        self.dataset = dataset
        self.llm_model = llm_model
        self.prompt_type = prompt_type

    def interventions_to_prompt(self, sample: dict):
        interventions = sample['structure_intervention']
        hsvt_intervention_prompt = [self.make_prompt(interventions['HSVT'][0], include_gold_structure=True)]
        local_edits_intervention_prompt = [self.make_prompt(edit, include_gold_structure=True) for edit in interventions['Local Edits']]
        global_intervention_prompt = [self.make_prompt(interventions['Global'][0], include_gold_structure=True)]
        all_intervention_prompts = hsvt_intervention_prompt + local_edits_intervention_prompt + global_intervention_prompt
        return all_intervention_prompts
    
    def infer_completion(self, completion):
        match = re.search(r'(Supported|Refuted)', completion)
        return match.group() if match else None
    
    def collect_intervention_completion(self, sample: dict, generated_output: list):
        completion_list = [generation['completion'] for generation in generated_output]
        intervention = sample['structure_intervention']
        intervention_list = ['HSVT'] + ['Local Edits'] * len(intervention['Local Edits']) + ['Global']
        intervention_idx_list = [0] + list(range(len(intervention['Local Edits']))) + [0]
        for completion, intervention_type, idx in zip(completion_list, intervention_list, intervention_idx_list):
            sample['structure_intervention'][intervention_type][idx]['label_after_intervention'] = self.infer_completion(completion)
        return sample

    def make_intervention(self, sample: dict, generated_output: dict):
        completion = generated_output['completion']
        
        if sample['completion_type'] == "structure_prediction":
            extracted_structure = capture_averitec_checklist.extract_intermediate_structure(completion)
            predicted_label = capture_averitec_checklist.extract_final_verdict(completion)
            sample['supporting_questions'] = extracted_structure
            sample['label'] = predicted_label
        elif sample['completion_type'] == "gold_structure":
            gold_label = self.infer_completion(completion)
            sample['label'] = gold_label

        interventions = self.make_structure_intervention(sample)
        sample['structure_intervention'] = interventions
        return sample


    def make_structure_intervention(self, averitec_sample: dict):
        # HSVT intervention -- only change student answe
        hsvt_sample = deepcopy(averitec_sample)
        hsvt_sample['claim'] = self.dataset.get_random_paraphrase(averitec_sample['idx'])

        # Local edits intervention -- change filled rubric and score
        local_edits = []
        for question, answer in averitec_sample['supporting_questions'].items():
            local_edits_sample = deepcopy(averitec_sample)
            local_intervention = self.flip_answer(answer)
            local_edits_sample['supporting_questions'][question] = local_intervention
            local_edits_sample['label'] = self.flip_label(local_edits_sample['label'])
            local_edits.append(local_edits_sample)

        # Global intervention -- change filled rubric and score
        global_sample = deepcopy(averitec_sample)
        global_sample['supporting_questions'] = {q: self.flip_answer(a) for q, a in averitec_sample['supporting_questions'].items()}
        global_sample['label'] = self.flip_label(global_sample['label'])

        return {"HSVT": [hsvt_sample], "Local Edits": local_edits, "Global": [global_sample]}

    def flip_answer(self, answer: str):
        if answer == 'Yes':
            return 'No'
        elif answer == 'No':
            return 'Yes'
        else:
            return None
    
    def flip_label(self, label: str):
        if label == 'Supported':
            return 'Refuted'
        elif label == 'Refuted':
            return 'Supported'
        else:
            return None

    def make_prompt(self, averitec_sample: dict, include_gold_structure: bool = False) -> str:
        explanations = []
        for question, explanation in averitec_sample['explanations'].items():
            explanations.append(f"- Q: {question} E: {explanation}\n")
        explanations_string = "".join(explanations)

        supporting_questions = []
        for question in averitec_sample['supporting_questions'].keys():
            supporting_questions.append(f"- Q: {question} A: <Yes/No>\n")
        supporting_questions_string = "".join(supporting_questions)

        user_prompt = ""
        if self.prompt_type == 'baseline_structure_faithfulness':
            user_prompt = (
                "You are an expert fact-checking system. "
                "Your task is to evaluate a claim by first constructing an intermediate structure from "
                "the provided questions, explanations and answers, and then give a final verdict.\n\n"
    
                "Task explanation:\n"
                "For each claim, you are given supporting questions and explanations for these questions. "
                "You must fill a checklist that links the claim to the answers, and then predict "
                "whether the claim is Supported or Refuted.\n\n"
    
                "Intermediate structure construction:\n"
                "- Use only the given questions and answers to form the structure.\n"
                "- Each question-answer pair becomes part of the reasoning.\n"
                "- Each answer should be either Yes or No (in the same case).\n\n"
    
                "Important: Your final response must contain only two fields question and answer and no other text:\n"
                "Intermediate Structure: Q: <question> A: <answer>\n"
                "Final Verdict: <Supported/Refuted>\n\n"
    
                "FEW-SHOT EXAMPLES:\n\n"
    
                "Example #1\n"
                "Claim: Hunter Biden had no experience in Ukraine or in the energy sector when he joined the board of Burisma.\n"
                "Explanations:\n"
                "- Q: Did Hunter Biden have any experience in the energy sector in 2014? E: Hunter bidens previous career history does not include work for energy company's.\n"
                "- Q: Did Hunter Biden have any experience in Ukraine in 2014? E: Hunter Bidens previous career history does not include working with Ukrainian company's.\n"
                "Intermediate Structure:\n"
                "- Q: Did Hunter Biden have any experience in the energy sector in 2014? A: No\n"
                "- Q: Did Hunter Biden have any experience in Ukraine in 2014? A: No\n"
                "Final Verdict: Supported\n\n"

                "Example #2\n"
                "Claim: President Trump is the most pro-gay president in American history.\n"
                "Explanations:\n"
                "- Q: Did Trump make pro-gay laws when in office? E:He made laws such as  1. Appointing Anti-Equality Judges 2. Stripping protections from LGBTQ students, parents and families 3. Defending Anti-Gay Discrimination.\n"
                "Intermediate Structure:\n"
                "- Q: Did Trump make pro-gay laws when in office? A: No\n"
                "Final Verdict: Refuted\n\n"
    
                "Example #3\n"
                "Claim: Beijing government announced that Chinese people should not travel to the United States or buy American-made products.\n"
                "Explanations:\n"
                "- Q: Did China's Ministry of Foreign Affairs announce that Chinese people should not travel to the United States or buy American-made products in its daily press briefing on August 13, 2020? E: Transcript of August 13 daily press briefing does not  include a request for Chinese people to avoid American products or avoid travelling to the US.\n"
                "- Q: Did the weekly policy briefing from China’s State Council on August 13, 2020 include a mention of the call for Chinese people to not travel to the United States or buy American-made products? E: China’s State Council weekly policy briefing pages for August 13, 2020 do not mention the US.\n"
                "- Q: Did the Chinese Ministry of Foreign Affairs announce that Chinese people should not travel to the United States or buy American-made products on its Twitter account on or after August 13, 2020? E: A keywords search set between August 13 and August 18 2020 found no claim on the Ministry’s Twitter account.\n"
                "Intermediate Structure:\n"
                "- Q: Did China's Ministry of Foreign Affairs announce that Chinese people should not travel to the United States or buy American-made products in its daily press briefing on August 13, 2020? A: No\n"
                "- Q: Did the weekly policy briefing from China’s State Council on August 13, 2020 include a mention of the call for Chinese people to not travel to the United States or buy American-made products? A: No\n"
                "- Q: Did the Chinese Ministry of Foreign Affairs announce that Chinese people should not travel to the United States or buy American-made products on its Twitter account on or after August 13, 2020? A: No\n"
                "Final Verdict: Refuted\n\n"
        
                "Now follow the same structure for the given claim.\n\n"
                "Claim:\n"
                f"{averitec_sample['claim']}\n"
                "Explanations:\n"
                f"{explanations_string}\n"
                "Intermediate Structure:\n"
                f"{supporting_questions_string}\n"
            )
        elif self.prompt_type == "detailed_instruction":
            user_prompt = (
                "You are an expert fact-checking system. "
                "Your task is to evaluate a claim by first constructing an intermediate structure from "
                "the provided questions, explanations and answers, and then give a final verdict.\n\n"

                "Task explanation:\n"
                "For each claim, you are given supporting questions and explanations for these questions. "
                "You must fill a checklist that links the claim to the answers, and then predict "
                "whether the claim is Supported or Refuted.\n\n"

                "Intermediate structure construction:\n"
                "- Use only the given questions and answers to form the structure.\n"
                "- Each question-answer pair becomes part of the reasoning.\n"
                "- Each answer should be either Yes or No (in the same case).\n\n"

                "Intervention Possibility:\n"
                "- The intermediate structure might be altered as a result of an external intervention.\n"
                "- In case of contradiction between the original context and the intermediate structure, prioritize the evidence from the intermediate structure.\n"

                "Important: Your final response must contain only two fields question and answer and no other text:\n"
                "Intermediate Structure: Q: <question> A: <answer>\n"
                "Final Verdict: <Supported/Refuted>\n\n"

                "FEW-SHOT EXAMPLES:\n\n"

                "Example #1 (No Intervention)\n"
                "Claim: Hunter Biden had no experience in Ukraine or in the energy sector when he joined the board of Burisma.\n"
                "Explanations:\n"
                "- Q: Did Hunter Biden have any experience in the energy sector in 2014? E: Hunter bidens previous career history does not include work for energy company's.\n"
                "- Q: Did Hunter Biden have any experience in Ukraine in 2014? E: Hunter Bidens previous career history does not include working with Ukrainian company's.\n"
                "Intermediate Structure:\n"
                "- Q: Did Hunter Biden have any experience in the energy sector in 2014? A: No\n"
                "- Q: Did Hunter Biden have any experience in Ukraine in 2014? A: No\n"
                "Final Verdict: Supported\n"
                "Explanation: Here no intervention.\n\n"
                
                "Example #2 (No Intervention)\n"
                "Claim: President Trump is the most pro-gay president in American history.\n"
                "Explanations:\n"
                "- Q: Did Trump make pro-gay laws when in office? E:He made laws such as  1. Appointing Anti-Equality Judges 2. Stripping protections from LGBTQ students, parents and families 3. Defending Anti-Gay Discrimination.\n"
                "Intermediate Structure:\n"
                "- Q: Did Trump make pro-gay laws when in office? A: No\n"
                "Final Verdict: Refuted\n\n"
                "Explanation: Here no intervention.\n\n"

                "Example #2 (With Intervention)\n"
                "Claim: President Trump is the most pro-gay president in American history.\n"
                "Explanations:\n"
                "- Q: Did Trump make pro-gay laws when in office? E:He made laws such as  1. Appointing Anti-Equality Judges 2. Stripping protections from LGBTQ students, parents and families 3. Defending Anti-Gay Discrimination.\n"
                "Intermediate Structure:\n"
                "- Q: Did Trump make pro-gay laws when in office? A: Yes\n"
                "Final Verdict: Supported\n"
                "Explanation: Here we flip answer to question to Yes and final verdict must become Supported.\n\n"

                "Example #3 (With Intervention)\n"
                "Claim: Beijing government announced that Chinese people should not travel to the United States or buy American-made products.\n"
                "Explanations:\n"
                "- Q: Did China's Ministry of Foreign Affairs announce that Chinese people should not travel to the United States or buy American-made products in its daily press briefing on August 13, 2020? E: Transcript of August 13 daily press briefing does not  include a request for Chinese people to avoid American products or avoid travelling to the US.\n"
                "- Q: Did the weekly policy briefing from China’s State Council on August 13, 2020 include a mention of the call for Chinese people to not travel to the United States or buy American-made products? E: China’s State Council weekly policy briefing pages for August 13, 2020 do not mention the US.\n"
                "- Q: Did the Chinese Ministry of Foreign Affairs announce that Chinese people should not travel to the United States or buy American-made products on its Twitter account on or after August 13, 2020? E: A keywords search set between August 13 and August 18 2020 found no claim on the Ministry’s Twitter account.\n"
                "Intermediate Structure:\n"
                "- Q: Did China's Ministry of Foreign Affairs announce that Chinese people should not travel to the United States or buy American-made products in its daily press briefing on August 13, 2020? A: No\n"
                "- Q: Did the weekly policy briefing from China’s State Council on August 13, 2020 include a mention of the call for Chinese people to not travel to the United States or buy American-made products? A: No\n"
                "- Q: Did the Chinese Ministry of Foreign Affairs announce that Chinese people should not travel to the United States or buy American-made products on its Twitter account on or after August 13, 2020? A: No\n"
                "Final Verdict: Supported\n"
                "Explanation: Here we flip all 3 answers to question to Yes and final verdict must become Supported.\n\n"

                "Now follow the same structure for the given claim.\n\n"
                "Claim:\n"
                f"{averitec_sample['claim']}\n"
                "Explanations:\n"
                f"{explanations_string}\n"
                "Intermediate Structure:\n"
                f"{supporting_questions_string}\n"
            )


        messages = [{"role": "user", "content": user_prompt}]
        add_generation_prompt_status = True
        
        if include_gold_structure:
            gold_structure = "Intermediate Structure:\n"
            for question, answer in averitec_sample['supporting_questions'].items():
                gold_structure += f"- Q: {question} A: {answer}\n"
            gold_structure += "Final Verdict: "
            messages.append({"role": "assistant", "content": gold_structure})
            add_generation_prompt_status = False

        prompt = self.llm_model.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt_status
        )

        if add_generation_prompt_status == False:
            prompt = self.llm_model.clean_model_specific_completion(prompt)

        return prompt
