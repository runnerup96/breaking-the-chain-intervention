import random
from copy import deepcopy
import re

class WildsReviewsIntervention:
    def __init__(self, llm_stop_token):
        self.QUESTIONS_TO_OPTIONS = {
            "What is the emotional tone of the review? (positive/neutral/negative)": ["positive", "neutral", "negative"],
            "Were any product issues mentioned (e.g. damage, defects, failures)? (yes/no)": ["yes", "no"],
            "Did the product meet or fall short of expectations? (met/exceeded/fell short)": ["met", "exceeded", "fell short"],
            "Was there any mention of support interaction, and was it positive or negative? (yes/no)": ["yes", "no"],
            "Does the reviewer express satisfaction or dissatisfaction with value for money? (yes/no)": ["yes", "no"],
            "Did the reviewer explicitly recommend or discourage others from buying it? (recommend/discourage/neutral)": ["recommend", "discourage", "neutral"]
        }

        self.NUMBER_OF_INTERVENTIONS = len(self.QUESTIONS_TO_OPTIONS)

        self.QUESTIONS_LIST = list(self.QUESTIONS_TO_OPTIONS.keys())

        self.INTERVENTION_DICT = dict()
        for question, options in self.QUESTIONS_TO_OPTIONS.items():
            self.INTERVENTION_DICT[question] = dict()
            for option in options:
                self.INTERVENTION_DICT[question][option] = f"{question}: {option}"

        self.llm_stop_token = llm_stop_token

    def make_prompt(self, sample: dict) -> str:
                
        checklist = []
        for question in self.QUESTIONS_LIST:
            checklist_item = f"{question}: <{'/'.join(self.QUESTIONS_TO_OPTIONS[question])}>\n"
            checklist.append(checklist_item)
        checklist_string = "".join(checklist)
        
        return (
        "You are a sentiment analysis assistant. Your task is to evaluate the sentiment "
        "of the following product review by completing a structured checklist of reasoning steps. "
        "Then provide a final classification ONLY BASED on the filled checklist: 0 (negative) or 1 (positive). "
        "After completing the checklist, you have to make the final classification only based on the checklist, "
        "without looking back at the review.\n\n"
        "Review:\n"
        f"\"\"\"{sample['text']}\"\"\"\n\n"
        f"OUTPUT FORMAT ({len(checklist)} lines + final classification):\n"
        f"Checklist:\n"
        f"\"\"\"{checklist_string}\"\"\"\n\n"
        "Final classification (0/1): <0/1>\n\n"
        "EXAMPLE of expected output for an arbitrary review:\n"
        "Checklist:\n"
        "What is the emotional tone of the review? (positive/neutral/negative): positive\n"
        "Were any product issues mentioned (e.g. damage, defects, failures)? (yes/no): no\n"
        "Did the product meet or fall short of expectations? (met/exceeded/fell short): exceeded\n"
        "Was there any mention of support interaction, and was it positive or negative? (yes/no): no\n"
        "Does the reviewer express satisfaction or dissatisfaction with value for money? (yes/no): yes\n"
        "Did the reviewer explicitly recommend or discourage others from buying it? (recommend/discourage/neutral): recommend\n"
        "Final classification (0/1): 1\n"
        )


    def find_question_in_prompt(self, prompt: str, question: str):
        for option, intervented_question in self.INTERVENTION_DICT[question].items():
            # Use regex to ensure exact match
            if re.search(rf"{re.escape(intervented_question)}\b", prompt):
                return option, intervented_question
        return None, None

    def question_to_random_option(self, question, option):
        options = deepcopy(self.QUESTIONS_TO_OPTIONS[question])
        options.remove(option)
        random_option = random.choice(options)
        return self.INTERVENTION_DICT[question][random_option]


    def reformat_prompt(self, prompt, current_question, random_question):
        # Use regex to ensure exact match and handle edge cases
        # Use regex to ensure exact match when replacing the question
        reformat_prompt = re.sub(rf"\b{re.escape(current_question)}\b", random_question, prompt)
        reformat_prompt = re.sub(r"Final classification \(0/1\): \d", "Final classification (0/1):", reformat_prompt)
        reformat_prompt = reformat_prompt.replace(self.llm_stop_token, "")
        return reformat_prompt

    def make_intervention(self, prompt):
        interviened_prompts = []
        for question in self.QUESTIONS_LIST:
            option, intervented_question = self.find_question_in_prompt(prompt, question)
            if option and intervented_question:
                # we need to add option to look after think token
                original_question = intervented_question
                random_question = self.question_to_random_option(question, option)
                new_prompt = self.reformat_prompt(prompt, intervented_question, random_question)
                interviened_prompts.append({"prompt": new_prompt, 
                                            "original_mediator":original_question,
                                            "intervention_mediator":random_question})

        if len(interviened_prompts) == self.NUMBER_OF_INTERVENTIONS:
            return interviened_prompts
        else:
            return None

    def validate_intervention(self, original_prompt, intervened_prompt, original_question, intervention_question):
        # revert the mediated prompt
        revert_to_original_prompt = intervened_prompt.replace(intervention_question, original_question)
        #revert the originl prompt
        original_prompt = re.sub(r"Final classification \(0/1\): \d", "Final classification (0/1):", original_prompt)
        original_prompt = original_prompt.replace('<|im_end|>', "")
        # compare them
        return revert_to_original_prompt == original_prompt

    def validate_all_interventions(self, original_prompt, intervention_data):
        check_list = []
        for intervention in intervention_data:
            check = self.validate_intervention(original_prompt, intervention['prompt'],
                                          intervention['original_mediator'], intervention['intervention_mediator'])
            check_list.append(check)
        return all(check_list)
    
    def extract_target_from_prompt(self, prompt):
        match = re.search(r"Final classification \(0/1\): (\d)", prompt)
        if match and match.group(1) in ['0', '1']:
            return int(match.group(1))
        return None

    def infer_completion(self, completion):
        if "0" in completion:
            return 0
        elif "1" in completion:
            return 1
        else:
            return None