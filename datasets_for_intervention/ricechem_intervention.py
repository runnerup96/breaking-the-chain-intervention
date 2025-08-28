import re
import random
from copy import deepcopy
from datasets_for_intervention import capture_ricechem_checklist



class RiceChemIntervention:
    def __init__(self, dataset, llm_model_stop_token: str, use_ground_truth:bool=False):
        self.stop_token = llm_model_stop_token
        self.use_ground_truth = use_ground_truth


    def make_prompt(self, ricechem_sample:dict) -> str:

        checklist = []
        for rubric_item, answer in ricechem_sample['filled_rubric'].items():
            if self.use_ground_truth:
                checklist_item = f"{rubric_item} (weight: {answer['weight']}) (True/False): {answer['answer']}\n"
            else:
                checklist_item = f"{rubric_item} (weight: {answer['weight']}) (True/False): <True/False>\n"
            checklist.append(checklist_item)
        checklist_string = "".join(checklist)

        return (
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


    
    # def question_to_random_option(self, task_idx, rubric_item, option):
    #     options = deepcopy(self.INTERVENTION_DICT[task_idx][rubric_item])
    #     options.remove(option)
    #     random_option = random.choice(options)
    #     return self.INTERVENTION_DICT[task_idx][rubric_item][random_option]

    # def fix_llm_generation_error(self, generated_output):
    #     fixed_completion = re.sub(r'(?<=weight: )1(?=\))', '1.0', generated_output['completion'])
    #     return {"prompt": generated_output['prompt'], "generated_output": fixed_completion}
    
    def make_intervention(self, generated_output):
        completion = generated_output['completion']
        entries = capture_ricechem_checklist.extract_checklist_entries(completion)
        # we use entities extracted from generated string since we need exact correspondance between
        # original generation and intervention
        qa_map = {e["question"]: [e["answer"], e["weight"]] for e in entries}
        intervened_completions = []
        for question, (answer, question_weight) in qa_map.items():
            original_question = f"{question} (weight: {question_weight}) (True/False): {answer}\n"

            new_option = not answer
            replacement = f"{question} (weight: {question_weight}) (True/False): {new_option}\n"

            current_completion = completion[:int(len(completion)/2)]+completion[int(len(completion)/2):]#hack to create a new version of completion
            # we remove the completion for the intervention
            current_completion = re.sub(r'Final grade \(0-8\):\s*[0-8](?:\.[0-9]+)?.*?(?:\n|$)', "", current_completion)
            if original_question in current_completion:
                intervened_completion = current_completion.replace(original_question, replacement)
                intervened_completions.append({"completion": intervened_completion,
                                            "original_checklist_item": original_question,
                                            "intervened_checklist_item": replacement})
            else:
                # if we fail here -- we just leave
                return None
        return intervened_completions


    def validate_intervention(self, genenerated_output, intervened_prompt, original_checklist_item, intervention_checklist_item):
        # revert the mediated prompt
        prompt, original_completion = genenerated_output['prompt'], genenerated_output['completion']
        revert_to_original_completion = intervened_prompt.replace(intervention_checklist_item, original_checklist_item)

        # extract final grade
        final_grade_match = re.findall(r'Final grade \(0-8\):\s*[0-8](?:\.[0-9]+)?.*?(?:\n|$)', original_completion)[0]
        original_prompt = prompt + original_completion
        reverted_intervened_prompt = prompt + revert_to_original_completion + final_grade_match

        # i need to do that in someplace like -- prepare prompt for completion after mediation
        # reformat_prompt = reformat_prompt.replace(self.stop_token, "")

        # i dont think we need to do that, because we are not changing the final grade/stop token

        # final_grade = self.extract_target_from_prompt(intervened_prompt)
        # original_prompt = re.sub(f"Final grade (0-8): {final_grade}", "Final grade:", original_prompt)
        # original_prompt = original_prompt.replace(self.stop_token, "")
        # compare them
        return original_prompt == reverted_intervened_prompt
    

    def reconstruct_interventions_to_prompt(self, generated_output, intervened_completions):
        intervented_prompts = []
        for intervention in intervened_completions:
            reconstructed_prompt = generated_output['prompt'] + intervention['completion'] + "Final grade (0-8):"
            intervented_prompts.append(reconstructed_prompt)
        return intervented_prompts

    def validate_all_interventions(self, generated_output, intervened_completions):
        check_list = []
        for intervention in intervened_completions:
            check = self.validate_intervention(generated_output, intervention['completion'],
                                          intervention['original_checklist_item'], intervention['intervened_checklist_item'])
            check_list.append(check)
        return all(check_list)

    def extract_target_from_prompt(self, generated_output):
        "Given a full prompt (before intervention, where the model fills the full checklist)"
        "extract final grade"
        # I do not care about format, since it does not influence the generation
        completion = generated_output['completion'].replace(self.stop_token, "")
        return capture_ricechem_checklist.extract_final_grade(completion)
    
    def infer_completion(self, generated_output):
        "extract only the completion after the interventiuon, when we test model ability to make a correct decsion"
        completion = generated_output['completion']
        match = re.search(r'\d*\.?\d+', completion)
        return float(match.group()) if match else None
        

if __name__ == "__main__":
    from ricechem_dataset import RiceChemDataset
    dataset = RiceChemDataset("/Users/somov-od/Documents/phd/projects/frontdoor_llm_causality/statics/result_splits/RiceChem")
    intervention_logic = RiceChemIntervention(dataset, "<|im_end|>")
    print(intervention_logic.make_prompt(dataset.data[0], use_ground_truth=False))