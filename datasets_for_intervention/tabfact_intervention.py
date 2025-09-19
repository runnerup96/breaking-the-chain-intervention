from .tabfact_intervention_helper import intervene_random_semantic_flip
from copy import deepcopy
import re


class TabFactIntervention:
    def __init__(self, dataset, llm_model):
        self.dataset = dataset
        self.llm_model = llm_model

        self.query_prefix = "Verifier Query:"
        self.final_verdict_prefix = "final verdict:"

    def interventions_to_prompt(self, sample:dict):
        interventions = sample['structure_intervention']
        hsvt_intervention_prompt = [self.make_prompt(interventions['HSVT'][0], include_gold_structure=True)]
        local_edits_intervention_prompt = [ self.make_prompt(edit, include_gold_structure=True) for edit in interventions['Local Edits']]
        global_intervention_prompt = [self.make_prompt(interventions['Global'][0], include_gold_structure=True)]
        all_intervention_prompts = hsvt_intervention_prompt + local_edits_intervention_prompt + global_intervention_prompt
        return all_intervention_prompts

    def infer_completion(self, completion: str) -> bool:
        decision_prefixes = [
            "final verdict:",
            "final decision:",
            "final answer:",
            "answer:",
            "decision:",
            "conclusion:",
        ]

        expr_pattern = r'[a-z_]+{.*?}=(?:true|false)'
        bool_pattern = r'\b(true|false)\b'

        completion_lower = completion.lower()
        lines = completion_lower.split('\n')

        for line in lines:
            line_stripped = line.strip()
            for prefix in decision_prefixes:
                if line_stripped.startswith(prefix):
                    after_prefix = line_stripped[len(prefix):].strip()
                    match = re.search(bool_pattern, after_prefix)
                    if match:
                        return True if match.group(1) == "true" else False

        expr_spans = []
        for match in re.finditer(expr_pattern, completion_lower, re.DOTALL):
            expr_spans.append((match.start(), match.end()))

        candidates = []
        for match in re.finditer(bool_pattern, completion_lower):
            bool_start = match.start()
            inside_expr = any(start <= bool_start < end for start, end in expr_spans)
            if not inside_expr:
                candidates.append(match.group(1))

        if candidates:
            result = candidates[-1]
            return True if result == "true" else False

        print(f"[WARNING] Unexpected verdict: {completion}")

        return None

    # def infer_completion(self, completion: str) -> bool:
    #     lines = completion.strip().split('\n')
    #     if not lines:
    #         return None
    #     last_line = lines[-1].strip()
    #     print('!!!!!!!!!', completion)
    #     if not last_line.lower().startswith(self.final_verdict_prefix):
            
    #         print(f"[WARNING] Unexpected format: {last_line}")
    #         return None
    #     print('OK')
    #     verdict = last_line.split(":", 1)[1].strip()
    #     if verdict == "True":
    #         return True
    #     elif verdict == "False":
    #         return False
    #     else:
    #         print(f"[WARNING] Unexpected verdict: {verdict}")
    #         return None

    def collect_intervention_completion(self, sample:dict, generated_output:list):
        completion_list = [generation['completion'] for generation in generated_output]
        intervention = sample['structure_intervention']
        intervention_list = ['HSVT'] + ['Local Edits'] * len(intervention['Local Edits']) + ['Global']
        intervention_idx_list = [0] + list(range(len(intervention['Local Edits']))) + [0]
        for completion, intervention_type, idx in zip(completion_list, intervention_list, intervention_idx_list):
            sample['structure_intervention'][intervention_type][idx]['result_after_intervention'] = self.infer_completion(completion)
        return sample

    # def _extract_verifier_expression(self, sample, completion: str) -> str:
    #     prefixes = [
    #         "Verifier Query:",
    #         "Logical expression:",
    #         "Expression:",
    #         "Query:",
    #     ]

    #     expr_pattern = r'([a-zA-Z_]+{.*?}=(?:True|False))'

    #     lines = completion.split('\n')

    #     for line in lines:
    #         line_stripped = line.strip()
    #         for prefix in prefixes:
    #             if line_stripped.startswith(prefix):
    #                 expr_candidate = line_stripped[len(prefix):].strip()
    #                 match = re.search(expr_pattern, expr_candidate, re.DOTALL)
    #                 if match:
    #                     return match.group(1)

    #     all_matches = re.findall(expr_pattern, completion, re.DOTALL)
    #     if all_matches:
    #         return all_matches[-1]

    #     return sample.get('verifier_query_gt', "")

    def _extract_verifier_expression(self, sample, completion: str) -> str:
        lines = completion.strip().split('\n')
        if not lines:
            print(f"[WARNING] Empty completion for sample {sample.get('idx', 'unknown')}")
            return sample.get('verifier_query_gt', "")

        first_line = lines[0].strip()
        
        if not first_line.startswith(self.query_prefix):
            print(f"[WARNING] First line does not start with '{self.query_prefix}'. Sample {sample.get('idx', 'unknown')}. Line: '{first_line}'")
            return sample.get('verifier_query_gt', "")

        # Extract everything after "Verifier Query:"
        expr = first_line[len(self.query_prefix):].strip()
        
        dsl_pattern = r'([a-zA-Z_]+{.*?}=(?:True|False))'
        
        if not re.fullmatch(dsl_pattern, expr):
            print(f"[WARNING] Extracted expression fails DSL syntax validation. Sample {sample.get('idx', 'unknown')}. Expr: '{expr}'")
            return sample.get('verifier_query_gt', "")
        
        # --- ADDITIONAL CHECK: Must end with =True or =False ---
        if not (expr.endswith("=True") or expr.endswith("=False")):
            print(f"[WARNING] Extracted expression does not end with =True/=False. Sample {sample.get('idx', 'unknown')}. Expr: '{expr}'")
            # We still return it, as the model might be correct, but the format is off.
        
        return expr
    
    def make_intervention(self, sample: dict, generated_output: dict) -> dict:
        completion = generated_output['completion']

        if sample['completion_type'] == "structure_prediction":
            predicted_expression = self._extract_verifier_expression(sample, completion)
            sample['verifier_query_gt'] = predicted_expression

        interventions = self.make_structure_intervention(sample)
        sample['structure_intervention'] = interventions
        return sample

    def _count_differences(self, str1: str, str2: str) -> int:
        """Count the number of different words between two strings."""
        words1 = set(str1.split())
        words2 = set(str2.split())
        return len(words1.symmetric_difference(words2))
    
    def make_structure_intervention(self, sample: dict) -> dict:
        original_expression = sample['verifier_query_gt']

        distractors = sample.get('distractors', {})
        table_columns = distractors.get('columns', [])
        column_values = distractors.get('values', {})
        entity_swaps = distractors.get('entity_swaps', [])

        # 1. HSVT: Replace the original question with another one related to the same table ---
        hsvt_sample = deepcopy(sample)
        hsvt_sample['statement'] = self.dataset.get_random_alternate_question(sample)

        # 2. Local Edits: Randomly replace several entities in the arguments of expression functions with others from the same table ---
        local_edits = []

        col_distractors = {'filter_eq': table_columns, 'hop': table_columns, 'aggregation': table_columns}
        value_distractors = column_values
        entity_swaps_dict = {'entity': entity_swaps}

        strategies = [
            {'num_changes': 1, 'seed_offset': 0},
            {'num_changes': 2, 'seed_offset': 1000},
            {'num_changes': 3, 'seed_offset': 2000}
        ]
        
        for i, strategy in enumerate(strategies):
            for attempt in range(5):
                local_sample = deepcopy(sample)
                new_expression = intervene_random_semantic_flip(
                    prog=original_expression,
                    col_distractors=col_distractors,
                    value_distractors=value_distractors,
                    entity_swaps=entity_swaps_dict,
                    seed=hash(sample['idx']) + strategy['seed_offset'] + attempt,
                    num_changes=strategy['num_changes']
                )
                
                if new_expression != original_expression:
                    local_sample['verifier_query_gt'] = new_expression
                    local_edits.append(local_sample)
                    break
            else:
                # Fallback: минимальное изменение
                local_sample = deepcopy(sample)
                new_expression = intervene_random_semantic_flip(
                    prog=original_expression,
                    col_distractors=col_distractors,
                    value_distractors=value_distractors,
                    entity_swaps=entity_swaps_dict,
                    seed=hash(sample['idx']) + 9999,
                    num_changes=1
                )
                local_sample['verifier_query_gt'] = new_expression
                local_edits.append(local_sample)

        # 3. Global Edit: Completely replace the expression from the model with another one related to the same table ---
        global_sample = deepcopy(sample)
        global_sample['verifier_query_gt'] = self.dataset.get_random_alternate_program(sample)

        return {
            "HSVT": [hsvt_sample],
            "Local Edits": local_edits,
            "Global": [global_sample]
        }

    def make_prompt(self, sample: dict, include_gold_structure: bool = False) -> str:
        user_prompt = (
            "You are an expert table fact-checking system. "
            "Your task is to evaluate a claim against tabular data by first constructing a verifier query "
            "using the provided Domain Specific Language (DSL), and then give a final verdict.\n\n"

            "### TASK EXPLANATION\n"
            "1. **Construct a Verifier Query**: Generate a logical expression using the DSL functions below. "
            "This query should precisely capture the logical steps needed to verify the statement against the table.\n"
            "2. **Make a Final Verdict**: Based SOLELY on the result of evaluating your Verifier Query, "
            "output a final verdict: `True` if the statement is supported by the table, `False` otherwise.\n\n"

            "### DOMAIN SPECIFIC LANGUAGE (DSL)\n"
            "Use these functions to build your verifier query:\n"
            "- `greater{A, B}`: Returns True if A > B\n"
            "- `less{A, B}`: Returns True if A < B\n"
            "- `eq{A, B}`: Returns True if A == B\n"
            "- `not_eq{A, B}`: Returns True if A != B\n"
            "- `and{A, B, ...}`: Returns True if all arguments are True\n"
            "- `hop{Row, Field}`: Extracts the value of 'Field' from the given 'Row'\n"
            "- `count{C}`: Returns the number of rows in the set 'C'\n"
            "- `only{C}`: Returns True if the set 'C' contains exactly one row\n"
            "- `filter_eq{C, Field, Value}`: Returns rows from 'C' where 'Field' equals 'Value'\n"
            "- `filter_greater{C, Field, Value}`: Returns rows from 'C' where 'Field' > 'Value'\n"
            "- `argmax{C, Field}`: Returns the row from 'C' with the maximum value in 'Field'\n"
            "- `max{C}`: Returns the maximum value in the set 'C'\n"
            "- `all_rows`: A special constant representing all rows in the table\n\n"

            "### OUTPUT FORMAT\n"
            "Your response must contain ONLY two lines and no other text:\n"
            "Verifier Query: <your DSL expression ending with =True or =False>\n"
            "Final Verdict: <True or False>\n\n"
            "Use only True or False in your answer. If somethin wrong with the data use ERROR"

            "### FEW-SHOT EXAMPLES\n\n"

            "Example #1\n"
            "Table:\n"
            "rank#athlete#nation#gold\n"
            "1#Usain Bolt#Jamaica#2\n"
            "2#Shawn Crawford#United States#1\n\n"
            "Claim: Usain Bolt won more gold medals than Shawn Crawford.\n"
            "Verifier Query: greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True\n"
            "Final Verdict: True\n\n"

            "Example #2\n"
            "Table:\n"
            "player#team#goals\n"
            "Messi#PSG#30\n"
            "Ronaldo#AlNassr#25\n\n"
            "Claim: Ronaldo scored more goals than Messi.\n"
            "Verifier Query: greater{hop{filter_eq{all_rows; player; Ronaldo}; goals}; hop{filter_eq{all_rows; player; Messi}; goals}}=True\n"
            "Final Verdict: False\n\n"

            "Example #3\n"
            "Table:\n"
            "event#year#location\n"
            "Olympics#2020#Tokyo\n"
            "World Cup#2022#Qatar\n\n"
            "Claim: The World Cup was held after the Olympics.\n"
            "Verifier Query: greater{hop{filter_eq{all_rows; event; World Cup}; year}; hop{filter_eq{all_rows; event; Olympics}; year}}=True\n"
            "Final Verdict: True\n\n"

            "Now follow the same structure for the given input.\n\n"
            "Table:\n"
            f"{sample['table_html_csv']}\n\n"
            "Claim:\n"
            f"{sample['statement']}\n\n"
            "Verifier Query: <YOUR QUERY>\n"
        )

        messages = [{"role": "user", "content": user_prompt}]
        add_generation_prompt_status = True

        if include_gold_structure:
            assistant_prefix = f"Verifier Query: {sample['verifier_query_gt']}\nFinal Verdict: "
            messages.append({"role": "assistant", "content": assistant_prefix})
            add_generation_prompt_status = False

        prompt = self.llm_model.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt_status
        )

        if not add_generation_prompt_status:
            prompt = self.llm_model.clean_model_specific_completion(prompt)

        return prompt
