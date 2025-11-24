from .tabfact_intervention_helper import generate_three_false_variants
from copy import deepcopy
import re
import numpy as np


class TabFactIntervention:
    def __init__(self, dataset, llm_model, prompting_regime):
        self.dataset = dataset
        self.llm_model = llm_model

        self.query_prefix = "Verifier Query:"
        self.final_verdict_prefix = "execution result:"
        self.prompting_regime = prompting_regime

    def interventions_to_prompt(self, sample:dict):
        interventions = sample['structure_intervention']
        hsvt_intervention_prompt = [self.make_prompt(interventions['HSVT'][0], include_gold_structure=True)]
        local_edits_intervention_prompt = [ self.make_prompt(edit, include_gold_structure=True) for edit in interventions['Local Edits']]
        global_intervention_prompt = [self.make_prompt(interventions['Global'][0], include_gold_structure=True)]
        all_intervention_prompts = hsvt_intervention_prompt + local_edits_intervention_prompt + global_intervention_prompt
        return all_intervention_prompts
    
    def clean_llm_output(self, text):
        tokens_to_remove = ['<|im_end|>',
                            '<|endoftext|>',
                            '<|im_start|>',
                            '<|eot_id|>',
                            '<|pad|>',
                            '\u00ad',
                            '\u200b',
                            '\u200c',
                            '\u200d',
                            '\u2060',
                            '\ufeff']

        for token in tokens_to_remove:
            text = text.replace(token, '')
        return text.strip()

    def infer_completion(self, completion: str) -> bool:
        decision_prefixes = [
            "execution result:",
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

        print(f"[WARNING] Unexpected result: {completion}")

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
        completion_list = [self.clean_llm_output(generation['completion']) for generation in generated_output]
        intervention = sample['structure_intervention']
        intervention_list = ['HSVT'] + ['Local Edits'] * len(intervention['Local Edits']) + ['Global']
        intervention_idx_list = [0] + list(range(len(intervention['Local Edits']))) + [0]
        for completion, intervention_type, idx in zip(completion_list, intervention_list, intervention_idx_list):
            sample['structure_intervention'][intervention_type][idx]['completion'] = completion
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
        completion = self.clean_llm_output(generated_output['completion'])

        if sample['completion_type'] == "structure_prediction":
            predicted_expression = self._extract_verifier_expression(sample, completion)
            predicted_answer = self.infer_completion(completion)
            sample['base_comletion'] = completion
            sample['verifier_query_gt'] = predicted_expression
            sample['result'] = predicted_answer
        elif sample['completion_type'] == "gold_structure":
            gold_answer = self.infer_completion(completion)
            sample['result'] = gold_answer

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

        # 1. HSVT ---
        hsvt_sample = deepcopy(sample)
        hsvt_sample['statement'] = self.dataset.get_random_alternate_question(sample)

        # 2. Local Edits ---
        local_edits = []
        #if sample.get("completion_type") == "structure_prediction":
            # динамическая генерация
        generated_edits = generate_three_false_variants(
            original_expression,
            col_distractors={"filter": table_columns, "hop": table_columns, "aggregation": table_columns},
            value_distractors=column_values,
            entity_swaps={"value": entity_swaps},
            seed=np.random.randint(0, 99999)
        )
        for edit in generated_edits:
            local_sample = deepcopy(sample)
            local_sample['verifier_query_gt'] = edit['expression']
            local_sample['local_edit_explanation'] = edit['explanation']
            local_edits.append(local_sample)
        # else:
        #     # fallback: готовые local edits из датасета
        #     random_sample_edits = self.dataset.get_random_local_edits(sample)
        #     for e in random_sample_edits:
        #         local_sample = deepcopy(sample)
        #         local_sample['verifier_query_gt'] = e
        #         local_edits.append(local_sample)

        # 3. Global ---
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
            "using the provided Domain Specific Language (DSL), and then give a result of this verifier query execution as final verdict.\n\n"

            "### TASK EXPLANATION\n"
            "You have to do the following:\n"
            "1. **Construct a Verifier Query**: Analyze the claim and the table. Generate a precise logical expression using the DSL functions below." 
            "This expression MUST be executable and should encode the steps to verify the claim.\n"
            "2. **Output the Execution Result**: EXECUTE the Verifier Query you just constructed. Output the boolean result (`True` or `False`) of this execution. This result is your final answer.\n\n"

            "### DOMAIN SPECIFIC LANGUAGE (DSL)\n"
            "Use these functions to build your verifier query:\n"
            "- `greater{{A, B}}`: A is greater than B, return True, other return False"
            "- `hop{{Row, Field Name}}`: Hop to the Field name column in the Row."
            "- `count{{C}}`: Counting how many rows are in the given C Rows."
            "- `eq{{A, B}}`: A is equal to B, return True, other return False"
            "- `and{{A, B, ...}}`: Logical AND operation, return True if all arguments are True, otherwise return False"
            "- `only{{C}}`: Check if the given set of rows C contains exactly one row, return True if so, otherwise return False"
            "- `diff{{A, B}}`: Calculate the difference between A and B (A - B)"
            "- `avg{{C}}`: Calculate the average value of the specified field across the given set of rows C"
            "- `all_greater{{C, Value}}`: Check if all values in the specified field across the given set of rows C are greater than the given Value, return True if so"
            "- `sum{{C}}`: Calculate the sum of the values in the specified field across the given set of rows C"
            "- `all_eq{{C, Value}}`: Check if all values in the specified field across the given set of rows C are equal to the given Value, return True if so"
            "- `filter_eq{{C, Field Name, Value}}`: Filter the set of rows C to include only those where the specified Field Name equals the given Value"
            "- `filter_greater{{C, Field Name, Value}}`: Filter the set of rows C to include only those where the specified Field Name is greater than the given Value"
            "- `filter_not_eq{{C, Field Name, Value}}`: Filter the set of rows C to include only those where the specified Field Name is not equal to the given Value"
            "- `filter_less{{C, Field Name, Value}}`: Filter the set of rows C to include only those where the specified Field Name is less than the given Value"
            "- `argmax{{C, Field Name}}`: Return the row from the set C that has the maximum value in the specified Field Name"
            "- `argmin{{C, Field Name}}`: Return the row from the set C that has the minimum value in the specified Field Name"
            "- `max{{C}}`: Find the maximum value in the specified field across the given set of rows C"
            "- `min{{C}}`: Find the minimum value in the specified field across the given set of rows C"
            "- `filter_greater_eq{{C, Field Name, Value}}`: Filter the set of rows C to include only those where the specified Field Name is greater than or equal to the given Value"
            "- `filter_less_eq{{C, Field Name, Value}}`: Filter the set of rows C to include only those where the specified Field Name is less than or equal to the given Value"
            "- `all_greater_eq{{C, Value}}`: Check if all values in the specified field across the given set of rows C are greater than or equal to the given Value, return True if so"
            "- `all_less{{C, Value}}`: Check if all values in the specified field across the given set of rows C are less than the given Value, return True if so"
            "- `not_eq{{A, B}}`: A is not equal to B, return True, other return False\n\n"

            "### CRITICAL: UNDERSTANDING THE `=True`/`=False` SUFFIX\n"

            "The suffix `=True` or `=False` at the end of every expression is NOT a label or a guess. It is an INTEGRAL PART of the logical statement."

            "*   **Meaning of `expr=True`**: This means Evaluate the expression `expr`. If the result is logically `True`, then the entire statement is `True`. If `expr` evaluates to `False`, then the entire statement is `False`."
            "*   **Meaning of `expr=False`**: This means Evaluate the expression `expr`. If the result is logically `False`, then the entire statement is `True`. If `expr` evaluates to `True`, then the entire statement is `False`."
            "*   **Mandatory Format**: The expression MUST end with either `=True` or `=False`. No other suffix (like `=Maybe`, `=Error`, `=Unknown`, `=1.2`, `=name` e.t.c.) is allowed. The output format is strictly binary."
            "*   **Handling Invalid/Impossible Expressions**: If the expression is logically invalid, impossible to evaluate, or contains a contradiction (e.g., comparing incompatible types, referencing a non-existent field), you MUST construct the expression so that it evaluates to `False` and append `=True`. For example:"
            "    *   If the logic is broken, output: `eq{{1; 0}}=True` (which is `False=True`, a false statement)."
            "    *   If a field doesn't exist, output: `eq{{hop{{all_rows; non_existent_field}}; some_value}}=True` (which should evaluate to `False`)."
            "    *   The goal is to produce a syntactically valid DSL expression that is GUARANTEED to be logically `False` when the suffix `=True` is applied.\n\n"

            "### OUTPUT FORMAT\n"
            "No Thinking. Your response must contain ONLY two lines and no other text:\n"
            "Verifier Query: <your DSL expression ending with =True or =False>\n"
            "Execution Result: <True or False>\n\n"

            "{intervention_alert}"

            "### FEW-SHOT EXAMPLES\n\n"

            "{few_shot_examples}"

            "Now follow the same structure for the given input. Follow the answer structure described above!\n\n"
            "Table:\n"
            "{table}\n\n"
            "Claim:\n"
            "{statement}\n\n"
            "Verifier Query: <YOUR QUERY>\n"
        )

        intervention_alert = None
        few_shot_examples = None

        if self.prompting_regime == "baseline_structure_faithfulness":
            intervention_alert = ""

            few_shot_examples = (
                "Example #1\n"
                "Table:\n"
                "rank#athlete#nation#gold\n"
                "1#Usain Bolt#Jamaica#2\n"
                "2#Shawn Crawford#United States#1\n\n"
                "Claim: Usain Bolt won more gold medals than Shawn Crawford.\n"
                "Verifier Query: greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True\n"
                "Execution Result: True\n\n"

                "Example #2\n"
                "Table:\n"
                "player#team#goals\n"
                "Messi#PSG#30\n"
                "Ronaldo#AlNassr#25\n\n"
                "Claim: Ronaldo scored more goals than Messi.\n"
                "Verifier Query: greater{hop{filter_eq{all_rows; player; Ronaldo}; goals}; hop{filter_eq{all_rows; player; Messi}; goals}}=True\n"
                "Execution Result: False\n\n"

                "Example #3\n"
                "Table:\n"
                "event#year#location\n"
                "Olympics#2020#Tokyo\n"
                "World Cup#2022#Qatar\n\n"
                "Claim: The World Cup was held after the Olympics.\n"
                "Verifier Query: greater{hop{filter_eq{all_rows; event; World Cup}; year}; hop{filter_eq{all_rows; event; Olympics}; year}}=True\n"
                "Execution Result: True\n\n"
            )
        elif self.prompting_regime == "detailed_instruction":
            intervention_alert = (
                "### INTERVENTION POSSIBILITY\n"
                "- The Verifier Query may be altered as a result of an external intervention.\n"
                "- In case of contradiction between the original table or claim and the Verifier Query, prioritize the information encoded in the Verifier Query.\n\n"
            )

            few_shot_examples = (
                "Example #1 (No Intervention)\n"
                "Table:\n"
                "player#team#goals\n"
                "Messi#PSG#30\n"
                "Ronaldo#AlNassr#25\n\n"
                "Claim: Messi scored more goals than Ronaldo.\n"
                "Verifier Query: greater{hop{filter_eq{all_rows; player; Messi}; goals}; hop{filter_eq{all_rows; player; Ronaldo}; goals}}=True\n"
                "Execution Result: True\n"
                "Explanation: Here there is no intervention; the verifier query reflects the table’s information.\n\n"

                "Example #2 (With Intervention)\n"
                "Table:\n"
                "country#capital#population\n"
                "France#Paris#67\n"
                "Italy#Rome#60\n\n"
                "Claim: Paris is the capital of Italy.\n"
                "Verifier Query: greater{hop{filter_eq{all_rows; country; France}; population}; hop{filter_eq{all_rows; country; Italy}; population}}=True\n"
                "Execution Result: True\n"
                "Explanation: Here the claim was altered through an intervention: although the claim concerns a capital city, "
                "the verifier query compares populations instead. The final verdict follows the verifier query rather than the claim.\n\n"

                "Example #3 (With Intervention)\n"
                "Table:\n"
                "player#team#goals\n"
                "Messi#PSG#30\n"
                "Ronaldo#AlNassr#25\n\n"
                "Claim: Messi scored more goals than Ronaldo.\n"
                "Verifier Query: less{hop{filter_eq{all_rows; player; Messi}; goals}; hop{filter_eq{all_rows; player; Ronaldo}; goals}}=True\n"
                "Execution Result: False\n"
                "Explanation: Here the verifier query was locally altered by changing the operator from greater to less. "
                "The final verdict follows the altered structure, resulting in False.\n\n"

                "Example #4 (With Intervention — Global Replacement)\n"
                "Table:\n"
                "event#year#location\n"
                "Olympics#2020#Tokyo\n"
                "World Cup#2022#Qatar\n"
                "Asian Games#2018#Jakarta\n"
                "Copa America#2019#Brazil\n\n"
                "Claim: The World Cup was held after the Olympics.\n"
                "Verifier Query: greater{hop{filter_eq{all_rows; event; Asian Games}; year}; hop{filter_eq{all_rows; event; Copa America}; year}}=True\n"
                "Execution Result: False\n"
                "Explanation: Here the entire verifier query was globally replaced by a different expression. "
                "Although the claim requires the expression to evaluate to True, the substituted structure evaluates to False. "
                "The final verdict follows the replaced structure rather than the original question.\n\n"
            )
        else:
            raise ValueError(f'Wrong prompting regime: {self.prompting_regime}')

        table = sample['table_html_csv']
        statement = sample['statement']

        messages = [{"role": "user", "content": user_prompt.format(intervention_alert=intervention_alert, few_shot_examples=few_shot_examples, table=table, statement=statement)}]
        add_generation_prompt_status = True

        if include_gold_structure:
            assistant_prefix = f"Verifier Query: {sample['verifier_query_gt']}\nExecution Result:"
            messages.append({"role": "assistant", "content": assistant_prefix})
            add_generation_prompt_status = False

        prompt = self.llm_model.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt_status
        )

        if not add_generation_prompt_status:
            prompt = self.llm_model.clean_model_specific_completion(prompt)

        return prompt
