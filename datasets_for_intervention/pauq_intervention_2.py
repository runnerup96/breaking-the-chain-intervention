import copy
from copy import deepcopy
import json
import random
import re
from .utils import parse_model_response


class PAUQIntervention:
    def __init__(self, dataset, llm_model):
        self.dataset = dataset
        self.llm_model = llm_model

    def remove_special_tokens(self, generated_text: str) -> str:
        tok = [
            self.llm_model.tokenizer.eos_token,
            "<|im_end|>", "<|endoftext|>", "</s>", "<eos>",
            "<pad>", "<|eot_id|>", "<|pad|>"
        ]
        escaped = map(re.escape, tok)
        pattern = re.compile(r'(?:' + "|".join(escaped) + r')+$')
        return pattern.sub("", generated_text).rstrip()

    def interventions_to_prompt(self, sample:dict):
        interventions = sample['structure_intervention']
        hsvt_intervention_prompt = [self.make_prompt(interventions['HSVT'][0], include_gold_structure=True)]
        local_edits_intervention_prompt = [ self.make_prompt(edit, include_gold_structure=True) for edit in interventions['Local Edits']]
        global_intervention_prompt = [self.make_prompt(interventions['Global'][0], include_gold_structure=True)]
        all_intervention_prompts = hsvt_intervention_prompt + local_edits_intervention_prompt + global_intervention_prompt
        return all_intervention_prompts

    def collect_intervention_completion(self, sample: dict, generated_output: list):
        completion_list = [self.remove_special_tokens(generation['completion']) for generation in generated_output]
        intervention = sample['structure_intervention']
        intervention_list = ['HSVT'] + ['Local Edits'] * len(intervention['Local Edits']) + ['Global']
        intervention_idx_list = [0] + list(range(len(intervention['Local Edits']))) + [0]
        for completion, intervention_type, idx in zip(completion_list, intervention_list, intervention_idx_list):
            extracted = completion.strip()
            sample['structure_intervention'][intervention_type][idx]['generated_output'] = completion
            sample['structure_intervention'][intervention_type][idx]['generated_sql'] = extracted
        return sample

    def update_slots(self, sample: dict, name: str, new_name: str):
        # assert name in sample["slots"]
        if name in sample["slots"]:
            idx = sample["slots"].index(name)
            sample["slots"][idx] = new_name

    def make_local_intervention(self, sample: dict, intervention_type: str = "column"):
        table_name = random.choice(list(sample["schema_links"].keys()))
        db_schema = sample["db_schema"]
        columns = sample["schema_links"][table_name]
        intervention = None
        if intervention_type == "column":
            # Column intervention
            if not columns:
                raise RuntimeError("No columns for table in schema linking!")
            column_idx = random.randint(0, len(columns) - 1)
            column_name = columns[column_idx]
            table_columns = self.dataset.get_table_columns(sample["db"], table_name)
            try:
                table_columns.remove(column_name)
            except Exception:
                pass
            other_column = random.choice(table_columns)
            intervention = {"type": "column", "before": column_name, "after": other_column}
            if table_name not in db_schema:
                db_schema[table_name] = [other_column]
            else:
                db_schema[table_name].append(other_column)
                db_schema[table_name] = list(set(db_schema[table_name]))
            columns[column_idx] = other_column

            # Intervene slots
            self.update_slots(sample, column_name, other_column)
            
        elif intervention_type == "table":
            # Table name
            table_names = sample["db"]["table_names_original"]
            try:
                table_names.remove(table_name)
            except Exception:
                raise Exception(f"No such table: {table_names}, {list(sample['schema_links'].keys())}")
            other_table_name = random.choice(table_names)
            intervention = {"type": "table", "before": table_name, "after": other_table_name}
            if other_table_name not in db_schema:
                db_schema[other_table_name] = db_schema[table_name][:]
            else:
                db_schema[other_table_name].extend(db_schema[table_name][:])
                db_schema[other_table_name] = list(set(db_schema[other_table_name]))
            columns = sample["schema_links"][table_name][:]
            del sample["schema_links"][table_name]
            sample["schema_links"][other_table_name] = columns

            # Intervene slots
            self.update_slots(sample, table_name, other_table_name)

        sample["local_intervention"] = intervention

    def make_global_intervention(self, sample: dict):
        intervention = []
        schema_links = sample["schema_links"]
        db_schema = sample["db_schema"]
        num_tables = len(schema_links)
        if num_tables > len(self.dataset.dummy_tables):
            raise Exception("Not enough dummy tables")
        random_tables = random.sample(list(self.dataset.dummy_tables.keys()), num_tables)
        random_schema_links = {}
        for random_table, table_name in zip(random_tables, schema_links):
            num_columns = len(schema_links[table_name])
            random_columns = self.dataset.dummy_tables[random_table]
            if num_columns <= len(random_columns):
                random_columns_link = random.sample(random_columns, num_columns)
            else:
                raise Exception("Not enough columns in dummy table")
            random_schema_links[random_table] = random_columns_link
            for old_column, new_column in zip(schema_links[table_name], random_columns_link):
                intervention.append({"type": "column", "before": old_column, "after": new_column})
                self.update_slots(sample, old_column, new_column)
            intervention.append({"type": "table", "before": table_name, "after": random_table})
            self.update_slots(sample, table_name, random_table)
            db_schema[random_table] = self.dataset.dummy_tables[random_table][:]
        sample["schema_links"] = random_schema_links
        sample["global_intervention"] = intervention
    
    def make_intervention(self, sample: dict, generated_output: dict):
        completion = self.remove_special_tokens(generated_output['completion'])
        sample["generated_output"] = completion
        assert isinstance(completion, str)
        if sample["completion_type"] == "gold_structure":
            sample["generated_sql"] = completion.strip()
            sample["schema_links"] = copy.deepcopy(sample["true_schema_links"])
            sample["skeleton"] = copy.deepcopy(sample["true_skeleton"])
            sample["slots"] = copy.deepcopy(sample["true_slots"])
        elif sample["completion_type"] == "structure_prediction":
            # json_completion = extract_data_from_response(completion)
            json_completion = parse_model_response(completion)
            schema_links_json = json_completion["schema_links"]
            schema_links = {}
            for item in schema_links_json:
                schema_links[item["table"]] = item["columns"]
            sample["schema_links"] = schema_links
            sample["generated_sql"] = json_completion["sql"]
            sample["skeleton"] = json_completion["skeleton"]
            sample["slots"] = json_completion["slots"]
        else:
            raise NotImplementedError
        interventions = self.make_structure_intervention(sample)
        sample["structure_intervention"] = interventions
        return sample

    def make_structure_intervention(self, sample: dict):
        # HSVT 
        hsvt_sample = deepcopy(sample)
        intervened_question = sample["paraphrase"]
        hsvt_sample["question"] = intervened_question

        # Local Edits
        local_edits = []
        local_intervention_types = ["column", "table"]
        for intervention_type in local_intervention_types:
          local_sample = deepcopy(sample)
          self.make_local_intervention(local_sample, intervention_type=intervention_type)
          local_edits.append(local_sample)

        # Global intervention
        global_sample = deepcopy(sample)
        self.make_global_intervention(global_sample)
        
        return {"HSVT": [hsvt_sample], "Local Edits": local_edits, "Global": [global_sample]}

    def make_prompt(self, sample: dict, include_gold_structure: bool = False):
        question = sample["question"]
        db_schema = ""
        db = sample["db"]
        for i, table_name in enumerate(db["table_names_original"]):
            db_schema += f"Table: {table_name.lower()}\n"
            db_schema += "Columns: "
            for col_name in db["column_names_original"][1:]:
                if col_name[0] == i:
                    db_schema += f"{col_name[1].lower()}, "
            db_schema = db_schema[:-2]
            db_schema += "\n"

        user_prompt = (
            f"You are an expert in natural language understanding and SQL queries generation.\n"
            f"Given a natural language question and a database schema, perform 4 steps:\n\n"
        
            f"1. SQL Skeleton Generation: Derive the abstract syntactic structure of the SQL query that answers the question."
            f"Replace all concrete identifiers (table names, column names, literals, etc.) with generic placeholders SLOT_1, SLOT_2, ..., in the order they appear."
            f"The skeleton must be valid SQL syntax except for the use of SLOT_* tokens.\n"
            f"2. Schema Linking: Based on the question identify which tables and columns in the schema are relevant to answer the question.\n"
            f"3. Slot Matching: For each SLOT_i in the skeleton, map it to the exact identifier from the provided schema or a literal value derived from the question."
            f"Use only:\n"
            f"Column names (e.g., capacity)\n"
            f"Table names (e.g., stadium)\n"
            f"Numeric or string literals (e.g., 56, '2020%')\n"
            f"Do not infer semantics; match based on the question intent and schema content.\n"
            f"4. SQL Generation: Substitute all SLOT_i tokens in the skeleton with their matched identifiers to produce a syntactically correct, executable SQL query."
            f"The query must:\n"
            f"- Use only tables and columns from the given schema\n"
            f"- Answer the question exactly\n"
            f"Be written in standard SQL (no markdown, no comments, no extra text)\n\n"
        
            f"Output Format\n"
            f"Return only the following, with no additional explanations or formatting:\n"
        
            f"Output:\n"
            f"===SKELETON===\n"
            f"[SQL skeleton with SLOT tokens]\n"
            f"===SCHEMA_LINKS===\n"
            f"table1:col1,col2\n"
            f"table2:col1,col2\n"
            f"...\n"
            f"===SLOT_MATCHING===\n"
            f"SLOT_1:[value]\n"
            f"SLOT_2:[value]\n"
            f"...\n"
            f"===SQL===\n"
            f"[final executable SQL query]\n\n"
        
            f"Rules:\n"
            f"- Never include real identifiers in the skeleton.\n"
            f"- Every SLOT_i must appear in both SKELETON and SLOT_MATCHING.\n"
            f"- Write a valid SQL query string without markdown, without extra text\n"
            f"- Only output in the specified format. Do not add explanations.\n"
            f"- Do not pay attention to the name semantics of columns and tables, rely more on slot matching.\n"
            f"- String literals must be enclosed in single quotes; numeric literals must not be quoted.\n\n"
        
            f"Few-Shot Examples\n"
            f"Example 1\n"
            f'Question: "How many heads of the departments are older than 56?"\n\n'
        
            f"Schema:\n"
            f"Table: head\n"
            f"Columns: head_id, name, age\n"
            f"Table: department\n"
            f"Columns: dept_id, name, budget\n\n"
        
            f"Output:\n"
            f"===SKELETON===\n"
            f"SELECT COUNT(SLOT_1) FROM SLOT_2 WHERE SLOT_3 > SLOT_4;\n"
            f"===SCHEMA_LINKS===\n"
            f"head:age\n"
            f"===SLOT_MATCHING===\n"
            f"SLOT_1:*\n"
            f"SLOT_2:head\n"
            f"SLOT_3:age\n"
            f"SLOT_4:56\n"
            f"===SQL===\n"
            f"SELECT COUNT(*) FROM head WHERE age > 56;\n\n"
        
            f"Example 2\n"
            f'Question: "List the names of departments with budget over 1 million."\n\n'
        
            f"Schema:\n"
            f"Table: department\n"
            f"Columns: dept_id, name, budget\n"
            f"Table: employee\n"
            f"Columns: emp_id, name, salary\n\n"
        
            f"Output:\n"
            f"===SKELETON===\n"
            f"SELECT SLOT_1 FROM SLOT_2 WHERE SLOT_3 > SLOT_4;\n"
            f"===SCHEMA_LINKS===\n"
            f"department:name,budget\n"
            f"===SLOT_MATCHING===\n"
            f"SLOT_1:name\n"
            f"SLOT_2:department\n"
            f"SLOT_3:budget\n"
            f"SLOT_4:1000000\n"
            f"===SQL===\n"
            f"SELECT name FROM department WHERE budget > 1000000;\n\n"
        
            f"Example 3\n"
            f'Question: "What is the average salary of employees hired in 2020?"\n\n'
        
            f"Schema:\n"
            f"Table: employee\n"
            f"Columns: emp_id, name, salary, hire_date\n\n"
        
            f"Output:\n"
            f"===SKELETON===\n"
            f"SELECT AVG(SLOT_1) FROM SLOT_2 WHERE SLOT_3 LIKE SLOT_4;\n"
            f"===SCHEMA_LINKS===\n"
            f"employee:salary,hire_date\n"
            f"===SLOT_MATCHING===\n"
            f"SLOT_1:salary\n"
            f"SLOT_2:employee\n"
            f"SLOT_3:hire_date\n"
            f"SLOT_4:'2020%'\n"
            f"===SQL===\n"
            f"SELECT AVG(salary) FROM employee WHERE hire_date LIKE '2020%';\n\n"
        
            f"Now process the following:\n"
            f'Question: "{question}"\n'
            f"Schema:\n"
            f"{db_schema}\n"
            f"Output:\n"
        )
        
        if include_gold_structure:
            if "schema_links" not in sample:
                sample["schema_links"] = deepcopy(sample["true_schema_links"])
            if "skeleton" not in sample:
                sample["skeleton"] = deepcopy(sample["true_skeleton"])
            if "slots" not in sample:
                sample["slots"] = deepcopy(sample["true_slots"])
            
        
        messages = [{"role": "user", "content": user_prompt}]
        add_generation_prompt_status = True

        if include_gold_structure:
            # Add skeleton
            mediator_string = "===SKELETON===\n"
            mediator_string += sample["skeleton"]
            mediator_string += "\n"

            # Add schema links
            schema_links_string = "===SCHEMA_LINKS===\n"
            for table_name in sample['schema_links']:
                schema_links_string += table_name
                schema_links_string += ":"
                for column_name in sample['schema_links'][table_name]:
                    schema_links_string += column_name
                    schema_links_string += ","
                schema_links_string = schema_links_string[:-1]
                schema_links_string += "\n"
            mediator_string += schema_links_string

            # Add slots matching
            mediator_string += "===SLOT_MATCHING===\n"
            for i, slot in enumerate(sample["slots"]):
                mediator_string += f"SLOT_{i}:{slot}\n"

            # Add final SQL
            mediator_string += "===SQL==="
            messages.append({"role": "assistant", "content": mediator_string})
            add_generation_prompt_status = False

        prompt = self.llm_model.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt_status
        )

        if include_gold_structure:
            sample['prompt_gold'] = prompt
        else:
            sample['prompt'] = prompt
        
        if add_generation_prompt_status == False:
            prompt = self.llm_model.clean_model_specific_completion(prompt)

        return prompt

def print_dict(d):
    for k, v in d.items():
        print(k, end=": ")
        print(v)
        print()

if __name__ == "__main__":
    text_response = '''
        Schema links: [
            {
              "table": "head",
              "columns": ["age"]
            }
          ]
        SQL: SELECT COUNT(*) FROM head WHERE age > 56;
    '''
    response = {"completion": text_response}
    intervention_logic = PAUQIntervention(dataset, None)
    sample = dataset[0]
    intervention = intervention_logic.make_intervention(sample, response)
    print_dict(intervention["structure_intervention"])

    for local_edit in intervention["structure_intervention"]["HSVT"]:
        print_dict(local_edit)
        
