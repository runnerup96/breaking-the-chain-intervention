import copy
from copy import deepcopy
import json
import random
import re
from .utils import extract_data_from_response


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
            sample['structure_intervention'][intervention_type][idx]['generated_sql'] = extracted
        return sample
    
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
        elif intervention_type == "table":
            # Table name
            table_names = sample["db"]["table_names_original"]
            try:
                table_names.remove(table_name)
            except Exception:
                raise Exception(f"No such table: {table_names}, {list(sample['schema_links'].keys())}")
            if table_names:
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
            intervention.append({"type": "table", "before": table_name, "after": random_table})
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
        elif sample["completion_type"] == "structure_prediction":
            json_completion = extract_data_from_response(completion)
            schema_links_json = json_completion["schema_links"]
            schema_links = {}
            for item in schema_links_json:
                schema_links[item["table"]] = item["columns"]
            sample["schema_links"] = schema_links
            sample["generated_sql"] = json_completion["sql"]
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
            f"Given a natural language question and a database schema, perform two steps:\n\n"
    
            f"1. Schema Linking: Based on the question identify which tables and columns in the schema are relevant to answer the question.\n"
            f"2. SQL Generation: Write a correct, executable SQL query that answers the question, using the linked schema elements.\n\n"
    
            f"Output Format\n"
            f"Return the answer in the exact following format:\n"
    
            f"Output:\n"
            f"===SCHEMA_LINKS===\n"
            f"table1:col1,col2\n"
            f"table2:col1,col2\n"
            f"===SQL===\n"
            f"SELECT ... FROM ... WHERE ...;\n\n"
    
            f"Rules:\n"
            f"- List all relevant tables and their columns in schema links\n"
            f"- Write a valid SQL query string without markdown, without extra text\n"
            f"- Only output in the specified format. Do not add explanations.\n\n"
    
            f"Few-Shot Examples\n"
            f"Example 1\n"
            f'Question: "How many heads of the departments are older than 56?"\n\n'
    
            f"Schema:\n"
            f"Table: head\n"
            f"Columns: head_id, name, age\n"
            f"Table: department\n"
            f"Columns: dept_id, name, budget\n\n"
    
            f"Output:\n"
            f"===SCHEMA_LINKS===\n"
            f"head:age\n"
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
            f"===SCHEMA_LINKS===\n"
            f"department:name,budget\n"
            f"===SQL===\n"
            f"SELECT name FROM department WHERE budget > 1000000;\n\n"
    
            f"Example 3\n"
            f'Question: "What is the average salary of employees hired in 2020?"\n\n'
    
            f"Schema:\n"
            f"Table: employee\n"
            f"Columns: emp_id, name, salary, hire_date\n\n"
    
            f"Output:\n"
            f"===SCHEMA_LINKS===\n"
            f"employee:salary,hire_date\n"
            f"===SQL===\n"
            f"SELECT AVG(salary) FROM employee WHERE hire_date LIKE '2020%';\n\n"
    
            f"Now process the following:\n"
            f'Question: "{question}"\n'
            f"Schema:\n"
            f"{db_schema}\n"
            f"Output:\n"
        )

        # sample["prompt"] = user_prompt
        
        if include_gold_structure and not "schema_links" in sample:
            sample["schema_links"] = deepcopy(sample["true_schema_links"])
        
        messages = [{"role": "user", "content": user_prompt}]
        add_generation_prompt_status = True

        if include_gold_structure:
            schema_links_string = "===SCHEMA_LINKS===\n"
            for table_name in sample['schema_links']:
                schema_links_string += table_name
                schema_links_string += ":"
                for column_name in sample['schema_links'][table_name]:
                    schema_links_string += column_name
                    schema_links_string += ","
                schema_links_string = schema_links_string[:-1]
                schema_links_string += "\n"
            schema_links_string += "===SQL==="
            messages.append({"role": "assistant", "content": schema_links_string})
            # sample["schema_links_string"] = schema_links_string
            add_generation_prompt_status = False

        prompt = self.llm_model.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt_status
        )

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
        
