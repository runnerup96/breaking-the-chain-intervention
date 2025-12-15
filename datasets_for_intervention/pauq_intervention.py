import copy
from copy import deepcopy
import json
import random
import re
from .utils import extract_data_from_response
# import pauq_dataset


class PAUQIntervention:
    def __init__(self, dataset, llm_model):
        self.dataset = dataset
        self.llm_model = llm_model

    def interventions_to_prompt(self, sample:dict):
        interventions = sample['structure_intervention']
        hsvt_intervention_prompt = [self.make_prompt(interventions['HSVT'][0], include_gold_structure=True)]
        local_edits_intervention_prompt = [ self.make_prompt(edit, include_gold_structure=True) for edit in interventions['Local Edits']]
        global_intervention_prompt = [self.make_prompt(interventions['Global'][0], include_gold_structure=True)]
        all_intervention_prompts = hsvt_intervention_prompt + local_edits_intervention_prompt + global_intervention_prompt
        return all_intervention_prompts

    def collect_intervention_completion(self, sample: dict, generated_output: list):
        completion_list = [generation['completion'] for generation in generated_output]
        intervention = sample['structure_intervention']
        intervention_list = ['HSVT'] + ['Local Edits'] * len(intervention['Local Edits']) + ['Global']
        intervention_idx_list = [0] + list(range(len(intervention['Local Edits']))) + [0]
        for completion, intervention_type, idx in zip(completion_list, intervention_list, intervention_idx_list):
            extracted = completion.strip()
            sample['structure_intervention'][intervention_type][idx]['generated_sql'] = extracted
        return sample
    
    def make_local_intervention(self, sample: dict, intervention_type: str = "column"):
        # random_link = random.choice(sample["schema_links"])
        table_name = random.choice(list(sample["schema_links"].keys()))
        columns = sample["schema_links"][table_name]
        intervention = None
        if intervention_type == "column":
            # Column intervention
            if not columns:
                raise RuntimeError("No columns for table in schema linking!")
            column_idx = random.randint(0, len(columns) - 1)
            column_name = columns[column_idx]
            table_columns = self.dataset.get_table_columns(sample["db"], table_name, True)
            table_columns.remove(column_name)
            other_column = random.choice(table_columns)
            intervention = {"type": "column", "before": column_name, "after": other_column}
            columns[column_idx] = other_column
        elif intervention_type == "table":
            # Table name
            table_names = sample["db"]["table_names_original"]
            table_names.remove(table_name)
            if table_names:
                other_table_name = random.choice(table_names)
                intervention = {"type": "table", "before": table_name, "after": other_table_name}
                columns = sample["schema_links"][table_name][:]
                del sample["schema_links"][table_name]
                sample["schema_links"][other_table_name] = columns

        sample["local_intervention"] = intervention

    def _get_random_table(self) -> str:
        return random.choice(list(self.dataset.tables.keys()))
    
    def _get_random_column(self, table_name: str) -> str:
        return random.choice(self.dataset.tables[table_name])
    
    def _get_random_columns(self, table_name: str, n_columns: int) -> str:
        return random.sample(self.dataset.tables[table_name], n_columns)
    
    def make_global_intervention(self, sample: dict):
        intervention = []
        schema_links = [{"table": key, "columns": value.copy()} for key, value in sample["schema_links"].items()]
        for link in schema_links:
            old_table = link["table"]
            old_columns = link["columns"].copy()
            new_columns = []
            while len(new_columns) < len(old_columns):
                new_table = self._get_random_table()
                new_columns.extend(self._get_random_columns(new_table, 1))

            link["table"] = self._get_random_table()
            intervention.append({"type": "table", "before": old_table, "after": new_table})
            link["columns"] = []
            for old_column, new_column in zip(old_columns, new_columns):
                link["columns"].append(new_column)
                intervention.append({"type": "column", "before": old_column, "after": new_column})

        sample["schema_links"] = {item["table"]: item["columns"] for item in schema_links}
        sample["global_intervention"] = intervention
    
    def make_intervention(self, sample: dict, generated_output: dict):
        completion = generated_output['completion']
        assert isinstance(completion, str)
        json_completion = extract_data_from_response(completion)
        schema_links_json = json_completion["schema_links"]
        schema_links = {}
        for item in schema_links_json:
            schema_links[item["table"]] = item["columns"]
        sample["schema_links"] = schema_links
        sample["generated_sql"] = json_completion["sql"]
        interventions = self.make_structure_intervention(sample)
        sample["structure_intervention"] = interventions
        return sample

    def make_structure_intervention(self, sample: dict):
        # HSVT 
        hsvt_sample = deepcopy(sample)
        intervened_question = self.dataset[sample["index"]]["paraphrase"]
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
            db_schema += f"Table: {table_name}\n"
            db_schema += "Columns: "
            for col_name in db["column_names_original"][1:]:
                if col_name[0] == i:
                    db_schema += f"{col_name[1]}, "
            db_schema = db_schema[:-2]
            db_schema += "\n"
        user_prompt = f"""
        You are an expert in natural language understanding and SQL queries generation.
        Given a natural language question and a database schema, perform two steps:

        1. Schema Linking: Identify which tables and columns in the schema are relevant to answer the question.
        2. SQL Generation: Write a correct, executable SQL query that answers the question, using the linked schema elements.

        Output Format
        Return the answer in the exact following format:

        Output:
        Schema links: [
                    {{
                      "table": "table_name",
                      "columns": ["column1", "column2", ...]
                    }}
                  ]
        SQL: SELECT ... FROM ... WHERE ...;

        Rules:
        - List all relevant tables and their columns in schema links
        - Write a valid SQL query string without markdown, without extra text
        - Only output in the specified format. Do not add explanations.

        Few-Shot Examples
        Example 1
        Question: "How many heads of the departments are older than 56?"

        Schema:
        Table: head
        Columns: head_id, name, age
        Table: department
        Columns: dept_id, name, budget

        Output:
        Schema links: [
                    {{
                      "table": "head",
                      "columns": ["age"]
                    }}
                  ]
        SQL: SELECT COUNT(*) FROM head WHERE age > 56;

        Example 2
        Question: "List the names of departments with budget over 1 million."

        Schema:
        Table: department
        Columns: dept_id, name, budget
        Table: employee
        Columns: emp_id, name, salary

        Output:
        Schema links: [
                    {{
                      "table": "department",
                      "columns": ["name", "budget"]
                    }}
                  ]
        SQL: SELECT name FROM department WHERE budget > 1000000;

        Example 3
        Question: "What is the average salary of employees hired in 2020?"

        Schema:
        Table: employee
        Columns: emp_id, name, salary, hire_date

        Output:
        Schema links: [
                    {{
                      "table": "employee",
                      "columns": ["salary", "hire_date"]
                    }}
                  ]
        SQL: SELECT AVG(salary) FROM employee WHERE hire_date LIKE '2020%';

        Now process the following:
        Question: "{question}"
        Schema:
        {db_schema}
        Output:
        """

        if not "true_schema_links" in sample:
            include_gold_structure = False

        messages = [{"role": "user", "content": user_prompt}]
        add_generation_prompt_status = True

        if include_gold_structure:
            schema_links_string = f"Schema links: {sample['true_schema_links']}\n"
            schema_links_string += "SQL: "
            messages.append({"role": "assistant", "content": schema_links_string})
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
    # json_model_response = extract_json_from_model_response(response)
    # dataset = pauq_dataset.PAUQDataset("./pauq", train=True)
    intervention_logic = PAUQIntervention(dataset, None)
    sample = dataset[0]
    # print_dict(sample)
    intervention = intervention_logic.make_intervention(sample, response)
    print_dict(intervention["structure_intervention"])

    for local_edit in intervention["structure_intervention"]["HSVT"]:
        print_dict(local_edit)
    # sql_before = "SELECT name FROM students WHERE age > 20"
    # sql_after = "SELECT full_name FROM students WHERE age > 20"

    # info_before = extract_tables_and_columns(sql_before)
    # info_after = extract_tables_and_columns(sql_after)

    # print("Before:", info_before)
    # print("After:", info_after)
