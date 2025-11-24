import copy
import json
import random
import re
from utils import extract_tables_and_columns


def extract_json_from_model_response(model_response: str) -> dict | None:
    json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", model_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"({.*})", model_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            return None

    try:
        data = json.loads(json_str)
        return data
    except (json.JSONDecodeError, TypeError):
        return None


class PAUQIntervention:
    def __init__(self, dataset, llm_model):
        self.dataset = dataset
        self.llm_model = llm_model

    def get_table_columns(self, db, table_name):
        table_db_idx = db["table_names_original"].index(table_name)
        table_columns = []
        for i, col_name in db["column_names_original"][1:]:
            if i == table_db_idx:
                table_columns.append(col_name)
        return table_columns

    def make_intervention(self, sample: dict, json_model_response: dict, intervention_type: str = "column"):
        intervened_schema = copy.deepcopy(json_model_response)
        random_link = random.choice(intervened_schema["schema_links"])
        # check the output correctness
        # global intervention: change the full db schema
        # HSVT: change the text question
        # SQL Parse
        table_name = random_link["table"]
        intervention = None
        if intervention_type == "column":
            # Column intervention
            if not random_link["columns"]:
                raise RuntimeError("No columns for table in schema linking!")
            column_idx = random.randint(0, len(random_link["columns"]) - 1)
            column_name = random_link["columns"][column_idx]
            table_columns = self.get_table_columns(sample["db"], table_name)
            table_columns.remove(column_name)
            other_column = random.choice(table_columns)
            intervention = {"type": "column", "before": column_name, "after": other_column}
            random_link["columns"][column_idx] = other_column
        elif intervention_type == "table":
            # Table name
            table_names = sample["db"]["table_names_original"]
            table_names.remove(table_name)
            if table_names:
                other_table_name = random.choice(table_names)
                intervention = {"type": "table", "before": random_link["table"], "after": other_table_name}
                random_link["table"] = other_table_name

        return {"intervened_schema": intervened_schema, "intervention": intervention}


    def make_hsvt_intervention(self, sample: dict):
        pass

    def make_prompt(self, sample: dict):
        question = sample["question"]["en"]
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
        
        1. Schema Linking: Identify which words or phrases in the question refer to which tables or columns in the schema.
        2. SQL Generation: Write a correct, executable SQL query that answers the question, using the linked schema elements.
        
        Output Format
        Return a JSON object with two keys:
        - "schema_links": a list of [question_token, schema_element] pairs.
        Use "table" for table references.
        Use "table.column" for column references.
        Link values (numbers, dates, etc.) to the column they constrain.
        - "sql": a valid SQL query string (without markdown, without extra text).
        Only output valid JSON. Do not add explanations.

        Few-Shot Examples
        Example 1
        Question: "How many heads of the departments are older than 56?"
        
        Schema:
        Table: head
        Columns: head_id, name, age
        Table: department
        Columns: dept_id, name, budget
        
        Output:
        {{
          "schema_links": [
            {{
              "table": "head",
              "columns": ["age"]
            }}
          ],
          "sql": "SELECT COUNT(*) FROM head WHERE age > 56;"
        }}
        
        Example 2
        Question: "List the names of departments with budget over 1 million."
        
        Schema:
        Table: department
        Columns: dept_id, name, budget
        Table: employee
        Columns: emp_id, name, salary
        
        Output:
        {{
          "schema_links": [
            {{
              "table": "department",
              "columns": ["name", "budget"]
            }}
          ],
          "sql": "SELECT name FROM department WHERE budget > 1000000;"
        }}
        
        Example 3
        Question: "What is the average salary of employees hired in 2020?"
        
        Schema:
        Table: employee
        Columns: emp_id, name, salary, hire_date
        
        Output:
        {{
          "schema_links": [
            {{
              "table": "employee",
              "columns": ["salary", "hire_date"]
            }}
          ],
          "sql": "SELECT AVG(salary) FROM employee WHERE hire_date LIKE '2020%';"
        }}
        
        Now process the following:
        Question: "{question}"
        Schema:
        {db_schema}
        Output:
        """

        messages = [{"role": "user", "content": user_prompt}]
        add_generation_prompt_status = True

        prompt = self.llm_model.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt_status
        )

        return prompt


if __name__ == "__main__":
    response = '''
    Example 1
        Question: "How many heads of the departments are older than 56?"

        Schema:
        Table: head
        Columns: head_id, name, age
        Table: department
        Columns: dept_id, name, budget

        Output:
        {
          "schema_links": [
            {
              "table": "head",
              "columns": ["age"]
            }
          ],
          "sql": "SELECT COUNT(*) FROM head WHERE age > 56;"
        }
    '''
    # json_model_response = extract_json_from_model_response(response)
    # dataset = pauq_dataset.PAUQDataset("./pauq")
    # intervention = PAUQIntervention(dataset, None)
    # sample = dataset[0]
    # intervened = intervention.make_intervention(sample, json_model_response)
    # print(json_model_response)
    # print(intervened)
    sql_before = "SELECT name FROM students WHERE age > 20"
    sql_after = "SELECT full_name FROM students WHERE age > 20"

    info_before = extract_tables_and_columns(sql_before)
    info_after = extract_tables_and_columns(sql_after)

    print("Before:", info_before)
    print("After:", info_after)
