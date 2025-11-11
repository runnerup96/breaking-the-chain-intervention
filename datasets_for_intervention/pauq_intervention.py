import copy
import json
import random
import re
import pauq_dataset


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

    def make_intervention(self, sample: dict, json_model_response: dict):
        intervention = copy.deepcopy(json_model_response)
        random_link = random.choice(intervention["schema_links"])
        table_element = random_link[1]
        # check the output correctness
        # global intervention: change the full db schema
        # HSVT: change the text question
        # SQL Parse
        if '.' in table_element:
            # Column name
            table_name, column_name = table_element.split(".")
            table_columns = self.get_table_columns(sample["db"], table_name)
            table_columns.remove(column_name)
            other_column = random.choice(table_columns)
            random_link[1] = f"{table_name}.{other_column}"
        else:
            # Table name
            table_name = table_element
            table_names = sample["db"]["table_names_original"]
            table_names.remove(table_name)
            if table_names:
                other_table_name = random.choice(table_names)
                random_link[1] = other_table_name
        return intervention

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
        #TODO fix schema linking: filter db schema
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
            ["heads", "head"],
            ["departments", "department"],
            ["older", "head.age"],
            ["56", "head.age"]
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
            ["names", "department.name"],
            ["departments", "department"],
            ["budget", "department.budget"],
            ["1 million", "department.budget"]
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
            ["salary", "employee.salary"],
            ["employees", "employee"],
            ["hired", "employee.hire_date"],
            ["2020", "employee.hire_date"]
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
            ["heads", "head"],
            ["departments", "department"],
            ["older", "head.age"],
            ["56", "head.age"]
          ],
          "sql": "SELECT COUNT(*) FROM head WHERE age > 56;"
        }
    '''
    json_model_response = extract_json_from_model_response(response)
    dataset = pauq_dataset.PAUQDataset("./pauq")
    intervention = PAUQIntervention(dataset, None)
    sample = dataset[0]
    intervened = intervention.make_intervention(sample, json_model_response)
    print(json_model_response)
    print(intervened)
