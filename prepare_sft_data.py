import argparse
import copy
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from datasets_for_intervention.pauq_dataset import PAUQDataset


def build_user_prompt(question: str, db: dict) -> str:
    db_schema = ""
    for i, table_name in enumerate(db["table_names_original"]):
        db_schema += f"Table: {table_name.lower()}\n"
        db_schema += "Columns: "
        for col_name in db["column_names_original"][1:]:
            if col_name[0] == i:
                db_schema += f"{col_name[1].lower()}, "
        db_schema = db_schema[:-2]
        db_schema += "\n"

    return (
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


def reconstruct_sql_from_skeleton(skeleton: str, slots: list) -> str:
    """Replace each SLOT_i in skeleton with slots[i-1] to produce executable SQL."""
    result = skeleton
    for i, slot_value in enumerate(slots, start=1):
        result = result.replace(f"SLOT_{i}", slot_value)
    return result


def update_slots(slots: list, name: str, new_name: str):
    """Replace first occurrence of name in slots with new_name (in-place)."""
    if name in slots:
        idx = slots.index(name)
        slots[idx] = new_name


def make_local_column_intervention(sample: dict, dataset: PAUQDataset):
    """Swap one column in schema_links/slots with another column from the same table.
    Returns modified sample copy or None if intervention is impossible."""
    s = copy.deepcopy(sample)
    schema_links = s["true_schema_links"]
    slots = s["true_slots"]

    table_name = random.choice(list(schema_links.keys()))
    columns = schema_links[table_name]
    if not columns:
        return None

    column_idx = random.randint(0, len(columns) - 1)
    column_name = columns[column_idx]

    table_columns = dataset.get_table_columns(s["db"], table_name)
    try:
        table_columns.remove(column_name)
    except ValueError:
        pass
    if not table_columns:
        return None

    other_column = random.choice(table_columns)
    columns[column_idx] = other_column
    update_slots(slots, column_name, other_column)

    s["true_schema_links"] = schema_links
    s["true_slots"] = slots
    s["query"] = reconstruct_sql_from_skeleton(s["true_skeleton"], slots)
    return s


def make_local_table_intervention(sample: dict, dataset: PAUQDataset):
    """Swap one table name in schema_links/slots with another table from the DB.
    Returns modified sample copy or None if intervention is impossible."""
    s = copy.deepcopy(sample)
    schema_links = s["true_schema_links"]
    slots = s["true_slots"]

    table_name = random.choice(list(schema_links.keys()))
    table_names = list(s["db"]["table_names_original"])
    try:
        table_names.remove(table_name)
    except ValueError:
        pass
    if not table_names:
        return None

    other_table_name = random.choice(table_names)
    columns = schema_links[table_name]
    del schema_links[table_name]
    schema_links[other_table_name] = columns
    update_slots(slots, table_name, other_table_name)

    s["true_schema_links"] = schema_links
    s["true_slots"] = slots
    s["query"] = reconstruct_sql_from_skeleton(s["true_skeleton"], slots)
    return s


def make_global_intervention(sample: dict, dataset: PAUQDataset):
    """Replace all tables/columns with dummy equivalents.
    Returns modified sample copy or None if intervention is impossible."""
    s = copy.deepcopy(sample)
    schema_links = s["true_schema_links"]
    slots = s["true_slots"]
    dummy_tables = dataset.dummy_tables

    num_tables = len(schema_links)
    if num_tables > len(dummy_tables):
        return None

    random_tables = random.sample(list(dummy_tables.keys()), num_tables)
    new_schema_links = {}

    for random_table, table_name in zip(random_tables, list(schema_links.keys())):
        num_columns = len(schema_links[table_name])
        available_columns = dummy_tables[random_table]
        if num_columns > len(available_columns):
            return None

        random_columns = random.sample(available_columns, num_columns)
        new_schema_links[random_table] = random_columns

        for old_column, new_column in zip(schema_links[table_name], random_columns):
            update_slots(slots, old_column, new_column)
        update_slots(slots, table_name, random_table)

    s["true_schema_links"] = new_schema_links
    s["true_slots"] = slots
    s["query"] = reconstruct_sql_from_skeleton(s["true_skeleton"], slots)
    return s


INTERVENTION_FUNCS = {
    "local_column": make_local_column_intervention,
    "local_table": make_local_table_intervention,
    "global": make_global_intervention,
}


def build_assistant_response(sample: dict) -> str:
    lines = ["===SKELETON===", sample["true_skeleton"], "===SCHEMA_LINKS==="]
    for table_name, columns in sample["true_schema_links"].items():
        lines.append(f"{table_name}:{','.join(columns)}")
    lines.append("===SLOT_MATCHING===")
    for i, slot in enumerate(sample["true_slots"]):
        lines.append(f"SLOT_{i + 1}:{slot}")
    lines.append("===SQL===")
    lines.append(sample["query"])
    return "\n".join(lines)


def make_record(sample, tokenizer=None):
    user_content = build_user_prompt(sample["question"], sample["db"])
    assistant_content = build_assistant_response(sample)
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    record = {"messages": messages}
    if tokenizer is not None:
        record["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return record


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from PAUQ dataset")
    parser.add_argument("--data-path", required=True, help="Path to pauq/ folder")
    parser.add_argument("--output", required=True, help="Output .jsonl file path")
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--tokenizer", default=None, help="HuggingFace tokenizer name/path for 'text' field")
    parser.add_argument("--include-interventions", action="store_true",
                        help="Add intervened samples to increase faithfulness")
    parser.add_argument("--intervention-types", default="local_column,local_table,global",
                        help="Comma-separated intervention types (default: local_column,local_table,global)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for interventions")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = PAUQDataset(args.data_path, train=(args.split == "train"))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    intervention_types = []
    if args.include_interventions:
        random.seed(args.seed)
        intervention_types = [t.strip() for t in args.intervention_types.split(",")]
        for t in intervention_types:
            if t not in INTERVENTION_FUNCS:
                parser.error(f"Unknown intervention type: {t}. Choose from: {list(INTERVENTION_FUNCS.keys())}")

    total_original = 0
    total_intervened = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for sample in dataset:
            # Original gold sample
            record = make_record(sample, tokenizer)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_original += 1

            # Intervened samples
            for itype in intervention_types:
                intervened = INTERVENTION_FUNCS[itype](sample, dataset)
                if intervened is None:
                    continue
                record = make_record(intervened, tokenizer)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_intervened += 1

    print(f"Split:      {args.split}")
    print(f"Original:   {total_original}")
    print(f"Intervened: {total_intervened}")
    print(f"Total:      {total_original + total_intervened}")
    print(f"Output:     {args.output}")


if __name__ == "__main__":
    main()
