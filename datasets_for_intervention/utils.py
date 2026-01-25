import re
import json
from typing import List, Dict, Any
import sys
from sql_metadata import Parser

sys.path.append("/home/jovyan/kmvafin/research/test-suite-sql-eval")
import evaluation


def parse_sql(query, db_schema):
    """
    Parse an SQL query into its components.

    Args:
        query (str): The SQL query.
        db_path (str): Path to the database.

    Returns:
        dict: Parsed SQL query components.
    """
    schema = evaluation.Schema(db_schema)
    try:
        parsed_query = evaluation.get_sql(schema, query)
    except:
        parsed_query = {
            "except": None,
            "from": {
                "conds": [],
                "table_units": []
            },
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [
                False,
                []
            ],
            "union": None,
            "where": []
        }
    return parsed_query


def extract_schema_links(parsed_sql: dict) -> dict:
    """
    Возвращает schema_links в формате
        {"table": ["col1", "col2", ...], ...}
    """
    _LIT = re.compile(r'__([^_]+)__')

    tc_pairs = set()

    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)
        elif isinstance(obj, str):
            for lit in _LIT.findall(obj):
                if '.' in lit:
                    tbl, col = lit.split('.', 1)
                    tc_pairs.add((tbl, col))
                else:
                    tc_pairs.add((lit, '*'))

    _walk(parsed_sql)

    schema_links = {}
    for tbl, col in tc_pairs:
        if col != '*':
            schema_links.setdefault(tbl, []).append(col)

    return {t: sorted(set(cols)) for t, cols in schema_links.items()}


def extract_data_from_response(model_response: str) -> Dict[str, Any]:
    """
    Извлекает schema-links и SQL из ответа модели.
    Возвращает:
        {
          "sql": "SELECT ...",
          "schema_links": {"table1": ["col1", "col2"], ...}
        }
    """
    m = re.search(
        r'===SCHEMA_LINKS===(.*?)===SQL===(.+)',
        model_response,
        flags=re.S
    )
    if not m:
        raise ValueError("Не найдены блоки SCHEMA_LINKS / SQL")

    links_raw = m.group(1).strip()
    sql = m.group(2).strip()

    schema_links: Dict[str, Any] = {}
    for line in links_raw.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        table, cols = line.split(':', 1)
        cols_list = [c.strip() for c in cols.split(',') if c.strip()]
        if table.strip():
            schema_links[table.strip()] = cols_list
    schema_links_list = [{"table": table_name, "columns": columns} for table_name, columns in schema_links.items()]

    return {"sql": sql, "schema_links": schema_links_list}


def validate_generated_sql(true_sql: str, generated_sql: str, db_schema: dict) -> bool:
    """
    Валидация SQL через Spider-official evaluator.
    Возвращает True, если структуры запросов совпадают (с учётом всех
    ограничений Spider-оценщика: порядок колонок, регистр, скобки и т.д.).
    """
    schema = evaluation.Schema(db_schema)

    try:
        true_parsed = evaluation.get_sql(schema, true_sql)
        pred_parsed = evaluation.get_sql(schema, generated_sql)
    except Exception:
        return False

    return true_parsed == pred_parsed


def convert_db_schema(db_schema):
    result = {}
    result["schema_items"] = []
    for table in db_schema:
        result["schema_items"].append({"table_name_original": table, "column_names_original": db_schema[table]})
    return result


def isNegativeInt(string):
    if string.startswith("-") and string[1:].isdigit():
        return True
    else:
        return False


def isFloat(string):
    if string.startswith("-"):
        string = string[1:]
    
    s = string.split(".")
    if len(s)>2:
        return False
    else:
        for s_i in s:
            if not s_i.isdigit():
                return False
        return True
        
    
def extract_skeleton_and_slots(sql: str, db_schema: dict) -> str:
    table_names_original, column_names_original, table_dot_column_names_original = [], [], []
    db_schema = convert_db_schema(db_schema)
    
    for table in db_schema["schema_items"]:
        t_name = table["table_name_original"]
        table_names_original.append(t_name)
        for col in ["*"] + table["column_names_original"]:
            table_dot_column_names_original.append(f"{t_name}.{col}")
            column_names_original.append(col)

    parsed_sql = Parser(sql)
    masked_tokens = []
    slots = []
    i = 1
    for tok in parsed_sql.tokens:
        val = tok.value.strip()
        if not val:
            continue

        if val in table_names_original:
            masked_tokens.append(f"SLOT_{i}")
            i += 1
            slots.append(val)
        elif val in column_names_original or val in table_dot_column_names_original:
            masked_tokens.append(f"SLOT_{i}")
            i += 1
            slots.append(val)
        elif val.startswith("'") and val.endswith("'"):
            masked_tokens.append(f"SLOT_{i}")
            i += 1
            slots.append(val)
        elif val.isdigit() or isNegativeInt(val) or isFloat(val):
            masked_tokens.append(f"SLOT_{i}")
            i += 1
            slots.append(val)
        else:
            masked_tokens.append(val)

    skeleton = " ".join(masked_tokens)

    while "  " in skeleton:
        skeleton = skeleton.replace("  ", " ")
    while " ," in skeleton:
        skeleton = skeleton.replace(" ,", ",")
    while " ;" in skeleton:
        skeleton = skeleton.replace(" ;", ";")
    return skeleton.strip(), slots


def parse_model_response(output_str):
    """
    Парсит строку с результатом и извлекает skeleton, schema_links, slot_matching и sql.
    
    Args:
        output_str (str): Строка в формате с метками ===SKELETON===, ===SCHEMA_LINKS=== и т.д.
    
    Returns:
        dict: Словарь с ключами 'sql', 'schema_links', 'slots', 'skeleton'
    """
    lines = output_str.strip().split('\n')
    
    skeleton = None
    schema_links_str = None
    slot_matching_lines = []
    sql = None
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('==='):
            if 'SKELETON===' in line:
                current_section = 'SKELETON'
            elif 'SCHEMA_LINKS===' in line:
                current_section = 'SCHEMA_LINKS'
            elif 'SLOT_MATCHING===' in line:
                current_section = 'SLOT_MATCHING'
            elif 'SQL===' in line:
                current_section = 'SQL'
            else:
                current_section = None
        elif current_section:
            if line:
                if current_section == 'SKELETON' and skeleton is None:
                    skeleton = line
                elif current_section == 'SCHEMA_LINKS' and schema_links_str is None:
                    schema_links_str = line
                elif current_section == 'SLOT_MATCHING':
                    slot_matching_lines.append(line)
                elif current_section == 'SQL' and sql is None:
                    sql = line
    
    schema_links = {}
    if schema_links_str:
        if ':' in schema_links_str:
            table, columns = schema_links_str.split(':', 1)
            table = table.strip()
            columns_list = [col.strip() for col in columns.split(',')]
            schema_links[table] = columns_list
    
    slots = []
    if slot_matching_lines:
        slot_dict = {}
        for slot_line in slot_matching_lines:
            if ':' in slot_line:
                slot_name, slot_value = slot_line.split(':', 1)
                if slot_name.startswith('SLOT_'):
                    slot_num = int(slot_name.replace('SLOT_', ''))
                    slot_dict[slot_num] = slot_value.strip()
        for i in range(1, max(slot_dict.keys()) + 1):
            if i in slot_dict:
                slots.append(slot_dict[i])

    schema_links_list = [{"table": table_name, "columns": columns} for table_name, columns in schema_links.items()]
    
    result = {
        'sql': sql,
        'schema_links': schema_links_list,
        'slots': slots,
        'skeleton': skeleton
    }
    
    return result
    

if __name__ == '__main__':
    query = "SELECT name, email FROM users;"
    query2 = "SELECT name, email FROM users"
    db_schema = {
        "users": ["id", "name", "email"]
    }
    parsed = parse_sql(query, db_schema)
    print(parsed)
    print(extract_schema_links(parsed))
    print(validate_generated_sql(query, query2, db_schema))

    print(extract_skeleton_and_slots(query, {"users": ["name", "email"]}))
    test_string = f"Output:\n" \
                      f"===SKELETON===\n" \
                      f"SELECT COUNT(SLOT_1) FROM SLOT_2 WHERE SLOT_3 > SLOT_4;\n" \
                      f"===SCHEMA_LINKS===\n" \
                      f"head:age\n" \
                      f"===SLOT_MATCHING===\n" \
                      f"SLOT_1:*\n" \
                      f"SLOT_2:head\n" \
                      f"SLOT_3:age\n" \
                      f"SLOT_4:56\n" \
                      f"===SQL===\n" \
                      f"SELECT COUNT(*) FROM head WHERE age > 56;\n\n"
    
    result = parse_model_response(test_string)
    print("Результат парсинга:")
    print(result)
    print()
    
    # Проверяем соответствие ожидаемому результату
    expected = {
        'sql': "SELECT COUNT(*) FROM head WHERE age > 56;",
        'schema_links': {"head": ["age"]},
        'slots': ['*', 'head', 'age', '56'],
        'skeleton': "SELECT COUNT(SLOT_1) FROM SLOT_2 WHERE SLOT_3 > SLOT_4;"
    }
    
    print("Ожидаемый результат:")
    print(expected)
    print()
    print("Результаты совпадают?", result == expected)
