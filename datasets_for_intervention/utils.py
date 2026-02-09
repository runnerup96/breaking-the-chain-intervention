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
    # _LIT = re.compile(r'__([^_]+)__')
    _LIT = re.compile(r'__(.+?)__')

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

    table_names_lc = {t.lower() for t in table_names_original}
    column_names_lc = {c.lower() for c in column_names_original}
    table_dot_column_names_lc = {tc.lower() for tc in table_dot_column_names_original}

    parsed_sql = Parser(sql)
    masked_tokens = []
    slots = []
    i = 1
    for tok in parsed_sql.tokens:
        val = tok.value.strip()
        if not val:
            continue

        if val.startswith('"') and val.endswith('"') and len(val) >= 2:
            inner = val[1:-1]
            inner_lc = inner.lower()
            # Quoted identifier
            if inner_lc in table_names_lc or inner_lc in column_names_lc or inner_lc in table_dot_column_names_lc:
                val = inner
            else:
                # Treat as a literal string (keep quotes as in original).
                masked_tokens.append(f"SLOT_{i}")
                i += 1
                slots.append(val)
                continue

        val_lc = val.lower()
        if val_lc in table_names_lc:
            masked_tokens.append(f"SLOT_{i}")
            i += 1
            slots.append(val)
        elif val_lc in column_names_lc or val_lc in table_dot_column_names_lc:
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


def parse_model_response(output_str: str):
    lines = [ln.rstrip() for ln in output_str.strip().splitlines()]

    sections = {"SKELETON": [], "SCHEMA_LINKS": [], "SLOT_MATCHING": [], "SQL": []}
    current = None

    header_map = {
        "===SKELETON===": "SKELETON",
        "===SCHEMA_LINKS===": "SCHEMA_LINKS",
        "===SLOT_MATCHING===": "SLOT_MATCHING",
        "===SQL===": "SQL",
    }

    for ln in lines:
        s = ln.strip()
        if s in header_map:
            current = header_map[s]
            continue
        if current is not None:
            if s != "":
                sections[current].append(s)

    skeleton = "\n".join(sections["SKELETON"]).strip() or None
    sql = "\n".join(sections["SQL"]).strip() or None

    schema_links = {}
    for row in sections["SCHEMA_LINKS"]:
        if ":" not in row:
            continue
        table, cols = row.split(":", 1)
        table = table.strip()
        cols_list = [c.strip() for c in cols.split(",") if c.strip()]
        if table:
            schema_links[table] = cols_list

    schema_links_list = [{"table": t, "columns": cols} for t, cols in schema_links.items()]

    slot_dict = {}
    for row in sections["SLOT_MATCHING"]:
        if ":" not in row:
            continue
        name, val = row.split(":", 1)
        name = name.strip()
        m = re.fullmatch(r"SLOT_(\d+)", name)
        if not m:
            continue
        idx = int(m.group(1))
        slot_dict[idx] = val.strip()

    slots = []
    if slot_dict:
        min_k = min(slot_dict.keys())
        max_k = max(slot_dict.keys())
        start = 0 if min_k == 0 else 1
        for i in range(start, max_k + 1):
            if i in slot_dict:
                slots.append(slot_dict[i])

    if sql is None:
        sql = ""
        
    return {
        "sql": sql,
        "schema_links": schema_links_list,
        "slots": slots,
        "skeleton": skeleton,
    }


def compare_schema_links(true_schema_links: dict, generated_schema_links: dict) -> bool:
    true_tables = set(true_schema_links.keys())
    generated_tables = set(generated_schema_links.keys())
    if true_tables != generated_tables:
        return False
    
    for table_name in generated_schema_links:
        if set(true_schema_links[table_name]) != set(generated_schema_links[table_name]):
            return False
        
    return True
    

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

    # sql1 = "select t2.name, t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id where t1.year > 2013 group by t2.stadium_id order by count ( 1 ) desc limit *;"
    # sql2 = "select t2.name, t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id where t1.year > 2013 group by t2.stadium_id order by count ( 1 ) desc limit *;"
    # print(sql1 == sql2)
    # print(validate_generated_sql(sql1, sql2, {'stadium': ['stadium_id', 'location', 'name', 'capacity', 'highest', 'lowest', 'average'], 'singer': ['singer_id', 'name', 'country', 'song_name', 'song_release_year', 'age', 'is_male'], 'concert': ['concert_id', 'concert_name', 'theme', 'stadium_id', 'year'], 'singer_in_concert': ['concert_id', 'singer_id']}))
