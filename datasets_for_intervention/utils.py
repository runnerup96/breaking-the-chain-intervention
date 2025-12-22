import re
import json
from typing import List, Dict, Any
import sys

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

    
# def extract_data_from_response(model_response: str) -> Dict[str, Any]:
#     schema_pattern = r"Schema links:\s*(\[[\s\S]*?\])"
#     schema_match = re.search(schema_pattern, model_response)

#     raw_schema = schema_match.group(1) if schema_match else "[]"

#     try:
#         schema_links = json.loads(raw_schema)
#     except Exception:
#         cleaned = raw_schema
#         cleaned = cleaned.replace("'", '"')
#         cleaned = cleaned.replace("“", '"').replace("”", '"')
#         cleaned = cleaned.replace("‘", '"').replace("’", '"')

#         cleaned = re.sub(r",\s*}", "}", cleaned)
#         cleaned = re.sub(r",\s*]", "]", cleaned)

#         try:
#             schema_links = json.loads(cleaned)
#         except Exception:
#             schema_links = fallback_extract_schema_links(cleaned)

#     sql_pattern = r"SQL:\s*(SELECT[\s\S]*?;)"
#     sql_match = re.search(sql_pattern, model_response, re.IGNORECASE)
#     sql_query = sql_match.group(1).strip() if sql_match else ""

#     return {
#         "sql": sql_query,
#         "schema_links": schema_links,
#     }


# def fallback_extract_schema_links(text: str) -> List[Dict[str, Any]]:
#     results = []

#     entry_pattern = r'table"\s*:\s*"([^"]+)"[\s\S]*?columns"\s*:\s*\[([^\]]*)\]'
#     for match in re.finditer(entry_pattern, text):
#         table = match.group(1)
#         cols_raw = match.group(2).strip()

#         cols = re.findall(r'"([^"]+)"', cols_raw)
#         results.append({"table": table, "columns": cols})

#     return results


# def extract_json_from_model_response(model_response: str) -> dict | None:
#     json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", model_response, re.DOTALL)
#     if json_match:
#         json_str = json_match.group(1)
#     else:
#         json_match = re.search(r"({.*})", model_response, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(1)
#         else:
#             return None

#     try:
#         data = json.loads(json_str)
#         return data
#     except (json.JSONDecodeError, TypeError):
#         return None


# from process_sql import get_sql, Schema

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

# def _is_subselect(parsed):
#     if not parsed.is_group:
#         return False
#     return any(token.ttype is DML and token.value.upper() == "SELECT"
#                for token in parsed.tokens)


# def _extract_from_part(statement):
#     """
#     Возвращает токены, соответствующие части FROM ... (включая JOIN).
#     """
#     from_seen = False
#     from_tokens = []
#     for token in statement.tokens:
#         if from_seen:
#             if token.ttype is Keyword and token.value.upper() in (
#                 "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "UNION", "INTERSECT", "EXCEPT"
#             ):
#                 break
#             from_tokens.append(token)
#         if token.ttype is Keyword and token.value.upper() == "FROM":
#             from_seen = True
#     return from_tokens


# def _get_tables(statement):
#     """
#     Возвращает set с именами таблиц (без алиасов).
#     """
#     tables = set()
#     from_tokens = _extract_from_part(statement)

#     def extract_from_identifier(idt: Identifier):
#         real_name = idt.get_real_name() or idt.get_name()
#         if real_name:
#             tables.add(real_name)

#     for token in from_tokens:
#         if isinstance(token, IdentifierList):
#             for idt in token.get_identifiers():
#                 if isinstance(idt, Identifier):
#                     extract_from_identifier(idt)
#         elif isinstance(token, Identifier):
#             extract_from_identifier(token)
#         elif _is_subselect(token):
#             # подзапросы во FROM
#             for sub in token.tokens:
#                 if isinstance(sub, sqlparse.sql.Statement):
#                     tables |= _get_tables(sub)

#     return tables


# def _extract_select_part(statement):
#     """
#     Возвращает токены, соответствующие части SELECT ... (до FROM).
#     """
#     select_seen = False
#     select_tokens = []
#     for token in statement.tokens:
#         if token.ttype is DML and token.value.upper() == "SELECT":
#             select_seen = True
#             continue
#         if select_seen:
#             if token.ttype is Keyword and token.value.upper() == "FROM":
#                 break
#             select_tokens.append(token)
#     return select_tokens


# def _get_columns_from_select(statement):
#     """
#     Возвращает set имён колонок, встречающихся в SELECT.
#     """
#     select_tokens = _extract_select_part(statement)
#     cols = set()

#     for token in select_tokens:
#         if isinstance(token, IdentifierList):
#             for idt in token.get_identifiers():
#                 if isinstance(idt, Identifier):
#                     col = idt.get_real_name() or idt.get_name()
#                     if col:
#                         cols.add(col)
#         elif isinstance(token, Identifier):
#             col = token.get_real_name() or token.get_name()
#             if col:
#                 cols.add(col)
#         elif isinstance(token, Function):
#             # Например COUNT(col) -> попробуем вытащить аргументы
#             for t in token.tokens:
#                 if isinstance(t, Identifier):
#                     col = t.get_real_name() or t.get_name()
#                     if col:
#                         cols.add(col)
#         # остальное (звёздочки, литералы) игнорируем
#     return cols


# def _get_where_token(statement):
#     for token in statement.tokens:
#         if isinstance(token, Where):
#             return token
#     return None


# def _get_columns_from_where(where_token):
#     """
#     Возвращает set имён колонок, встречающихся в WHERE.
#     """
#     if where_token is None:
#         return set()

#     cols = set()

#     def walk(token):
#         if isinstance(token, IdentifierList):
#             for t in token.get_identifiers():
#                 walk(t)
#         elif isinstance(token, Identifier):
#             col = token.get_real_name() or token.get_name()
#             if col:
#                 cols.add(col)
#         elif token.is_group:
#             for t in token.tokens:
#                 walk(t)

#     walk(where_token)
#     return cols


# def extract_tables_and_columns(sql: str):
#     """
#     Парсит SQL-запрос и возвращает словарь вида:
#     {
#         "tables": [...],
#         "column": [...],
#     }

#     Таблицы берутся из FROM / JOIN.
#     Колонки — из SELECT и WHERE (можно расширить при необходимости).
#     """
#     parsed = sqlparse.parse(sql)
#     if not parsed:
#         return {"tables": [], "column": []}

#     statement = parsed[0]

#     tables = _get_tables(statement)
#     cols_select = _get_columns_from_select(statement)
#     where_token = _get_where_token(statement)
#     cols_where = _get_columns_from_where(where_token)

#     columns = cols_select | cols_where

#     return {
#         "tables": sorted(tables),
#         "column": sorted(columns),
#     }


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
