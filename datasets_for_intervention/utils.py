import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Function, Where
from sqlparse.tokens import Keyword, DML
import re
import json


def extract_data_from_response(model_response: str):
    """
    Извлекает SQL запрос и schema links из ответа модели.
    
    Args:
        model_response: Ответ модели в формате:
            Schema links: [{...}]
            SQL: SELECT ...;
    
    Returns:
        Словарь с ключами (sql_query, schema_links_list)
    """
    # Найти начало Schema links
    schema_start = model_response.find("Schema links:")
    if schema_start == -1:
        raise ValueError("Schema links not found in the response")
    
    # Найти начало SQL
    sql_start = model_response.find("SQL:")
    if sql_start == -1:
        raise ValueError("SQL query not found in the response")
    
    # Извлечь текст между "Schema links:" и "SQL:"
    schema_json_str = model_response[schema_start + len("Schema links:"):sql_start].strip()
    
    # Теперь попробуем распарсить как JSON
    try:
        schema_links_list = json.loads(schema_json_str)
    except json.JSONDecodeError:
        # Попытка исправить одинарные кавычки
        try:
            schema_json_str = schema_json_str.replace("'", '"')
            schema_links_list = json.loads(schema_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse schema links as JSON: {e}")

    # Извлечь SQL
    sql_part = model_response[sql_start + len("SQL:"):]
    sql_query = sql_part.strip()
    # Удалить возможный завершающий ; и/или лишние пробелы — по желанию
    # Но обычно оставляем как есть

    return {
        "sql": sql_query,
        "schema_links": schema_links_list
    }


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


def _is_subselect(parsed):
    if not parsed.is_group:
        return False
    return any(token.ttype is DML and token.value.upper() == "SELECT"
               for token in parsed.tokens)


def _extract_from_part(statement):
    """
    Возвращает токены, соответствующие части FROM ... (включая JOIN).
    """
    from_seen = False
    from_tokens = []
    for token in statement.tokens:
        if from_seen:
            if token.ttype is Keyword and token.value.upper() in (
                "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "UNION", "INTERSECT", "EXCEPT"
            ):
                break
            from_tokens.append(token)
        if token.ttype is Keyword and token.value.upper() == "FROM":
            from_seen = True
    return from_tokens


def _get_tables(statement):
    """
    Возвращает set с именами таблиц (без алиасов).
    """
    tables = set()
    from_tokens = _extract_from_part(statement)

    def extract_from_identifier(idt: Identifier):
        real_name = idt.get_real_name() or idt.get_name()
        if real_name:
            tables.add(real_name)

    for token in from_tokens:
        if isinstance(token, IdentifierList):
            for idt in token.get_identifiers():
                if isinstance(idt, Identifier):
                    extract_from_identifier(idt)
        elif isinstance(token, Identifier):
            extract_from_identifier(token)
        elif _is_subselect(token):
            # подзапросы во FROM
            for sub in token.tokens:
                if isinstance(sub, sqlparse.sql.Statement):
                    tables |= _get_tables(sub)

    return tables


def _extract_select_part(statement):
    """
    Возвращает токены, соответствующие части SELECT ... (до FROM).
    """
    select_seen = False
    select_tokens = []
    for token in statement.tokens:
        if token.ttype is DML and token.value.upper() == "SELECT":
            select_seen = True
            continue
        if select_seen:
            if token.ttype is Keyword and token.value.upper() == "FROM":
                break
            select_tokens.append(token)
    return select_tokens


def _get_columns_from_select(statement):
    """
    Возвращает set имён колонок, встречающихся в SELECT.
    """
    select_tokens = _extract_select_part(statement)
    cols = set()

    for token in select_tokens:
        if isinstance(token, IdentifierList):
            for idt in token.get_identifiers():
                if isinstance(idt, Identifier):
                    col = idt.get_real_name() or idt.get_name()
                    if col:
                        cols.add(col)
        elif isinstance(token, Identifier):
            col = token.get_real_name() or token.get_name()
            if col:
                cols.add(col)
        elif isinstance(token, Function):
            # Например COUNT(col) -> попробуем вытащить аргументы
            for t in token.tokens:
                if isinstance(t, Identifier):
                    col = t.get_real_name() or t.get_name()
                    if col:
                        cols.add(col)
        # остальное (звёздочки, литералы) игнорируем
    return cols


def _get_where_token(statement):
    for token in statement.tokens:
        if isinstance(token, Where):
            return token
    return None


def _get_columns_from_where(where_token):
    """
    Возвращает set имён колонок, встречающихся в WHERE.
    """
    if where_token is None:
        return set()

    cols = set()

    def walk(token):
        if isinstance(token, IdentifierList):
            for t in token.get_identifiers():
                walk(t)
        elif isinstance(token, Identifier):
            col = token.get_real_name() or token.get_name()
            if col:
                cols.add(col)
        elif token.is_group:
            for t in token.tokens:
                walk(t)

    walk(where_token)
    return cols


def extract_tables_and_columns(sql: str):
    """
    Парсит SQL-запрос и возвращает словарь вида:
    {
        "tables": [...],
        "columns": [...],
    }

    Таблицы берутся из FROM / JOIN.
    Колонки — из SELECT и WHERE (можно расширить при необходимости).
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return {"tables": [], "columns": []}

    statement = parsed[0]

    tables = _get_tables(statement)
    cols_select = _get_columns_from_select(statement)
    where_token = _get_where_token(statement)
    cols_where = _get_columns_from_where(where_token)

    columns = cols_select | cols_where

    return {
        "tables": sorted(tables),
        "columns": sorted(columns),
    }
