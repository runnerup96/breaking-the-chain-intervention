from utils import extract_tables_and_columns


class PAUQEvaluation:
    def __init__(self, dataset=None):
        self.dataset = dataset

    def compare_sql_queries(self, query_before: str, query_after: str, intervention: dict[str, str]) -> bool:
        type = intervention["type"]
        before = intervention["before"]
        after = intervention["after"]

        parsed_query_before = extract_tables_and_columns(query_before)
        parsed_query_after = extract_tables_and_columns(query_after)

        if before not in parsed_query_before[type]:
            raise ValueError("Query before and intervention do not match!")

        if before in parsed_query_after[type]:
            return False

        before_idx = parsed_query_before[type].index(before)
        parsed_query_before[type][before_idx] = after

        return set(parsed_query_after[type]) == set(parsed_query_before[type])

    def evaluate(self):
        pass


if __name__ == "__main__":
    sql_before = "SELECT name FROM students WHERE age > 20"
    sql_after = "SELECT full_name FROM students WHERE age > 20"
    intervention = {"type": "columns", "before": "name", "after": "full_name"}

    info_before = extract_tables_and_columns(sql_before)
    info_after = extract_tables_and_columns(sql_after)

    print("Before:", info_before)
    print("After:", info_after)

    eval = PAUQEvaluation()
    print(eval.compare_sql_queries(sql_before, sql_after, intervention))
