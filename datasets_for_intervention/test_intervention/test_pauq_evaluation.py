import unittest
from copy import deepcopy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_mocks import FakeLLMModel
from pauq_mocks import PAUQDatasetMock
from datasets_for_intervention.pauq_evaluation import PAUQEvaluation
import mock


class TestPAUQEvaluation(unittest.TestCase):
    def setUp(self):
        self.dataset = PAUQDatasetMock()
        self.evaluator = PAUQEvaluation(self.dataset)
        self.sample = deepcopy(self.dataset[0])

    def test_compare_schema_links_exact_match(self):
        gold = {"table1": ["col1", "col2"], "table2": ["col3"]}
        pred = {"table1": ["col2", "col1"], "table2": ["col3"]}
        self.assertTrue(self.evaluator.compare_schema_links(gold, pred))

    def test_compare_schema_links_missing_table(self):
        gold = {"table1": ["col1"], "table2": ["col2"]}
        pred = {"table1": ["col1"]}
        self.assertFalse(self.evaluator.compare_schema_links(gold, pred))

    def test_compare_schema_links_extra_column(self):
        gold = {"table1": ["col1"]}
        pred = {"table1": ["col1", "col2"]}
        self.assertFalse(self.evaluator.compare_schema_links(gold, pred))

    def test_compare_sql_queries_column_intervention(self):
        query_before = "SELECT col1 FROM table1 WHERE col2 > 10"
        query_after  = "SELECT col1 FROM table1 WHERE col3 > 10"
        intervention = {"type": "column", "before": "col2", "after": "col3"}
        db_schema = {
            "table1": ["col1", "col2", "col3"],
        }
        self.assertTrue(
            self.evaluator.compare_sql_queries(query_before, query_after, intervention, db_schema)
        )

    def test_compare_sql_queries_table_intervention(self):
        query_before = "SELECT col1 FROM table1"
        query_after  = "SELECT col1 FROM table2"
        intervention = {"type": "table", "before": "table1", "after": "table2"}
        db_schema = {
            "table1": ["col1"],
            "table2": ["col1"],
        }
        self.assertTrue(
            self.evaluator.compare_sql_queries(query_before, query_after, intervention, db_schema)
        )

    def test_validate_generated_sql_true(self):
        true_sql = "SELECT col1 FROM table1"
        gen_sql  = "SELECT col1 FROM table1"
        db_schema = {"dummy": "schema"}
        with mock.patch(
            "datasets_for_intervention.pauq_evaluation.validate_generated_sql",
            return_value=True
        ):
            self.assertTrue(
                self.evaluator.validate_generated_sql(true_sql, gen_sql, db_schema)
            )
