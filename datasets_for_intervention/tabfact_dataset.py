import os
import io
import json
import hashlib
import pandas as pd
import numpy as np
from copy import deepcopy

from datasets_for_intervention.tabfact_dsl_engine import TabFactEngine


class TabFactDataset:
    """
    Dataset for TabFact: table-based fact-checking.

    Each sample contains a table (CSV-formatted string), a natural-language claim,
    the gold DSL verifier query (mediator), and pre-computed local edits.

    Standard architecture keys:
        - gold_query:     gold DSL query string (the mediator M_gold)
        - mediator_query: copy of gold_query at load time; replaced during interventions
        - gold_target:    True for all samples (all main questions are entailed by construction)

    sample_id2local_edits stores only edits verified at load time to:
        (a) parse without error, and
        (b) execute to a result different from gold_target on the actual table.
    Each entry: {"query": str, "expected_target": bool}
    """

    def __init__(self, queries_json_path: str, tables_dir: str):
        self.tables_dir = tables_dir
        self._engine = TabFactEngine()   # used only at load time to filter local edits

        queries_data = self._load_queries(queries_json_path)
        table_id2content = self._load_tables(tables_dir, list(queries_data.keys()))

        self.sample_id2local_edits: dict = {}
        self.data: list = self._process(queries_data, table_id2content)

    def _load_queries(self, json_path: str) -> dict:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_tables(self, tables_dir: str, table_ids: list) -> dict:
        wanted = set(table_ids)
        table_dict = {}
        for filename in os.listdir(tables_dir):
            if filename in wanted:
                with open(os.path.join(tables_dir, filename), 'r', encoding='utf-8') as f:
                    table_dict[filename] = f.read().strip()
        return table_dict

    def _preprocess_text(self, text: str) -> str:
        if text is None:
            return ""
        return text.strip().replace('\n', ' ').replace('\r', ' ')

    def _generate_sample_id(self, table_id: str, statement: str) -> str:
        h = hashlib.md5(statement.encode('utf-8')).hexdigest()[:8]
        return f"{table_id.replace('.html.csv', '')}@{h}"

    def _process(self, queries_data: dict, table_id2content: dict) -> list:
        data = []

        for table_id, table_entries in queries_data.items():
            if table_id not in table_id2content:
                continue

            table_content = table_id2content[table_id]

            try:
                df = pd.read_csv(io.StringIO(table_content), sep="#", header=0, dtype=str)
            except Exception:
                df = None

            for entry in table_entries:
                statement   = self._preprocess_text(entry["main_question"])
                gold_query  = self._preprocess_text(entry["main_program"])
                gold_target = True   # all bootstrap main questions are entailed

                sample_id = self._generate_sample_id(table_id, statement)

                # Filter local edits at load time — keep only those that flip the result
                valid_local_edits = []
                if df is not None:
                    for raw in entry.get("local_edits", []):
                        q = self._preprocess_text(raw)
                        r = self._engine.execute(q, df)
                        if r.executable and r.final != gold_target:
                            valid_local_edits.append({
                                "query": q,
                                "expected_target": r.final,
                            })

                self.sample_id2local_edits[sample_id] = valid_local_edits

                data.append({
                    # standard architecture keys
                    "idx":            sample_id,
                    "gold_query":     gold_query,
                    "mediator_query": deepcopy(gold_query),
                    "gold_target":    gold_target,
                    # X fields
                    "table_id":       table_id,
                    "table_html_csv": table_content,
                    "statement":      statement,
                    "table_caption":  entry.get("table_name", ""),
                })

        return data

    def get_local_edits(self, sample: dict, n: int = None) -> list:
        """
        Return verified local-edit entries for this sample.

        Each entry: {"query": str, "expected_target": bool}

        Args:
            sample: a dataset sample dict (must have "idx").
            n:      if given, return a random subset of at most n edits.
                    If the pool is smaller than n, returns the whole pool.

        Returns [] if the sample has no valid local edits.
        """
        pool = self.sample_id2local_edits.get(sample["idx"], [])
        if n is None or n >= len(pool):
            return pool
        indices = np.random.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]