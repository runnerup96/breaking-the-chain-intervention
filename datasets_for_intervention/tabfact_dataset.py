import os
import json
import hashlib
import pandas as pd
import numpy as np

class TabFactDataset:
    def __init__(self, queries_json_path: str, tables_dir: str):
        self.tables_dir = tables_dir
        self.queries_json_path = queries_json_path
        self.data = None
        
        self.queries_data = self._load_queries(queries_json_path)
        self.table_id2content = self._load_tables(tables_dir, list(self.queries_data.keys()))

        self.table_id2alt_questions = {}
        self.table_id2alt_programs = {}
        self.sample_id2local_edits = {}

        self.process_data()

    def _load_queries(self, json_path: str) -> dict:
        with open(json_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        return queries

    def _load_tables(self, tables_dir: str, table_ids: list) -> dict:
        table_dict = {}
        for filename in os.listdir(tables_dir):
            if filename in table_ids:
                table_id = filename
                file_path = os.path.join(tables_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    table_content = f.read().strip()
                table_dict[table_id] = table_content
        return table_dict
    
    def _preprocess_text(self, text: str) -> str:
        if text is None:
            return ""
        return text.strip().replace('\n', ' ').replace('\r', ' ')
    
    def _generate_sample_id(self, table_id: str, statement: str) -> str:
        statement_hash = hashlib.md5(statement.encode('utf-8')).hexdigest()[:8]
        return f"{table_id.replace('.html.csv', '')}@{statement_hash}"
    
    def _parse_table_for_distractors(self, table_path: str) -> dict:
        try:
            df = pd.read_csv(table_path, sep='#', header=0, dtype=str)
            columns = df.columns.tolist()

            column_values = {}
            all_entities = []
            for col in columns:
                unique_vals = df[col].dropna().astype(str).str.strip()
                unique_vals = unique_vals[unique_vals != ''].unique().tolist()
                column_values[col] = unique_vals
                all_entities.extend(unique_vals)

            entity_swaps = list(set(all_entities))

            return {
                "columns": columns,
                "values": column_values,
                "entity_swaps": entity_swaps
            }

        except Exception as e:
            print(f"Warning: Failed to parse table {table_path} with pandas: {e}")
            return {
                "columns": [],
                "values": {},
                "entity_swaps": []
            }

    
    def process_data(self):
        self.data = []

        for table_id, table_entries in self.queries_data.items():
            if table_id not in self.table_id2content:
                continue

            table_content = self.table_id2content[table_id]

            table_path = os.path.join(self.tables_dir, table_id)
            distractors = self._parse_table_for_distractors(table_path)

            all_alt_questions = []
            all_alt_programs = []

            for entry in table_entries:
                main_question = self._preprocess_text(entry["main_question"])
                table_caption = entry["table_name"]
                main_program = self._preprocess_text(entry["main_program"])
                label_gt = True

                alt_questions = [self._preprocess_text(q) for q in entry.get("alternate_questions", [])]
                alt_programs = [self._preprocess_text(p) for p in entry.get("alternate_programs", [])]
                local_edits = [self._preprocess_text(p) for p in entry.get("local_edits", [])]

                all_alt_questions.extend(alt_questions)
                all_alt_programs.extend(alt_programs)

                sample_id = self._generate_sample_id(table_id, main_question)

                sample = {
                    "idx": sample_id,
                    "table_id": table_id,
                    "table_html_csv": table_content,
                    "statement": main_question,
                    "table_caption": table_caption,
                    "verifier_query_gt": main_program,
                    "label_gt": label_gt,
                    "distractors": distractors,
                }
                self.data.append(sample)
                self.sample_id2local_edits[sample_id] = local_edits

            self.table_id2alt_questions[table_id] = list(set(all_alt_questions))
            self.table_id2alt_programs[table_id] = list(set(all_alt_programs))
            

    def get_random_alternate_question(self, sample: dict) -> str:
        table_id = sample['table_id']
        pool = self.table_id2alt_questions.get(table_id, [])
        if pool:
            idx = np.random.choice(len(pool))
            return pool[idx]
        else:
            return sample['statement']

    def get_random_alternate_program(self, sample: dict) -> str:
        table_id = sample['table_id']
        pool = self.table_id2alt_programs.get(table_id, [])
        if pool:
            idx = np.random.choice(len(pool))
            return pool[idx]
        else:
            orig_prog = sample['verifier_query_gt']
            if orig_prog.endswith("=True"):
                return orig_prog[:-len("=True")] + "=False"
            elif orig_prog.endswith("=False"):
                return orig_prog[:-len("=False")] + "=True"
            else:
                return orig_prog
            
    def get_random_local_edits(self, sample: dict, n=3) -> str:
        sample_id = sample['idx']
        pool = self.sample_id2local_edits.get(sample_id, [])
        return np.random.choice(pool, size=n, replace=False).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]