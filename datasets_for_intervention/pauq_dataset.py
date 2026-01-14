import json
import os
if __name__ == "__main__":
    from utils import extract_schema_links, parse_sql
else:
    from .utils import extract_schema_links, parse_sql
import copy


class PAUQDataset:
    def __init__(self, data_path: str, train: bool = False):
        """
        Args:
            data_path: path to the folder with the following files:
            pauq_train.json
            pauq_dev.json
            tables.json
            pauq_dev_paraphrase.json
        """
        self.data_path = data_path
        if train:
            json_path = os.path.join(data_path, "pauq_train.json")
        else:
            json_path = os.path.join(data_path, "pauq_dev.json")
        
        with open(json_path) as input_json:
            train_data = json.loads(input_json.read())
        tables_path = os.path.join(data_path, "tables.json")
        with open(tables_path) as input_json:
            db_data = json.loads(input_json.read())
        db_keys = [
            "column_names",
            "column_names_original",
            "column_types",
            "foreign_keys",
            "primary_keys",
            "table_names",
            "table_names_original",
        ]
        self.databases = {}
        self.tables = {}
        for db in db_data:
            self.databases[db["db_id"]] = {key: db[key] for key in db_keys}
            for table_name in db["table_names_original"]:
                if table_name in self.tables:
                    continue
                columns = PAUQDataset.get_table_columns(db, table_name)
                self.tables[table_name] = columns
                
        for row in train_data:
            row["db"] = self.databases[row["db_id"]]
        self.data = []
        self.sample_idx2idx = {}
        i = 0
        for idx, sample in enumerate(train_data):
            row = {
                "index": idx,
                "query": sample["query"]["en"],
                "question": sample["question"]["en"],
                "db": sample["db"],
                "db_schema": PAUQDataset.get_db_schema(sample["db"]),
            }
            parsed_sql = parse_sql(row["query"], row["db_schema"])
            row["true_schema_links"] = extract_schema_links(parsed_sql)
            if row["true_schema_links"]:
                self.data.append(row)
                self.sample_idx2idx[idx] = i
                i += 1

        paraphrase_train_path = os.path.join(data_path, "pauq_dev_paraphrase.json")
        paraphrase_train = json.load(open(paraphrase_train_path))
        for sample in paraphrase_train:
            if sample["sample_idx"] in self.sample_idx2idx:
                self.data[self.sample_idx2idx[sample["sample_idx"]]]["paraphrase"] = sample["paraphrase"]

        self.dummy_tables = {
            "space_missions": ["id", "name", "launch_date", "country", "success", "budget_mlns"],
            "albums": ["id", "artist_id", "title", "year", "genre", "duration_sec"],
            "matches": ["id", "tournament_id", "team_a", "team_b", "score_a", "score_b", "match_date"],
            "recipes": ["id", "dish_name", "cuisine", "calories", "cook_time_min", "difficulty"],
            "air_quality": ["id", "city", "measure_date", "pm25", "pm10", "aqi"]
        }
        
    @staticmethod
    def get_table_columns(db, table_name, original: bool = True):
        table_key = "table_names_original" if original else "table_names"
        column_key = "column_names_original" if original else "column_names"
        table_db_idx = db[table_key].index(table_name)
        table_columns = []
        for i, col_name in db[column_key][1:]:
            if i == table_db_idx:
                table_columns.append(col_name.lower())
        return table_columns
    
    @staticmethod
    def get_db_schema(db):
        schema = {}
        for table_name in db['table_names_original']:
            schema[table_name] = PAUQDataset.get_table_columns(db, table_name)
        return schema
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]



if __name__ == "__main__":
    dataset = PAUQDataset("./pauq")
    tables = {}
    for i in range(1, 100, 20):
        print("=" * 100)
        for k, v in dataset[i].items():
            if k == "db_schema":
                for table_name in v:
                    tables[table_name] = v[table_name]
            print("\t"*4, end="")
            if isinstance(v, str):
                print(f'"{k}": "{v}",')
            else:
                print(f'"{k}": {v},')
        
    print(tables)
    
