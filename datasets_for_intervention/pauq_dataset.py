import json
import os


class PAUQDataset:
    def __init__(self, data_path: str, train: bool = False):
        """
        Args:
            data_path: path to the folder with the following files:
            pauq_train.json
            pauq_dev.json
            tables.json
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
            "table_names_original"
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
        for idx, sample in enumerate(train_data):
            self.data.append({
                "index": idx,
                "query": sample["query"]["en"],
                "question": sample["question"]["en"],
                "db": sample["db"]
            })

        paraphrase_train_path = os.path.join(data_path, "pauq_dev_paraphrase.json")
        paraphrase_train = json.load(open(paraphrase_train_path))
        for sample in paraphrase_train:
            self.data[sample["sample_idx"]]["paraphrase"] = sample["paraphrase"]

    @staticmethod
    def get_table_columns(db, table_name):
        table_db_idx = db["table_names_original"].index(table_name)
        table_columns = []
        for i, col_name in db["column_names_original"][1:]:
            if i == table_db_idx:
                table_columns.append(col_name)
        return table_columns
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]



if __name__ == "__main__":
    dataset = PAUQDataset("./pauq")
    print(dataset[0])
    i = 0
    for k, v in dataset[0].items():
        print(k, ":", v)
        i += 1
        if i > 25:
            break
