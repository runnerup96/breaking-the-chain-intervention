import json
import os


class PAUQDataset:
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the folder with the following files:
            pauq_train.json
            pauq_dev.json
            tables.json
        """
        self.data_path = data_path
        train_path = os.path.join(data_path, "pauq_train.json")
        with open(train_path) as input_json:
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
        for row in train_data:
            row["db"] = self.databases[row["db_id"]]
        self.data = train_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]



if __name__ == "__main__":
    dataset = PAUQDataset("./pauq")
    for k, v in dataset[0].items():
        print(k, ":", v)
