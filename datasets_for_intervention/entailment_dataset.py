import json

def read_jsonl(file_path: str):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

class EntailmentDataset:
    def __init__(self, file_path: str):
        self.data = []
        
        raw_data = read_jsonl(file_path)
    
        for entry in raw_data:
            example = {
                "id": entry["id"],
                "proof": entry["meta"]["step_proof"] if "meta" in entry and "step_proof" in entry["meta"] else entry["step_proof"],
                "question": entry["meta"]["question_text"],
                "answer": entry["meta"]["answer_text"],
                "context": entry["meta"]["triples"],
                "hypothesis": entry["hypothesis"],
                "intermediate_conclusions": entry["meta"]["intermediate_conclusions"],
                "hypothesis_id": entry["meta"]["hypothesis_id"],
                "distractors": entry["meta"]["distractors"],
            }

            self.data.append(example)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    path = "entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl"
    dataset = EntailmentDataset(file_path=path)
    for i in range(3):
        print(json.dumps(dataset[i], indent=2))
        print("--------------------------------")