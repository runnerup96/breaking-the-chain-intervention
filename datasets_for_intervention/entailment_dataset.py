import json
from typing import Optional

def read_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

def read_jsonl(file_path: str):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

class EntailmentDataset:
    def __init__(self, file_path: str, paraphrases_path: Optional[str] = None):
        self.data = []
        
        raw_data = read_jsonl(file_path)
        paraphrases_data = read_json(paraphrases_path) if paraphrases_path is not None else [None] * len(raw_data)
    
        for entry, paraphrases in zip(raw_data, paraphrases_data):
            if paraphrases is not None:
                paraphrases = paraphrases["response"]["response"].split("|")
                paraphrases = [phrase.strip() for phrase in paraphrases]
            else:
                paraphrases = None

            # All scores initially are set to True since the proofs are presumed correct.
            example = {
                "id": entry["id"],
                "proof": entry["meta"]["step_proof"] if "meta" in entry and "step_proof" in entry["meta"] else entry["step_proof"],
                "question": entry["meta"]["question_text"],
                "question_paraphrases": paraphrases,
                "answer": entry["meta"]["answer_text"],
                "context": entry["meta"]["triples"],
                "hypothesis": entry["hypothesis"],
                "intermediate_conclusions": entry["meta"]["intermediate_conclusions"],
                "hypothesis_id": entry["meta"]["hypothesis_id"],
                "distractors": entry["meta"]["distractors"],
                "score": True,
            }
            assert example["distractors"] is not None, "Distractors are required for entailment intervention"
            assert example["proof"] is not None, "Proof is required for entailment intervention"
            assert example["question"] is not None, "Question is required for entailment intervention"
            assert example["context"] is not None, "Context is required for entailment intervent"

            self.data.append(example)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
