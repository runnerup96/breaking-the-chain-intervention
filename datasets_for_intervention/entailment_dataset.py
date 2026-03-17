"""
EntailmentBank dataset loader for the Breaking the Chain pipeline.

Original sample keys are preserved exactly as in the original codebase:
    id, proof, score, question, answer, context,
    hypothesis, intermediate_conclusions, hypothesis_id, distractors

One pipeline-standard alias is added:
    idx  (str)  — unique sample identifier, equal to id

The proof string (e.g. "sent21 & sent3 -> int1; int1 & sent15 -> hypothesis; ")
is the canonical "mediator" structure for entailment. It lives in sample['proof']
and is never wrapped in a dict. EntailmentStructureProcessor and
EntailmentIntervention work directly with this string.

score is always True on load (every gold proof is presumed correct).
"""

import json
from typing import Dict, List, Optional


def _read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class EntailmentDataset:
    def __init__(self, file_path: str):
        self.data: List[Dict] = []

        raw_data = _read_jsonl(file_path)

        for entry in raw_data:
            # Gold proof string
            step_proof: str = (
                entry["meta"]["step_proof"]
                if "meta" in entry and "step_proof" in entry["meta"]
                else entry.get("step_proof", "")
            )

            sample: Dict = {
                # ---- original keys (unchanged) ----
                "id":                       entry["id"],
                "proof":                    step_proof,
                "question":                 entry["meta"]["question_text"],
                "answer":                   entry["meta"]["answer_text"],
                "context":                  entry["meta"]["triples"],  # dict {sent_id: text}
                "hypothesis":               entry["hypothesis"],
                "intermediate_conclusions": entry["meta"]["intermediate_conclusions"],
                "hypothesis_id":            entry["meta"]["hypothesis_id"],
                "distractors":              entry["meta"]["distractors"],
                # All gold proofs are presumed correct
                "score":                    True,
                # ---- pipeline standard keys ----
                "idx":                      entry["id"],
                "gold_proof":               step_proof,          # M_gold — never modified
                "gold_score":               True,                # Y_gold — never modified
                "mediator_proof":           step_proof,          # M' — replaced during inference
            }

            assert sample["distractors"] is not None, \
                f"Sample {sample['id']}: distractors required for intervention"
            assert step_proof, \
                f"Sample {sample['id']}: step_proof must be non-empty"
            assert sample["question"] is not None, \
                f"Sample {sample['id']}: question required"
            assert sample["context"] is not None, \
                f"Sample {sample['id']}: context required"

            self.data.append(sample)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
