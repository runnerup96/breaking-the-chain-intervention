import random
from copy import deepcopy


class RiceChemDatasetMock:
    """
    Lightweight dataset mock aligned with the current unified RiceChem pipeline.

    Each sample contains:
    - idx, task_idx, task, student_answer, score_range
    - gold_rubric (dict[item->bool]), gold_score (float)
    - mediator_rubric (dict[item->bool]) as a mutable copy of gold (default)
    - optional bad_rubric / bad_score for Correction tests
    """
    def __init__(self):
        self.task2rubric_weights = {
            1: {
                "A item": 1.0,
                "B item": 1.5,
                "C item": 0.5,
            }
        }
        self.task2student_answers = {
            1: ["Sample answer A", "Sample answer B", "Sample answer C"]
        }

        self.data = []

        gold_0 = {"A item": True, "B item": False, "C item": True}   # score 1.5
        bad_0  = {"A item": False, "B item": True,  "C item": False}  # score 1.5

        gold_1 = {"A item": False, "B item": True, "C item": True}   # score 2.0
        bad_1  = {"A item": True,  "B item": False, "C item": False} # score 1.0

        samples = [
            ("mock_0@Task1", "Sample answer A", gold_0, bad_0),
            ("mock_1@Task1", "Sample answer B", gold_1, bad_1),
        ]

        for idx, ans, gold, bad in samples:
            weights = self.task2rubric_weights[1]
            gold_score = float(sum(weights[k] for k, v in gold.items() if v))
            bad_score = float(sum(weights[k] for k, v in bad.items() if v))
            self.data.append({
                "idx": idx,
                "task_idx": 1,
                "task": "Mock task text.",
                "student_answer": ans,
                "score_range": "0-3",
                "gold_rubric": deepcopy(gold),
                "gold_score": gold_score,
                "mediator_rubric": deepcopy(gold),
                "bad_rubric": deepcopy(bad),
                "bad_score": bad_score,
            })

    def get_random_student_answer(self, task_idx: int) -> str:
        return random.choice(self.task2student_answers[task_idx])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]