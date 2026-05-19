import json
import os
from copy import deepcopy


class AVeriTeCDataset:
    """
    AVeriTeC fact-checking dataset.

    Mediator (gold_rubric): dict {question: bool}
        True  = answer "Yes"  (in the original dataset)
        False = answer "No"   (in the original dataset)

    Target (gold_target): "Supported" | "Refuted"

    Filtering logic (required for deterministic Local Edit expectations):
      - gold_target == "Supported": any number of questions allowed.
        Flipping ANY single answer deterministically yields Refuted
        (all answers must agree for the claim to be Supported).
      - gold_target == "Refuted" AND len(gold_rubric) == 1: exactly one question.
        Flipping the single answer deterministically yields Supported.
      Multi-question Refuted samples are excluded: flipping one answer there
      does NOT necessarily change the verdict, so Local Edits cannot be evaluated.

    Claims used in few-shot examples are excluded to prevent data leakage.
    """

    # Claims used in few-shot prompts — excluded from evaluation to avoid data leakage
    EXCLUDED_CLAIMS = [
        "Hunter Biden had no experience in Ukraine or in the energy sector "
        "when he joined the board of Burisma.",
        "President Trump is the most pro-gay president in American history.",
        "Beijing government announced that Chinese people should not travel to "
        "the United States or buy American-made products.",
    ]

    def __init__(self, data_path: str, include_explanations: bool = True):
        """
        Args:
            data_path:            path to the folder containing dataset files.
                                  Expects "onlyboolean_samples.json" to be present.
            include_explanations: if False, the "explanations" field in every sample
                                  is set to an empty dict {}.  Use this for the
                                  no-explanations ablation experiment.
        """
        self.data_path = data_path
        self.include_explanations = include_explanations
        self.data = []

        raw_samples = json.load(
            open(os.path.join(data_path, "onlyboolean_samples.json"), "r", encoding="utf-8")
        )
        self.process_data(raw_samples)

    def _answers_to_bool(self, answer_str: str) -> bool:
        """Convert a string answer "Yes"/"No" to a boolean."""
        return answer_str.strip().lower() == "yes"

    def process_data(self, raw_samples: list):
        """
        Normalize raw records into a list of samples.

        Each sample contains:
          "idx"             -- unique string identifier
          "claim"           -- claim text (X)
          "explanations"    -- dict {question: explanation_text}  (part of X)
          "gold_rubric"     -- dict {question: bool}  (canonical mediator M_gold)
          "gold_target"     -- "Supported" | "Refuted"            (gold answer Y_gold)
          "mediator_rubric" -- deepcopy(gold_rubric), the starting mediator
                               (replaced during interventions)
        """
        for idx, raw in enumerate(raw_samples):
            claim = raw.get("claim", "")

            gold_rubric = {}
            explanations = {}
            for qa in raw.get("questions", []):
                question = qa.get("question", "").strip()
                if not question:
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                answer_str = answers[0].get("answer", "")
                if answer_str.strip().lower() not in ("yes", "no"):
                    continue
                gold_rubric[question] = self._answers_to_bool(answer_str)
                explanations[question] = answers[0].get("boolean_explanation", "")

            if not gold_rubric:
                continue

            gold_target = raw.get("label", "")

            is_supported = gold_target == "Supported"
            # Refuted with exactly 1 question: flipping the single answer yields Supported (deterministic)
            is_single_refuted = gold_target == "Refuted" and len(gold_rubric) == 1

            if not (is_supported or is_single_refuted):
                continue

            # Exclude few-shot examples
            if claim in self.EXCLUDED_CLAIMS:
                continue

            sample = {
                "idx": str(idx),               # string idx (architecture invariant)
                "claim": claim,                # claim text (X)
                # explanations are included only when include_explanations=True;
                # set to {} for the no-explanations ablation so the prompt builder
                # sees an empty dict and produces no "Explanations:" block.
                "explanations": explanations if self.include_explanations else {},
                "gold_rubric": gold_rubric,    # gold mediator (M_gold)
                "gold_target": gold_target,    # gold verdict  (Y_gold)
                # mediator_rubric: starting copy of gold_rubric;
                # replaced during interventions by Intervention.make_intervention
                "mediator_rubric": deepcopy(gold_rubric),
            }
            self.data.append(sample)

        print(f"AVeriTeC: Total samples after filtering = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]