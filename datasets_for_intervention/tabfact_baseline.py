"""
tabfact_baseline.py
--------------------
Baseline experiment for TabFact: direct X → Y (True/False) without any DSL mediator.

The model receives the table and claim and must output True or False directly —
no Verifier Query, no DSL, no reasoning trace.

Few-shot examples are taken from the same pool as TabFactIntervention.FEW_SHOT_EXAMPLES,
but the DSL Verifier Query is stripped: only table + claim + result are shown.
"""

from __future__ import annotations

import re
from typing import Optional

from datasets_for_intervention.prompt import Prompt


class TabFactBaseline:
    """
    Direct table fact-checking baseline for TabFact.

    Args:
        dataset:   TabFactDataset instance.
        llm_model: LLMModel wrapper.
    """

    # ── regex for True/False extraction ──────────────────────────────────────
    _RESULT_LABEL_RE = re.compile(
        r"(?:result|answer|verdict)\s*[:\-]\s*(?P<v>True|False)",
        re.IGNORECASE,
    )
    _RESULT_BARE_RE = re.compile(
        r"^\s*(?P<v>True|False)\s*[.!]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # ── few-shot: same tables/claims as TabFactIntervention.FEW_SHOT_EXAMPLES ─
    # The DSL Verifier Query is stripped — model only sees table + claim + result.
    FEW_SHOT_EXAMPLES = [
        # Example 1: True
        {
            "num": 1,
            "table": (
                "rank#athlete#nation#gold\n"
                "1#Usain Bolt#Jamaica#2\n"
                "2#Shawn Crawford#United States#1"
            ),
            "claim": "Usain Bolt won more gold medals than Shawn Crawford.",
            "result": "True",
        },
        # Example 2: False
        {
            "num": 2,
            "table": (
                "player#team#goals\n"
                "Messi#PSG#30\n"
                "Ronaldo#Al-Nassr#25"
            ),
            "claim": "Ronaldo scored more goals than Messi.",
            "result": "False",
        },
        # Example 3: True
        {
            "num": 3,
            "table": (
                "event#year#location\n"
                "Olympics#2020#Tokyo\n"
                "World Cup#2022#Qatar\n"
                "Asian Games#2018#Jakarta"
            ),
            "claim": "The World Cup was held after the Olympics.",
            "result": "True",
        },
        # Example 4: True
        {
            "num": 4,
            "table": (
                "country#sport#medals\n"
                "USA#swimming#12\n"
                "USA#athletics#8\n"
                "China#swimming#5\n"
                "China#athletics#7"
            ),
            "claim": "The USA won more medals than China across all sports.",
            "result": "True",
        },
    ]

    def __init__(self, dataset, llm_model):
        self.dataset   = dataset
        self.llm_model = llm_model

        instruction = (
            "You are a table fact-checking system.\n"
            "You are given a table and a claim about it.\n"
            "Your task is to determine whether the claim is True or False "
            "based solely on the data in the table.\n\n"
            "Important output rule:\n"
            "Your response must end with exactly one line:\n"
            "Result: True\n"
            "or\n"
            "Result: False\n"
            "Do not output anything after the result line."
        )

        few_shot_lines = ["FEW-SHOT EXAMPLES:\n"]
        for ex in self.FEW_SHOT_EXAMPLES:
            few_shot_lines.append(
                f"Example #{ex['num']}\n"
                f"Table:\n{ex['table']}\n\n"
                f"Claim:\n{ex['claim']}\n\n"
                f"Result: {ex['result']}\n"
            )
        few_shot_text = "\n".join(few_shot_lines)

        self.prompt = Prompt(
            prompting_regime="standard",
            use_tool_call=False,
            tool_call_instruction="",
            instruction=instruction,
            few_shot=few_shot_text,
            llm_model=self.llm_model,
        )

    # ── prompt ────────────────────────────────────────────────────────────────

    def make_prompt(self, sample: dict) -> str:
        """Build the direct True/False prompt — no DSL, no Verifier Query."""
        user_content = self._build_user_content(sample)

        messages = [{"role": "user", "content": user_content}]
        messages.append({"role": "assistant", "content": "Result: "})

        prompt = self.llm_model.apply_chat_template(messages, add_generation_prompt=False)
        return self.llm_model.clean_model_specific_completion(prompt)

    def _build_user_content(self, sample: dict) -> str:
        instruction = self.prompt.build_zeroshot_instruction()
        few_shot    = self.prompt.few_shot
        current = (
            "Now fact-check the following claim against the table.\n\n"
            f"Table:\n{sample['table_html_csv']}\n\n"
            f"Claim:\n{sample['statement']}\n"
        )
        parts = [instruction]
        if few_shot.strip():
            parts.append(few_shot)
        parts.append(current)
        return "\n\n".join(parts)

    # ── parsing ───────────────────────────────────────────────────────────────

    def clean_output(self, text: str) -> str:
        tokens = [
            "<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|eot_id|>",
            "<|end_of_text|>", "<|pad|>", "<end_of_turn>", "</s>",
            "\u00ad", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
        ]
        for tok in tokens:
            text = text.replace(tok, "")
        return text.strip()

    def parse_answer(self, completion: str) -> Optional[bool]:
        """
        Extract True or False from a completion string.
        Returns a Python bool or None if parsing fails.
        """
        text = self.clean_output(completion)
        if not text:
            return None

        m = self._RESULT_LABEL_RE.search(text)
        if m:
            return m.group("v").lower() == "true"

        for m in self._RESULT_BARE_RE.finditer(text):
            return m.group("v").lower() == "true"

        last_word = text.split()[-1].strip(".,!?") if text else ""
        if last_word.lower() == "true":
            return True
        if last_word.lower() == "false":
            return False

        return None

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, results: list) -> dict:
        n_total   = len(results)
        n_parsed  = 0
        n_correct = 0
        label_counts: dict = {}

        for s in results:
            pred     = s.get("predicted_answer")
            gold     = s.get("gold_answer")
            gold_key = str(gold)

            label_counts.setdefault(gold_key, {"correct": 0, "total": 0})
            label_counts[gold_key]["total"] += 1

            if pred is None:
                continue

            n_parsed += 1
            if pred == gold:
                n_correct += 1
                label_counts[gold_key]["correct"] += 1

        metrics = {
            "n_total":        n_total,
            "n_parsed":       n_parsed,
            "n_parse_error":  n_total - n_parsed,
            "n_correct":      n_correct,
            "parse_rate":     round(n_parsed  / n_total,  4) if n_total  else None,
            "accuracy":       round(n_correct / n_parsed, 4) if n_parsed else None,
            "accuracy_total": round(n_correct / n_total,  4) if n_total  else None,
            "per_label": {
                label: {
                    "accuracy":  round(v["correct"] / v["total"], 4) if v["total"] else None,
                    "n_correct": v["correct"],
                    "n_total":   v["total"],
                }
                for label, v in label_counts.items()
            },
        }

        print("\n=== TabFact Baseline Results ===")
        print(f"  Samples:      {n_total}")
        print(f"  Parsed:       {n_parsed}  ({metrics['parse_rate']:.1%})")
        print(f"  Correct:      {n_correct}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}  "
              f"(total: {metrics['accuracy_total']:.4f})")

        return metrics
