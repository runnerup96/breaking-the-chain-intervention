"""
averitec_baseline.py
--------------------
Baseline experiment for AVeriTeC: direct X → Y (verdict) without any mediator.

The model receives the claim and supporting evidence (Q&A explanations presented
as plain text) and must output "Supported" or "Refuted" directly — no Q&A
checklist structure.

Few-shot examples are taken from the same pool as AVeriTeCIntervention.FEW_SHOT,
but the checklist structure is stripped: only claim + evidence text + verdict shown.
"""

from __future__ import annotations

import re
from typing import Optional

from datasets_for_intervention.prompt import Prompt


class AVeriTeCBaseline:
    """
    Direct fact-checking baseline for AVeriTeC.

    Args:
        dataset:   AVeriTeCDataset instance.
        llm_model: LLMModel wrapper.
    """

    # ── regex for verdict extraction ──────────────────────────────────────────
    _VERDICT_LABEL_RE = re.compile(
        r"(?:final\s+)?verdict\s*[:\-]\s*(?P<v>Supported|Refuted)",
        re.IGNORECASE,
    )
    _VERDICT_BARE_RE = re.compile(
        r"^\s*(?P<v>Supported|Refuted)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # ── few-shot: same claims/explanations as AVeriTeCIntervention.FEW_SHOT ───
    # The Q&A checklist structure is hidden from the model — only the explanation
    # text (evidence) and the final verdict are shown.
    FEW_SHOT_EXAMPLES = [
        # Example 1: Supported, 2 questions
        {
            "num": 1,
            "claim": (
                "Hunter Biden had no experience in Ukraine or in the energy sector "
                "when he joined the board of Burisma."
            ),
            "explanations": {
                "Did Hunter Biden have any experience in the energy sector in 2014?":
                    "Hunter Biden's previous career history does not include work for energy companies.",
                "Did Hunter Biden have any experience in Ukraine in 2014?":
                    "Hunter Biden's previous career history does not include working with Ukrainian companies.",
            },
            "verdict": "Supported",
        },
        # Example 2: Refuted, 1 question
        {
            "num": 2,
            "claim": "President Trump is the most pro-gay president in American history.",
            "explanations": {
                "Did Trump make pro-gay laws when in office?":
                    "He made laws such as: 1. Appointing Anti-Equality Judges "
                    "2. Stripping protections from LGBTQ students, parents and families "
                    "3. Defending Anti-Gay Discrimination.",
            },
            "verdict": "Refuted",
        },
        # Example 3: Refuted, 3 questions
        {
            "num": 3,
            "claim": (
                "Beijing government announced that Chinese people should not travel "
                "to the United States or buy American-made products."
            ),
            "explanations": {
                "Did China's Ministry of Foreign Affairs announce that Chinese people should not "
                "travel to the United States or buy American-made products in its daily press "
                "briefing on August 13, 2020?":
                    "Transcript of August 13 daily press briefing does not include a request for "
                    "Chinese people to avoid American products or avoid travelling to the US.",
                "Did the weekly policy briefing from China's State Council on August 13, 2020 "
                "include a mention of the call for Chinese people to not travel to the United "
                "States or buy American-made products?":
                    "China's State Council weekly policy briefing pages for August 13, 2020 "
                    "do not mention the US.",
                "Did the Chinese Ministry of Foreign Affairs announce that Chinese people should "
                "not travel to the United States or buy American-made products on its Twitter "
                "account on or after August 13, 2020?":
                    "A keywords search set between August 13 and August 18 2020 found no claim "
                    "on the Ministry's Twitter account.",
            },
            "verdict": "Refuted",
        },
    ]

    def __init__(self, dataset, llm_model, include_explanations: bool = True):
        """
        Args:
            include_explanations: if False, evidence is stripped from every prompt
                                  (few-shot and current sample).
                                  Mirrors AVeriTeCIntervention(include_explanations=False).
        """
        self.dataset              = dataset
        self.llm_model            = llm_model
        self.include_explanations = include_explanations

        instruction = (
            "You are an expert fact-checking system.\n"
            "You are given a claim and supporting evidence gathered by fact-checkers.\n"
            "Your task is to decide whether the claim is Supported or Refuted "
            "based on the provided evidence.\n\n"
            "Important output rule:\n"
            "Your response must end with exactly one line in the format:\n"
            "Verdict: Supported\n"
            "or\n"
            "Verdict: Refuted\n"
            "Do not output anything after the verdict line."
        )

        few_shot_lines = ["FEW-SHOT EXAMPLES:\n"]
        for ex in self.FEW_SHOT_EXAMPLES:
            if self.include_explanations:
                evidence_block = (
                    f"Evidence:\n{self._format_evidence_from_dict(ex['explanations'])}\n\n"
                )
            else:
                evidence_block = ""
            few_shot_lines.append(
                f"Example #{ex['num']}\n"
                f"Claim:\n{ex['claim']}\n\n"
                f"{evidence_block}"
                f"Verdict: {ex['verdict']}\n"
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

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_evidence_from_dict(explanations: dict) -> str:
        """Join explanation texts into a single evidence paragraph."""
        lines = [e.strip() for e in explanations.values() if e and e.strip()]
        return " ".join(lines) if lines else "(no evidence provided)"

    def _format_evidence(self, sample: dict) -> str:
        """Convert a live sample's explanations dict into plain evidence text."""
        return self._format_evidence_from_dict(sample.get("explanations", {}))

    # ── prompt ────────────────────────────────────────────────────────────────

    def make_prompt(self, sample: dict) -> str:
        """Build the direct verdict prompt — no Q&A checklist structure."""
        evidence     = self._format_evidence(sample) if self.include_explanations else None
        user_content = self._build_user_content(sample, evidence)

        messages = [{"role": "user", "content": user_content}]
        messages.append({"role": "assistant", "content": "Verdict: "})

        prompt = self.llm_model.apply_chat_template(messages, add_generation_prompt=False)
        return self.llm_model.clean_model_specific_completion(prompt)

    def _build_user_content(self, sample: dict, evidence) -> str:
        instruction = self.prompt.build_zeroshot_instruction()
        few_shot    = self.prompt.few_shot
        evidence_block = f"Evidence:\n{evidence}\n\n" if evidence else ""
        current = (
            "Now fact-check the following claim.\n\n"
            f"Claim:\n{sample['claim']}\n\n"
            f"{evidence_block}"
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

    def parse_answer(self, completion: str) -> Optional[str]:
        """
        Extract 'Supported' or 'Refuted' from a completion string.
        Tries labelled pattern first, then bare last word.
        """
        text = self.clean_output(completion)
        if not text:
            return None

        m = self._VERDICT_LABEL_RE.search(text)
        if m:
            return m.group("v").title()

        for m in self._VERDICT_BARE_RE.finditer(text):
            return m.group("v").title()

        last_word = text.split()[-1].strip(".,!?") if text else ""
        if last_word.lower() in ("supported", "refuted"):
            return last_word.title()

        return None

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, results: list) -> dict:
        n_total   = len(results)
        n_parsed  = 0
        n_correct = 0

        label_counts: dict = {}

        for s in results:
            pred = s.get("predicted_answer")
            gold = s.get("gold_answer")

            label_counts.setdefault(gold, {"correct": 0, "total": 0})
            label_counts[gold]["total"] += 1

            if pred is None:
                continue

            n_parsed += 1
            if pred == gold:
                n_correct += 1
                label_counts[gold]["correct"] += 1

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

        print("\n=== AVeriTeC Baseline Results ===")
        print(f"  Samples:      {n_total}")
        print(f"  Parsed:       {n_parsed}  ({metrics['parse_rate']:.1%})")
        print(f"  Correct:      {n_correct}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}  "
              f"(total: {metrics['accuracy_total']:.4f})")
        for label, v in metrics["per_label"].items():
            acc = f"{v['accuracy']:.4f}" if v["accuracy"] is not None else "N/A"
            print(f"    {label}: {v['n_correct']}/{v['n_total']}  acc={acc}")

        return metrics
