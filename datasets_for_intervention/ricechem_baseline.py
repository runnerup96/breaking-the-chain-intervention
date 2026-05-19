"""
ricechem_baseline.py
--------------------
Baseline experiment for RiceChem: direct X → Y (score) without a filled mediator.

The model receives:
  - the question
  - the student answer
  - the rubric criteria (what to check) with point values — but NOT pre-filled True/False

The model must decide which criteria are satisfied and output a total grade.
This is X → Y without an intermediate structure: the rubric here is part of X
(it defines the task), not a mediator (an intermediate reasoning variable).
"""

from __future__ import annotations

import re
from math import isclose
from typing import Optional

from datasets_for_intervention.prompt import Prompt


class RiceChemBaseline:
    """
    Direct grading baseline for RiceChem.

    The prompt shows rubric criteria + point values but NOT filled True/False.
    The model grades the answer directly.

    Args:
        dataset:   RiceChemDataset instance (used to retrieve task rubric weights).
        llm_model: LLMModel wrapper.
    """

    _GRADE_RE = re.compile(
        r"""
        (?:
            (?:final\s+)?grade\s*(?:\([^)]*\))?\s*[:\-]\s*
            | score\s*[:\-]\s*
        )
        (?P<num>[-+]?\d+(?:\.\d+)?)
        |
        ^\s*(?P<bare>[-+]?\d+(?:\.\d+)?)\s*$
        """,
        re.IGNORECASE | re.VERBOSE | re.MULTILINE,
    )

    # Few-shot: same Q/A as RiceChemIntervention.FEW_SHOT, rubric shown unfilled.
    # Grades = sum of True values in gold checklist.
    FEW_SHOT_EXAMPLES = [
        {
            "num": 1,
            "question": (
                "When studying the emission sources within the Milky Way, a satellite detected "
                "interplanetary clouds containing silicon atoms that have lost five electrons.\n"
                "b) The ionization energies corresponding to the removal of the third, fourth, "
                "and fifth electrons in silicon are 3231, 4356, and 16091 kJ/mol, respectively.\n"
                "Using core charge calculations and your understanding of Coulomb's Law, briefly "
                "explain 1) why the removal of each additional electron requires more energy than "
                "the removal of the previous one, and 2) the relative magnitude of the values "
                "observed.\nThis question can be answered reasonably in around 150 words or fewer.\n"
            ),
            "answer": (
                "With each removal of an electron, there is less electron-electron repulsion, "
                "which decreases the potential energy of the electrons as they are more strongly "
                "attracted to the nucleus, and ultimately increasing each successive ionization "
                "energy. The ionization energies of the third and fourth electron are similar due "
                "to the fact that both of these electrons reside in the same n quantum number (3), "
                "meaning they are basically the same radius away from the nucleus. Furthermore, "
                "these two electrons have the same core charge of +4. The difference in the 3rd "
                "and 4th IE is due to greater repulsion in 3p vs 3s. There is a large jump from "
                "4th to 5th because the 5th electron comes from n=2, so the core charge is +12 "
                "and the radius is much smaller, greatly increasing the ionization energy.\n"
            ),
            "rubric_weights": {
                "correctly cites decreased electron electron repulsion": 1.0,
                "relates decreased electron electron repulsion to decreased potential energy": 1.0,
                "3rd and 4th electrons ionized feel same core charge": 1.0,
                "3rd and 4th electrons ionized from n=3 shell and have same radius": 1.0,
                "5th electron ionized from n=2 shell and feels higher core charge": 1.0,
                "5th electron ionized from n=2 shell and has smaller radius": 1.0,
                "correctly explains relationship of potential energy to ionization energy": 1.0,
                "partially explains relationship between potential energy and ionization energy": 1.0,
            },
            "score_range": "0-8",
            "grade": 7.0,
        },
        {
            "num": 2,
            "question": (
                "In each statement below (a-c), two observations are given which seem to contrast "
                "with each other. Using your knowledge of electron configurations, orbitals, "
                "Coulomb's law, and/or atomic and molecular structures, briefly explain why both "
                "of these observations are true.\n\n"
                "b) If light is used to excite an electron to a higher energy level in an atom, "
                "only certain frequencies of light can be absorbed. However, if it is used to eject "
                "an electron from the atom, any value above a minimum threshold frequency can be "
                "absorbed.\n\n"
                "This question can be answered reasonably in around 150 words or fewer.\n"
            ),
            "answer": (
                "Energy levels in an atom are quantized. To excite an electron the photon energy "
                "must exactly equal the difference between two energy levels. For ejection, any "
                "energy above the ionization threshold works because the electron just needs enough "
                "to escape; any excess becomes kinetic energy."
            ),
            "rubric_weights": {
                "Correctly states that frequency is proportional to energy of light": 1.0,
                "Explaining sentence 1: energy levels of an electron in an atom are quantized": 1.0,
                "Explaining sentence 1: FULLY explains energy/frequency absorbed must equal the difference in energy levels in an electron": 1.0,
                "Explaining sentence 1: PARTIALLY explains energy/frequency absorbed must equal the difference in energy levels in an electron": 1.0,
                "Explaining sentence 2: a minimum amount of energy is needed to eject an electron": 1.0,
                "Explaining sentence 2: any additional energy becomes kinetic energy": 1.0,
            },
            "score_range": "0-6",
            "grade": 3.0,
        },
        {
            "num": 3,
            "question": (
                "A CHEM 121 student was asked what hybrid orbitals must be present to form "
                "methanimine (CH2NH). The student responded:\n"
                "Carbon cannot form four bonds because it only has two unpaired valence electrons. "
                "So, it has to form four sp3 hybrid orbitals to create the four bonds. Nitrogen "
                "doesn't need to hybridize because it already has three unpaired 2p valence "
                "electrons.\n"
                "Assess the accuracy and logic of the student's response.\n"
                "This question can be reasonably answered in 150 words or fewer."
            ),
            "answer": (
                "Sentence 1: Incorrect. VBT says carbon hybridizes to allow overlap, not because "
                "of unpaired electrons. Sentence 2: Carbon has 3 electron domains so it is sp2, "
                "not sp3. Sentence 3: Nitrogen is also sp2 hybridized with 3 electron domains; "
                "one sp2 orbital forms the N-H bond, another participates in the C=N double bond, "
                "and the unhybridized p orbital forms the pi bond."
            ),
            "rubric_weights": {
                "Sentence 1 is correct. Valence bond theory describes that atomic orbitals must be half-filled to participate in covalent bonding.": 1.0,
                "Sentence 2: Correct number of hybrid orbitals. In this molecule, carbon must form three hybrid orbitals to form three electron domains.": 1.0,
                "Sentence 2: Correct type of hybrid orbitals. Carbon must form sp2 hybrid orbitals (from using a 2s and two 2p orbitals)": 1.0,
                "Sentence 3: Correctly states that nitrogen is hybridized": 1.0,
                "Sentence 3: Correct type of hybridization. Nitrogen is sp2 hybridized to form 3 electron domains": 1.0,
                "Sentence 3: Correct description of hybrid orbital bonds in nitrogen. Two sp2 orbitals form two sigma bonds.": 1.0,
                "Sentence 3: Correct description of unhybridized orbital bonds in nitrogen. Unhybridized p orbital forms pi bond": 1.0,
            },
            "score_range": "0-7",
            "grade": 6.0,
        },
        {
            "num": 4,
            "question": (
                "How did the Law of Multiple Proportions lead to the conclusion that matter is "
                "made of atoms?\nThis question can be reasonably answered in around 75 words or "
                "fewer.\n"
            ),
            "answer": (
                "The Law of Multiple Proportions states that when two elements form more than one "
                "compound, the masses of one element that combine with a fixed mass of the other "
                "are in simple whole-number ratios. Whole numbers imply something indivisible is "
                "being counted. Since this is mass data, the indivisible unit must be a unit of "
                "mass — the atom."
            ),
            "rubric_weights": {
                "Fixed mass of one element": 1.0,
                "Mass data in LoMP": 1.0,
                "Combine to form compounds": 1.0,
                "Integer/whole number ratio": 1.0,
                "Whole numbers mean indivisible/discrete": 1.0,
                "Indivisible unit of mass = atom": 1.0,
            },
            "score_range": "0-6",
            "grade": 6.0,
        },
    ]

    def __init__(self, dataset, llm_model):
        self.dataset   = dataset
        self.llm_model = llm_model

        instruction = (
            "You are an automated grader for a college-level chemistry class.\n"
            "You are given a question, a student's answer, and a grading rubric.\n"
            "The rubric lists the criteria to check, each worth the indicated number of points.\n"
            "Your task is to decide which criteria are satisfied by the answer "
            "and compute the total grade.\n\n"
            "Important output rule:\n"
            "Your response must end with exactly one line in the format:\n"
            "Grade (<score_range>): <number>\n"
            "where <number> is a float equal to the sum of points for satisfied criteria. "
            "Do not output anything after the grade line."
        )

        few_shot_lines = ["FEW-SHOT EXAMPLES:\n"]
        for ex in self.FEW_SHOT_EXAMPLES:
            rubric_str = self._format_rubric(ex["rubric_weights"])
            few_shot_lines.append(
                f"Example #{ex['num']}\n"
                f"Question:\n{ex['question']}\n"
                f"Answer:\n{ex['answer']}\n\n"
                f"Grading rubric:\n{rubric_str}\n"
                f"Grade ({ex['score_range']}): {ex['grade']:.1f}\n"
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

    @staticmethod
    def _format_rubric(rubric_weights: dict) -> str:
        """Render rubric criteria with point values, no True/False."""
        lines = []
        for criterion, weight in rubric_weights.items():
            pts = int(weight) if weight == int(weight) else weight
            lines.append(f"  - {criterion} [{pts} pt]")
        return "\n".join(lines)

    def _get_rubric_weights(self, sample: dict) -> dict:
        """Retrieve rubric weights for this sample's task from the dataset."""
        task_idx = sample.get("task_idx")
        if task_idx is not None and hasattr(self.dataset, "task2rubric_weights"):
            return self.dataset.task2rubric_weights[task_idx]
        # Fallback: all weights 1.0 inferred from gold_rubric keys
        return {k: 1.0 for k in sample.get("gold_rubric", {}).keys()}

    def make_prompt(self, sample: dict) -> str:
        rubric_weights = self._get_rubric_weights(sample)
        rubric_str     = self._format_rubric(rubric_weights)

        instruction = self.prompt.build_zeroshot_instruction()
        few_shot    = self.prompt.few_shot

        current = (
            "Now grade the following answer.\n\n"
            f"Question:\n{sample['task']}\n\n"
            f"Answer:\n{sample['student_answer']}\n\n"
            f"Grading rubric:\n{rubric_str}\n"
        )

        user_content = "\n\n".join(
            p for p in [instruction, few_shot, current] if p.strip()
        )

        messages = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": f"Grade ({sample['score_range']}): "},
        ]

        prompt = self.llm_model.apply_chat_template(messages, add_generation_prompt=False)
        return self.llm_model.clean_model_specific_completion(prompt)

    def clean_output(self, text: str) -> str:
        for tok in [
            "<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|eot_id|>",
            "<|end_of_text|>", "<|pad|>", "<end_of_turn>", "</s>",
            "\u00ad", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
        ]:
            text = text.replace(tok, "")
        return text.strip()

    def parse_answer(self, completion: str) -> Optional[float]:
        text = self.clean_output(completion)
        if not text:
            return None
        for m in self._GRADE_RE.finditer(text):
            num_str = m.group("num") or m.group("bare")
            if num_str:
                try:
                    return float(num_str)
                except ValueError:
                    pass
        for line in reversed(text.splitlines()):
            try:
                return float(line.strip())
            except ValueError:
                pass
        return None

    def evaluate(self, results: list) -> dict:
        n_total = len(results)
        n_parsed = n_exact = 0
        abs_errors = []

        for s in results:
            pred = s.get("predicted_answer")
            gold = s.get("gold_answer")
            if pred is None:
                continue
            n_parsed += 1
            err = abs(float(pred) - float(gold))
            abs_errors.append(err)
            if isclose(float(pred), float(gold), abs_tol=0.01):
                n_exact += 1

        mae = round(sum(abs_errors) / n_parsed, 4) if n_parsed else None
        metrics = {
            "n_total":        n_total,
            "n_parsed":       n_parsed,
            "n_parse_error":  n_total - n_parsed,
            "n_exact_match":  n_exact,
            "parse_rate":     round(n_parsed / n_total, 4) if n_total  else None,
            "accuracy":       round(n_exact  / n_parsed, 4) if n_parsed else None,
            "accuracy_total": round(n_exact  / n_total,  4) if n_total  else None,
            "mae":            mae,
        }

        print("\n=== RiceChem Baseline Results ===")
        print(f"  Samples:      {n_total}")
        print(f"  Parsed:       {n_parsed}  ({metrics['parse_rate']:.1%})")
        print(f"  Exact match:  {n_exact}   accuracy={metrics['accuracy']:.4f}  "
              f"accuracy_total={metrics['accuracy_total']:.4f}")
        print(f"  MAE:          {mae}")
        return metrics
