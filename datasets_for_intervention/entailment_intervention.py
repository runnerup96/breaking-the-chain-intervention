"""
EntailmentBank intervention logic for the Breaking the Chain pipeline.

Module-level mutation functions (intervention primitives):
  _pick_rule_with_min_arity, _pick_distractor, _ensure_structural_change,
  _resolve_target_rhs                              — private helpers
  delete_one_antecedent, replace_antecedent_with_distractor,
  rewire_drop_support_creation, global_break       — structural mutation primitives
  intervene_step_proof                             — main mutation entry-point

These functions import Rule / parse_step_proof / serialize_step_proof /
collect_supporting_rules from entailment_structure_processor, where the
data structures and graph analysis live.

EntailmentIntervention class:
  - __init__ calls _get_prompt_structure()
  - clean_llm_output delegates to processor
  - infer_completion → (Optional[bool], dict)  — True/False/None
  - classify_generation → "correct" | "incorrect" | "error"
  - make_structure_intervention calls intervene_step_proof directly
"""

import hashlib
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from datasets_for_intervention.prompt import Prompt
from datasets_for_intervention.entailment_structure_processor import (
    Rule,
    parse_step_proof,
    serialize_step_proof,
    collect_supporting_rules,
)


# ===========================================================================
# Private helpers
# ===========================================================================

def _pick_rule_with_min_arity(
    rules: List[Rule], idxs: List[int], min_arity: int
) -> int:
    cand = [i for i in idxs if len(rules[i].lhs_ids) >= min_arity]
    return random.choice(cand) if cand else -1


def _pick_distractor(distractors: List[str], forbidden: List[str]) -> Optional[str]:
    pool = [d for d in distractors if d not in forbidden]
    return random.choice(pool) if pool else None


def _ensure_structural_change(old: List[Rule], new: List[Rule]) -> bool:
    return any(
        old[i].lhs_ids != new[i].lhs_ids or old[i].rhs_id != new[i].rhs_id
        for i in range(len(old))
    )


def _resolve_target_rhs(rules: List[Rule], preferred: str) -> str:
    rhs_set = {r.rhs_id for r in rules}
    if preferred in rhs_set:
        return preferred
    if 'hypothesis' in rhs_set:
        return 'hypothesis'
    return rules[-1].rhs_id if rules else preferred


# ===========================================================================
# Structural mutation primitives
# ===========================================================================

def delete_one_antecedent(
    rules: List[Rule], target_rules: List[int], rng: random.Random = None
) -> List[Rule]:
    """Delete exactly one antecedent from a supporting rule with arity >= 2."""
    rng = rng or random
    new = [Rule([*r.lhs_ids], r.rhs_id, r.annotation) for r in rules]
    i = _pick_rule_with_min_arity(new, target_rules, min_arity=2)
    if i == -1:
        return rules
    new[i].lhs_ids.pop(rng.randrange(len(new[i].lhs_ids)))
    assert _ensure_structural_change(rules, new)
    return new


def replace_antecedent_with_distractor(
    rules: List[Rule], target_rules: List[int], distractors: List[str],
    rng: random.Random = None
) -> List[Rule]:
    """Replace one antecedent of a supporting rule with a distractor (keeps arity)."""
    rng = rng or random
    if not target_rules or not distractors:
        return rules
    new  = [Rule([*r.lhs_ids], r.rhs_id, r.annotation) for r in rules]
    i    = rng.choice(target_rules)
    rule = new[i]
    if not rule.lhs_ids:
        return rules
    j = rng.randrange(len(rule.lhs_ids))
    d = _pick_distractor(distractors, forbidden=rule.lhs_ids + [rule.rhs_id])
    if d is None:
        return rules
    rule.lhs_ids[j] = d
    assert _ensure_structural_change(rules, new)
    return new


def rewire_drop_support_creation(
    rules: List[Rule], target_rules: List[int]
) -> List[Rule]:
    """Remove an int*-producing rule, creating a dangling reference."""
    new        = [Rule([*r.lhs_ids], r.rhs_id, r.annotation) for r in rules]
    candidates = [i for i in target_rules if new[i].rhs_id.startswith('int')]
    if not candidates:
        rng = random.Random(hash(str(rules)))
        print("WARNING: No int* candidates for rewire; falling back to delete")
        return delete_one_antecedent(rules, target_rules, rng)
    del new[candidates[0]]
    return new


def global_break(
    rules: List[Rule], target_rhs: str, distractors: List[str],
    rng: random.Random = None
) -> List[Rule]:
    """Destructively edit every rule on the path(s) to target_rhs."""
    rng  = rng or random
    supp = collect_supporting_rules(rules, target_rhs)
    new  = [Rule([*r.lhs_ids], r.rhs_id, r.annotation) for r in rules]
    for i in supp:
        rule = new[i]
        if len(rule.lhs_ids) >= 2:
            rule.lhs_ids.pop(0)
        elif len(rule.lhs_ids) == 1:
            d = _pick_distractor(distractors, forbidden=rule.lhs_ids + [rule.rhs_id])
            if d is not None:
                rule.lhs_ids[0] = d
    assert _ensure_structural_change(rules, new)
    return new


def intervene_step_proof(
    step_proof: Optional[str],
    hypothesis_id: str,
    distractors: List[str],
    mode: str = "replace",
    seed: int = 0,
    verbose: bool = True,
) -> Optional[str]:
    """Apply a structural intervention to a step_proof string.

    mode ∈ {"delete", "replace", "rewire", "global"}.
    Returns the modified proof string, or None if step_proof is None.
    """
    if step_proof is None:
        return None
    rng   = random.Random(seed)
    rules = parse_step_proof(step_proof)

    if verbose:
        print(f"[diag] parsed {len(rules)} rules")
        for i, r in enumerate(rules):
            print(f"  rule[{i}]: {' & '.join(r.lhs_ids)} -> {r.rhs_id}")

    target_rhs   = _resolve_target_rhs(rules, hypothesis_id)
    supp         = collect_supporting_rules(rules, target_rhs)
    target_rules = supp if supp else list(range(len(rules)))

    if verbose:
        if target_rhs != hypothesis_id:
            print(f"[diag] targeting '{target_rhs}' ('{hypothesis_id}' not a RHS)")
        print(f"[diag] supporting rule idx: {supp}")
        if not supp:
            print("[diag] no supporting path; editing any rule")

    if   mode == "delete":  edited = delete_one_antecedent(rules, target_rules, rng)
    elif mode == "replace": edited = replace_antecedent_with_distractor(rules, target_rules, distractors, rng)
    elif mode == "rewire":  edited = rewire_drop_support_creation(rules, target_rules)
    elif mode == "global":  edited = global_break(rules, target_rhs, distractors, rng)
    else: raise ValueError(f"Unknown mode: {mode!r}")

    if edited is rules:
        if verbose: print("[diag] no change; forcing replace on last rule")
        edited = replace_antecedent_with_distractor(
            rules, [len(rules) - 1] if rules else [], distractors, rng
        )

    result = serialize_step_proof(edited)
    if verbose: print("[diag] NEW step_proof:", result)
    return result


class EntailmentIntervention:
    """Pipeline orchestration for EntailmentBank.

    Mirrors RiceChem / TabFact interface:
      __init__                        calls _get_prompt_structure()
      clean_llm_output                delegates to processor
      infer_completion                (Optional[bool], dict) — True/False/None
      classify_generation             "correct" | "incorrect" | "error"
      make_intervention               full pipeline step
      make_structure_intervention     Local Edits / Correction
      collect_intervention_completion
      interventions_to_prompt
    """

    edit_modes: List[str] = ["delete", "replace", "rewire"]

    def __init__(
        self,
        dataset,
        llm_model,
        tool,
        processor,
        few_shot_examples: List[Dict],
        prompting_regime: str = "standard",
        tool_mode: str = "none",
    ):
        assert prompting_regime in ["standard", "detailed", "max_detailed"]
        assert tool_mode in ["none", "simple", "structured"]

        self.dataset = dataset
        self.llm_model = llm_model
        self.tool = tool
        self.processor = processor
        self.few_shot_examples = few_shot_examples
        self.prompting_regime = prompting_regime
        self.tool_mode = tool_mode if tool_mode != "none" else None

        self.question_prefix = "## Question\n"
        self.context_prefix = "## Context\n"
        self.hypothesis_prefix = "## Hypothesis\n"
        self.proof_prefix = "## Proof\n"
        self.final_answer_prefix = "## Final Answer\nIs the hypothesis correct? "
        self.small_proof_prefix = "Proof"
        self.small_final_answer_prefix = "Final Answer"

        instruction, tool_call_instruction, few_shot_text = self._get_prompt_structure()

        self.prompt = Prompt(
            prompting_regime=self.prompting_regime,
            use_tool_call=self.tool_mode is not None,
            tool_call_instruction=tool_call_instruction,
            instruction=instruction,
            few_shot=few_shot_text,
            llm_model=self.llm_model,
        )

    def clean_llm_output(self, text):
        tokens_to_remove = [
            '<|im_end|>', '<|endoftext|>', '<|im_start|>', '<|eot_id|>',
            '<|end_of_text|>', '<|pad|>',
            '<end_of_turn>',
            '</s>',
            '\u00ad', '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
        ]
        for token in tokens_to_remove:
            text = text.replace(token, '')
        return text.strip()

    def infer_completion(
        self, completion: str, sample: Dict = None, short_completion: bool = False
    ) -> Tuple[Optional[bool], Dict]:
        """Extract the Yes/No answer. Returns (True/False/None, tool_proof_nodes)."""
        if self.tool_mode:
            tool_args = self.processor.extract_tool_args(completion, short_completion)
            return (
                self.processor.extract_final_answer(completion, short_completion),
                tool_args if tool_args is not None else {},
            )
        return self.processor.extract_final_answer(completion, short_completion), {}

    def classify_generation(
        self, completion: str, mediator_proof: Optional[str], gold_proof: str
    ) -> str:
        if mediator_proof is None:
            return "error"
        if self.processor.check_generation_format_mistakes(completion):
            return "error"
        if self.processor.extract_final_answer(completion) is None:
            return "error"
        match = self.processor.compare_structures(mediator_proof, gold_proof)
        return "correct" if match == 1 else "incorrect"

    def make_intervention(self, sample: Dict, generated_output: Dict) -> Dict:
        completion = self.clean_llm_output(generated_output["completion"])
        sample["raw_generation"] = completion

        predicted_proof = self.processor.extract_mediator(completion)
        generation_status = self.classify_generation(
            completion, predicted_proof, sample["gold_proof"]
        )
        sample["generation_status"] = generation_status
        # Store the extracted mediator M' in mediator_proof.
        # sample["gold_proof"] / sample["gold_score"] are never modified.
        # None means the completion could not be parsed at all.
        sample["mediator_proof"] = predicted_proof

        if generation_status == "error":
            sample["score_before_intervention"] = None
            sample["tool_proof_nodes"] = None
            sample["structure_intervention"] = {"Local Edits": [], "Correction": []}
            return sample

        sample["score_before_intervention"], tool_proof_nodes = self.infer_completion(
            completion, sample, short_completion=False
        )
        sample["tool_proof_nodes"] = tool_proof_nodes
        sample["structure_intervention"] = self.make_structure_intervention(sample)
        return sample

    def make_structure_intervention(self, sample: Dict) -> Dict:
        status = sample.get("generation_status")

        if status == "correct":
            local_edits = []
            for mode in self.edit_modes:
                edit = deepcopy(sample)
                # Intervene on the model's predicted proof (not the gold proof).
                # edit["proof"] is what make_prompt shows the model as the "given" proof.
                edit["proof"] = intervene_step_proof(
                    edit["mediator_proof"],
                    mode=mode,
                    distractors=edit["distractors"],
                    hypothesis_id=edit["hypothesis_id"],
                    seed=hash(edit["id"]),
                    verbose=False,
                )
                edit["score"] = False
                edit["expected_score_after_intervention"] = False
                local_edits.append(edit)
            return {"Local Edits": local_edits, "Correction": []}

        if status == "incorrect":
            gold_idx = next(
                i for i in range(len(self.dataset))
                if self.dataset[i]["id"] == sample["id"]
            )
            corr = deepcopy(sample)
            # For Correction: show the model the gold proof as mediator.
            # corr["proof"] is what make_prompt/format_example renders.
            # corr["mediator_proof"] reflects what we're feeding back.
            corr["proof"] = self.dataset[gold_idx]["gold_proof"]
            corr["mediator_proof"] = self.dataset[gold_idx]["gold_proof"]
            corr["score"] = True
            corr["expected_score_after_intervention"] = True
            return {"Local Edits": [], "Correction": [corr]}

        return {"Local Edits": [], "Correction": []}

    def interventions_to_prompt(self, sample: Dict) -> List[str]:
        interv = sample["structure_intervention"]
        return (
            [self.make_prompt(e, include_gold_structure=True) for e in interv.get("Local Edits", [])]
            + [self.make_prompt(c, include_gold_structure=True) for c in interv.get("Correction", [])]
        )

    def collect_intervention_completion(
        self, sample: Dict, generated_output: List[Dict]
    ) -> Dict:
        """Order matches interventions_to_prompt: Local Edits first, then Correction."""
        completions = [self.clean_llm_output(g["completion"]) for g in generated_output]
        interv = sample["structure_intervention"]
        idx = 0

        for i in range(len(interv.get("Local Edits", []))):
            c = completions[idx]
            answer, nodes = self.infer_completion(c, interv["Local Edits"][i], short_completion=True)
            interv["Local Edits"][i]["raw_generation"] = c
            interv["Local Edits"][i]["score_after_intervention"] = answer
            if self.tool_mode:
                interv["Local Edits"][i]["tool_proof_nodes_after_intervention"] = nodes
            idx += 1

        for i in range(len(interv.get("Correction", []))):
            c = completions[idx]
            answer, nodes = self.infer_completion(c, interv["Correction"][i], short_completion=True)
            interv["Correction"][i]["raw_generation"] = c
            interv["Correction"][i]["score_after_intervention"] = answer
            if self.tool_mode:
                interv["Correction"][i]["tool_proof_nodes_after_intervention"] = nodes
            idx += 1

        return sample

    @staticmethod
    def get_few_shot_examples(
        train_dataset_path: str,
        prompting_regime: str,
        n_few_shot_examples: int = 5,
    ) -> List[Dict]:
        """Load few-shot examples from the EntailmentBank train split."""
        from datasets_for_intervention.entailment_dataset import EntailmentDataset

        train = EntailmentDataset(train_dataset_path)

        if prompting_regime == "standard":
            stride = max(1, len(train) // (n_few_shot_examples * 2))
            examples = [deepcopy(train[i]) for i in range(0, len(train), stride)]
            return examples[:n_few_shot_examples]

        if prompting_regime in ("detailed", "max_detailed"):
            if not train:
                raise ValueError("Train dataset is empty.")

            def stable_seed(sid: str, mode: str) -> int:
                digest = hashlib.sha256(f"{sid}::{mode}".encode()).digest()
                return int.from_bytes(digest[:4], "big")

            modes = ["rewire", "global", "delete", "replace"]
            mode_idx = 0
            examples = []
            stride = max(1, len(train) // max(1, (n_few_shot_examples + 1) // 2))

            for idx in range(0, len(train), stride):
                if len(examples) >= n_few_shot_examples:
                    break
                orig = deepcopy(train[idx])
                orig_id = orig["id"]
                orig["id"] = f"{orig_id}::orig"
                examples.append(orig)
                if len(examples) >= n_few_shot_examples:
                    break

                broken = deepcopy(train[idx])
                mode = modes[mode_idx % len(modes)]
                mode_idx += 1
                broken["id"] = f"{orig_id}::{mode}"
                broken["proof"] = intervene_step_proof(
                    broken["proof"],
                    hypothesis_id=broken["hypothesis_id"],
                    distractors=broken["distractors"],
                    mode=mode,
                    seed=stable_seed(orig_id, mode),
                    verbose=False,
                )
                broken["score"] = not broken["score"]
                examples.append(broken)

            assert len(examples) == n_few_shot_examples
            return examples

        raise ValueError(f"Unknown prompting_regime: {prompting_regime!r}")

    def _get_prompt_structure(self) -> Tuple[str, str, str]:
        instruction = (
            "You are an expert logical reasoning system specialized in hypothesis verification. "
            "Your task is to evaluate whether a given hypothesis is correct by first constructing "
            "a structured reasoning block (a step-by-step logical proof) and then providing a final answer.\n\n"
            "Task explanation:\n"
            "- You are given a question, context containing factual sentences, and a hypothesis to evaluate.\n"
            "- You must construct a logical proof that traces the reasoning from context sentences to intermediate conclusions.\n"
            "- The final answer determines whether the hypothesis is correct based on your proof.\n\n"
            "Structured reasoning block construction (Proof):\n"
            "- Use only the given context sentences and logical reasoning—do not assume or invent new facts.\n"
            "- Reference context sentences using identifiers (sent1, sent2, etc.) as they appear in the context.\n"
            "- Create intermediate conclusions (int1, int2, etc.) by combining sentences using logical rules.\n"
            "- Follow the format: \"sentX & sentY -> intZ\" for combining multiple sentences, or \"sentX -> intZ\" for single-sentence inferences.\n"
            "- Each step should represent a valid logical inference that brings you closer to evaluating the hypothesis.\n"
            "- Build your proof incrementally, where each intermediate conclusion can be used in subsequent steps.\n"
            "- The final step should connect your reasoning to the hypothesis being evaluated.\n\n"
            "Logical reasoning guidelines:\n"
            "- Ensure each inference step is logically sound and based on the information provided.\n"
            "- If multiple reasoning paths are possible, choose the most direct and clear one.\n\n"
        )
        if not self.tool_mode:
            instruction += (
                "Important output format:\n"
                "Your response must contain exactly two sections in this order:\n"
                "1) Proof: (step-by-step logical reasoning using the sentence reference format)\n"
                "2) Final Answer: Is the hypothesis correct? <Yes/No>\n"
            )

        tool_call_instruction = ""
        if self.tool_mode:
            tool_call_instruction = (
                "Tool usage (REQUIRED):\n"
                f"- After writing the proof, call the tool '{self.tool.name}' with the list "
                "of leaf sentence IDs (sentX) you used. Exclude intermediate conclusions (intX).\n\n"
                "Important output format:\n"
                "Your response must contain exactly three sections in this order:\n"
                f"1) {self.proof_prefix}   (step-by-step reasoning)\n"
                "2) Final tool call:\n"
                f"   TOOL: {self.tool.name}\n"
                '   ARGS: {"proof_nodes": ["sent16", "sent20", ...]}\n'
                f"3) {self.final_answer_prefix}<Yes/No>\n\n"
                "Tool specification:\n" + self.tool.spec_json() + "\n\n"
            )

        few_shot_text = "FEW-SHOT EXAMPLES:\n\n" + "\n\n".join(
            f"# Example {i}\n"
            + self.format_example(ex, True, True, True, True)
            for i, ex in enumerate(self.few_shot_examples)
        )
        return instruction, tool_call_instruction, few_shot_text

    def format_example(
        self,
        example: Dict,
        add_question_context_hypothesis: bool,
        add_proof: bool,
        add_final_answer_prefix: bool,
        add_gold_answer: bool,
    ) -> str:
        out = ""
        sep = "\n"
        if add_question_context_hypothesis:
            out += (
                f"{self.question_prefix}{example['question']}{sep}"
                f"{self.context_prefix}{example['context']}{sep}"
                f"{self.hypothesis_prefix}{example['hypothesis']}"
            )
        if add_proof:
            block = f"{self.proof_prefix}{example['proof']}"
            out = block if not out else out + sep + block
        if add_final_answer_prefix:
            out += sep + self.final_answer_prefix
        if add_gold_answer:
            out += "Yes" if example["score"] else "No"
        return out

    def make_prompt(self, sample: Dict, include_gold_structure: bool = False) -> str:
        current_sample = (
            f"# Example {len(self.few_shot_examples)}\n"
            + self.format_example(sample, True, False, False, False)
        )
        gold_structure = (
            self.format_example(sample, False, True, True, False)
            if include_gold_structure else None
        )
        return self.prompt.make_prompt(
            current_sample=current_sample,
            include_gold_structure=include_gold_structure,
            gold_structure=gold_structure,
        )
