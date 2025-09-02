
import json
import os
import re
import random
from typing import List, Tuple, Dict
from collections import defaultdict

Rule = Tuple[List[str], str]  # ([lhs_ids], rhs_id)

# ----------------------------
# Parsing / serialization
# ----------------------------

def parse_step_proof(step: str) -> List[Rule]:
    """
    Parse EntailmentBank step_proof into Rules.
    Supports optional trailing annotations after ':' and arbitrary LHS arity.
    Example chunk: "sent1 & sent17 -> int1: some text"
    """
    rules: List[Rule] = []
    for chunk in step.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Remove trailing annotation after colon (but only after the RHS)
        # Strategy: first split on '->', then clean RHS part optionally at ':'
        if '->' not in chunk:
            continue
        lhs_raw, rhs_raw = chunk.split('->', 1)
        lhs_raw = lhs_raw.strip()
        rhs_raw = rhs_raw.strip()
        # Remove trailing colon annotation from RHS side
        rhs_clean = re.sub(r':\s*.*$', '', rhs_raw).strip()
        # Validate token-ish RHS id
        m = re.match(r'^(\w+)$', rhs_clean)
        if not m:
            continue
        rhs = m.group(1)
        # Split LHS on '&' and clean tokens
        lhs_ids = [tok.strip() for tok in lhs_raw.split('&') if tok.strip()]
        rules.append((lhs_ids, rhs))
    return rules


def serialize_step_proof(rules: List[Rule]) -> str:
    """
    Serialize Rules back to EntailmentBank step_proof string.
    Keeps simple 'A & B -> C; ...' formatting. No annotations appended.
    """
    parts = []
    for lhs, rhs in rules:
        if len(lhs) == 0:
            parts.append(f'-> {rhs}')
        elif len(lhs) == 1:
            parts.append(f'{lhs[0]} -> {rhs}')
        else:
            parts.append(f' {" & ".join(lhs)} -> {rhs}')
    return '; '.join(parts) + '; '


# ----------------------------
# Graph utilities
# ----------------------------

def build_graph(rules: List[Rule]):
    parents = defaultdict(list)   # rhs -> list of lhs lists (each rule)
    children = defaultdict(list)  # lhs_id -> list of rhs ids
    for lhs_ids, rhs in rules:
        parents[rhs].append(lhs_ids)
        for x in lhs_ids:
            children[x].append(rhs)
    return parents, children


def collect_supporting_rules(rules: List[Rule], target_rhs: str) -> List[int]:
    """
    Return indices of rules on (some) path(s) to target_rhs by backtracking
    through intermediate nodes (ids starting with 'int').
    """
    idx_by_rhs = defaultdict(list)
    for i, (lhs, rhs) in enumerate(rules):
        idx_by_rhs[rhs].append(i)

    supporting = set()
    frontier = [target_rhs]
    seen = set()
    while frontier:
        rhs = frontier.pop()
        if rhs in seen:
            continue
        seen.add(rhs)
        for i in idx_by_rhs.get(rhs, []):
            if i in supporting:
                continue
            supporting.add(i)
            lhs_ids, _ = rules[i]
            for lhs in lhs_ids:
                if lhs.startswith('int'):
                    frontier.append(lhs)
    return sorted(supporting)


# ----------------------------
# Helpers for safe interventions
# ----------------------------

def _pick_rule_with_min_arity(rules: List[Rule], idxs: List[int], min_arity: int) -> int:
    cand = [i for i in idxs if len(rules[i][0]) >= min_arity]
    return random.choice(cand) if cand else -1

def _pick_distractor(distractors: List[str], forbidden: List[str]) -> str:
    pool = [d for d in distractors if d not in forbidden]
    return random.choice(pool) if pool else None

def _ensure_structural_change(old_rules: List[Rule], new_rules: List[Rule]) -> bool:
    return any(old_rules[i][0] != new_rules[i][0] or old_rules[i][1] != new_rules[i][1]
               for i in range(len(old_rules)))


# ----------------------------
# Interventions
# ----------------------------

def delete_one_antecedent(rules: List[Rule], target_rules: List[int], rng: random.Random = None) -> List[Rule]:
    """
    Delete exactly one antecedent from a supporting rule with arity >= 2.
    """
    rng = rng or random
    new_rules = [([*lhs], rhs) for lhs, rhs in rules]
    i = _pick_rule_with_min_arity(new_rules, target_rules, min_arity=2)
    if i == -1:
        return rules  # no valid deletion
    lhs, rhs = new_rules[i]
    del_idx = rng.randrange(len(lhs))
    lhs.pop(del_idx)
    new_rules[i] = (lhs, rhs)
    return new_rules if _ensure_structural_change(rules, new_rules) else rules


def replace_antecedent_with_distractor(
    rules: List[Rule], target_rules: List[int], distractors: List[str], rng: random.Random = None
) -> List[Rule]:
    """
    Replace one antecedent of a supporting rule with a distractor (keeps arity).
    """
    rng = rng or random
    if not target_rules or not distractors:
        return rules
    new_rules = [([*lhs], rhs) for lhs, rhs in rules]
    i = rng.choice(target_rules)
    lhs, rhs = new_rules[i]
    if not lhs:
        return rules
    j = rng.randrange(len(lhs))
    d = _pick_distractor(distractors, forbidden=lhs + [rhs])
    if d is None:
        return rules
    lhs[j] = d
    new_rules[i] = (lhs, rhs)
    return new_rules if _ensure_structural_change(rules, new_rules) else rules


def rewire_drop_support_creation(rules: List[Rule], target_rules: List[int]) -> List[Rule]:
    """
    Remove a rule that produces some intermediate 'int*' that is still used downstream.
    This creates a dangling reference (hard break) without touching texts.
    """
    new_rules = [([*lhs], rhs) for lhs, rhs in rules]
    # choose a supporting rule that produces an intermediate
    candidates = [i for i in target_rules if new_rules[i][1].startswith('int')]
    if not candidates:
        return rules
    i = candidates[0]
    del new_rules[i]
    return new_rules if len(new_rules) < len(rules) else rules


def global_break(
    rules: List[Rule], target_rhs: str, distractors: List[str], rng: random.Random = None
) -> List[Rule]:
    """
    Apply a destructive edit to every rule on the path(s) to target_rhs:
    - If arity >= 2: drop the first antecedent
    - If arity == 1: replace it with a distractor
    """
    rng = rng or random
    supp = collect_supporting_rules(rules, target_rhs)
    new_rules = [([*lhs], rhs) for lhs, rhs in rules]
    for i in supp:
        lhs, rhs = new_rules[i]
        if len(lhs) >= 2:
            lhs.pop(0)
        elif len(lhs) == 1:
            d = _pick_distractor(distractors, forbidden=lhs + [rhs])
            if d is None:
                # fallback: if no distractor, attempt to drop (results in vacuous rule)
                lhs.clear()
            else:
                lhs[0] = d
        new_rules[i] = (lhs, rhs)
    return new_rules if _ensure_structural_change(rules, new_rules) else rules


# ----------------------------
# Example usage on one sample
# ----------------------------

def _resolve_target_rhs(rules: List[Rule], preferred: str) -> str:
    """
    Pick the RHS token to target. If 'preferred' (e.g., 'int2') doesn’t appear
    as a RHS in the proof, but 'hypothesis' does, use 'hypothesis'. Otherwise,
    if neither appears, fall back to the last RHS.
    """
    rhs_set = {rhs for _, rhs in rules}
    if preferred in rhs_set:
        return preferred
    if 'hypothesis' in rhs_set:
        return 'hypothesis'
    # fallback: choose the final rule’s RHS if available
    return rules[-1][1] if rules else preferred


def intervene_step_proof(step_proof: str,
                         hypothesis_id: str,
                         distractors: List[str],
                         mode: str = "replace",
                         seed: int = 0,
                         verbose: bool = True) -> str:
    """
    mode ∈ {"delete", "replace", "rewire", "global"}.
    Returns a new step_proof string with a STRUCTURAL intervention applied.
    Adds robust target resolution and diagnostics.
    """
    rng = random.Random(seed)
    rules = parse_step_proof(step_proof)
    if verbose:
        print(f"[diag] parsed {len(rules)} rules")
        for i,(lhs,rhs) in enumerate(rules):
            print(f"  rule[{i}]: {' & '.join(lhs)} -> {rhs}")

    # --- resolve which RHS to aim at ---
    target_rhs = _resolve_target_rhs(rules, hypothesis_id)
    if verbose and target_rhs != hypothesis_id:
        print(f"[diag] targeting '{target_rhs}' (preferred '{hypothesis_id}' not present as RHS)")

    supp = collect_supporting_rules(rules, target_rhs)
    if verbose:
        print(f"[diag] supporting rule idx for '{target_rhs}': {supp}")

    # If no supporting rules found (e.g., unusual format), fallback to all rules
    target_rules = supp if supp else list(range(len(rules)))
    if verbose and not supp:
        print("[diag] no supporting path found; falling back to editing any rule")

    # --- perform edit ---
    if mode == "delete":
        edited = delete_one_antecedent(rules, target_rules, rng)
    elif mode == "replace":
        edited = replace_antecedent_with_distractor(rules, target_rules, distractors, rng)
    elif mode == "rewire":
        edited = rewire_drop_support_creation(rules, target_rules)
    elif mode == "global":
        edited = global_break(rules, target_rhs, distractors, rng)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Verify structural change; if none, force a small change as a last resort
    if edited == rules:
        if verbose:
            print("[diag] first attempt made no structural change; forcing a replace on the last rule")
        # Force a replace on the last rule if possible
        forced_targets = [len(rules) - 1] if rules else []
        edited = replace_antecedent_with_distractor(rules, forced_targets, distractors, rng)

    new_step = serialize_step_proof(edited)
    if verbose:
        print("[diag] NEW step_proof:", new_step)
    return new_step


class EntailmentIntervention:
    def __init__(self, dataset, llm_stop_token: str):
        """
        Initialize the intervention class with dataset and stop token.
        
        Args:
            dataset: The EntailmentBank dataset instance
            llm_stop_token: The stop token used by the LLM model
        """
        self.dataset = dataset
        self.llm_stop_token = llm_stop_token

        self.modes = ["delete", "replace", "rewire", "global"]

        self.question_verbalizer = "## Question"
        self.question_separator = ": "
        self.context_verbalizer = "## Context"
        self.context_separator = ": "
        self.hypothesis_verbalizer = "## Hypothesis"
        self.hypothesis_separator = ": "
        self.proof_verbalizer = "## Proof"
        self.proof_separator = ": "
        self.answer_verbalizer = "## Conclusion"
        self.answer_separator = ": "

    def make_prompt(self, sample: dict) -> str:
        """
        Create a prompt for the LLM to generate reasoning steps and final answer.
        
        Args:
            sample: Dictionary containing the entailment sample data
            
        Returns:
            str: Formatted prompt for the LLM
        """
        

        prompt = f"""You are a Proof-Checker. You are given a question, context, hypothesis, proof, and conclusion. Your task is to check whether the proof supports the conclusion.

{self.question_verbalizer}{self.question_separator}{sample["question"]}

{self.context_verbalizer}{self.context_separator}{sample["context"]}

{self.hypothesis_verbalizer}{self.hypothesis_separator}{sample["hypothesis"]}

{self.proof_verbalizer}{self.proof_separator}{sample["proof"]}

{self.answer_verbalizer}{self.answer_separator}{sample["answer"]}

If the proof supports the conclusion, return "Yes".
If the proof does not support the conclusion, return "No".

Only return "Yes" or "No".
"""
        return prompt
    
    def make_intervention(self, generated_output, mode: str):
        """
        Create interventions by flipping reasoning steps in the generated output.
        
        Args:
            generated_output: The original LLM generation (could be dict or str depending on format)
            
        Returns:
            List of dictionaries with intervention data, or None if intervention fails
        """
        # NOTE: In my case, checklists are in prompt, not in completion
        prompt = generated_output['prompt']
        completion = generated_output['completion']

        step_proof = prompt.split(self.proof_verbalizer + self.proof_separator)[1]\
            .split(self.answer_verbalizer + self.answer_separator)[0]

        intervened_prompts = []
        for i in range(5):
            # NOTE: Ideally I need access to the full dataset sample, to get the distractors, hypothesis_id, etc.
            # new_step_proof = intervene_step_proof(step_proof=step_proof, mode=mode)
            # new_prompt = prompt.replace(step_proof, new_step_proof)
            
            new_output = {'prompt': "INTERVENTION " + str(i) + ": " + prompt}
            intervened_prompts.append(new_output)

        return intervened_prompts

    def validate_intervention(self, original_output, intervened_output, original_reasoning, intervention_reasoning):
        """
        Validate that an intervention correctly modifies only the intended reasoning step.
        
        Args:
            original_output: The original LLM generation
            intervened_output: The intervened generation
            original_reasoning: The original reasoning step
            intervention_reasoning: The intervened reasoning step
            
        Returns:
            bool: True if intervention is valid, False otherwise
        """
        # TODO: Implement this
        return True

    def validate_all_interventions(self, original_output, intervention_data):
        """
        Validate all interventions for a given sample.
        
        Args:
            original_output: The original LLM generation
            intervention_data: List of intervention dictionaries
            
        Returns:
            bool: True if all interventions are valid, False otherwise
        """
        # TODO: Implement this
        return True

    def reconstruct_interventions_to_prompt(self, original_output, intervened_completions):
        """
        Reconstruct prompts for completion after intervention.
        
        Args:
            original_output: The original LLM generation
            intervened_completions: List of intervention dictionaries
            
        Returns:
            List of prompts ready for LLM completion
        """
        return [original_output['prompt']] + [intervention['prompt'] for intervention in intervened_completions]

    def extract_target_from_prompt(self, generated_output):
        """
        Extract the final answer/target from the generated output.
        
        Args:
            generated_output: The LLM generation (could be dict or str)
            
        Returns:
            The extracted target value (type depends on dataset)
        """
        pass

    def infer_completion(self, completion_output):
        """
        Extract the completion result after intervention.
        
        Args:
            completion_output: The LLM completion after intervention
            
        Returns:
            The inferred result (type depends on dataset)
        """
        return completion_output.split(self.answer_verbalizer + self.answer_separator)[1]



if __name__ == "__main__":
  dataset_path = "entailment_bank/dataset/task_2" #we look at problem with distractors

  train_path = os.path.join(dataset_path, "train.jsonl")
  dev_path = os.path.join(dataset_path, "dev.jsonl")
  test_path = os.path.join(dataset_path, "test.jsonl")

  def read_jsonl(path):
      with open(path, "r") as f:
          return [json.loads(line) for line in f]

  train_dataset = read_jsonl(train_path)
  dev_dataset = read_jsonl(dev_path)
  test_dataset = read_jsonl(test_path)

  print('Train dataset size: ', len(train_dataset))
  print('Dev dataset size: ', len(dev_dataset))
  print('Test dataset size: ', len(test_dataset))
  print('Total dataset size: ', len(train_dataset) + len(dev_dataset) + len(test_dataset))

  ex = train_dataset[0]  # your dict
  step = ex['meta']['step_proof'] if 'meta' in ex and 'step_proof' in ex['meta'] else ex['step_proof']

  new_step = intervene_step_proof(
      step_proof = step,
      hypothesis_id = ex["meta"]["hypothesis_id"],
      distractors = ex["meta"]["distractors"],
      mode = "replace",   # or "delete", "rewire", "global", "replace"
      seed = 42           # fix seed for reproducibility
  )

  print("Original:", step)
  print("Edited:", new_step)
