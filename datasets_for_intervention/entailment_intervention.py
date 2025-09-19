
import json
import os
import re
import random
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from copy import deepcopy
from datasets_for_intervention.entailment_dataset import EntailmentDataset

class Rule:
    """
    Represents a logical rule with optional annotation.
    Supports tuple unpacking for backward compatibility: lhs_ids, rhs_id = rule
    """
    def __init__(self, lhs_ids: List[str], rhs_id: str, annotation: Optional[str] = None):
        self.lhs_ids = lhs_ids
        self.rhs_id = rhs_id
        self.annotation = annotation
    
    def __iter__(self):
        """Support tuple unpacking: lhs, rhs = rule"""
        return iter((self.lhs_ids, self.rhs_id))
    
    def __getitem__(self, index):
        """Support indexing: rule[0] for lhs_ids, rule[1] for rhs_id"""
        if index == 0:
            return self.lhs_ids
        elif index == 1:
            return self.rhs_id
        else:
            raise IndexError("Rule index out of range")
    
    def __repr__(self):
        return f"Rule({self.lhs_ids}, {self.rhs_id}, {self.annotation!r})"

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
        # Parse annotation after colon (but only after the RHS)
        # Strategy: first split on '->', then extract annotation from RHS part
        if '->' not in chunk:
            continue
        lhs_raw, rhs_raw = chunk.split('->', 1)
        lhs_raw = lhs_raw.strip()
        rhs_raw = rhs_raw.strip()
        
        # Extract annotation if present
        annotation = None
        if ':' in rhs_raw:
            rhs_part, annotation_part = rhs_raw.split(':', 1)
            rhs_clean = rhs_part.strip()
            annotation = annotation_part.strip() if annotation_part.strip() else None
        else:
            rhs_clean = rhs_raw
        
        # Validate token-ish RHS id
        m = re.match(r'^(\w+)$', rhs_clean)
        if not m:
            continue
        rhs = m.group(1)
        # Split LHS on '&' and clean tokens
        lhs_ids = [tok.strip() for tok in lhs_raw.split('&') if tok.strip()]
        rules.append(Rule(lhs_ids, rhs, annotation))
    return rules


def serialize_step_proof(rules: List[Rule]) -> str:
    """
    Serialize Rules back to EntailmentBank step_proof string.
    Includes annotations if present.
    """
    parts = []
    for rule in rules:
        lhs, rhs = rule.lhs_ids, rule.rhs_id
        if len(lhs) == 0:
            rule_str = f'-> {rhs}'
        elif len(lhs) == 1:
            rule_str = f'{lhs[0]} -> {rhs}'
        else:
            rule_str = f'{" & ".join(lhs)} -> {rhs}'
        
        # Add annotation if present
        if rule.annotation:
            rule_str += f': {rule.annotation}'
        
        parts.append(rule_str)
    return '; '.join(parts) + '; '



# ----------------------------
# Graph utilities
# ----------------------------

def build_graph(rules: List[Rule]):
    parents = defaultdict(list)   # rhs -> list of lhs lists (each rule)
    children = defaultdict(list)  # lhs_id -> list of rhs ids
    for rule in rules:
        parents[rule.rhs_id].append(rule.lhs_ids)
        for x in rule.lhs_ids:
            children[x].append(rule.rhs_id)
    return parents, children


def collect_supporting_rules(rules: List[Rule], target_rhs: str) -> List[int]:
    """
    Return indices of rules on (some) path(s) to target_rhs by backtracking
    through intermediate nodes (ids starting with 'int').
    """
    idx_by_rhs = defaultdict(list)
    for i, rule in enumerate(rules):
        idx_by_rhs[rule.rhs_id].append(i)

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
            rule = rules[i]
            for lhs in rule.lhs_ids:
                if lhs.startswith('int'):
                    frontier.append(lhs)
    return sorted(supporting)


# ----------------------------
# Helpers for safe interventions
# ----------------------------

def _pick_rule_with_min_arity(rules: List[Rule], idxs: List[int], min_arity: int) -> int:
    cand = [i for i in idxs if len(rules[i].lhs_ids) >= min_arity]
    return random.choice(cand) if cand else -1

def _pick_distractor(distractors: List[str], forbidden: List[str]) -> str:
    pool = [d for d in distractors if d not in forbidden]
    return random.choice(pool) if pool else None

def _ensure_structural_change(old_rules: List[Rule], new_rules: List[Rule]) -> bool:
    return any(old_rules[i].lhs_ids != new_rules[i].lhs_ids or old_rules[i].rhs_id != new_rules[i].rhs_id
               for i in range(len(old_rules)))


# ----------------------------
# Interventions
# ----------------------------

def delete_one_antecedent(rules: List[Rule], target_rules: List[int], rng: random.Random = None) -> List[Rule]:
    """
    Delete exactly one antecedent from a supporting rule with arity >= 2.
    """
    rng = rng or random
    new_rules = [Rule([*rule.lhs_ids], rule.rhs_id, rule.annotation) for rule in rules]
    i = _pick_rule_with_min_arity(new_rules, target_rules, min_arity=2)
    if i == -1:
        return rules  # no valid deletion
    rule = new_rules[i]
    del_idx = rng.randrange(len(rule.lhs_ids))
    rule.lhs_ids.pop(del_idx)
    assert _ensure_structural_change(rules, new_rules), f"No structural change for delete_one_antecedent\n{rules}\n{new_rules}"
    return new_rules


def replace_antecedent_with_distractor(
    rules: List[Rule], target_rules: List[int], distractors: List[str], rng: random.Random = None
) -> List[Rule]:
    """
    Replace one antecedent of a supporting rule with a distractor (keeps arity).
    """
    rng = rng or random
    if not target_rules or not distractors:
        return rules
    new_rules = [Rule([*rule.lhs_ids], rule.rhs_id, rule.annotation) for rule in rules]
    i = rng.choice(target_rules)
    rule = new_rules[i]
    if not rule.lhs_ids:
        return rules
    j = rng.randrange(len(rule.lhs_ids))
    d = _pick_distractor(distractors, forbidden=rule.lhs_ids + [rule.rhs_id])
    if d is None:
        return rules
    rule.lhs_ids[j] = d
    assert _ensure_structural_change(rules, new_rules), f"No structural change for replace_antecedent_with_distractor\n{rules}\n{new_rules}"
    return new_rules


def rewire_drop_support_creation(rules: List[Rule], target_rules: List[int]) -> List[Rule]:
    """
    Remove a rule that produces some intermediate 'int*' that is still used downstream.
    This creates a dangling reference (hard break) without touching texts.
    """
    new_rules = [Rule([*rule.lhs_ids], rule.rhs_id, rule.annotation) for rule in rules]
    # choose a supporting rule that produces an intermediate
    candidates = [i for i in target_rules if new_rules[i].rhs_id.startswith('int')]
    if len(candidates) == 0:
        rng = random.Random(hash(str(rules)))
        print("WARNING: No candidates found for rewire_drop_support_creation, deleting one antecedent")
        return delete_one_antecedent(rules, target_rules, rng)
    # assert len(candidates) > 0, f"No candidates found for rewire_drop_support_creation\n{rules}\n{target_rules}"
    i = candidates[0]
    del new_rules[i]
    # no check for structural change, since we are deleting a rule
    return new_rules


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
    new_rules = [Rule([*rule.lhs_ids], rule.rhs_id, rule.annotation) for rule in rules]
    for i in supp:
        rule = new_rules[i]
        if len(rule.lhs_ids) >= 2:
            rule.lhs_ids.pop(0)
        elif len(rule.lhs_ids) == 1:
            d = _pick_distractor(distractors, forbidden=rule.lhs_ids + [rule.rhs_id])
            if d is None:
                # fallback: if no distractor, attempt to drop (results in vacuous rule)
                rule.lhs_ids.clear()
            else:
                rule.lhs_ids[0] = d
    assert _ensure_structural_change(rules, new_rules), f"No structural change for global_break\n{rules}\n{new_rules}"
    return new_rules


# ----------------------------
# Example usage on one sample
# ----------------------------

def _resolve_target_rhs(rules: List[Rule], preferred: str) -> str:
    """
    Pick the RHS token to target. If 'preferred' (e.g., 'int2') doesn't appear
    as a RHS in the proof, but 'hypothesis' does, use 'hypothesis'. Otherwise,
    if neither appears, fall back to the last RHS.
    """
    rhs_set = {rule.rhs_id for rule in rules}
    if preferred in rhs_set:
        return preferred
    if 'hypothesis' in rhs_set:
        return 'hypothesis'
    # fallback: choose the final rule's RHS if available
    return rules[-1].rhs_id if rules else preferred


def intervene_step_proof(step_proof: Optional[str],
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
    if step_proof is None:
        return None
    rng = random.Random(seed)
    rules = parse_step_proof(step_proof)
    if verbose:
        print(f"[diag] parsed {len(rules)} rules")
        for i, rule in enumerate(rules):
            print(f"  rule[{i}]: {' & '.join(rule.lhs_ids)} -> {rule.rhs_id}")

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
    def __init__(self, dataset: EntailmentDataset, few_shot_examples: List[Dict], hsvt_mode: str):
        """
        Initialize the intervention class with dataset and stop token.
        
        Args:
            dataset: The EntailmentBank dataset instance
            few_shot_examples: The few shot examples
            hsvt_mode: The mode of HSVT intervention -- whether to convert the question to lowercase or to use paraphrases
        """
        assert hsvt_mode in ["lower", "paraphrase"]

        self.dataset = dataset
        self.few_shot_examples = few_shot_examples

        self.edit_modes = ["delete", "replace", "rewire"]
        self.hsvt_mode = hsvt_mode
        if self.hsvt_mode == "paraphrase":
            assert all(example["question_paraphrases"] is not None for example in self.dataset), "Dataset must have question paraphrases when using 'paraphrase' HSVT mode"

        self.question_prefix = "## Question\n"
        self.context_prefix = "## Context\n"
        self.hypothesis_prefix = "## Hypothesis\n"
        self.proof_prefix = "## Proof\n"

        self.final_answer_prefix = "## Final Answer\nIs the hypothesis correct? "
        # Used for parsing later
        self.small_final_answer_prefix = "## Final Answer"
        assert self.final_answer_prefix.startswith(self.small_final_answer_prefix)
    
        self.system_prompt = """You are an expert at checking hypotheses correctness.
You will be given a question, some context and a hypothesis.
For each example, you first generate a proof, and then predict the final answer: whether the hypothesis is correct."""

    
    def interventions_to_prompt(self, sample:dict):
        interventions = sample['structure_intervention']
        hsvt_intervention_prompt = [self.make_prompt(interventions['HSVT'][0], include_gold_structure=True)]
        local_edits_intervention_prompt = [ self.make_prompt(edit, include_gold_structure=True) for edit in interventions['Local Edits']]
        global_intervention_prompt = [self.make_prompt(interventions['Global'][0], include_gold_structure=True)]
        all_intervention_prompts = hsvt_intervention_prompt + local_edits_intervention_prompt + global_intervention_prompt
        return all_intervention_prompts

    def infer_completion(self, completion):
        "extract only the completion after the intervention, when we test model ability to make a correct decision"
        if self.small_final_answer_prefix in completion:
            completion = completion.split(self.small_final_answer_prefix)[1].strip()

        if "Yes" in completion and "No" in completion:
            return -1
        elif "Yes" in completion and not "No" in completion:
            return 1
        elif "No" in completion and not "Yes" in completion:
            return 0
        else:
            return -1

    def collect_intervention_completion(self, sample:dict, generated_output:list):
        completion_list = [generation['completion'] for generation in generated_output]
        intervention = sample['structure_intervention']
        intervention_list = ['HSVT'] + ['Local Edits'] * len(intervention['Local Edits']) + ['Global']
        intervention_idx_list = [0] + list(range(len(intervention['Local Edits']))) + [0]
        for completion, intervention_type, idx in zip(completion_list, intervention_list, intervention_idx_list):
            sample['structure_intervention'][intervention_type][idx]['result_after_intervention'] = self.infer_completion(completion)
        return sample


    def format_example(self, example: Dict, add_question_context_hypothesis: bool, add_proof: bool, add_final_answer_prefix: bool, add_gold_answer: bool) -> str:
        """
        Format an example into a prompt.
        """

        formatted_example = ""
        common_sep = "\n"

        if add_question_context_hypothesis:
            formatted_question = f"{self.question_prefix}{example['question']}"
            formatted_context = f"{self.context_prefix}{example['context']}"
            formatted_hypothesis = f"{self.hypothesis_prefix}{example['hypothesis']}"

            formatted_example += formatted_question + common_sep + formatted_context + common_sep + formatted_hypothesis
        
        if add_proof:
            formatted_proof = f"{self.proof_prefix}{example['proof']}"
            formatted_example += formatted_proof if formatted_example == "" else common_sep + formatted_proof
        if add_final_answer_prefix:
            formatted_final_answer_prefix = f"{self.final_answer_prefix}"
            formatted_example += common_sep + formatted_final_answer_prefix
        if add_gold_answer:
            formatted_gold_answer = "Yes" if example["score"] else "No"
            formatted_example += formatted_gold_answer
        
        return formatted_example

    def _extract_entailment_proof(self, completion):
        if not (self.proof_prefix in completion):
            return None
        if not (self.small_final_answer_prefix in completion):
            return None
        proof_plus_something = completion.split(self.proof_prefix)[1].strip()
        proof = proof_plus_something.split(self.small_final_answer_prefix)[0].strip()
        return proof

    def _extract_entailment_answer(self, completion):
        if not (self.final_answer_prefix in completion):
            return None
        return completion.split(self.final_answer_prefix)[1].strip()

    def make_intervention(self, sample: dict, generated_output: dict):
        # TODO: support message list instead of prompts
        # i get the sample, make the intervention
        # here i have gold structure, predicted structure and make intervention on both of them.

        completion = generated_output['completion']
        # here we update the sample with the predicted structure, we have gold result in dataset
        if sample['completion_type'] == "structure_prediction":
            predicted_proof = self._extract_entailment_proof(completion)
            predicted_answer = self._extract_entailment_answer(completion)
            sample['proof'] = predicted_proof
            sample['score'] = predicted_answer
        elif sample['completion_type'] == "gold_structure":
            gold_answer = self.infer_completion(completion)
            sample['score'] = gold_answer

        interventions = self.make_structure_intervention(sample)
        sample['structure_intervention'] = interventions
        return sample

    def make_structure_intervention(self, entailment_sample: dict):
        # TODO: support message list instead of prompts
        # i get a entailment original sample and make a structure intervention
        # i do 3 types of interventions -- HSVT, local edits and global
        # I get a list 3 types of intervented samples -- 1 + M + 1 size, where M is the amount of local edits
        # then these samples are used to make a prompt
        # i also return a list of intervention types for each of the intervented samples
        # this will be tested in tests

        # TODO: change HSVT intervention to something more meaningful than converting the question to lowercase
        hsvt_sample = deepcopy(entailment_sample)
        sample_id = entailment_sample['id']

        if self.hsvt_mode == "lower":
            hsvt_sample['question'] = hsvt_sample["question"].lower()
        elif self.hsvt_mode == "paraphrase":
            n_paraphrases = len(hsvt_sample["question_paraphrases"])
            hash_based_idx = hash(sample_id) % n_paraphrases
            hsvt_sample['question'] = hsvt_sample["question_paraphrases"][hash_based_idx]

        # Local edits intervention -- invalidate the proof in one of the ways: delete, replace, rewire
        local_edits = []
        for mode in self.edit_modes:
            local_edits_sample = deepcopy(entailment_sample)
            local_edits_sample['proof'] = intervene_step_proof(
                local_edits_sample['proof'],
                mode=mode,
                distractors=local_edits_sample['distractors'],
                hypothesis_id=local_edits_sample['hypothesis_id'],
                seed=hash(local_edits_sample['id'])
            )
            # Any intervention invalidates the proof
            local_edits_sample['score'] = not local_edits_sample['score']
            local_edits.append(local_edits_sample)

        # Global intervention
        global_sample = deepcopy(entailment_sample)
        global_sample['proof'] = intervene_step_proof(
            global_sample['proof'],
            mode="global",
            distractors=global_sample['distractors'],
            hypothesis_id=global_sample['hypothesis_id'],
            seed=hash(global_sample['id'])
        )
        global_sample['score'] = not global_sample['score']

        return {"HSVT": [hsvt_sample], "Local Edits": local_edits, "Global": [global_sample]}


    def make_prompt(self, sample: dict, include_gold_structure: bool) -> str:
        """
        Create a prompt for the LLM to generate reasoning steps and final answer.
        
        Args:
            sample: Dictionary containing the entailment sample data
            
        Returns:
            str: Formatted prompt for the LLM
        """
        prompt = self.system_prompt + "\n\n"

        prompt += "\n\n".join(f"# Example {i}\n{self.format_example(example, add_question_context_hypothesis=True, add_proof=True, add_final_answer_prefix=True, add_gold_answer=True)}"
            for i, example in enumerate(self.few_shot_examples))
        # Proof constitutes the gold structure. Check part is only present if the gold structure is included, otherwise model must generate it.
        # Gold answer is never included, since the model always must generate it.
        prompt += f"\n\n# Example {len(self.few_shot_examples)}\n" + self.format_example(sample, add_question_context_hypothesis=True,
            add_proof=include_gold_structure, add_final_answer_prefix=include_gold_structure, add_gold_answer=False)
        return prompt

    def make_message_list(self, sample: dict, include_gold_structure: bool) -> List[Dict[str, str]]:
        """
        Create a message list for the LLM to generate reasoning steps and final answer.
        """
        message_list = []
        message_list.append({"role": "system", "content": self.system_prompt})

        for example in self.few_shot_examples:
            user_message = self.format_example(example, add_question_context_hypothesis=True,
                add_proof=False, add_final_answer_prefix=False, add_gold_answer=False)
            message_list.append({"role": "user", "content": user_message})
            assistant_message = self.format_example(example, add_question_context_hypothesis=False,
                add_proof=True, add_final_answer_prefix=True, add_gold_answer=True)
            message_list.append({"role": "assistant", "content": assistant_message})

        user_message = self.format_example(sample, add_question_context_hypothesis=True,
            add_proof=False, add_final_answer_prefix=False, add_gold_answer=False)
        message_list.append({"role": "user", "content": user_message})
        assistant_message = self.format_example(sample, add_question_context_hypothesis=False,
            add_proof=include_gold_structure, add_final_answer_prefix=include_gold_structure, add_gold_answer=False)
        message_list.append({"role": "assistant", "content": assistant_message})
        return message_list


if __name__ == "__main__":
    dataset_path = "entailment_trees_emnlp2021_data_v3/dataset/task_2" #we look at problem with distractors

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

    ex = train_dataset[14]  # your dict
    step = ex['meta']['step_proof'] if 'meta' in ex and 'step_proof' in ex['meta'] else ex['step_proof']
    print("Hypothesis:", ex["hypothesis"])
    print("Original:", step)
    print("-"*100)

    for mode in ["delete", "replace", "rewire", "global"]:
        new_step = intervene_step_proof(
            step_proof = step,
            hypothesis_id = ex["meta"]["hypothesis_id"],
            distractors = ex["meta"]["distractors"],
            mode = mode,
            seed = 42
        )

        print(f"Edited ({mode}):", new_step)
        print("-"*100)


    val_dataset = EntailmentDataset(dev_path)
    few_shot_dataset = EntailmentDataset(train_path)
    intervention = EntailmentIntervention(val_dataset, [sample for sample in few_shot_dataset[:5]], hsvt_mode="paraphrase")
    print("-"*100)
    print("Prompt:")
    print(intervention.make_prompt(val_dataset[0], include_gold_structure=True))
    print("-"*100)
    print("Message list:")
    message_list = intervention.make_message_list(val_dataset[0], include_gold_structure=True)
    for message in message_list:
        print(message['role'].upper() + ": " + message['content'])
        print("---")
    print()
    print("-"*100)
