"""
EntailmentBank Tool and StructureProcessor for the Breaking the Chain pipeline.

Module-level (shared data structures and pure analysis):
  Rule, parse_step_proof, serialize_step_proof  — proof I/O
  build_graph, collect_supporting_rules          — structural graph analysis

Structural mutations (delete_one_antecedent, intervene_step_proof, …) live in
entailment_intervention.py — they are intervention operations, not parsing.
They import Rule / parse_step_proof / serialize_step_proof / collect_supporting_rules
from here.

EntailmentStructureProcessor  (instance methods — parsing, extraction, comparison):
  clean_llm_output, extract_mediator, extract_final_answer, extract_tool_args,
  compare_structures, jaccard_similarity, check_generation_format_mistakes
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional


class EntailmentTool:
    """Records leaf sent-IDs from the model's proof for mediator_tool_match eval."""

    name: str = "verify_proof"

    @property
    def spec(self) -> Dict:
        return {
            "title": self.name, "type": "object",
            "description": (
                "Record the supporting sentence IDs (leaf nodes) of your logical proof. "
                "Exclude intermediate conclusions (intX)."
            ),
            "returns": {"type": "boolean",
                        "description": "True when the sentences logically support the hypothesis."},
            "properties": {
                "proof_nodes": {
                    "type": "array", "items": {"type": "string"},
                    "description": 'Sorted list of leaf sentX IDs. Example: ["sent16", "sent20"]',
                }
            },
            "required": ["proof_nodes"],
        }

    def spec_json(self) -> str:
        return json.dumps(self.spec, ensure_ascii=False)

    def validate_args(self, args: Dict) -> bool:
        if not isinstance(args, dict): return False
        nodes = args.get("proof_nodes")
        if not isinstance(nodes, list) or not nodes: return False
        return all(isinstance(n, str) and re.match(r'^sent\d+$', n) for n in nodes)

    def calculate_score(self, args: Dict, sample_meta: Dict) -> Optional[bool]:
        return True if self.validate_args(args) else None


class Rule:
    """One logical step: lhs_ids -> rhs_id [ : annotation ].

    Supports tuple unpacking:  lhs_ids, rhs_id = rule
    """
    def __init__(self, lhs_ids: List[str], rhs_id: str, annotation: Optional[str] = None):
        self.lhs_ids    = lhs_ids
        self.rhs_id     = rhs_id
        self.annotation = annotation

    def __iter__(self):
        return iter((self.lhs_ids, self.rhs_id))

    def __getitem__(self, i: int):
        if i == 0:
            return self.lhs_ids
        if i == 1:
            return self.rhs_id
        raise IndexError("Rule index out of range")

    def __repr__(self):
        return f"Rule({self.lhs_ids!r}, {self.rhs_id!r}, {self.annotation!r})"


def parse_step_proof(step: str) -> List[Rule]:
    """Parse an EntailmentBank step_proof string into a list of Rules."""
    rules: List[Rule] = []
    for chunk in step.split(';'):
        chunk = chunk.strip()
        if not chunk or '->' not in chunk:
            continue
        lhs_raw, rhs_raw = chunk.split('->', 1)
        annotation = None
        if ':' in rhs_raw:
            rhs_part, ann = rhs_raw.split(':', 1)
            rhs_raw    = rhs_part.strip()
            annotation = ann.strip() or None
        else:
            rhs_raw = rhs_raw.strip()
        m = re.match(r'^(\w+)$', rhs_raw)
        if not m:
            continue
        rules.append(Rule(
            [t.strip() for t in lhs_raw.split('&') if t.strip()],
            m.group(1),
            annotation,
        ))
    return rules


def serialize_step_proof(rules: List[Rule], include_annotations: bool = True) -> str:
    """Serialize a list of Rules back to a step_proof string."""
    parts = []
    for rule in rules:
        lhs = rule.lhs_ids
        if   not lhs:       s = f'-> {rule.rhs_id}'
        elif len(lhs) == 1: s = f'{lhs[0]} -> {rule.rhs_id}'
        else:               s = f'{" & ".join(lhs)} -> {rule.rhs_id}'
        if rule.annotation and include_annotations:
            s += f': {rule.annotation}'
        parts.append(s)
    return '; '.join(parts) + '; '


def build_graph(rules: List[Rule]):
    """Return (parents, children) adjacency dicts for the proof DAG."""
    parents  = defaultdict(list)
    children = defaultdict(list)
    for rule in rules:
        parents[rule.rhs_id].append(rule.lhs_ids)
        for x in rule.lhs_ids:
            children[x].append(rule.rhs_id)
    return parents, children


def collect_supporting_rules(rules: List[Rule], target_rhs: str) -> List[int]:
    """Return sorted indices of rules on any path to target_rhs."""
    idx_by_rhs: Dict[str, List[int]] = defaultdict(list)
    for i, rule in enumerate(rules):
        idx_by_rhs[rule.rhs_id].append(i)

    supporting: set = set()
    frontier = [target_rhs]
    seen:     set = set()
    while frontier:
        rhs = frontier.pop()
        if rhs in seen:
            continue
        seen.add(rhs)
        for i in idx_by_rhs.get(rhs, []):
            if i not in supporting:
                supporting.add(i)
                for lhs in rules[i].lhs_ids:
                    if lhs.startswith('int'):
                        frontier.append(lhs)
    return sorted(supporting)


class EntailmentStructureProcessor:
    """Parsing, extraction, and comparison of EntailmentBank proof structures.

    Responsible for: LLM output parsing (extract_mediator, extract_final_answer,
    extract_tool_args), structural comparison (compare_structures, jaccard_similarity),
    and format validation (check_generation_format_mistakes).

    Structural mutations live in entailment_intervention.py.
    """

    # Section prefixes — must stay in sync with EntailmentIntervention
    PROOF_PREFIX = "## Proof\n"
    FINAL_ANSWER_PREFIX = "## Final Answer\nIs the hypothesis correct? "
    SMALL_PROOF_PREFIX = "Proof"
    SMALL_FINAL_ANSWER_PREFIX = "Final Answer"

    TOOL_ARGS_BLOCK_RE = re.compile(r'(?is)\bARGS\s*:\s*(?P<block>.*)$')
    TOOL_PROOF_NODES_RE = re.compile(r'(?is)"proof_nodes"\s*:\s*\[(?P<items>.*?)(?:\]|$)')

    def __init__(self, tool_mode: str = 'none'):
        self.tool_mode = tool_mode if tool_mode != 'none' else None

    def extract_mediator(self, completion: str) -> Optional[str]:
        """Extract the proof string from a model completion.

        Requires exactly one Proof and one Final Answer section, FA after Proof.
        Falls back to short prefix forms; strips stray "2)" artefacts.
        Returns the raw proof string or None.
        """
        if completion.count(self.SMALL_PROOF_PREFIX) != 1:
            return None
        if completion.count(self.SMALL_FINAL_ANSWER_PREFIX) != 1:
            return None
        if completion.find(self.SMALL_FINAL_ANSWER_PREFIX) < completion.find(self.SMALL_PROOF_PREFIX):
            return None

        if self.PROOF_PREFIX in completion and self.FINAL_ANSWER_PREFIX in completion:
            proof = completion.split(self.PROOF_PREFIX)[1].strip()
            return proof.split(self.FINAL_ANSWER_PREFIX)[0].strip()
        else:
            proof = completion.split(self.SMALL_PROOF_PREFIX)[1]
            proof = proof.strip().strip(":#").strip()
            proof = proof.split(self.SMALL_FINAL_ANSWER_PREFIX)[0]
            return proof.replace("2)", "").strip().strip(":#").strip()

    def extract_final_answer(
        self, completion: str, short_completion: bool = False
    ) -> Optional[bool]:
        """Parse Yes/No. Returns True / False / None (None = ambiguous or missing)."""
        if not completion:
            return None
        if self.SMALL_FINAL_ANSWER_PREFIX in completion:
            tail = completion.split(self.SMALL_FINAL_ANSWER_PREFIX)[1].strip(":").strip()
        elif short_completion:
            tail = completion.strip()
        else:
            tail = completion

        has_yes = "Yes" in tail
        has_no  = "No"  in tail
        if has_yes and not has_no: return True
        if has_no  and not has_yes: return False
        return None

    def extract_tool_args(
        self, text: str, short_completion: bool = False
    ) -> Optional[Dict]:
        if not self.tool_mode:
            return None
        raw = (text or "").strip()
        if short_completion:
            block = raw
        else:
            m = self.TOOL_ARGS_BLOCK_RE.search(raw)
            block = m.group("block").strip() if m else ""
        if not block:
            return None
        block = re.sub(r'(?is)^\s*```[a-z0-9_-]*\s*', "", block)
        block = re.sub(r'(?is)\s*```\s*$',             "", block).strip()
        mp = self.TOOL_PROOF_NODES_RE.search(block)
        if not mp:
            return None
        items_str = mp.group("items").strip()
        if not items_str:
            return None
        parts = re.findall(r'"([^"]+)"', items_str) or \
                [p.strip() for p in items_str.split(",") if p.strip()]
        nodes = sorted(p for p in parts if re.match(r'^sent\d+$', p))
        return {"proof_nodes": nodes} if nodes else None

    def boollist_to_checklist(self, sample: Dict, payload) -> None:
        return None

    def compare_structures(
        self, a: Optional[str], b: Optional[str]
    ) -> Optional[int]:
        """Normalised proof-string comparison. Returns 1 / 0 / None."""
        if a is None or b is None:
            return None
        na = self._normalize_proof(a)
        nb = self._normalize_proof(b)
        if na is None or nb is None:
            return None
        return 1 if na == nb else 0

    def _normalize_proof(self, proof_text: str) -> Optional[str]:
        if not proof_text:
            return None
        rules = parse_step_proof(proof_text)
        return serialize_step_proof(rules) if rules else None

    def jaccard_similarity(
        self, a: Optional[str], b: Optional[str]
    ) -> Optional[float]:
        """Jaccard similarity between the sent* node sets of two proof strings."""
        if a is None or b is None:
            return None
        sa = frozenset(re.findall(r'\b(sent\d+)\b', a))
        sb = frozenset(re.findall(r'\b(sent\d+)\b', b))
        if not sa and not sb:
            return 1.0
        union = len(sa | sb)
        return None if union == 0 else len(sa & sb) / union

    def check_generation_format_mistakes(self, completion: str) -> bool:
        """True if the completion has structural format problems:
          - empty
          - preamble before the Proof section
          - multiple Final Answer sections
        """
        s = (completion or "").strip()
        if not s:
            return True
        if s.count(self.SMALL_FINAL_ANSWER_PREFIX) > 1:
            return True
        return not bool(re.match(r'(?i)(?:##\s*)?Proof\b', s))