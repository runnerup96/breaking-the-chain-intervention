"""
CRUXEval structure processor and tool.

Completion format (non-tool):
    Trace:
    line 2:
      x = 5
    line 3:
      x = 5
      y = 10
    ...
    Final Answer: <python repr of the value>

Completion format (tool):
    Trace:
    ...
    Final tool call:
    TOOL: simulate_output
    ARGS: {"trace": "<raw trace string>"}                     # simple
    ARGS: {"trace": [{"line": 2, "locals": {"x": "5"}}, ...]} # structured

Canonical mediator form (M): list[{"line": int, "locals": {name: <python value>},
                                   "nl_comment"?: str}].
"""

import json
import re

from datasets_for_intervention.cruxeval_trace import (
    canonicalize_trace,
    parse_trace_text,
    simulate_from_trace,
    trace_to_text,
)


class CRUXEvalTool:
    """
    Deterministic function: predicts the final value from a trace.

    calculate_score(args, sample_meta):
        args["trace"] may be:
            - str   (textual trace; parsed via parse_trace_text)
            - list  (already canonical list-of-dicts)
        sample_meta must contain "code" (function source).
    """

    def __init__(self, dataset, tool_mode: str = "none"):
        self.dataset = dataset
        self.tool_mode = tool_mode if tool_mode != "none" else None
        self.name = "simulate_output"

    @property
    def spec(self) -> dict:
        if self.tool_mode == "structured":
            return {
                "title": self.name,
                "type": "object",
                "description": (
                    "Predict the final return value of the function by evaluating "
                    "the return expression with the locals from the last trace step."
                ),
                "returns": {
                    "type": "string",
                    "description": "Python repr() of the final value.",
                },
                "properties": {
                    "trace": {
                        "type": "array",
                        "description": (
                            "Ordered list of execution steps. Each step is an object "
                            'with "line" (int) and "locals" (object: var -> repr string).'
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "line":   {"type": "integer"},
                                "locals": {"type": "object"},
                                "nl_comment": {"type": "string"},
                            },
                            "required": ["line", "locals"],
                        },
                    }
                },
                "required": ["trace"],
            }

        return {
            "title": self.name,
            "type": "object",
            "description": (
                "Predict the final return value of the function from a textual trace."
            ),
            "returns": {
                "type": "string",
                "description": "Python repr() of the final value.",
            },
            "properties": {
                "trace": {
                    "type": "string",
                    "description": "Raw textual trace (same format as in the prompt).",
                }
            },
            "required": ["trace"],
        }

    def spec_json(self) -> str:
        return json.dumps(self.spec, ensure_ascii=False)

    def validate_args(self, args: dict) -> bool:
        if not isinstance(args, dict) or "trace" not in args:
            return False
        t = args["trace"]
        if isinstance(t, str):
            return bool(t.strip())
        if isinstance(t, list):
            for step in t:
                if not isinstance(step, dict):
                    return False
                if "line" not in step or "locals" not in step:
                    return False
            return True
        return False

    def calculate_score(self, args: dict, sample_meta: dict):
        """Return repr(result) of the simulated answer or None on bad args."""
        if not self.validate_args(args):
            return None
        if not sample_meta or "code" not in sample_meta:
            return None

        trace = args["trace"]
        if isinstance(trace, str):
            trace = parse_trace_text(trace)
            if trace is None:
                return None

        # Re-pack list-form dicts with stringified locals back to python objects
        # so simulate_from_trace can evaluate the return expression.
        canonical_trace = []
        for step in trace:
            locs = {}
            for k, v in step.get("locals", {}).items():
                if isinstance(v, str):
                    # values may have arrived as repr strings (structured tool)
                    try:
                        import ast
                        locs[k] = ast.literal_eval(v)
                    except Exception:
                        locs[k] = v
                else:
                    locs[k] = v
            entry = {"line": int(step["line"]), "locals": locs}
            if step.get("nl_comment"):
                entry["nl_comment"] = step["nl_comment"]
            canonical_trace.append(entry)

        result = simulate_from_trace(canonical_trace, sample_meta["code"])
        return repr(result)


class CRUXEvalStructureProcessor:
    """Parsing helpers for CRUXEval completions."""

    # "Trace:" block until "Final Answer:" or "TOOL:" or end of text
    MEDIATOR_BLOCK_RE = re.compile(
        r"(?is)Trace\s*:\s*(?P<block>.*?)(?=\n\s*Final\s*Answer\b|\n\s*Final\s*tool\s*call\b|\n\s*TOOL\s*:|$)"
    )

    FINAL_ANSWER_RE = re.compile(
        r"(?is)\bfinal\s*answer\s*[:\-]\s*(?P<answer>.+?)\s*$"
    )

    ANSWER_ONLY_RE = re.compile(r"(?s)^\s*(?P<answer>.+?)\s*$")

    TOOL_ARGS_BLOCK_RE = re.compile(r"(?is)\bARGS\s*:\s*(?P<block>.*)$")

    def __init__(self, dataset, tool_mode: str = "none"):
        self.dataset = dataset
        self.tool_mode = tool_mode if tool_mode != "none" else None

    # ---- mediator (trace) ----

    def extract_mediator(self, completion_text: str):
        if not completion_text:
            return None
        m = self.MEDIATOR_BLOCK_RE.search(completion_text)
        if not m:
            return None
        block = m.group("block").strip()
        if not block:
            return None
        return parse_trace_text(block)

    # ---- final answer ----

    def extract_final_answer(self, text: str, short_completion: bool = False):
        s = (text or "").strip()
        if not s:
            return None
        # "Final Answer: X" — may be multi-line, capture greedily up to EOL
        # We use rsearch-style approach: find the LAST "Final Answer:" occurrence
        # and take the rest of the line(s).
        for m in self.FINAL_ANSWER_RE.finditer(s):
            answer = m.group("answer").strip()
            # if it spans multiple lines, take first line only
            return answer.splitlines()[0].strip()

        if short_completion:
            m = self.ANSWER_ONLY_RE.match(s)
            if m:
                return m.group("answer").splitlines()[0].strip()
        return None

    # ---- tool ARGS ----

    @staticmethod
    def _clean_tool_text(s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        s = re.sub(r"(?is)^\s*```[a-z0-9_-]*\s*", "", s)
        s = re.sub(r"(?is)\s*```\s*$", "", s)
        return s.strip()

    def extract_tool_args(self, completion_text: str, short_completion: bool = False):
        """
        Returns:
          simple     -> {"trace": <str>}
          structured -> {"trace": <list[dict]>}
          non-tool   -> None
        """
        if not self.tool_mode:
            return None
        raw = completion_text or ""
        if short_completion:
            block = self._clean_tool_text(raw)
        else:
            m = self.TOOL_ARGS_BLOCK_RE.search(raw)
            if not m:
                return None
            block = self._clean_tool_text(m.group("block"))
        if not block:
            return None

        # Tolerate trailing junk after the JSON object
        decoder = json.JSONDecoder()
        try:
            obj, _end = decoder.raw_decode(block)
        except json.JSONDecodeError:
            # Try a relaxed extraction: take the substring starting at first '{'.
            i = block.find("{")
            if i < 0:
                return None
            try:
                obj, _end = decoder.raw_decode(block[i:])
            except json.JSONDecodeError:
                return None

        if not isinstance(obj, dict) or "trace" not in obj:
            return None
        return obj

    # ---- canonicalization helpers ----

    def trace_to_text(self, trace) -> str:
        return trace_to_text(trace) if trace else ""

    def compare_structures(self, a, b):
        """
        Compare two traces canonically.
        Returns 1 (match), 0 (mismatch), or None if either is None.
        """
        if a is None or b is None:
            return None
        ca = canonicalize_trace(a)
        cb = canonicalize_trace(b)
        if ca is None or cb is None:
            return None
        return 1 if ca == cb else 0

    def compare_answers(self, a, b):
        """
        Compare two answers tolerantly: try literal_eval on both then str() equality;
        fall back to stripped-string equality.
        """
        if a is None or b is None:
            return None
        sa, sb = str(a).strip(), str(b).strip()
        if sa == sb:
            return 1
        import ast
        try:
            va = ast.literal_eval(sa)
            vb = ast.literal_eval(sb)
            return 1 if va == vb else 0
        except Exception:
            return 0

    def check_generation_format_mistakes(self, completion: str) -> bool:
        s = (completion or "").strip()
        if not s:
            return True
        if not re.match(r"(?i)trace\s*:", s):
            return True
        return False
