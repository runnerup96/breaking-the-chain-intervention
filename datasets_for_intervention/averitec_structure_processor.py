import re
import json


# ---------------------------------------------------------------------------
# AVeriTeCTool
# ---------------------------------------------------------------------------

class AVeriTeCTool:
    """
    Deterministic function: predicts the verdict ("Supported" / "Refuted")
    from a checklist {question: bool}.

    Logic of calculate_score:
      - If the provided checklist equals gold_rubric -> return gold_target.
      - Otherwise (at least one answer differs) -> return flip(gold_target).

    This is correct for filtered samples:
      - Supported (all answers "correct"):
            flipping any single answer -> Refuted.
      - Refuted with len == 1 (single "wrong" answer):
            flipping the single answer -> Supported.

    tool_mode only affects the spec (structured vs simple) and not the logic.
    """

    def __init__(self, dataset, tool_mode: str = "none"):
        self.dataset = dataset
        self.tool_mode = tool_mode if tool_mode != "none" else None
        self.name = "predict_verdict"

    # ------------------------------------------------------------------
    # Spec (argument description for prompt injection)
    # ------------------------------------------------------------------

    @property
    def spec(self) -> dict:
        if self.tool_mode == "structured":
            return {
                "title": self.name,
                "type": "object",
                "description": (
                    "Predict the verdict (Supported/Refuted) based on a boolean list "
                    "aligned with the checklist questions."
                ),
                "returns": {
                    "type": "string",
                    "enum": ["Supported", "Refuted"],
                    "description": "The predicted verdict.",
                },
                "properties": {
                    "rubric": {
                        "type": "array",
                        "items": {"type": "boolean"},
                        "description": (
                            "List of True/False values aligned with checklist question order. "
                            "True = Yes, False = No."
                        ),
                    }
                },
                "required": ["rubric"],
            }

        # simple / non-tool -> spec with dict format
        return {
            "title": self.name,
            "type": "object",
            "description": (
                "Predict the verdict (Supported/Refuted) based on the filled checklist."
            ),
            "returns": {
                "type": "string",
                "enum": ["Supported", "Refuted"],
                "description": "The predicted verdict.",
            },
            "properties": {
                "rubric": {
                    "type": "object",
                    "description": "Filled checklist as {question: True/False}.",
                    "additionalProperties": {"type": "boolean"},
                }
            },
            "required": ["rubric"],
        }

    def spec_json(self) -> str:
        return json.dumps(self.spec, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Argument validation -- type-based, not mode-based
    # ------------------------------------------------------------------

    def validate_args(self, args: dict) -> bool:
        """
        Accepts both dict and list rubric -- type is determined via isinstance.
        Does not depend on self.tool_mode.
        """
        if not isinstance(args, dict):
            return False
        if "rubric" not in args:
            return False

        r = args["rubric"]

        if r is None or (hasattr(r, "__len__") and len(r) == 0):
            return False

        if isinstance(r, list):
            # structured: all elements must be bool
            for x in r:
                if x is not True and x is not False:
                    return False
            return True

        if isinstance(r, dict):
            # simple: non-empty string keys, boolean values
            for k, v in r.items():
                if not isinstance(k, str) or not k:
                    return False
                if v is not True and v is not False:
                    return False
            return True

        return False

    # ------------------------------------------------------------------
    # Core function
    # ------------------------------------------------------------------

    def calculate_score(self, args: dict, sample_meta: dict):
        """
        Returns "Supported" | "Refuted" | None.

        Algorithm:
          1. Validate args.
          2. If rubric is list[bool] (structured), convert to dict using gold_rubric keys.
          3. Compare rubric with gold_rubric:
               - Exact match  -> gold_target.
               - Any mismatch -> flip(gold_target).
        """
        if not self.validate_args(args):
            return None

        rubric = args["rubric"]
        gold_rubric = sample_meta.get("gold_rubric") if sample_meta else None
        gold_target = sample_meta.get("gold_target") if sample_meta else None

        if gold_rubric is None or gold_target is None:
            return None

        # Structured (list[bool]) -> convert to dict
        if isinstance(rubric, list):
            if len(rubric) != len(gold_rubric):
                return None
            keys = list(gold_rubric.keys())
            rubric = {keys[i]: rubric[i] for i in range(len(keys))}

        # Compare rubric with gold_rubric
        if self._rubrics_equal(rubric, gold_rubric):
            return gold_target
        # At least one difference -> flip the verdict
        return "Refuted" if gold_target == "Supported" else "Supported"

    @staticmethod
    def _rubrics_equal(a: dict, b: dict) -> bool:
        """Strict element-wise comparison of two {question: bool} dicts."""
        if len(a) != len(b):
            return False
        for k, v in a.items():
            if k not in b or b[k] != v:
                return False
        return True


# ---------------------------------------------------------------------------
# AVeriTeCStructureProcessor
# ---------------------------------------------------------------------------

class AVeriTeCStructureProcessor:
    """
    Parsing and canonicalization of completions for AVeriTeC.

    Expected completion format (non-tool):
      Checklist:
      - Q: <question> (True/False): <True|False>
      ...
      Final Verdict: <Supported|Refuted>

    Expected completion format (tool):
      Checklist:
      - Q: <question> (True/False): <True|False>
      ...
      Final tool call:
         TOOL: predict_verdict
         ARGS: {"rubric": [True, False, ...]}    <- structured
         ARGS: {"rubric": "Q: ..."}              <- simple

    Canonical mediator form: dict {question_str: bool}
    """

    # ------------------------------------------------------------------
    # Regular expressions
    # ------------------------------------------------------------------

    # Checklist line pattern:
    #   - Q: <question> (True/False): True
    #   - Q: <question>: No
    #   Q: <question> (True/False): False
    # Accepts True/False and Yes/No (for robustness)
    QA_LINE_RE = re.compile(
        r"""
        ^\s*
        (?:[-\u2022]?\s*)?          # optional bullet  - or bullet
        Q\s*:\s*                    # Q: prefix
        (?P<question>.+?)           # question text (non-greedy)
        (?:\s*\([^)]*\))?           # optional hint (True/False)
        \s*[:\-]\s*                 # delimiter : or -
        (?P<answer>True|False|Yes|No)  # answer
        (?:\b.*)?                   # optional trailing parenthetical
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # "Checklist:" block up to "Final Verdict:", "TOOL:", or end of text
    MEDIATOR_BLOCK_RE = re.compile(
        r"(?is)Checklist\s*:\s*(?P<block>.*?)(?=\n\s*Final\s*Verdict\b|\n\s*TOOL\s*:|$)"
    )

    # "Final Verdict: Supported" / "Final Verdict: Refuted"
    FINAL_VERDICT_RE = re.compile(
        r"(?i)\bfinal\s*verdict\s*[:\-]\s*(?P<verdict>Supported|Refuted)\b"
    )

    # "Supported" or "Refuted" as the full string (short_completion)
    VERDICT_ONLY_RE = re.compile(
        r"^\s*(?P<verdict>Supported|Refuted)\s*$", re.IGNORECASE
    )

    # ARGS block inside a tool call
    TOOL_ARGS_BLOCK_RE = re.compile(
        r"(?is)\bARGS\s*:\s*(?P<block>.*)$"
    )

    # "rubric": "<string>" (simple mode)
    TOOL_RUBRIC_STRING_RE = re.compile(
        r'(?is)"rubric"\s*:\s*"(?P<rubric>.*?)(?:"\s*(?:,|\}|$)|$)'
    )

    # "rubric": [True, False, ...] (structured mode)
    TOOL_RUBRIC_BOOL_LIST_RE = re.compile(
        r'(?is)"rubric"\s*:\s*\[(?P<items>.*?)(?:\]|\}|$)'
    )

    def __init__(self, dataset, tool_mode: str = "none"):
        self.dataset = dataset
        self.tool_mode = tool_mode if tool_mode != "none" else None

    # ------------------------------------------------------------------
    # extract_mediator
    # ------------------------------------------------------------------

    def extract_mediator(self, completion_text: str) -> dict | None:
        """
        Extract the "Checklist:" block and parse it into a canonical dict {question: bool}.
        Returns None on format error.
        """
        m = self.MEDIATOR_BLOCK_RE.search(completion_text or "")
        if not m:
            return None
        block = m.group("block").strip()
        if not block:
            return None
        return self._parse_checklist_block(block)

    def _parse_checklist_block(self, block: str) -> dict | None:
        """
        Line-by-line parsing of a checklist block.
        Each line has the form "- Q: <question> (True/False): <True|False|Yes|No>".
        Returns {question: bool} or None if no lines were recognized.
        """
        checklist = {}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = self.QA_LINE_RE.match(line)
            if not m:
                continue
            question = m.group("question").strip()
            # Remove stray quotes left by escaping
            question = re.sub(r"[\'\"]{1,3}", "", question).strip()
            answer_raw = m.group("answer").strip().lower()
            answer_bool = answer_raw in ("true", "yes")
            checklist[question] = answer_bool
        return checklist if checklist else None

    # ------------------------------------------------------------------
    # extract_final_answer
    # ------------------------------------------------------------------

    def extract_final_answer(self, text: str, short_completion: bool = False) -> str | None:
        """
        Parse the verdict ("Supported" | "Refuted") from text.

        short_completion=True: text contains only "Supported" or "Refuted"
        (the model appended only the tail in an intervention prompt).
        """
        s = (text or "").strip()
        if not s:
            return None

        # Try full pattern "Final Verdict: X" first
        m = self.FINAL_VERDICT_RE.search(s)
        if m:
            return m.group("verdict").title()  # normalize casing

        # For short_completion accept "Supported" / "Refuted" as the entire string
        if short_completion:
            m2 = self.VERDICT_ONLY_RE.match(s)
            if m2:
                return m2.group("verdict").title()

        return None

    # ------------------------------------------------------------------
    # extract_tool_args
    # ------------------------------------------------------------------

    def _clean_tool_text(self, s: str) -> str:
        """Strip Markdown wrappers and unescape escape sequences from an ARGS block."""
        if not s:
            return ""
        s = s.strip()
        s = s.replace("{{", "{").replace("}}", "}")
        # Remove ```json ... ``` wrappers
        s = re.sub(r"(?is)^\s*```[a-z0-9_-]*\s*", "", s)
        s = re.sub(r"(?is)\s*```\s*$", "", s)
        # Unwrap double-escaping before converting \n to newline
        s = s.replace("\\\\", "\\")
        s = s.replace("\\r\\n", "\n").replace("\\n", "\n")
        s = s.replace('\\"', '"')
        return s.strip()

    def extract_tool_args(self, completion_text: str, short_completion: bool = False):
        """
        Return tool arguments:
          structured -> list[bool]
          simple     -> dict {question: bool}
          non-tool   -> None
        """
        if not self.tool_mode:
            return None

        raw = completion_text or ""

        if short_completion:
            # In short_completion mode the entire text is already the ARGS tail
            block = self._clean_tool_text(raw)
        else:
            m = self.TOOL_ARGS_BLOCK_RE.search(raw)
            if not m:
                return None
            block = self._clean_tool_text(m.group("block"))

        if not block:
            return None

        # SIMPLE: {"rubric": "<raw checklist string>"} -> dict
        if self.tool_mode == "simple":
            ms = self.TOOL_RUBRIC_STRING_RE.search(block)
            if not ms:
                return None
            rubric_str = self._clean_tool_text(ms.group("rubric"))
            if not rubric_str:
                return None
            return self._parse_checklist_block(rubric_str)

        # STRUCTURED: {"rubric": [True, False, ...]} -> list[bool]
        if self.tool_mode == "structured":
            ml = self.TOOL_RUBRIC_BOOL_LIST_RE.search(block)
            if not ml:
                return None
            items = (ml.group("items") or "").strip()
            if not items:
                return None
            parts = [p.strip() for p in items.split(",") if p.strip()]
            out = []
            for p in parts:
                pl = p.lower()
                if pl == "true":
                    out.append(True)
                elif pl == "false":
                    out.append(False)
                else:
                    return None
            return out if out else None

        return None

    # ------------------------------------------------------------------
    # boollist_to_checklist (structured only)
    # ------------------------------------------------------------------

    def boollist_to_checklist(self, sample: dict, payload) -> dict | None:
        """
        Convert list[bool] to dict {question: bool} using gold_rubric key order.
        Used in Intervention.infer_completion immediately after extract_tool_args.

        Structured mode only.
        """
        if payload is None:
            return None
        if not isinstance(payload, list) or len(payload) == 0:
            return None
        for x in payload:
            if not isinstance(x, bool):
                return None

        gold = sample.get("gold_rubric") if sample else None
        if not isinstance(gold, dict) or not gold:
            return None
        keys = list(gold.keys())

        if len(payload) != len(keys):
            return None

        return {keys[i]: payload[i] for i in range(len(keys))}

    # ------------------------------------------------------------------
    # compare_structures
    # ------------------------------------------------------------------

    def compare_structures(self, a, b) -> int | None:
        """
        Strict comparison of two canonical {question: bool} dicts.
        Returns:
          1    -- exact match
          0    -- mismatch
          None -- at least one argument is None
        """
        if a is None or b is None:
            return None
        if len(a) != len(b):
            return 0
        for k, v in a.items():
            if k not in b or b[k] != v:
                return 0
        return 1

    # ------------------------------------------------------------------
    # check_generation_format_mistakes
    # ------------------------------------------------------------------

    def check_generation_format_mistakes(self, completion: str) -> bool:
        """
        Check for garbage in the XM part of the completion.

        Garbage means ONLY a preamble before the "Checklist:" block
        (any text the model inserted before the structure starts).
        The tail after "Final Verdict:" is NOT considered garbage:
        infer_completion will either parse the verdict or return None --
        both outcomes are handled correctly.
        We only care about the cleanliness of XM' because XM' is what
        gets compared to XM_gold when substituted in Correction.

        Returns True  -> garbage detected -> generation_status = "error".
        Returns False -> no garbage detected.
        """
        s = (completion or "").strip()
        if not s:
            return True
        # Completion must start exactly with "Checklist:"
        if not re.match(r"(?i)checklist\s*:", s):
            return True
        return False