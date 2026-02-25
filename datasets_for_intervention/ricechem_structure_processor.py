import re
import json


class RiceChemTool:
    def __init__(self, dataset, tool_mode):
        self.dataset = dataset
        self.tool_mode = tool_mode
        self.name = "calculate_score"

    @property
    def spec(self):
        if self.tool_mode == "structured":
            return {
                "title": self.name,
                "type": "object",
                "description": "Compute final grade by summing weights for True positions in the boolean list.",
                "returns": {
                    "type": "number",
                    "description": "Final grade as a single float (sum of weights for True items).",
                },
                "properties": {
                    "rubric": {
                        "type": "array",
                        "items": {"type": "boolean"},
                        "description": "Single list of True/False values aligned with the rubric items order.",
                    }
                },
                "required": ["rubric"],
            }
        
        return {
            "title": self.name,
            "type": "object",
            "description": "Compute final grade by summing weights of rubric items marked True.",
            "returns": {
                "type": "number",
                "description": "Final grade as a single float (sum of weights for True items).",
            },
            "properties": {
                "rubric": {
                    "type": "object",
                    "description": "Parsed checklist as mapping: {rubric_item_text: True/False}.",
                    "additionalProperties": {"type": "boolean"},
                }
            },
            "required": ["rubric"],
        }


    def spec_json(self):
        return json.dumps(self.spec, ensure_ascii=False)

    def validate_args(self, args):
        if not isinstance(args, dict):
            return False
        if "rubric" not in args:
            return False

        r = args["rubric"]

        if self.tool_mode == "structured":
            if not isinstance(r, list) or len(r) == 0:
                return False
            for x in r:
                if x is not True and x is not False:
                    return False
            return True

        if not isinstance(r, dict) or len(r) == 0:
            return False
        for k, v in r.items():
            if not isinstance(k, str) or not k:
                return False
            if v is not True and v is not False:
                return False
        return True


    def calculate_score(self, args, sample_meta):
        if not self.validate_args(args):
            return None

        weights = self._get_weights(sample_meta)
        if weights is None:
            return None

        rubric = args.get("rubric")

        # structured: list[bool]
        if isinstance(rubric, list):
            gold = sample_meta.get("gold_rubric")
            if isinstance(gold, dict) and gold:
                keys = list(gold.keys())
            else:
                keys = list(weights.keys())

            if len(rubric) != len(keys):
                return None

            score = 0.0
            for i, key in enumerate(keys):
                if rubric[i]:
                    score += float(weights[key])
            return float(score)

        # simple: dict[item -> bool]
        if isinstance(rubric, dict):
            score = 0.0
            for k, v in rubric.items():
                if v and k in weights:
                    score += float(weights[k])
            return float(score)

        return None

    def _get_weights(self, sample_meta):
        if sample_meta is None:
            return None
        if "weights" in sample_meta and sample_meta["weights"] is not None:
            return sample_meta["weights"]
        if "task_idx" in sample_meta:
            return self.dataset.task2rubric_weights[sample_meta["task_idx"]]
        return None


class RiceChemStructureProcessor:
    LINE_RE = re.compile(
        r"""
        ^\s*
        (?P<question>.+?)
        (?:\s*\(\s*weight:\s*(?P<weight>[-+]?\d+(?:\.\d+)?)\s*\))?
        (?:\s*\(\s*(?P<options>[^()]+?)\s*\))?          # <-- options now OPTIONAL
        \s*[:\-]\s*(?P<answer>True|False)\s*$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    MEDIATOR_BLOCK_RE = re.compile(
        r"(?is)Checklist\s*:\s*(?P<block>.*?)(?=\n\s*Final\s*grade\b|\n\s*TOOL\s*:|$)"
    )

    TOOL_ARGS_BLOCK_RE = re.compile(
        r"(?is)\bARGS\s*:\s*(?P<block>.*)$"
    )

    TOOL_RUBRIC_STRING_RE = re.compile(
        r'(?is)"rubric"\s*:\s*"(?P<rubric>.*?)(?:"\s*(?:,|\}|$)|$)'
    )

    TOOL_RUBRIC_BOOL_LIST_RE = re.compile(
        r'(?is)"rubric"\s*:\s*\[(?P<items>.*?)(?:\]|\}|$)'
    )

    FINAL_GRADE_RE = re.compile(
        r"""(?i)\bfinal\s*grade(?:\s*\([^)]*\))?\s*[:\-]\s*(?P<grade>[-+]?\d+(?:\.\d+)?)\b"""
    )

    NUM_ONLY_RE = re.compile(r"^\s*(?P<num>[-+]?\d+(?:\.\d+)?)\s*$")

    def __init__(self, dataset, tool_mode='none'):
        self.dataset = dataset
        self.tool_mode = tool_mode

    def extract_final_answer(self, text: str, short_completion: bool = False) -> float | None:
        s = (text or "").strip()
        if not s:
            return None

        m = self.FINAL_GRADE_RE.search(s)
        if m:
            try:
                return float(m.group("grade"))
            except ValueError:
                return None

        if short_completion:
            m2 = self.NUM_ONLY_RE.match(s)
            if not m2:
                return None
            try:
                return float(m2.group("num"))
            except ValueError:
                return None

        return None

    def _clean_tool_text(self, s):
        if not s:
            return ""
        s = s.strip()
        s = s.replace("{{", "{").replace("}}", "}")

        # strip optional code fences
        s = re.sub(r"(?is)^\s*```[a-z0-9_-]*\s*", "", s)
        s = re.sub(r"(?is)\s*```\s*$", "", s)

        # IMPORTANT: collapse double escaping BEFORE turning \n into newline
        s = s.replace("\\\\", "\\")
        s = s.replace("\\r\\n", "\n").replace("\\n", "\n")
        s = s.replace('\\"', '"')
        return s.strip()

    def _parse_checklist_block(self, block):
        checklist = {}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = self.LINE_RE.match(line)
            if not m:
                continue
            q = m.group("question").strip()
            q = re.sub(r"[\'\"]{1,3}", "", q)
            a = m.group("answer").strip().lower() == "true"
            checklist[q] = a
        return checklist if checklist else None

    def extract_mediator(self, completion_text):
        m = self.MEDIATOR_BLOCK_RE.search(completion_text or "")
        if not m:
            return None
        block = m.group("block").strip()
        if not block:
            return None
        return self._parse_checklist_block(block)

    def extract_tool_args(self, completion_text, short_completion: bool = False):
        """
        Returns TOOL arguments in the format expected by Tool.calculate_score:
        - tool_mode == "simple"     -> dict checklist {item: bool}
        - tool_mode == "structured" -> list[bool]
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

        # SIMPLE: {"rubric": "<raw checklist string>"} -> parse into dict checklist
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

            parts = [p.strip() for p in items.split(",") if p.strip() != ""]
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

    def boollist_to_checklist(self, sample, payload):
        """
        STRUCTURED ONLY HELPER:
        payload: list[bool] -> {rubric_item_text: bool}
        Used for logging/tool_rubric canonicalization and comparisons.
        """
        if payload is None:
            return None
        if not isinstance(payload, list) or len(payload) == 0:
            return None
        for x in payload:
            if not isinstance(x, bool):
                return None

        gold = sample.get("gold_rubric")
        if isinstance(gold, dict) and gold:
            keys = list(gold.keys())
        else:
            keys = list(self.dataset.task2rubric_weights[sample["task_idx"]].keys())

        if len(payload) != len(keys):
            return None

        return {keys[i]: payload[i] for i in range(len(keys))}

    def compare_structures(self, checklist_a, checklist_b):
        if checklist_a is None or checklist_b is None:
            return None
        if len(checklist_a) != len(checklist_b):
            return 0
        for k, v in checklist_a.items():
            if k not in checklist_b or checklist_b[k] != v:
                return 0
        return 1
