import re
import json


class RiceChemTool:
    def __init__(self, dataset, prompting_regime):
        self.dataset = dataset
        self.prompting_regime = prompting_regime
        self.name = "calculate_score"

    @property
    def spec(self):
        if self.prompting_regime == "tool_heavy":
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

        if self.prompting_regime == "tool_heavy":
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

        if self.prompting_regime == "tool_heavy":
            bools_list = args["rubric"]
            keys = list(weights.keys())
            if len(bools_list) != len(keys):
                return None

            score = 0.0
            for i in range(len(keys)):
                if bools_list[i]:
                    score += float(weights[keys[i]])
            return float(score)

        rubric_dict = args["rubric"]
        score = 0.0
        for k, v in rubric_dict.items():
            if v and k in weights:
                score += float(weights[k])
        return float(score)

    def _get_weights(self, sample_meta):
        if sample_meta is None:
            return None
        if "weights" in sample_meta and sample_meta["weights"] is not None:
            return sample_meta["weights"]
        if "task_idx" in sample_meta:
            return self.dataset.task2rubric_weights[sample_meta["task_idx"]]
        return None



class MediatorProcessor:
    LINE_RE = re.compile(
        r"""
        ^\s*
        (?P<question>.+?)
        (?:\s*\(\s*weight:\s*(?P<weight>[-+]?\d+(?:\.\d+)?)\s*\))?
        \s*\(\s*(?P<options>[^()]+?)\s*\)
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

    def __init__(self, dataset, prompting_regime):
        self.dataset = dataset
        self.prompting_regime = prompting_regime

    def _clean_tool_text(self, s):
        if not s:
            return ""
        s = s.strip()
        s = s.replace("{{", "{").replace("}}", "}")
        s = s.replace("\\r\\n", "\n").replace("\\n", "\n")
        s = s.replace('\\"', '"')
        s = s.replace("\\\\", "\\")
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

    def parse_mediator_checklist(self, completion_text):
        m = self.MEDIATOR_BLOCK_RE.search(completion_text or "")
        if not m:
            return None
        block = m.group("block").strip()
        if not block:
            return None
        return self._parse_checklist_block(block)

    def extract_final_grade(self, text: str) -> float | None:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.search(r"(?i)final\s*grade.*?[:\-]\s*([\d.]+)", line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
        return None

    def parse_tool_payload(self, completion_text):
        m = self.TOOL_ARGS_BLOCK_RE.search(completion_text or "")
        if not m:
            return None

        block = self._clean_tool_text(m.group("block"))
        if not block:
            return None

        if self.prompting_regime != "tool_heavy":
            ms = self.TOOL_RUBRIC_STRING_RE.search(block)
            if not ms:
                return None
            rubric = self._clean_tool_text(ms.group("rubric"))
            return rubric if rubric else None

        else:
            ml = self.TOOL_RUBRIC_BOOL_LIST_RE.search(block)
            if not ml:
                return None
            items = (ml.group("items") or "").strip()
            if not items:
                return None

            parts = [p.strip() for p in items.split(",")]
            parts = [p for p in parts if p != ""]

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

    def tool_payload_to_checklist(self, sample, payload):
        if payload is None:
            return None

        if self.prompting_regime != "tool_heavy":
            return self._parse_checklist_block(payload)

        keys = list(self.dataset.task2rubric_weights[sample["task_idx"]].keys())
        if len(payload) != len(keys):
            return None
        return {keys[i]: payload[i] for i in range(len(keys))}

    def compare_checklists(self, checklist_a, checklist_b):
        if checklist_a is None or checklist_b is None:
            return None
        for k, v in checklist_a.items():
            if k not in checklist_b or checklist_b[k] != v:
                return 0
        return 1
