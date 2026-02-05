import re
from typing import List, Dict, Optional, Tuple

# one checklist line, e.g.:
# "correctly cites ... (weight: 1.5) (True/False): True"
# LINE_RE = re.compile(
#     r"""
#     ^\s*
#     (?P<question>.+?)
#     \s*\(\s*weight:\s*(?P<weight>[-+]?\d+(?:\.\d+)?)\s*\)
#     \s*\(\s*(?P<options>[^()]+?)\s*\)
#     \s*[:\-]\s*(?P<answer>True|False)\s*$
#     """,
#     re.IGNORECASE | re.VERBOSE
# )

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

# final grade line, e.g. "Final grade (0-8): 7.5" or "Final grade: 7.5"
FINAL_GRADE_RE = re.compile(
    r"""(?i)^\s*final\s*grade(?:\s*\([^)]*\))?\s*[:\-]\s*(?P<grade>[-+]?\d+(?:\.\d+)?)\s*(?!\w)"""
)

MEDIATOR_BLOCK_RE = re.compile(
    r"(?is)Checklist\s*:\s*(?P<block>.*?)(?=\n\s*Final\s*grade\b|\n\s*TOOL\s*:|$)"
)

TOOL_ARGS_BLOCK_RE = re.compile(
    r"(?is)\bARGS\s*:\s*(?P<block>.*)$"
)

# Extract value of "rubric": "<...>" (supports literal newlines inside quotes)
TOOL_RUBRIC_STRING_RE = re.compile(
    r'(?is)"rubric"\s*:\s*"(?P<rubric>(?:\\.|[^"\\])*)"\s*(?:,|\})'
)

def extract_final_grade(text: str) -> Optional[float]:
    """
    Extract final grade from text.
    Returns float grade or None if not found/invalid.
    """
    final_grade: Optional[float] = None
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        mg = FINAL_GRADE_RE.match(line)
        if mg:
            try:
                final_grade = float(mg.group("grade"))
                break
            except ValueError:
                pass
                
    return final_grade

def extract_checklist_entries(text: str) -> List[Dict]:
    """
    Extract structured checklist entries from text.
    Returns list of dicts with keys: question, weight, options, answer
    """
    checklist: Dict[str, str] = {}
    
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = LINE_RE.match(line)
        if m:
            question = m.group("question").strip()
            question = re.sub(r'[\'\"]{1,3}', '', question)
            # Clean triple quotes from question text (anywhere in the string)
            options = [p.strip() for p in re.split(r"\s*[/,\|]\s*", m.group("options")) if p.strip()]
            answer = m.group("answer").strip()

            # Normalize answer to one of the options if it matches case-insensitively
            if options:
                for opt in options:
                    if answer.lower() == opt.lower():
                        answer = opt == "True"
                        break

            checklist[question] = answer

    return checklist


def extract_mediator_rubric_raw(text: str) -> Optional[str]:
    """
    Returns raw checklist text produced under:
      Checklist:
        ...
    Stops before 'Final grade' or 'TOOL:'.
    """
    m = MEDIATOR_BLOCK_RE.search(text)
    if not m:
        return None
    block = m.group("block").strip()
    return block if block else None


def extract_tool_rubric_raw(text: str) -> Optional[str]:
    """
    Returns raw checklist text that was passed inside:
      ARGS: {"rubric": "<MULTILINE CHECKLIST>"}

    We do NOT parse JSON. We only extract the value of "rubric" as raw text.
    """
    m = TOOL_ARGS_BLOCK_RE.search(text)
    if not m:
        return None

    block = m.group("block").strip()
    if not block:
        return None

    block = block.replace("\\r\\n", "\n").replace("\\n", "\n")

    ms = TOOL_RUBRIC_STRING_RE.search(block)
    if not ms:
        return None

    rubric = ms.group("rubric")
    rubric = rubric.replace("\\r\\n", "\n").replace("\\n", "\n")
    rubric = rubric.replace('\\"', '"').replace("\\\\", "\\")
    rubric = rubric.strip()
    return rubric if rubric else None
