import re
from typing import List, Dict, Optional

# Lines inside "Intermediate Structure" section, e.g.:
# "- Q: Did MFA announce this on August 13, 2020 press briefing? A: No"
# "- Q: Did Trump make pro-gay laws when in office? A: No (evidence of ...)"
QA_LINE_RE = re.compile(
    r"""
    ^\s*
    (?:.*?)?                                # ignore any header text
    (?:[-\u2022]?\s*)?                      # optional bullet (- or •)
    \s*\bQ\s*:\s*                           # optional 'Q:' prefix
    (?P<question>.+?)                       # question text (non-greedy)
    \s*\bA\s*:\s*                           # 'A:' separator
    (?P<answer>Yes|No|True|False)           # canonical boolean-like answer
    (?:\b.*)?                               # optionally anything trailing (e.g., parenthetical evidence)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE
)

# Section headers
FINAL_VERDICT_RE = re.compile(r"""(?i)^\s*final\s*verdict\s*[:\-]\s*(?P<verdict>Supported|Refuted)\s*""")



def extract_intermediate_structure(text: str) -> List[Dict]:
    """
    Extract Q/A entries from the 'Intermediate Structure' section.
    Returns a list of dicts with keys: question, answer_raw, answer_bool.
    Robust to optional bullets and trailing parentheticals after the answer.
    """
    lines = text.splitlines()
    in_section = False
    entries: List[Dict] = []
    supporting_questions = dict()
    for raw in lines:
        line = raw.rstrip("\n")

        if FINAL_VERDICT_RE.match(line):
            break

        m = QA_LINE_RE.match(line.strip())
        if m:
            q = m.group("question").strip()
            a_raw = m.group("answer").strip()
            supporting_questions[q] = a_raw

    return supporting_questions


def extract_final_verdict(text: str) -> Optional[str]:
    """
    Extract final verdict from text.
    Returns 'Supported' | 'Refuted' | None.
    """
    for line in text.splitlines():
        mg = FINAL_VERDICT_RE.match(line.strip())
        if mg:
            return mg.group("verdict").title()
    return None


