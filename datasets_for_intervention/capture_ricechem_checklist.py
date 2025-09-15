import re
from typing import List, Dict, Optional, Tuple

# one checklist line, e.g.:
# "correctly cites ... (weight: 1.5) (True/False): True"
LINE_RE = re.compile(
    r"""
    ^\s*
    (?P<question>.+?)
    \s*\(\s*weight:\s*(?P<weight>[-+]?\d+(?:\.\d+)?)\s*\)
    \s*\(\s*(?P<options>[^()]+?)\s*\)
    \s*[:\-]\s*(?P<answer>True|False)\s*$
    """,
    re.IGNORECASE | re.VERBOSE
)

# final grade line, e.g. "Final grade (0-8): 7.5" or "Final grade: 7.5"
FINAL_GRADE_RE = re.compile(
    r"""(?i)^\s*final\s*grade(?:\s*\([^)]*\))?\s*[:\-]\s*(?P<grade>[-+]?\d+(?:\.\d+)?)\s*(?!\w)"""
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


if __name__ == "__main__":
    sample = """
    Checklist:
    correctly cites decreased electron electron repulsion (weight: 1) (True/False): True
    relates decreased electron electron repulsion to decreased potential energy (weight: 1) (True/False): True
    3rd and 4th electrons ionized feel same core charge (weight: 1) (True/False): True
    3rd and 4th electrons ionized from n=3 shell and have same radius (weight: 1) (True/False): True
    5th electron ionized from n=2 shell and feels higher core charge (weight: 1) (True/False): True
    5th electron ionized from n=2 shell and has smaller radius (weight: 1) (True/False): True
    correctly explains relationship of potential energy to ionization energy (weight: 1.5) (True/False): True
    partially explains relationship between potential energy and ionization energy (weight: 0.5) (True/False): False
    Final grade (0-8): 7.5
    """

    checklist = extract_checklist_entries(sample)
    final_grade = extract_final_grade(sample)
    print(checklist)
    print(final_grade)
