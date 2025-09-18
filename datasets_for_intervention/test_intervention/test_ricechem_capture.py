import unittest
from datasets_for_intervention.capture_ricechem_checklist import LINE_RE, FINAL_GRADE_RE, extract_final_grade, extract_checklist_entries

REF_1 = """\
Checklist:
A (weight: 1) (True/False): True
B (weight: 1) (True/False): True
C (weight: 1) (True/False): True
D (weight: 1) (True/False): True
E (weight: 1) (True/False): True
F (weight: 1) (True/False): True
G (weight: 1.5) (True/False): True
H (weight: 0.5) (True/False): False
Final grade (0-8): 7.5
"""

REF_2 = '''Checklist:
"""A (weight: 1) (True/False): True
B (weight: 1) (True/False): True
C (weight: 1) (True/False): True
D (weight: 1) (True/False): True
E (weight: 1) (True/False): True
F (weight: 1) (True/False): True
G (weight: 1.5) (True/False): True
H (weight: 0.5) (True/False): False
Final grade (0-8): 7.5
'''



class TestChecklistRegex(unittest.TestCase):
    def test_line_re_matches_question_and_answer(self):
        m = LINE_RE.match("A (weight: 1) (True/False): True")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("question"), "A")
        self.assertEqual(m.group("answer"), "True")

        m = LINE_RE.match("H (weight: 0.5) (True/False): False")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("question"), "H")
        self.assertEqual(m.group("answer"), "False")

    def test_final_grade_re_basic_and_signed(self):
        self.assertIsNotNone(FINAL_GRADE_RE.match("Final grade (0-8): 7.5"))
        self.assertEqual(extract_final_grade("Final grade: 2<\|im_end\|>"), 2.0)
        self.assertIsNotNone(FINAL_GRADE_RE.match("Final grade: 7.5pts"))

    def test_extractors_on_reference_1(self):
        entries = extract_checklist_entries(REF_1)
        grade = extract_final_grade(REF_1)

        self.assertEqual(grade, 7.5)
        # 8 checklist lines (A..H)
        self.assertEqual(len(entries), 8)
        self.assertTrue(entries["A"])
        self.assertFalse(entries["H"])

        entries = extract_checklist_entries(REF_2)
        grade = extract_final_grade(REF_2)

        self.assertEqual(grade, 7.5)
        # 8 checklist lines (A..H)
        self.assertEqual(len(entries), 8)
        self.assertTrue(entries["A"])
        self.assertFalse(entries["H"])


    def test_case_and_separator_normalization(self):
        s = """\
        X (weight: 1) (True/False): True
        Y (weight: 1) (True,False): false
        Z (weight: 1) (TRUE|FALSE): True
        """
        entries = extract_checklist_entries(s)
        self.assertTrue(entries["X"])
        self.assertFalse(entries["Y"])
        self.assertFalse(entries["Z"])

        s = """X (weight: 1) (True/False): True"""
        entries = extract_checklist_entries(s)
        self.assertTrue(entries["X"])



    def test_non_matching_lines_are_ignored(self):
        s = """\
        Not a checklist line
        Q (weight: 1) (Yes/No): Maybe
        """
        entries = extract_checklist_entries(s)
        # "Q" kept, answer preserved since it's not in True/False options
        self.assertNotIn("Q", entries)

