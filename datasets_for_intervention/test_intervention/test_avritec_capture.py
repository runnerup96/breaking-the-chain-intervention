import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets_for_intervention.capture_averitec_checklist import (
    QA_LINE_RE, FINAL_VERDICT_RE,
    extract_intermediate_structure, extract_final_verdict
)

REF_1 = """\
Intermediate Structure:
- Q: Did Hunter Biden have any experience in the energy sector in 2014? A: No
- Q: Did Hunter Biden have any experience in Ukraine in 2014? A: No
Final Verdict: Supported
"""

REF_2 = """\
Intermediate Structure:
- Q: Did Trump make pro-gay laws when in office? A: No
Final Verdict: Refuted
"""

REF_3 = """\
Intermediate Structure:
- Q: Did MFA announce this on August 13, 2020 press briefing? A: No
- Q: Did China's State Council briefing include this? A: No
- Q: Did MFA Twitter account post this between Aug 13–18, 2020? A: No
Final Verdict: Refuted
"""

REF_4 = """\
Intermediate Structure:
- Q: Did MFA announce this on August 13, 2020 press briefing? A: No (evidence of ...)
- Q: Did China's State Council briefing include this? A: No
Final Verdict: Refuted
"""


class TestChecklistRegex(unittest.TestCase):
    def test_qa_line_re_matches_question_and_answer(self):
        m = QA_LINE_RE.match("- Q: Did Hunter Biden have experience? A: No")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("question"), "Did Hunter Biden have experience?")
        self.assertEqual(m.group("answer"), "No")

        m = QA_LINE_RE.match("Q: Did Trump make pro-gay laws? A: Yes")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("question"), "Did Trump make pro-gay laws?")
        self.assertEqual(m.group("answer"), "Yes")

        m = QA_LINE_RE.match("• Q: Did MFA announce this? A: No (evidence of ...)")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("question"), "Did MFA announce this?")
        self.assertEqual(m.group("answer"), "No")

    def test_qa_line_re_case_insensitive(self):
        m = QA_LINE_RE.match("- Q: Test question? A: YES")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("answer"), "YES")

        m = QA_LINE_RE.match("- Q: Test question? A: true")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("answer"), "true")

    # def test_intermediate_header_re(self):
    #     self.assertIsNotNone(INTERMEDIATE_HEADER_RE.match("Intermediate Structure:"))
    #     self.assertIsNotNone(INTERMEDIATE_HEADER_RE.match("intermediate structure:"))
    #     self.assertIsNotNone(INTERMEDIATE_HEADER_RE.match("  Intermediate Structure:  "))

    def test_final_verdict_re(self):
        self.assertIsNotNone(FINAL_VERDICT_RE.match("Final Verdict: Supported"))
        self.assertIsNotNone(FINAL_VERDICT_RE.match("final verdict: Refuted"))
        self.assertIsNotNone(FINAL_VERDICT_RE.match("Final Verdict - Supported"))

    def test_extract_intermediate_structure_basic(self):
        structure = extract_intermediate_structure(REF_1)
        expected = {
            "Did Hunter Biden have any experience in the energy sector in 2014?": "No",
            "Did Hunter Biden have any experience in Ukraine in 2014?": "No"
        }
        self.assertEqual(structure, expected)

    def test_extract_intermediate_structure_single_question(self):
        structure = extract_intermediate_structure(REF_2)
        expected = {
            "Did Trump make pro-gay laws when in office?": "No"
        }
        self.assertEqual(structure, expected)

    def test_extract_intermediate_structure_with_evidence(self):
        structure = extract_intermediate_structure(REF_4)
        expected = {
            "Did MFA announce this on August 13, 2020 press briefing?": "No",
            "Did China's State Council briefing include this?": "No"
        }
        self.assertEqual(structure, expected)

    def test_extract_intermediate_structure_empty(self):
        structure = extract_intermediate_structure("No structure here")
        self.assertEqual(structure, {})

    def test_extract_final_verdict_basic(self):
        verdict = extract_final_verdict(REF_1)
        self.assertEqual(verdict, "Supported")

        verdict = extract_final_verdict(REF_2)
        self.assertEqual(verdict, "Refuted")

    def test_extract_final_verdict_case_insensitive(self):
        text = "final verdict: supported"
        verdict = extract_final_verdict(text)
        self.assertEqual(verdict, "Supported")

    def test_extract_final_verdict_not_found(self):
        verdict = extract_final_verdict("No verdict here")
        self.assertIsNone(verdict)

    def test_extract_intermediate_structure_stops_at_final_verdict(self):
        text = """\
        Intermediate Structure:
        - Q: Question 1? A: Yes
        Final Verdict: Supported
        - Q: Question 2? A: No
        """
        structure = extract_intermediate_structure(text)
        expected = {"Question 1?": "Yes"}
        self.assertEqual(structure, expected)
