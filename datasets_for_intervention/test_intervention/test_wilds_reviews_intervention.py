import unittest
from datasets_for_intervention import wilds_reviews_intervention
import re


class TestWildsReviewsIntervention(unittest.TestCase):

    def setUp(self):
        self.llm_stop_token = '<|im_end|>'
        self.intervention_logic = wilds_reviews_intervention.WildsReviewsIntervention(self.llm_stop_token)

        self.review_text = "I love this product! It's so easy to use and the quality is amazing. I would definitely recommend it to anyone."

        self.model_completion = """
        '<think>\nOkay, let\'s tackle this step by step. The review is: "These mics have to be literally against you lips to pick up any sound." \n\nFirst, the emotional tone. The user is saying that the mics need to be very close to the lips to capture sound. That sounds like a problem, so the tone is probably negative.\n\nNext, product issues. The user is pointing out a flaw in the product\'s performance. They need to be right up against their lips, which is a clear issue. So yes, there\'s a product issue mentioned.\n\nDid the product meet or fall short? The user is frustrated because the mics don\'t work well unless placed very close. So it\'s falling short of expectations. \n\nSupport interaction? The review doesn\'t mention any interaction with customer support. So no.\n\nValue for money? The user hasn\'t commented on the price or whether it\'s worth it. So no mention of that.\n\nRecommendation? The user is dissatisfied, so they might discourage others. But the review doesn\'t explicitly say they don\'t recommend it. However, the negative tone implies that. But the question is if they explicitly recommend or discourage. Since the review doesn\'t say "don\'t buy," maybe it\'s neutral? Wait, but the sentiment is negative. Hmm. The user is expressing dissatisfaction, so they might be discouraging others. But the instruction says to check if they explicitly mention it. The review doesn\'t say "don\'t buy," so maybe neutral. But I need to check the exact wording. The review is negative, but does it explicitly recommend or discourage? The original statement is about the product\'s performance, not a direct recommendation. So maybe neutral? But the user is not satisfied, so maybe discourage. Wait, the question is whether they explicitly recommend or discourage. The review doesn\'t say "don\'t buy," so maybe neutral. But the sentiment is negative, so perhaps the answer is neutral here. But I need to be careful. The instruction says to check if they explicitly mention. Since the review doesn\'t mention anything about recommending, it\'s neutral. \n\nNow, the final classification. The checklist has all the steps filled. The emotional tone is negative. Product issues: yes. Met/exceeded/fell short: fell short. Support interaction: no. Value for money: no. Recommendation: neutral. \n\nBut the final classification is 0 (negative) or 1 (positive). Based on the checklist, the main points are negative sentiment, product issues, fell short. So the final classification is 0.\n</think>\n\nWhat is the emotional tone of the review? (positive/neutral/negative): negative  \nWere any product issues mentioned (e.g. damage, defects, failures)? (yes/no): yes  \nDid the product meet or fall short of expectations? (met/exceeded/fell short): fell short  \nWas there any mention of support interaction, and was it positive or negative? (yes/no): no  \nDoes the reviewer express satisfaction or dissatisfaction with value for money? (yes/no): no  \nDid the reviewer explicitly recommend or discourage others from buying it? (recommend/discourage/neutral): neutral  \nFinal classification (0/1): 0<|im_end|>'
        """
        # we compare the prepared model completion with the reverted one
        self.model_completion = re.sub(r"Final classification \(0/1\): \d", "Final classification (0/1):", self.model_completion)
        self.model_completion = self.model_completion.replace(self.llm_stop_token, "")
    
    def test_find_question_in_prompt(self):
        """
        Test that the find_question_in_prompt method is working correctly.
        """
        question = "What is the emotional tone of the review? (positive/neutral/negative)"
        expected_result = "What is the emotional tone of the review? (positive/neutral/negative): negative"
        self.assertEqual(self.intervention_logic.find_question_in_prompt(self.model_completion, question), ("negative", expected_result))
    
    def test_make_intervention(self):
        """
        Test that the intervention logic is working correctly -- we get the same prompt when we revert intervened mediator to the original prompt.
        """
        # Make interventions on the model completion
        intervention_outputs = self.intervention_logic.make_intervention(self.model_completion)

        # Validate that all interventions can be reverted back to original prompt
        validation_result = self.intervention_logic.validate_all_interventions(
            self.model_completion, 
            intervention_outputs
        )

        self.assertTrue(validation_result)

    def test_extract_target_from_prompt(self):
        # Test prompt with target 0
        prompt_with_zero = "Some text\nFinal classification (0/1): 0"
        self.assertEqual(self.intervention_logic.extract_target_from_prompt(prompt_with_zero), 0)

        # Test prompt with target 1 
        prompt_with_one = "Some text\nFinal classification (0/1): 1"
        self.assertEqual(self.intervention_logic.extract_target_from_prompt(prompt_with_one), 1)

        # Test prompt with no target
        prompt_no_target = "Some text without target"
        self.assertIsNone(self.intervention_logic.extract_target_from_prompt(prompt_no_target))

        # Test prompt with invalid target
        prompt_invalid_target = "Some text\nFinal classification (0/1): 2"
        self.assertIsNone(self.intervention_logic.extract_target_from_prompt(prompt_invalid_target))

    def test_infer_completion(self):
        # Test completion with 0
        completion_with_zero = "Some text with 0"
        self.assertEqual(self.intervention_logic.infer_completion(completion_with_zero), 0)

        # Test completion with 1
        completion_with_one = "Some text with 1" 
        self.assertEqual(self.intervention_logic.infer_completion(completion_with_one), 1)

        # Test completion with no valid target
        completion_no_target = "Some text without target"
        self.assertIsNone(self.intervention_logic.infer_completion(completion_no_target))

        # Test completion with both 0 and 1
        completion_both = "Text with both 0 and 1"
        self.assertEqual(self.intervention_logic.infer_completion(completion_both), 0)