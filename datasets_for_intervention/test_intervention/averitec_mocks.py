"""
Mocks for AVeriTeC unit tests.

Dataset samples use the new architecture keys:
  idx, claim, explanations, gold_rubric {question: bool}, gold_target, mediator_rubric
"""
from copy import deepcopy
from datasets_for_intervention.prompt import Prompt


class FakeLLMModel:
    """Minimal LLMModel stub -- no real tokenizer or generation."""
    def apply_chat_template(self, messages, add_generation_prompt=True):
        return messages[-1]["content"]

    def clean_model_specific_completion(self, text: str) -> str:
        return text


GOLD_RUBRIC_2Q = {
    "Did Hunter Biden have any experience in the energy sector in 2014?": False,
    "Did Hunter Biden have any experience in Ukraine in 2014?": False,
}
GOLD_RUBRIC_1Q = {
    "Did Trump make pro-gay laws when in office?": False,
}
GOLD_RUBRIC_3Q = {
    "Did China's Ministry of Foreign Affairs announce that Chinese people should not "
    "travel to the United States or buy American-made products in its daily press "
    "briefing on August 13, 2020?": False,
    "Did the weekly policy briefing from China's State Council on August 13, 2020 "
    "include a mention of the call for Chinese people to not travel to the United "
    "States or buy American-made products?": False,
    "Did the Chinese Ministry of Foreign Affairs announce that Chinese people should "
    "not travel to the United States or buy American-made products on its Twitter "
    "account on or after August 13, 2020?": False,
}


class AVeriTeCDatasetMock:
    """
    Three canonical samples matching the new architecture.
      [0]: Supported, 2 questions  (all False)
      [1]: Refuted,   1 question   (single False -- eligible for Local Edits)
      [2]: Refuted,   3 questions  (multi-False  -- NOT eligible for Local Edits)
    """
    def __init__(self):
        self.data = [
            {
                "idx": "0",
                "claim": (
                    "Hunter Biden had no experience in Ukraine or in the energy sector "
                    "when he joined the board of Burisma."
                ),
                "explanations": {
                    "Did Hunter Biden have any experience in the energy sector in 2014?":
                        "Hunter Biden's previous career history does not include work for energy companies.",
                    "Did Hunter Biden have any experience in Ukraine in 2014?":
                        "Hunter Biden's previous career history does not include working with Ukrainian companies.",
                },
                "gold_rubric":    deepcopy(GOLD_RUBRIC_2Q),
                "gold_target":    "Supported",
                "mediator_rubric": deepcopy(GOLD_RUBRIC_2Q),
            },
            {
                "idx": "1",
                "claim": "President Trump is the most pro-gay president in American history.",
                "explanations": {
                    "Did Trump make pro-gay laws when in office?":
                        "He made laws such as: 1. Appointing Anti-Equality Judges "
                        "2. Stripping protections from LGBTQ students, parents and families "
                        "3. Defending Anti-Gay Discrimination.",
                },
                "gold_rubric":    deepcopy(GOLD_RUBRIC_1Q),
                "gold_target":    "Refuted",
                "mediator_rubric": deepcopy(GOLD_RUBRIC_1Q),
            },
            {
                "idx": "2",
                "claim": (
                    "Beijing government announced that Chinese people should not travel "
                    "to the United States or buy American-made products."
                ),
                "explanations": {
                    "Did China's Ministry of Foreign Affairs announce that Chinese people should not "
                    "travel to the United States or buy American-made products in its daily press "
                    "briefing on August 13, 2020?":
                        "Transcript of August 13 daily press briefing does not include a request.",
                    "Did the weekly policy briefing from China's State Council on August 13, 2020 "
                    "include a mention of the call for Chinese people to not travel to the United "
                    "States or buy American-made products?":
                        "China's State Council weekly policy briefing pages for August 13, 2020 "
                        "do not mention the US.",
                    "Did the Chinese Ministry of Foreign Affairs announce that Chinese people should "
                    "not travel to the United States or buy American-made products on its Twitter "
                    "account on or after August 13, 2020?":
                        "A keywords search found no claim on the Ministry's Twitter account.",
                },
                "gold_rubric":    deepcopy(GOLD_RUBRIC_3Q),
                "gold_target":    "Refuted",
                "mediator_rubric": deepcopy(GOLD_RUBRIC_3Q),
            },
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __iter__(self):
        return iter(self.data)
