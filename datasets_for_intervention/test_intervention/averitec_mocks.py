import random

class AVeriTeCDatasetMock:
    def __init__(self):
        self.data = [
            {
                "idx": 0,
                "claim": "Hunter Biden had no experience in Ukraine or in the energy sector when he joined the board of Burisma.",
                "label": "Supported",
                "supporting_questions": {
                    "Did Hunter Biden have any experience in the energy sector in 2014?": "No",
                    "Did Hunter Biden have any experience in Ukraine in 2014?": "No"
                },
                "explanations": {
                    "Did Hunter Biden have any experience in the energy sector in 2014?": "Hunter bidens previous career history does not include work for energy company's.",
                    "Did Hunter Biden have any experience in Ukraine in 2014?": "Hunter Bidens previous career history does not include working with Ukrainian company's."
                }
            },
            {
                "idx": 1,
                "claim": "President Trump is the most pro-gay president in American history.",
                "label": "Refuted",
                "supporting_questions": {
                    "Did Trump make pro-gay laws when in office?": "No"
                },
                "explanations": {
                    "Did Trump make pro-gay laws when in office?": "He made laws such as  1. Appointing Anti-Equality Judges 2. Stripping protections from LGBTQ students, parents and families 3. Defending Anti-Gay Discrimination."
                }
            },
            {
                "idx": 2,
                "claim": "Beijing government announced that Chinese people should not travel to the United States or buy American-made products.",
                "label": "Refuted",
                "supporting_questions": {
                    "Did MFA announce this on August 13, 2020 press briefing?": "No",
                    "Did China's State Council briefing include this?": "No",
                    "Did MFA Twitter account post this between Aug 13–18, 2020?": "No"
                },
                "explanations": {
                    "Did MFA announce this on August 13, 2020 press briefing?": "Transcript of August 13 daily press briefing does not  include a request for Chinese people to avoid American products or avoid travelling to the US.",
                    "Did China's State Council briefing include this?": "China’s State Council weekly policy briefing pages for August 13, 2020 do not mention the US.",
                    "Did MFA Twitter account post this between Aug 13–18, 2020?": "A keywords search set between August 13 and August 18 2020 found no claim on the Ministry’s Twitter account."
                }
            }
        ]
        
        self.idx2paraphrases = {
            0: ["Hunter Biden lacked experience in Ukraine and energy sectors when joining Burisma board.",
                "When Hunter Biden joined Burisma, he had no background in Ukraine or energy."],
            1: ["Trump was the most LGBTQ-friendly president in US history.",
                "Donald Trump was the most pro-gay president America has ever had."],
            2: ["Chinese government told citizens not to visit the US or purchase American goods.",
                "Beijing instructed Chinese people to avoid traveling to America and buying US products."]
        }

    def get_random_paraphrase(self, idx):
        return random.choice(self.idx2paraphrases[idx])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]


class FakeCapture:
    def __init__(self):
        self.predicted_structure = None
        self.predicted_verdict = None

    def extract_intermediate_structure(self, completion: str):
        if self.predicted_structure is None:
            return {}
        return self.predicted_structure

    def extract_final_verdict(self, completion: str):
        if self.predicted_verdict is None:
            import re
            match = re.search(r'(Supported|Refuted)', completion)
            return match.group() if match else None
        return self.predicted_verdict
