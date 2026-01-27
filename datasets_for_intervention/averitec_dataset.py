import json
import os
import random


class AVeriTeCDataset:
    def __init__(self, data_path: str):
        self.data_path = data_path

        self.filtered_samples = json.load(open(os.path.join(self.data_path, "onlyboolean_samples.json"), 'r', encoding='utf-8'))
        self.paraphrases = json.load(open(os.path.join(self.data_path, "onlyboolean_paraphrases.json"), 'r', encoding='utf-8'))

        self.idx2paraphrases = dict()
        for sample in self.paraphrases:
            self.idx2paraphrases[sample['sample_idx']] = sample['response']['response'].split('|')

        self.excluded_few_shot_examples = [
            "Hunter Biden had no experience in Ukraine or in the energy sector when he joined the board of Burisma.",
            "President Trump is the most pro-gay president in American history.",
            "Beijing government announced that Chinese people should not travel to the United States or buy American-made products."
        ]
        self.data = []
        self.process_data()


    def get_random_paraphrase(self, idx):
        return random.choice(self.idx2paraphrases[idx])

    def process_data(self):

        for idx, sample in enumerate(self.filtered_samples):
            question_answer_dict,question_explanation_dict = {}, {}
            for question_data in sample['questions']:
                question = question_data['question']
                if question_data.get('answers') and len(question_data['answers']) > 0:
                    answer = question_data['answers'][0]['answer']# just get the first answer
                    explanation = question_data['answers'][0]['boolean_explanation']
                    question_answer_dict[question] = answer
                    question_explanation_dict[question] = explanation


            # filtering of appropriate samples
            support_label_check = sample['label'] == 'Supported'
            correct_refuted_check = True if sample['label'] == 'Refuted' and len(question_answer_dict) == 1 else False
            few_shot_exclusion = True if sample['claim'] not in self.excluded_few_shot_examples else False
            
            # if (support_label_check or correct_refuted_check) and len(question_answer_dict) > 0 and few_shot_exclusion:
            if len(question_answer_dict) > 0 and few_shot_exclusion:
                processed_sample = {
                    "idx": idx,
                    "claim": sample['claim'],
                    "label": sample['label'],
                    "supporting_questions": question_answer_dict,
                    "explanations": question_explanation_dict
                }
            
                self.data.append(processed_sample)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]


def _norm_yesno(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "y", "true", "t", "1"):
            return "Yes"
        if s in ("no", "n", "false", "f", "0"):
            return "No"
    raise ValueError(f"Invalid Yes/No value: {v!r}")


def _norm_label(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s in ("Supported", "Refuted"):
            return s
    raise ValueError(f"Invalid label (Supported/Refuted): {v!r}")


class AVeriTeCCorrectionDataset:
    def __init__(
        self,
        path: str,
        *,
        allow_missing_in_bad: bool = True,
        forbid_extra_in_bad: bool = True,
        validate_gold_in_explanations: bool = True,
    ):
        with open(path, "r", encoding="utf-8") as f:
            idx2payload = json.load(f)
        if not isinstance(idx2payload, dict):
            raise ValueError("AVeriTeC correction file must be a JSON dict: idx -> payload")

        self.data = []
        self.idx2claim = {}

        for raw_idx, p in idx2payload.items():
            if not isinstance(p, dict):
                raise ValueError(f"Payload for idx={raw_idx} must be a dict")

            idx = int(raw_idx) if str(raw_idx).isdigit() else raw_idx

            for k in ("claim", "explanations", "golden_supporting_questions", "golden_label", "bad_supporting_questions"):
                if k not in p:
                    raise ValueError(f"Missing '{k}' for idx={raw_idx}")

            claim = str(p["claim"])

            explanations = p["explanations"]
            if not isinstance(explanations, dict):
                raise ValueError(f"'explanations' must be dict for idx={raw_idx}")
            explanations = {str(q): str(e) for q, e in explanations.items()}

            gold_sq_raw = p["golden_supporting_questions"]
            bad_sq_raw = p["bad_supporting_questions"]
            if not isinstance(gold_sq_raw, dict) or not isinstance(bad_sq_raw, dict):
                raise ValueError(f"'golden_supporting_questions' and 'bad_supporting_questions' must be dicts for idx={raw_idx}")

            golden_supporting_questions = {str(q): _norm_yesno(a) for q, a in gold_sq_raw.items()}
            bad_supporting_questions = {str(q): _norm_yesno(a) for q, a in bad_sq_raw.items()}

            golden_label = _norm_label(p["golden_label"])
            bad_label = _norm_label(p.get("bad_label", None))  # may be None

            gold_qs = set(golden_supporting_questions.keys())
            bad_qs = set(bad_supporting_questions.keys())
            expl_qs = set(explanations.keys())

            if validate_gold_in_explanations:
                missing_expl = sorted(gold_qs - expl_qs)
                if missing_expl:
                    raise ValueError(f"[idx={raw_idx}] some gold questions missing in explanations: {missing_expl[:5]}")

            if forbid_extra_in_bad:
                extra_bad = sorted(bad_qs - gold_qs)
                if extra_bad:
                    raise ValueError(f"[idx={raw_idx}] bad_supporting_questions has extra questions not in gold: {extra_bad[:5]}")

            if not allow_missing_in_bad:
                missing_bad = sorted(gold_qs - bad_qs)
                if missing_bad:
                    raise ValueError(f"[idx={raw_idx}] bad_supporting_questions is missing gold questions: {missing_bad[:5]}")

            missing_bad_expl = sorted(bad_qs - expl_qs)
            if missing_bad_expl:
                raise ValueError(f"[idx={raw_idx}] some bad questions missing in explanations: {missing_bad_expl[:5]}")

            sample = {
                "idx": idx,
                "claim": claim,
                "explanations": explanations,

                "golden_supporting_questions": golden_supporting_questions,
                "golden_label": golden_label,

                "bad_supporting_questions": bad_supporting_questions,
                "bad_label": bad_label,

                "supporting_questions": golden_supporting_questions,
                "label": golden_label,
            }

            self.data.append(sample)
            self.idx2claim[idx] = claim

        self.data.sort(key=lambda x: str(x["idx"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]