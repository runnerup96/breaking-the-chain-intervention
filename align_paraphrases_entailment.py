import json

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def load_jsonl(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


data = load_jsonl("/mnt/extremessd10tb/seleznev/breaking-the-chain-intervention/entailment_trees_emnlp2021_data_v3/dataset/task_2/test.jsonl")
paraphrases = load_json("/mnt/extremessd10tb/seleznev/breaking-the-chain-intervention/entailment_trees_emnlp2021_data_v3/dataset/task_2/test_question_paraphases.json")

id_to_paraphrases = {para["sample_idx"]: para for para in paraphrases}

aligned_paraphrases = [id_to_paraphrases[sample["id"]] for sample in data]

for i, (sample, paraphrases) in enumerate(zip(data, aligned_paraphrases)):
    sample_id = sample["id"]
    para_id = paraphrases["sample_idx"]

    assert sample_id == para_id, f"Mismatch on {i}th sample: Sample ID {sample_id} does not match paraphrase ID {para_id}"

print("All samples are aligned")

json.dump(aligned_paraphrases, open("/mnt/extremessd10tb/seleznev/breaking-the-chain-intervention/entailment_trees_emnlp2021_data_v3/dataset/task_2/aligned_test_question_paraphases.json", "w"), ensure_ascii=False, indent=4)