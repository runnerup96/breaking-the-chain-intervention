import os
import json
from pauq_dataset import PAUQDataset
from utils import compare_schema_links, compare_skeletons, compare_slots

path = '/Users/kmvafin/research/breaking-the-chain-intervention/intervention_analysis/intervention_predictions/pauq'
files = os.listdir(path)

dataset = PAUQDataset('./pauq')

for file in files:
    data = None
    with open(f'{path}/{file}', 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'result' not in data:
        continue

    generated = {}

    for i in range(len(data['result'])):
        if data['result'][i]['completion_type'] == 'gold_structure':
            continue
        idx = data['result'][i]['index']

        if not "skeleton" in data['result'][i]:
            continue
        if not "slots" in data['result'][i]:
            continue
        schema_links = data['result'][i]['schema_links']
        skeleton = data['result'][i]['skeleton']
        slots = data['result'][i]['slots']
        generated[idx] = {'schema_links': schema_links, 'skeleton': skeleton, 'slots': slots}

    
    new_data = {}

    cnt = 0

    for i in range(len(dataset)):
        golden = dataset[i]
        idx = golden['index']
        if idx not in generated:
            continue
        gen = generated[idx]
        if gen['schema_links'] == {}:
            continue

        if (
            compare_schema_links(gen['schema_links'], golden['true_schema_links']) and
            # compare_skeletons(gen["skeleton"], golden["true_skeleton"]) and
            compare_slots(gen["slots"], golden["true_slots"])
           ):
            new_data[idx] = {
                "query": golden["query"],
                "question": golden["question"],
                "db": golden["db"],
                "db_schema": golden["db_schema"],
                "true_schema_links": golden["true_schema_links"],
                "true_skeleton": golden["true_skeleton"],
                "true_slots": golden["true_slots"],
                "bad_schema_links": gen["schema_links"],
                "bad_skeleton": gen["skeleton"],
                "bad_slots": gen["slots"],
                "paraphrase": golden["paraphrase"],
            }
        elif cnt < 0:
            cnt += 1
            print(gen['schema_links'])
            print(golden['true_schema_links'])
            print("--------------------------------")
            print(gen["skeleton"])
            print(golden["true_skeleton"])
            print("--------------------------------")
            print(gen["slots"])
            print(golden["true_slots"])
            print("--------------------------------")
            print("--------------------------------")
            
    
    file = file.split('_')[0] + '_intervention.json'
    print(file, len(new_data))

    if len(new_data) != 0:
        with open(f'./pauq/intervention/{file}', 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
