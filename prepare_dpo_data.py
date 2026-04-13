import argparse
import copy
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from datasets_for_intervention.pauq_dataset import PAUQDataset
from prepare_sft_data import (
    INTERVENTION_FUNCS,
    build_assistant_response,
    build_user_prompt,
    reconstruct_sql_from_skeleton,
)


def make_chosen(sample, dataset, intervention_types, intervention_prob):
    if intervention_types and random.random() < intervention_prob:
        itype = random.choice(intervention_types)
        intervened = INTERVENTION_FUNCS[itype](sample, dataset)
        if intervened is not None:
            return intervened
    return copy.deepcopy(sample)


def make_rejected(base_sample, chosen_query, dataset, intervention_types, max_attempts):
    """Build a sample whose mediator does NOT reconstruct to its printed SQL.

    Strategy: intervene on the original mediator (skeleton/schema_links/slots)
    but keep the chosen sample's SQL as the printed answer, breaking the
    mediator/answer correspondence. We always intervene from the original gold
    sample so that intervention helpers see real db identifiers.
    """
    gold_query = chosen_query
    types = list(intervention_types)
    for _ in range(max_attempts):
        if not types:
            return None
        itype = random.choice(types)
        intervened = INTERVENTION_FUNCS[itype](base_sample, dataset)
        if intervened is None:
            continue
        rebuilt = reconstruct_sql_from_skeleton(
            intervened["true_skeleton"], intervened["true_slots"]
        )
        if rebuilt == gold_query:
            continue
        intervened["query"] = gold_query
        return intervened
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO data from PAUQ dataset")
    parser.add_argument("--data-path", required=True, help="Path to pauq/ folder")
    parser.add_argument("--output", required=True, help="Output .jsonl file path")
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--intervention-types", default="local_column,local_table,global",
                        help="Comma-separated intervention types")
    parser.add_argument("--chosen-intervention-prob", type=float, default=0.5,
                        help="Probability of using an intervened mediator for the chosen response")
    parser.add_argument("--max-rejected-attempts", type=int, default=5,
                        help="Retries to find a valid mediator/SQL mismatch for rejected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default=None,
                        help="HuggingFace tokenizer name/path; if set, also emit chat-templated fields")
    args = parser.parse_args()

    intervention_types = [t.strip() for t in args.intervention_types.split(",") if t.strip()]
    for t in intervention_types:
        if t not in INTERVENTION_FUNCS:
            parser.error(f"Unknown intervention type: {t}. Choose from: {list(INTERVENTION_FUNCS.keys())}")

    random.seed(args.seed)

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = PAUQDataset(args.data_path, train=(args.split == "train"))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    total = 0
    chosen_gold = 0
    chosen_intervened = 0
    skipped = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for sample in dataset:
            chosen = make_chosen(sample, dataset, intervention_types, args.chosen_intervention_prob)
            rejected = make_rejected(sample, chosen["query"], dataset, intervention_types, args.max_rejected_attempts)
            if rejected is None:
                skipped += 1
                continue

            prompt = build_user_prompt(sample["question"], sample["db"])
            chosen_text = build_assistant_response(chosen)
            rejected_text = build_assistant_response(rejected)

            record = {
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }

            if tokenizer is not None:
                record["chosen_text"] = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": chosen_text}],
                    tokenize=False, add_generation_prompt=False,
                )
                record["rejected_text"] = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": rejected_text}],
                    tokenize=False, add_generation_prompt=False,
                )

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            if chosen["query"] == sample["query"]:
                chosen_gold += 1
            else:
                chosen_intervened += 1

    print(f"Split:             {args.split}")
    print(f"Total written:     {total}")
    print(f"Chosen gold:       {chosen_gold}")
    print(f"Chosen intervened: {chosen_intervened}")
    print(f"Skipped:           {skipped}")
    print(f"Output:            {args.output}")


if __name__ == "__main__":
    main()
