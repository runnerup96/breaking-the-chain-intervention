import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict

import pandas as pd


SPLIT_NAMES = ("train", "val", "test")


RICECHEM_TASK_RUBRIC_KEYS = {
    1: [
        "correctly cites decreased electron electron repulsion",
        "relates decreased electron electron repulsion to decreased potential energy",
        "3rd and 4th electrons ionized feel same core charge",
        "3rd and 4th electrons ionized from n=3 shell and have same radius",
        "5th electron ionized from n=2 shell and feels higher core charge",
        "5th electron ionized from n=2 shell and has smaller radius",
        "correctly explains relationship of potential energy to ionization energy",
        "partially explains relationship between potential energy and ionization energy",
    ],
    2: [
        "Correctly states that frequency is proportional to energy of light",
        "Explaining sentence 1: energy levels of an electron in an atom are quantized",
        "Explaining sentence 1: FULLY explains energy/frequency absorbed must equal the difference in energy levels in an electron",
        "Explaining sentence 1: PARTIALLY explains energy/frequency absorbed must equal the difference in energy levels in an electron",
        "Explaining sentence 2: a minimum amount of energy is needed to eject an electron",
        "Explaining sentence 2: any additional energy becomes kinetic energy",
    ],
    3: [
        "Sentence 1 is correct. Valence bond theory describes that atomic orbitals must be half-filled to participate in covalent bonding.",
        "Sentence 2: Correct number of hybrid orbitals. In this molecule, carbon must form three hybrid orbitals to form three electron domains.",
        "Sentence 2: Correct type of hybrid orbitals. Carbon must form sp2 hybrid orbitals (from using a 2s and two 2p orbitals)",
        "Sentence 3: Correctly states that nitrogen is hybridized",
        "Sentence 3: Correct type of hybridization. Nitrogen is sp2 hybridized to form 3 electron domains",
        "Sentence 3: Correct description of hybrid orbital bonds in nitrogen. Two sp2 orbitals form two sigma bonds.",
        "Sentence 3: Correct description of unhybridized orbital bonds in nitrogen. Unhybridized p orbital forms pi bond",
    ],
    4: [
        "Fixed mass of one element",
        "Mass data in LoMP",
        "Combine to form compounds",
        "Integer/whole number ratio",
        "Whole numbers mean indivisible/discrete",
        "Indivisible unit of mass = atom",
    ],
}


def _partition_indices(n: int, ratios):
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _split_list(items, ratios, rng: random.Random):
    items = list(items)
    rng.shuffle(items)
    n_train, n_val, n_test = _partition_indices(len(items), ratios)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def _ensure_split_dirs(base_path: str, force: bool):
    for split in SPLIT_NAMES:
        target = os.path.join(base_path, split)
        if os.path.exists(target) or os.path.islink(target):
            if not force:
                sys.exit(f"Error: {target} already exists. Use --force to overwrite.")
            if os.path.islink(target) or os.path.isfile(target):
                os.unlink(target)
            else:
                shutil.rmtree(target)
        os.makedirs(target, exist_ok=False)


def split_ricechem(data_path: str, ratios, seed: int):
    summary = defaultdict(int)
    tasks_summary = {}

    for t in (1, 2, 3, 4):
        ans_path = os.path.join(data_path, f"Student Answers Q{t}.csv")
        rub_path = os.path.join(data_path, f"Graded Rubric Q{t}.csv")
        ans_df = pd.read_csv(ans_path)
        rub_df = pd.read_csv(rub_path)

        ans_sid_col = ans_df.columns[0]
        ans_answer_col = ans_df.columns[1]
        rub_sid_col = "SID"

        rubric_keys = RICECHEM_TASK_RUBRIC_KEYS[t]
        rub_filtered = rub_df.dropna(subset=["Score"] + rubric_keys)
        valid_rubric_sids = set(rub_filtered[rub_sid_col].astype(str))

        ans_valid_mask = ans_df[ans_answer_col].notna() & ans_df[ans_answer_col].astype(str).str.len().gt(0)
        ans_valid_sids = set(ans_df.loc[ans_valid_mask, ans_sid_col].astype(str))

        valid_sids = sorted(valid_rubric_sids & ans_valid_sids)

        rng = random.Random(seed + t)
        per_split = _split_list(valid_sids, ratios, rng)
        tasks_summary[t] = {s: len(per_split[s]) for s in SPLIT_NAMES}

        ans_row0 = ans_df.iloc[[0]]
        rub_row0 = rub_df.iloc[[0]]

        for split in SPLIT_NAMES:
            split_sid_set = set(per_split[split])
            ans_rest = ans_df.iloc[1:]
            rub_rest = rub_df.iloc[1:]
            ans_split = ans_rest[ans_rest[ans_sid_col].astype(str).isin(split_sid_set)]
            rub_split = rub_rest[rub_rest[rub_sid_col].astype(str).isin(split_sid_set)]

            out_ans = pd.concat([ans_row0, ans_split], ignore_index=True)
            out_rub = pd.concat([rub_row0, rub_split], ignore_index=True)

            split_dir = os.path.join(data_path, split)
            out_ans.to_csv(os.path.join(split_dir, f"Student Answers Q{t}.csv"), index=False)
            out_rub.to_csv(os.path.join(split_dir, f"Graded Rubric Q{t}.csv"), index=False)

            summary[split] += len(split_sid_set)

    print("== ricechem splits (SIDs per task) ==")
    for t, counts in tasks_summary.items():
        print(f"  Q{t}: train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"  TOTAL: train={summary['train']} val={summary['val']} test={summary['test']}")


def split_averitec(data_path: str, ratios, seed: int):
    src_path = os.path.join(data_path, "onlyboolean_samples.json")
    with open(src_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    groups = defaultdict(list)
    for sample in raw_samples:
        groups[sample.get("label", "")].append(sample)

    rng = random.Random(seed)
    per_split = {s: [] for s in SPLIT_NAMES}
    for label, samples in sorted(groups.items()):
        split_map = _split_list(samples, ratios, random.Random(seed + hash(label) % (2**31)))
        for s in SPLIT_NAMES:
            per_split[s].extend(split_map[s])

    for s in SPLIT_NAMES:
        rng.shuffle(per_split[s])
        out_path = os.path.join(data_path, s, "onlyboolean_samples.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(per_split[s], f, ensure_ascii=False, indent=2)

    print("== averitec splits ==")
    for s in SPLIT_NAMES:
        labels = defaultdict(int)
        for sample in per_split[s]:
            labels[sample.get("label", "")] += 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(labels.items()))
        print(f"  {s}: total={len(per_split[s])} ({breakdown})")
    print(f"  source raw size: {len(raw_samples)}")


def split_tabfact(data_path: str, ratios, seed: int):
    src_path = os.path.join(data_path, "bootstrap_full.json")
    with open(src_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    table_ids = sorted(queries.keys())
    rng = random.Random(seed)
    per_split_keys = _split_list(table_ids, ratios, rng)

    for s in SPLIT_NAMES:
        split_dir = os.path.join(data_path, s)
        split_queries = {k: queries[k] for k in per_split_keys[s]}
        with open(os.path.join(split_dir, "bootstrap_full.json"), "w", encoding="utf-8") as f:
            json.dump(split_queries, f, ensure_ascii=False)

        link_path = os.path.join(split_dir, "data")
        try:
            os.symlink("../data", link_path)
        except OSError as e:
            print(f"  WARN: could not symlink {link_path} -> ../data ({e}); falling back to copy.")
            shutil.copytree(os.path.join(data_path, "data"), link_path)

    print("== tabfact splits ==")
    for s in SPLIT_NAMES:
        keys = per_split_keys[s]
        entries = sum(len(queries[k]) for k in keys)
        print(f"  {s}: tables={len(keys)} entries={entries}")
    print(f"  source: tables={len(table_ids)} entries={sum(len(v) for v in queries.values())}")


def parse_ratios(s: str):
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ratios must be three comma-separated floats")
    if abs(sum(parts) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"ratios must sum to 1.0, got {sum(parts)}")
    return parts


def main():
    parser = argparse.ArgumentParser(
        description="Physically split a dataset directory into train/val/test subdirectories. "
                    "Each split mirrors the source layout so prepare_dpo_data_intervention.py "
                    "can consume it via --data-path <source>/{train,val,test}."
    )
    parser.add_argument("--dataset", required=True, choices=["ricechem", "averitec", "tabfact"])
    parser.add_argument("--data-path", required=True,
                        help="Source dataset path (same convention as prep script).")
    parser.add_argument("--ratios", type=parse_ratios, default="0.7,0.15,0.15",
                        help='Comma-separated train,val,test ratios summing to 1.0 (default: "0.7,0.15,0.15").')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing train/val/test subdirs.")
    args = parser.parse_args()

    if isinstance(args.ratios, str):
        args.ratios = parse_ratios(args.ratios)

    _ensure_split_dirs(args.data_path, args.force)

    if args.dataset == "ricechem":
        split_ricechem(args.data_path, args.ratios, args.seed)
    elif args.dataset == "averitec":
        split_averitec(args.data_path, args.ratios, args.seed)
    else:
        split_tabfact(args.data_path, args.ratios, args.seed)

    print(f"Splits written under: {args.data_path}/{{train,val,test}}")


if __name__ == "__main__":
    main()
