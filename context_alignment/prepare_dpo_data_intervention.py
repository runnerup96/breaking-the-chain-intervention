import argparse
import json
import os
import random
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets_for_intervention import (
    averitec_dataset,
    averitec_intervention,
    averitec_structure_processor,
    cruxeval_dataset,
    cruxeval_intervention,
    cruxeval_structure_processor,
    ricechem_dataset,
    ricechem_intervention,
    ricechem_structure_processor,
    tabfact_dataset,
    tabfact_dsl_engine,
    tabfact_intervention,
    tabfact_structure_processor,
)
from datasets_for_intervention.cruxeval_trace import trace_to_text


class _NoOpLLM:
    """Stub passed to Intervention classes; only stored, never invoked here."""

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return ""

    def clean_model_specific_completion(self, text):
        return text


def build_current_sample_ricechem(sample: dict) -> str:
    checklist_string = "".join(
        f"{rubric_item} (True/False): <True/False>\n"
        for rubric_item in sample["gold_rubric"]
    )
    return (
        "Now follow the same structure for the given input.\n\n"
        "Question:\n"
        f"{sample['task']}\n\n"
        "Answer:\n"
        f"{sample['student_answer']}\n\n"
        "Checklist:\n"
        f"{checklist_string}"
    )


def build_current_sample_averitec(sample: dict) -> str:
    checklist_template = "".join(
        f"- Q: {q} (True/False): <True/False>\n" for q in sample["gold_rubric"]
    )
    explanations = sample.get("explanations") or {}
    if explanations:
        explanations_str = "".join(
            f"- Q: {q} E: {e}\n" for q, e in explanations.items()
        )
        return (
            "Now follow the same structure for the given claim.\n\n"
            "Claim:\n"
            f"{sample['claim']}\n\n"
            "Explanations:\n"
            f"{explanations_str}\n"
            "Checklist:\n"
            f"{checklist_template}"
        )
    return (
        "Now follow the same structure for the given claim.\n\n"
        "Claim:\n"
        f"{sample['claim']}\n\n"
        "Checklist:\n"
        f"{checklist_template}"
    )


def build_current_sample_tabfact(sample: dict) -> str:
    return (
        "Now follow the same structure for the given input.\n\n"
        "Table:\n"
        f"{sample['table_html_csv']}\n\n"
        "Claim:\n"
        f"{sample['statement']}\n\n"
        "Verifier Query: <YOUR QUERY>\n"
    )


def build_response_ricechem(local: dict, score) -> str:
    lines = ["Checklist:"]
    for rubric_item, val in local["mediator_rubric"].items():
        lines.append(f"{rubric_item} (True/False): {val}")
    lines.append(f"Final grade ({local['score_range']}): {float(score):.1f}")
    return "\n".join(lines)


def build_response_averitec(local: dict, verdict: str) -> str:
    lines = ["Checklist:"]
    for q, val in local["mediator_rubric"].items():
        lines.append(f"- Q: {q} (True/False): {val}")
    lines.append(f"Final Verdict: {verdict}")
    return "\n".join(lines)


def build_response_tabfact(local: dict, target: bool) -> str:
    return f"Verifier Query: {local['mediator_query']}\nExecution Result: {target}"


def build_current_sample_cruxeval(sample: dict) -> str:
    return (
        "Now follow the same structure for the given code and call.\n\n"
        "Code:\n"
        f"```python\n{sample['code']}```\n"
        f"Call: f({sample['input_str']})\n"
        "Trace: <YOUR TRACE>\n"
        "Final Answer: <YOUR ANSWER>\n"
    )


def build_response_cruxeval(local: dict, target: str) -> str:
    trace_text = trace_to_text(local["mediator_trace"])
    return f"Trace:\n{trace_text}\nFinal Answer: {target}"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DPO data using the dataset/intervention classes "
                    "for ricechem, averitec, or tabfact."
    )
    parser.add_argument("--dataset", required=True,
                        choices=["ricechem", "averitec", "tabfact", "cruxeval"])
    parser.add_argument("--data-path", required=True,
                        help="Per-dataset path (matches the convention of make_intervention.py): "
                             "ricechem -> directory with the four CSV pairs; "
                             "averitec -> directory with onlyboolean_samples.json; "
                             "tabfact  -> directory containing bootstrap_full.json and data/all_csv/; "
                             "cruxeval -> directory containing test.jsonl.")
    parser.add_argument("--output", required=True, help="Output .jsonl path")
    parser.add_argument("--prompting-regime", default="standard",
                        choices=["standard", "detailed", "max_detailed"])
    parser.add_argument("--max-edits-per-sample", type=int, default=None,
                        help="If set, randomly subsample this many local edits per gold sample.")
    parser.add_argument("--no-explanations", action="store_true",
                        help="AVeriTeC ablation: strip explanations from X. "
                             "Ignored for other datasets.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    llm = _NoOpLLM()

    if args.dataset == "ricechem":
        dataset = ricechem_dataset.RiceChemDataset(data_path=args.data_path)
        tool = ricechem_structure_processor.RiceChemTool(dataset, "none")
        processor = ricechem_structure_processor.RiceChemStructureProcessor(dataset, "none")
        intervention = ricechem_intervention.RiceChemIntervention(
            dataset=dataset, llm_model=llm, tool=tool, processor=processor,
            prompting_regime=args.prompting_regime, tool_mode="none",
        )
        build_current = build_current_sample_ricechem
        gold_mediator_field = "gold_rubric"
        mediator_field = "mediator_rubric"
    elif args.dataset == "averitec":
        include_explanations = not args.no_explanations
        dataset = averitec_dataset.AVeriTeCDataset(
            data_path=args.data_path, include_explanations=include_explanations,
        )
        tool = averitec_structure_processor.AVeriTeCTool(dataset, "none")
        processor = averitec_structure_processor.AVeriTeCStructureProcessor(dataset, "none")
        intervention = averitec_intervention.AVeriTeCIntervention(
            dataset=dataset, llm_model=llm, tool=tool, processor=processor,
            prompting_regime=args.prompting_regime, tool_mode="none",
            include_explanations=include_explanations,
        )
        build_current = build_current_sample_averitec
        gold_mediator_field = "gold_rubric"
        mediator_field = "mediator_rubric"
    elif args.dataset == "tabfact":
        queries_json_path = os.path.join(args.data_path, "bootstrap_full.json")
        tables_dir = os.path.join(args.data_path, "data", "all_csv")
        dataset = tabfact_dataset.TabFactDataset(
            queries_json_path=queries_json_path, tables_dir=tables_dir,
        )
        engine = tabfact_dsl_engine.TabFactEngine()
        tool = tabfact_structure_processor.TabFactTool(engine)
        processor = tabfact_structure_processor.TabFactStructureProcessor(engine)
        intervention = tabfact_intervention.TabFactIntervention(
            dataset=dataset, llm_model=llm, tool=tool, processor=processor,
            prompting_regime=args.prompting_regime, tool_mode="none",
        )
        build_current = build_current_sample_tabfact
        gold_mediator_field = "gold_query"
        mediator_field = "mediator_query"
    else:  # cruxeval
        dataset = cruxeval_dataset.CRUXEvalDataset(data_path=args.data_path)
        tool = cruxeval_structure_processor.CRUXEvalTool(dataset, "none")
        processor = cruxeval_structure_processor.CRUXEvalStructureProcessor(dataset, "none")
        intervention = cruxeval_intervention.CRUXEvalIntervention(
            dataset=dataset, llm_model=llm, tool=tool, processor=processor,
            prompting_regime=args.prompting_regime, tool_mode="none",
        )
        build_current = build_current_sample_cruxeval
        gold_mediator_field = "gold_trace"
        mediator_field = "mediator_trace"

    instruction = intervention.prompt.build_zeroshot_instruction()
    few_shot = intervention.prompt.few_shot

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = 0
    n_edit_pairs = 0
    n_gold_pairs = 0
    no_edits = 0
    skipped_degenerate = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for sample in dataset:
            s = deepcopy(sample)
            s["generation_status"] = "correct"
            s[mediator_field] = deepcopy(s[gold_mediator_field])

            interv = intervention.make_structure_intervention(s)
            local_edits = list(interv.get("Local Edits", []))
            if not local_edits:
                no_edits += 1
                continue

            if args.max_edits_per_sample is not None and len(local_edits) > args.max_edits_per_sample:
                local_edits = random.sample(local_edits, args.max_edits_per_sample)

            current_sample_text = build_current(s)
            parts = [instruction]
            if few_shot.strip():
                parts.append(few_shot)
            parts.append(current_sample_text)
            prompt_text = "\n\n".join(parts)

            for local in local_edits:
                if args.dataset == "ricechem":
                    expected = local["expected_score_after_intervention"]
                    gold_target = local["gold_score"]
                    build_fn = build_response_ricechem
                elif args.dataset == "averitec":
                    expected = local["expected_target_after_intervention"]
                    gold_target = local["gold_target"]
                    build_fn = build_response_averitec
                elif args.dataset == "tabfact":
                    expected = local["expected_target_after_intervention"]
                    gold_target = local["gold_target"]
                    build_fn = build_response_tabfact
                else:  # cruxeval
                    expected = local["expected_target_after_intervention"]
                    gold_target = local["gold_target"]
                    build_fn = build_response_cruxeval

                if expected is None:
                    skipped_degenerate += 1
                    continue

                # Two DPO pairs per local edit, kept 1:1 balanced:
                #   edit direction -- faithful answer for the EDITED mediator is `expected`
                #   gold direction -- faithful answer for the GOLD mediator is `gold_target`
                # Without the gold direction the faithful answer is 100% correlated
                # with `expected`, and DPO collapses onto a constant shortcut.
                gold_local = dict(local)
                gold_local[mediator_field] = s[gold_mediator_field]

                pairs = (
                    (build_fn(local, expected), build_fn(local, gold_target), "edit"),
                    (build_fn(gold_local, gold_target), build_fn(gold_local, expected), "gold"),
                )

                for chosen, rejected, direction in pairs:
                    if chosen == rejected:
                        skipped_degenerate += 1
                        continue

                    f.write(json.dumps({
                        "prompt": prompt_text,
                        "chosen": chosen,
                        "rejected": rejected,
                    }, ensure_ascii=False) + "\n")
                    total += 1
                    if direction == "edit":
                        n_edit_pairs += 1
                    else:
                        n_gold_pairs += 1

    print(f"Dataset:                      {args.dataset}")
    print(f"Prompting regime:             {args.prompting_regime}")
    print(f"Total written:                {total}")
    print(f"  edit-direction pairs:       {n_edit_pairs}")
    print(f"  gold-direction pairs:       {n_gold_pairs}")
    print(f"Samples without local edits:  {no_edits}")
    print(f"Skipped (chosen==rejected):   {skipped_degenerate}")
    print(f"Output:                       {args.output}")


if __name__ == "__main__":
    main()
