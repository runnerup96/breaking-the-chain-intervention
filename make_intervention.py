import argparse
import json
import os
import random
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
import hashlib

import llm_model


def _safe_json_default(o):
    """Last-resort encoder for json.dump: any value that isn't natively
    JSON-serializable degrades to a `repr()` string (or the obvious list
    conversion for sets/ranges/ndarrays) instead of crashing the save.
    """
    if isinstance(o, (bytes, bytearray)):
        return repr(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, complex):
        return repr(o)
    if isinstance(o, range):
        return list(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return repr(o)


def _coerce_json_keys(obj):
    """Walk obj and coerce dict keys that JSON cannot represent (anything that
    is not str / int / float / bool / None) into their `repr()`. Returns a
    sanitized copy; the original object is not mutated.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, (str, int, float, bool)) and k is not None:
                k = repr(k)
            out[k] = _coerce_json_keys(v)
        return out
    if isinstance(obj, list):
        return [_coerce_json_keys(x) for x in obj]
    if isinstance(obj, tuple):
        return [_coerce_json_keys(x) for x in obj]
    return obj


def _stringify_cruxeval_locals(samples_list):
    """CRUXEval-only post-processing applied just before saving the JSON.

    Keeps the `list[{"line": int, "locals": {...}}]` shape of `gold_trace` /
    `mediator_trace`, but renders every value in each `locals` dict via
    `repr()` so the saved value matches what the LLM sees in the prompt
    (`trace_to_text` already serializes locals via `repr(v)`).

    Side effects of doing this here, after evaluation has run:
      - `gold_trace` and `mediator_trace` end up with the same shape
        (locals: dict[str, str]), removing the current asymmetry between
        native-Python gold locals and parsed-string mediator locals.
      - Bytes / sets / dicts-with-tuple-keys hiding inside Python locals no
        longer reach the JSON encoder, so the save can't crash mid-write.

    Walks the two trace fields at the top level AND inside
    `structure_intervention.{Local Edits, Correction}[i]`. Mutates in place.
    """

    def _strify_locals(locs):
        if not isinstance(locs, dict):
            return locs
        return {str(k): repr(v) for k, v in locs.items()}

    def _strify_trace(t):
        if not isinstance(t, list):
            return t
        out = []
        for step in t:
            if not isinstance(step, dict):
                out.append(step)
                continue
            new_step = dict(step)
            if "locals" in new_step:
                new_step["locals"] = _strify_locals(new_step["locals"])
            out.append(new_step)
        return out

    def _apply(sample):
        if not isinstance(sample, dict):
            return
        if "gold_trace" in sample:
            sample["gold_trace"] = _strify_trace(sample["gold_trace"])
        if "mediator_trace" in sample:
            sample["mediator_trace"] = _strify_trace(sample["mediator_trace"])
        si = sample.get("structure_intervention")
        if isinstance(si, dict):
            for sub_list in si.values():
                if isinstance(sub_list, list):
                    for sub in sub_list:
                        _apply(sub)

    for s in samples_list:
        _apply(s)


from datasets_for_intervention import (
    ricechem_intervention, ricechem_dataset, ricechem_evaluation, ricechem_structure_processor,
    averitec_intervention, averitec_dataset, averitec_evaluation, averitec_structure_processor,
    tabfact_intervention, tabfact_dataset, tabfact_evaluation, tabfact_dsl_engine, tabfact_structure_processor,
    cruxeval_intervention, cruxeval_dataset, cruxeval_evaluation, cruxeval_structure_processor,
)

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

model_name2simple = {
    "Qwen/Qwen3-1.7B": "qwen3-1.7B",
    "Qwen/Qwen3-4B": "qwen3-4B",
    "Qwen/Qwen3-8B": "qwen3-8B",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen3-235B-a22B",
    "tiiuae/Falcon3-3B-Instruct": "falcon3-3B",
    "tiiuae/Falcon3-7B-Instruct": "falcon3-7B",
    "alpindale/Llama-3.2-1B-Instruct": "llama32-1B",
    "alpindale/Llama-3.2-3B-Instruct": "llama32-3B",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "llama31-8B",
    "google/gemma-2-2b-it": "gemma2-2B",
    "Openai/Gpt-oss-120b": "gpt-oss-120b",
    "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit": "llama31-70B",
    "unsloth/Qwen3-32B-bnb-4bit": "qwen3-32B",
    "unsloth/Qwen3-14B-bnb-4bit": "qwen3-14B"
}


GEN_MAX_NEW_TOKENS = {
    "default": {
        "pred": {"none": 512, "simple": 512, "structured": 512},
        "interv": {"none": 10, "simple": 200, "structured": 200},
    },
    "ricechem": {
        "pred": {"none": 350, "simple": 350, "structured": 350},
        "interv": {"none": 10, "simple": 200, "structured": 200},
    },
    "averitec": {
        "pred": {"none": 512, "simple": 512, "structured": 512},
        "interv": {"none": 10, "simple": 200, "structured": 200},
    },
    "tabfact": {
        "pred": {"none": 512, "simple": 512, "structured": 512},
        "interv": {"none": 10, "simple": 200, "structured": 200},
    },
    "cruxeval": {
        # CRUXEval prediction requires emitting the full trace -> larger budget.
        "pred":   {"none": 1024, "simple": 1024, "structured": 1024},
        # Intervention: assistant prefix already contains the trace; only the
        # final answer (or ARGS) remains to be generated.
        "interv": {"none": 32, "simple": 256, "structured": 256},
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--evaluation_dataset", type=str, required=True,
                        choices=["ricechem", "averitec", "tabfact", "cruxeval"])

    parser.add_argument("--prompting_regime", type=str, choices=["standard", "detailed", "max_detailed"], default="standard")
    parser.add_argument("--tool_mode", type=str, choices=["none", "simple", "structured"], default="none")
    parser.add_argument("--no_explanations", action="store_true",
                        help="AVeriTeC ablation: strip explanations from X. "
                             "Ignored for other datasets.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--try_one_batch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_api", action="store_true")
    parser.add_argument("--api_base_url", type=str, default='https://inference.airi.net:46783/v1')
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # CRUXEval-only: which perturbation levels to consider as Local Edits, and
    # whether to emit one Local Edit per applicable level ("all") or exactly
    # one randomly-chosen Local Edit per sample ("one").
    parser.add_argument(
        "--cruxeval_levels", type=str, default="1,2,3,4,5,6",
        help="Comma-separated subset of {1..6} (CRUXEval only)."
    )
    parser.add_argument(
        "--cruxeval_sampling", type=str, choices=["all", "one"], default="all",
        help='CRUXEval Local Edit sampling: "all" or "one" per sample.'
    )

    # sic! Llama-3.1-8B-Instruct under short prompts (`standard` / `tool:structured`)
    # wraps its output in ```python ... ``` and echoes the function source before
    # emitting `Trace:`. The default strict format gate rejects 788/788 such
    # completions even though the trace/answer/args parsers tolerate the wrap.
    # This opt-in flag relaxes the gate (re.match -> re.search) and tags the
    # saved JSON with a `_stripmd` filename suffix so it can be distinguished
    # from strict-mode dumps. CRUXEval-only.
    parser.add_argument(
        "--strip_md_wrap", action="store_true",
        help=("CRUXEval-only / Llama-3.1-8B carve-out: relax the strict "
              "'completion must start with Trace:' format gate to accept a "
              "Trace: block anywhere in the completion. Saved file gets a "
              "`_stripmd` suffix so it does not collide with strict-mode dumps.")
    )

    args = parser.parse_args()
    fix_seed(args.seed)

    llm = llm_model.LLMModel(args.model_name, use_api=args.use_api, api_base_url=args.api_base_url)

    project_path = os.environ["PROJECT_PATH"]

    evaluator = None

    if args.evaluation_dataset == "ricechem":
        dataset = ricechem_dataset.RiceChemDataset(
            data_path=os.path.join(project_path, "statics/datasets/RiceChem/data"),
        )
        tool = ricechem_structure_processor.RiceChemTool(dataset, args.tool_mode)
        processor = ricechem_structure_processor.RiceChemStructureProcessor(dataset, args.tool_mode)
        intervention_logic = ricechem_intervention.RiceChemIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=args.prompting_regime,
            tool_mode=args.tool_mode
        )
        evaluator = ricechem_evaluation.RiceChemEvaluation(dataset, processor, args.tool_mode)
    elif args.evaluation_dataset == "averitec":
        dataset_path = os.path.join(project_path, "statics/datasets/AVeriTeC/data")
        include_explanations = not args.no_explanations
        dataset = averitec_dataset.AVeriTeCDataset(
            dataset_path,
            include_explanations=include_explanations,
        )
        tool = averitec_structure_processor.AVeriTeCTool(dataset, args.tool_mode)
        processor = averitec_structure_processor.AVeriTeCStructureProcessor(dataset, args.tool_mode)
        intervention_logic = averitec_intervention.AVeriTeCIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=args.prompting_regime,
            tool_mode=args.tool_mode,
            include_explanations=include_explanations,
        )
        evaluator = averitec_evaluation.AVeriTeCEvaluation(dataset, processor, args.tool_mode)
    elif args.evaluation_dataset == "tabfact":
        dataset_path = os.path.join(project_path, "statics/datasets/TabFact")
        dataset = tabfact_dataset.TabFactDataset(
            queries_json_path=os.path.join(dataset_path, "bootstrap_full.json"),
            tables_dir=os.path.join(dataset_path, "data/all_csv"),
        )
        engine = tabfact_dsl_engine.TabFactEngine()
        tool = tabfact_structure_processor.TabFactTool(engine)
        processor = tabfact_structure_processor.TabFactStructureProcessor(engine)
        intervention_logic = tabfact_intervention.TabFactIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=args.prompting_regime,
            tool_mode=args.tool_mode,
        )
        evaluator = tabfact_evaluation.TabFactEvaluation(
            dataset=dataset,
            processor=processor,
            tool=tool,
            tool_mode=args.tool_mode,
        )
    elif args.evaluation_dataset == "cruxeval":
        dataset_path = os.path.join(project_path, "statics/datasets/CRUXEval")
        dataset = cruxeval_dataset.CRUXEvalDataset(data_path=dataset_path)
        tool = cruxeval_structure_processor.CRUXEvalTool(dataset, args.tool_mode)
        # sic! `lenient_format` propagates `--strip_md_wrap` to the format gate;
        # see CRUXEvalStructureProcessor.__init__ for the rationale.
        processor = cruxeval_structure_processor.CRUXEvalStructureProcessor(
            dataset, args.tool_mode, lenient_format=args.strip_md_wrap
        )
        cruxeval_levels = [int(x) for x in args.cruxeval_levels.split(",") if x.strip()]
        intervention_logic = cruxeval_intervention.CRUXEvalIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=args.prompting_regime,
            tool_mode=args.tool_mode,
            intervention_levels=cruxeval_levels,
            local_edit_sampling=args.cruxeval_sampling,
        )
        evaluator = cruxeval_evaluation.CRUXEvalEvaluation(
            dataset=dataset,
            processor=processor,
            tool=tool,
            tool_mode=args.tool_mode,
        )
    else:
        raise NotImplementedError(f"No implementation for {args.evaluation_dataset} dataset"
                                  f"Currently -- [ricechem, averitec, tabfact, cruxeval]")

    print(f"Loaded {args.evaluation_dataset} | prompting_regime={args.prompting_regime}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: b, shuffle=False)
    if args.try_one_batch:
        dataloader = [next(iter(dataloader))]

    processed_samples_list = []

    for batch in tqdm(dataloader, desc=f"Intervention {args.evaluation_dataset}"):
        pred_prompts = [intervention_logic.make_prompt(s, include_gold_structure=False) for s in batch]
        pred_outputs = llm.generate(
            pred_prompts,
            max_new_tokens=GEN_MAX_NEW_TOKENS[args.evaluation_dataset]['pred'][args.tool_mode],
            skip_special_tokens=False
        )

        for orig_sample, model_out in zip(batch, pred_outputs):
            sample = deepcopy(orig_sample)

            sample_with_interv = intervention_logic.make_intervention(sample, model_out)

            if sample_with_interv.get('generation_status') == 'error':
                processed_samples_list.append(sample_with_interv)
                continue

            prompt_list = intervention_logic.interventions_to_prompt(sample_with_interv)
            if prompt_list:
                interv_outputs = llm.generate(
                    prompt_list,
                    max_new_tokens=GEN_MAX_NEW_TOKENS[args.evaluation_dataset]['interv'][args.tool_mode],
                    skip_special_tokens=False
                )
                final_sample = intervention_logic.collect_intervention_completion(sample_with_interv, interv_outputs)
            else:
                final_sample = sample_with_interv

            processed_samples_list.append(final_sample)

    print("\n=== Running evaluation ===")
    if evaluator is not None:
        try:
            evaluation_metrics = evaluator.evaluate(processed_samples_list)
            print("Evaluation completed successfully")
        except Exception as e:
            print(f"[WARNING] Evaluation failed: {type(e).__name__}: {e}")
            evaluation_metrics = {"error": str(e)}
    else:
        evaluation_metrics = {"note": "No evaluator configured for this dataset"}
        print("No evaluator for this dataset")

    if args.evaluation_dataset == "averitec" and args.no_explanations:
        subdir = f"{args.prompting_regime}_no_expl" if args.tool_mode == "none" else "tool_no_expl"
    else:
        subdir = args.prompting_regime if args.tool_mode == "none" else "tool"
    save_dir = os.path.join(project_path, "intervention_analysis", "intervention_predictions", args.evaluation_dataset, subdir)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    # sic! Lenient-format dumps get a `_stripmd` suffix so they don't shadow
    # strict-mode files for the same (model, regime, tool_mode).
    fname_suffix = "_stripmd" if (args.evaluation_dataset == "cruxeval" and args.strip_md_wrap) else ""
    filename = f"{model_name2simple[args.model_name]}_{args.prompting_regime}_{timestamp}{fname_suffix}.json"

    # CRUXEval: render every trace `locals` value via repr() so the saved
    # `gold_trace` / `mediator_trace` have the same shape and match what the
    # LLM sees in the prompt. Done AFTER evaluation, so metrics are unaffected.
    if args.evaluation_dataset == "cruxeval":
        _stringify_cruxeval_locals(processed_samples_list)

    n_total     = len(processed_samples_list)
    n_correct   = sum(1 for s in processed_samples_list if s.get("generation_status") == "correct")
    n_incorrect = sum(1 for s in processed_samples_list if s.get("generation_status") == "incorrect")
    n_error     = sum(1 for s in processed_samples_list if s.get("generation_status") == "error")

    final_dict = {
        "meta": {
            "model":          args.model_name,
            "model_simple":   model_name2simple.get(args.model_name, args.model_name),
            "use_api":        args.use_api,
            "api_base_url":   args.api_base_url if args.use_api else None,
            "tokenizer_name": args.tokenizer_name,

            "dataset":            args.evaluation_dataset,
            "prompting_regime":   args.prompting_regime,
            "tool_mode":          args.tool_mode,
            "include_explanations": not args.no_explanations,

            "batch_size":    args.batch_size,
            "seed":          args.seed,
            "try_one_batch": args.try_one_batch,
            "timestamp":     timestamp,
            "strip_md_wrap": bool(args.strip_md_wrap),

            "total_samples":  n_total,
            "n_correct":      n_correct,
            "n_incorrect":    n_incorrect,
            "n_error":        n_error,
            "correct_rate":   round(n_correct   / max(1, n_correct + n_incorrect), 3),
            "incorrect_rate": round(n_incorrect / max(1, n_correct + n_incorrect), 3),
            "error_rate":     round(n_error     / max(1, n_total), 3),
        },
        "metrics": evaluation_metrics,
        "result": processed_samples_list,
    }

    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
        json.dump(
            _coerce_json_keys(final_dict),
            f,
            ensure_ascii=False,
            indent=2,
            default=_safe_json_default,
        )

    print(f"\nSaved {len(processed_samples_list)} samples + metrics → {filename}")