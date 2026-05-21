import os
import sys
from copy import deepcopy

from torch.utils.data import DataLoader
from transformers import TrainerCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_model import LLMModel
from make_intervention import GEN_MAX_NEW_TOKENS
from datasets_for_intervention import (
    ricechem_dataset,
    ricechem_intervention,
    ricechem_evaluation,
    ricechem_structure_processor,
    averitec_dataset,
    averitec_intervention,
    averitec_evaluation,
    averitec_structure_processor,
    tabfact_dataset,
    tabfact_intervention,
    tabfact_evaluation,
    tabfact_dsl_engine,
    tabfact_structure_processor,
    cruxeval_dataset,
    cruxeval_intervention,
    cruxeval_evaluation,
    cruxeval_structure_processor,
)


class EvalLLMAdapter(LLMModel):
    """Wrap an externally owned model+tokenizer (the in-training model) with the
    LLMModel interface, without reloading weights."""

    def __init__(self, model, tokenizer, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.use_api = False
        self.api_base_url = None
        self.tokenizer_name = model_name
        self.model_family = self.identify_model_family(model_name)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id


def _build_pipeline(dataset_type, data_path, prompting_regime, tool_mode, llm, no_explanations=False):
    if dataset_type == "ricechem":
        dataset = ricechem_dataset.RiceChemDataset(data_path=data_path)
        tool = ricechem_structure_processor.RiceChemTool(dataset, tool_mode)
        processor = ricechem_structure_processor.RiceChemStructureProcessor(dataset, tool_mode)
        intervention_logic = ricechem_intervention.RiceChemIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=prompting_regime,
            tool_mode=tool_mode,
        )
        evaluator = ricechem_evaluation.RiceChemEvaluation(dataset, processor, tool_mode)
    elif dataset_type == "averitec":
        include_explanations = not no_explanations
        dataset = averitec_dataset.AVeriTeCDataset(data_path, include_explanations=include_explanations)
        tool = averitec_structure_processor.AVeriTeCTool(dataset, tool_mode)
        processor = averitec_structure_processor.AVeriTeCStructureProcessor(dataset, tool_mode)
        intervention_logic = averitec_intervention.AVeriTeCIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=prompting_regime,
            tool_mode=tool_mode,
            include_explanations=include_explanations,
        )
        evaluator = averitec_evaluation.AVeriTeCEvaluation(dataset, processor, tool_mode)
    elif dataset_type == "tabfact":
        dataset = tabfact_dataset.TabFactDataset(
            queries_json_path=os.path.join(data_path, "bootstrap_full.json"),
            tables_dir=os.path.join(data_path, "data", "all_csv"),
        )
        engine = tabfact_dsl_engine.TabFactEngine()
        tool = tabfact_structure_processor.TabFactTool(engine)
        processor = tabfact_structure_processor.TabFactStructureProcessor(engine)
        intervention_logic = tabfact_intervention.TabFactIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=prompting_regime,
            tool_mode=tool_mode,
        )
        evaluator = tabfact_evaluation.TabFactEvaluation(
            dataset=dataset,
            processor=processor,
            tool=tool,
            tool_mode=tool_mode,
        )
    elif dataset_type == "cruxeval":
        dataset = cruxeval_dataset.CRUXEvalDataset(data_path=data_path)
        tool = cruxeval_structure_processor.CRUXEvalTool(dataset, tool_mode)
        processor = cruxeval_structure_processor.CRUXEvalStructureProcessor(dataset, tool_mode)
        intervention_logic = cruxeval_intervention.CRUXEvalIntervention(
            dataset=dataset,
            llm_model=llm,
            tool=tool,
            processor=processor,
            prompting_regime=prompting_regime,
            tool_mode=tool_mode,
        )
        evaluator = cruxeval_evaluation.CRUXEvalEvaluation(
            dataset=dataset,
            processor=processor,
            tool=tool,
            tool_mode=tool_mode,
        )
    else:
        raise ValueError(f"Unsupported faithfulness dataset {dataset_type}")
    return dataset, intervention_logic, evaluator


def _flatten_metrics(metrics, prefix="eval_faithfulness"):
    out = {}

    def rec(node, path):
        if isinstance(node, dict):
            if "mean" in node and "n_total" in node:
                if node.get("mean") is not None:
                    out[f"{path}/mean"] = float(node["mean"])
                if node.get("std") is not None:
                    out[f"{path}/std"] = float(node["std"])
                out[f"{path}/n_valid"] = int(node.get("n_valid", 0))
                return
            for k, v in node.items():
                rec(v, f"{path}/{str(k).replace(' ', '_')}")
        elif isinstance(node, (int, float)):
            out[path] = float(node)

    rec(metrics, prefix)
    return out


class FaithfulnessEvalCallback(TrainerCallback):
    def __init__(
        self,
        dataset_type,
        data_path,
        model_name,
        batch_size,
        prompting_regime="standard",
        tool_mode="none",
        no_explanations=False,
    ):
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.prompting_regime = prompting_regime
        self.tool_mode = tool_mode
        self.no_explanations = no_explanations
        self._gen_budget = GEN_MAX_NEW_TOKENS.get(dataset_type, GEN_MAX_NEW_TOKENS["default"])
        self._trainer = None
        self._dataset = None
        self._intervention_logic = None
        self._evaluator = None

    def attach(self, trainer):
        self._trainer = trainer

    def _ensure_pipeline(self, llm):
        if self._dataset is None:
            self._dataset, self._intervention_logic, self._evaluator = _build_pipeline(
                self.dataset_type,
                self.data_path,
                self.prompting_regime,
                self.tool_mode,
                llm,
                no_explanations=self.no_explanations,
            )
        else:
            self._intervention_logic.llm_model = llm

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None or tokenizer is None or self._trainer is None:
            return control

        was_training = model.training
        model.eval()
        try:
            llm = EvalLLMAdapter(model=model, tokenizer=tokenizer, model_name=self.model_name)
            self._ensure_pipeline(llm)

            loader = DataLoader(
                self._dataset,
                batch_size=self.batch_size,
                collate_fn=lambda b: b,
                shuffle=False,
            )

            processed = []
            for batch in loader:
                pred_prompts = [
                    self._intervention_logic.make_prompt(s, include_gold_structure=False)
                    for s in batch
                ]
                pred_outputs = llm.generate(
                    pred_prompts,
                    max_new_tokens=self._gen_budget["pred"][self.tool_mode],
                    skip_special_tokens=False,
                )
                for orig_sample, model_out in zip(batch, pred_outputs):
                    sample = deepcopy(orig_sample)
                    sample_iv = self._intervention_logic.make_intervention(sample, model_out)
                    if sample_iv.get("generation_status") == "error":
                        processed.append(sample_iv)
                        continue
                    prompts_iv = self._intervention_logic.interventions_to_prompt(sample_iv)
                    if prompts_iv:
                        outs_iv = llm.generate(
                            prompts_iv,
                            max_new_tokens=self._gen_budget["interv"][self.tool_mode],
                            skip_special_tokens=False,
                        )
                        final_sample = self._intervention_logic.collect_intervention_completion(
                            sample_iv, outs_iv,
                        )
                    else:
                        final_sample = sample_iv
                    processed.append(final_sample)

            print(f"\n=== Faithfulness eval at step {state.global_step} ===")
            metrics = self._evaluator.evaluate(processed)
            scalar_logs = _flatten_metrics(metrics)
            scalar_logs["faithfulness_step"] = state.global_step
            self._trainer.log(scalar_logs)
        finally:
            if was_training:
                model.train()
        return control
