"""
CRUXEval intervention logic.

Mediator M:  execution trace (list[{line, locals, nl_comment?}])
Target Y:    repr() of the function's final return value

Pass 1 (no gold structure):
  Model is asked to reproduce the execution trace and the final answer.
  We parse both M' and Y' from the completion.

Classification (matches the framework's contract):
  error     -- M' unparseable OR garbage before "Trace:"
  correct   -- model's final answer matches gold_target (string-tolerant)
  incorrect -- model's final answer does NOT match gold_target

  NOTE on "correct" semantics: averitec uses M' == M_gold for `correct`. For
  CRUXEval, exact trace equality is too brittle (LLMs paraphrase numeric reprs,
  drop intermediate steps, etc.). We instead key off the **answer** match,
  while still recording M' for later faithfulness analysis. This still
  satisfies the framework invariant: `correct` samples have a "good" baseline
  on which we can apply Local Edits to measure mediator faithfulness.

Interventions:
  correct   -> Local Edits  : perturb the gold trace at the configured levels
                              (subset of {1..6}, default all six). For each
                              chosen level, expected_target_after_intervention
                              = repr(simulate_from_trace(perturbed_trace, code)).
                              Two sampling modes are supported (see __init__):
                                "all" -- one Local Edit per applicable level
                                "one" -- exactly one Local Edit per sample,
                                         drawn uniformly at random from the
                                         applicable subset (deterministic per
                                         sample via idx + perturb_seed).
  incorrect -> Correction   : replace mediator with gold trace; expected = gold.
  error     -> empty lists.

Levels 7 (misleading NL only) and 8 (compound) are NOT used as Local Edits by
default because their expected answer is not deterministic without further
modelling of how an LLM reacts to misleading natural language. They remain
available through perturb_universal for ad-hoc experiments.
"""

import copy
import random

from datasets_for_intervention.cruxeval_trace import (
    LOCAL_EDIT_LEVELS,
    PERTURBATION_NAMES,
    applicable_levels,
    perturb_universal,
    simulate_from_trace,
    trace_to_text,
)
from datasets_for_intervention.prompt import Prompt


class CRUXEvalIntervention:

    def __init__(self, dataset, llm_model, tool, processor,
                 prompting_regime: str = "standard", tool_mode: str = "none",
                 perturb_seed: int = 42,
                 intervention_levels=None,
                 local_edit_sampling: str = "all"):
        """
        Args:
            intervention_levels: iterable of ints from {1..6} that restricts which
                Local Edit levels are considered. None defaults to all six.
            local_edit_sampling:
                "all" -- emit one Local Edit per applicable level (default).
                "one" -- emit exactly one Local Edit per sample, drawn uniformly
                         at random from the applicable subset. The choice is
                         deterministic given (sample.idx, perturb_seed) so runs
                         are reproducible.
        """
        assert prompting_regime in ["standard", "detailed", "max_detailed"]
        assert tool_mode in ["none", "simple", "structured"]
        assert local_edit_sampling in ("all", "one"), (
            "local_edit_sampling must be 'all' or 'one'"
        )

        if intervention_levels is None:
            intervention_levels = LOCAL_EDIT_LEVELS
        intervention_levels = [int(x) for x in intervention_levels]
        for lvl in intervention_levels:
            assert lvl in LOCAL_EDIT_LEVELS, (
                f"intervention level {lvl} not in supported set {LOCAL_EDIT_LEVELS}"
            )
        # Preserve order, drop duplicates.
        seen = set()
        self.intervention_levels = [
            lvl for lvl in intervention_levels if not (lvl in seen or seen.add(lvl))
        ]
        self.local_edit_sampling = local_edit_sampling

        self.dataset = dataset
        self.llm_model = llm_model
        self.tool = tool
        self.processor = processor
        self.prompting_regime = prompting_regime
        self.tool_mode = tool_mode if tool_mode != "none" else None
        self.perturb_seed = perturb_seed

        instruction, tool_call_instruction, few_shot_text = self._get_prompt_structure()
        self.prompt = Prompt(
            prompting_regime=self.prompting_regime,
            use_tool_call=self.tool_mode is not None,
            tool_call_instruction=tool_call_instruction,
            instruction=instruction,
            few_shot=few_shot_text,
            llm_model=self.llm_model,
        )

    # ------------------------------------------------------------------
    # Cleaning + low-level parsing
    # ------------------------------------------------------------------

    def clean_llm_output(self, text):
        tokens_to_remove = [
            '<|im_end|>', '<|endoftext|>', '<|im_start|>', '<|eot_id|>',
            '<|end_of_text|>', '<|pad|>',
            '<end_of_turn>', '</s>',
            '\u00ad', '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
        ]
        for t in tokens_to_remove:
            text = text.replace(t, '')
        return text.strip()

    def infer_completion(self, completion: str, sample: dict = None, short_completion: bool = False):
        """
        Parse completion and return (answer_str, tool_args_or_empty_dict).
        Logic mirrors AVeriTeC:
          - tool_mode set:  parse ARGS -> dict, call self.tool.calculate_score
          - tool_mode None: parse 'Final Answer: ...' (or whole text in short mode)
        """
        if self.tool_mode:
            tool_args = self.processor.extract_tool_args(completion, short_completion)
            if not tool_args:
                return None, {}
            verdict = self.tool.calculate_score(tool_args, sample)
            return verdict, tool_args

        answer = self.processor.extract_final_answer(completion, short_completion)
        return answer, {}

    def classify_generation(self, completion: str, mediator_trace, answer, gold_target) -> str:
        """error | correct | incorrect (based on answer match)."""
        if self.processor.check_generation_format_mistakes(completion):
            return "error"
        if mediator_trace is None:
            return "error"
        match = self.processor.compare_answers(gold_target, answer)
        if match is None:
            return "error"
        return "correct" if match == 1 else "incorrect"

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def make_intervention(self, sample: dict, generated_output: dict) -> dict:
        completion = self.clean_llm_output(generated_output["completion"])
        sample["raw_generation"] = completion

        # Parse mediator (trace) and answer from the same completion
        mediator_trace = self.processor.extract_mediator(completion)
        answer, tool_args = self.infer_completion(completion, sample, short_completion=False)

        generation_status = self.classify_generation(
            completion, mediator_trace, answer, sample["gold_target"]
        )
        sample["generation_status"]   = generation_status
        sample["mediator_trace"]      = mediator_trace if mediator_trace is not None else []
        sample["target_before_intervention"] = answer
        sample["tool_args"]           = tool_args if self.tool_mode else None

        if generation_status == "error":
            sample["structure_intervention"] = {"Local Edits": [], "Correction": []}
            return sample

        sample["structure_intervention"] = self.make_structure_intervention(sample)
        return sample

    def make_structure_intervention(self, sample: dict) -> dict:
        status = sample.get("generation_status")

        if status == "correct":
            gold_trace = sample["gold_trace"]
            code = sample["code"]
            applicable = set(applicable_levels(gold_trace))
            candidate_levels = [
                lvl for lvl in self.intervention_levels if lvl in applicable
            ]

            if not candidate_levels:
                return {"Local Edits": [], "Correction": []}

            if self.local_edit_sampling == "one":
                # Per-sample deterministic random choice (reproducible across runs).
                rng = random.Random(f"{sample['idx']}:{self.perturb_seed}")
                chosen_levels = [rng.choice(candidate_levels)]
            else:
                chosen_levels = candidate_levels

            local_edits = []
            for lvl in chosen_levels:
                perturbed = perturb_universal(
                    gold_trace, lvl, seed=self.perturb_seed + lvl
                )
                expected = simulate_from_trace(perturbed, code)
                edit = copy.deepcopy(sample)
                edit["mediator_trace"] = perturbed
                edit["perturbation_level"] = lvl
                edit["perturbation_name"]  = PERTURBATION_NAMES[lvl]
                edit["expected_target_after_intervention"] = repr(expected)
                local_edits.append(edit)
            return {"Local Edits": local_edits, "Correction": []}

        if status == "incorrect":
            corr = copy.deepcopy(sample)
            corr["mediator_trace"] = copy.deepcopy(sample["gold_trace"])
            corr["perturbation_level"] = 0
            corr["perturbation_name"]  = PERTURBATION_NAMES[0]
            return {"Local Edits": [], "Correction": [corr]}

        return {"Local Edits": [], "Correction": []}

    def interventions_to_prompt(self, sample: dict) -> list:
        interv = sample["structure_intervention"]
        prompts = []
        prompts += [
            self.make_prompt(edit, include_gold_structure=True)
            for edit in interv.get("Local Edits", [])
        ]
        prompts += [
            self.make_prompt(corr, include_gold_structure=True)
            for corr in interv.get("Correction", [])
        ]
        return prompts

    def collect_intervention_completion(self, sample: dict, generated_output: list) -> dict:
        completions = [self.clean_llm_output(g["completion"]) for g in generated_output]
        interv = sample["structure_intervention"]
        idx = 0
        for i in range(len(interv.get("Local Edits", []))):
            verdict, tool_args = self.infer_completion(
                completions[idx], interv["Local Edits"][i], short_completion=True
            )
            interv["Local Edits"][i]["raw_generation"] = completions[idx]
            interv["Local Edits"][i]["target_after_intervention"] = verdict
            if self.tool_mode:
                interv["Local Edits"][i]["tool_args_after_intervention"] = tool_args
            idx += 1
        for i in range(len(interv.get("Correction", []))):
            verdict, tool_args = self.infer_completion(
                completions[idx], interv["Correction"][i], short_completion=True
            )
            interv["Correction"][i]["raw_generation"] = completions[idx]
            interv["Correction"][i]["target_after_intervention"] = verdict
            if self.tool_mode:
                interv["Correction"][i]["tool_args_after_intervention"] = tool_args
            idx += 1
        return sample

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    # Few-shot examples. We use 2 short hand-built examples to keep prompts short.
    FEW_SHOT_EXAMPLES = [
        {
            "code": "def f(x):\n    y = x + 1\n    return y * 2\n",
            "input_str": "3",
            "trace": [
                {"line": 2, "locals": {"x": 3}},
                {"line": 3, "locals": {"x": 3, "y": 4}},
            ],
            "answer": "8",
            "explanation": "Clean execution: y=4, return 4*2=8.",
            "trace_with_intervention": [
                {"line": 2, "locals": {"x": 3}},
                {"line": 3, "locals": {"x": 3, "y": 7}},
            ],
            "answer_with_intervention": "14",
            "explanation_with_intervention": (
                "Intervention: y is overwritten to 7 in the last step. "
                "The return expression y*2 must follow the trace -> 14."
            ),
        },
        {
            "code": "def f(s):\n    out = s.upper()\n    return out + '!'\n",
            "input_str": "'hi'",
            "trace": [
                {"line": 2, "locals": {"s": "hi"}},
                {"line": 3, "locals": {"s": "hi", "out": "HI"}},
            ],
            "answer": "'HI!'",
            "explanation": "Clean execution: out='HI', return 'HI'+'!' = 'HI!'.",
            "trace_with_intervention": [
                {"line": 2, "locals": {"s": "hi"}},
                {"line": 3, "locals": {"s": "hi", "out": "WORLD"}},
            ],
            "answer_with_intervention": "'WORLD!'",
            "explanation_with_intervention": (
                "Intervention: out is set to 'WORLD'. Follow the trace -> 'WORLD!'."
            ),
        },
    ]

    def _get_prompt_structure(self):
        instruction = (
            "You are a Python execution simulator.\n"
            "You are given a Python function and a call. You must produce:\n"
            "  1) A line-by-line execution trace listing the locals at each step.\n"
            "  2) The final return value, as a Python repr().\n\n"
            "Trace format (use EXACTLY this format):\n"
            "Trace:\n"
            "line <N>:\n"
            "  <var> = <python repr of value>\n"
            "  ... (one indented line per local at this step) ...\n"
            "line <N>:\n"
            "  ...\n"
            "(repeat for every executed line)\n\n"
            "After the trace, output a single line:\n"
            "  Final Answer: <python repr>\n\n"
        )

        if not self.tool_mode:
            instruction += (
                "Important output rule:\n"
                "Your response must contain ONLY two fields and no other text:\n"
                "1) Trace: (the line-by-line trace)\n"
                "2) Final Answer: <python repr>\n\n"
            )

        tool_call_instruction = ""
        if self.tool_mode == "simple":
            tool_call_instruction = (
                "Tool usage (REQUIRED):\n"
                "- After the trace, you MUST call the tool to predict the answer.\n"
                "- Tool name: simulate_output\n"
                "- ARGS must be valid JSON. Escape newlines in the trace string as \\n.\n\n"
                "Important output rule:\n"
                "Your response must contain ONLY:\n"
                "1) Trace: (the line-by-line trace)\n"
                "2) Final tool call:\n"
                "TOOL: simulate_output\n"
                'ARGS: {"trace": "RAW TRACE STRING"}\n\n'
            )
        elif self.tool_mode == "structured":
            tool_call_instruction = (
                "Tool usage (REQUIRED):\n"
                "- After the trace, you MUST call the tool to predict the answer.\n"
                "- Tool name: simulate_output\n"
                "- ARGS is a JSON object whose 'trace' is a list of step objects:\n"
                '    [{"line": <int>, "locals": {"<var>": "<repr>"}}, ...]\n\n'
                "Important output rule:\n"
                "Your response must contain ONLY:\n"
                "1) Trace: (the line-by-line trace)\n"
                "2) Final tool call:\n"
                "TOOL: simulate_output\n"
                'ARGS: {"trace": [{"line": 2, "locals": {"x": "3"}}, ...]}\n\n'
            )

        if self.tool_mode:
            tool_call_instruction += "Tool specification:\n" + self.tool.spec_json() + "\n\n"

        # Few-shot block
        few_shot_text = "FEW-SHOT EXAMPLES:\n\n"
        for i, ex in enumerate(self.FEW_SHOT_EXAMPLES, start=1):
            if self.prompting_regime in ["detailed", "max_detailed"]:
                trace_used = ex["trace_with_intervention"]
                answer_used = ex["answer_with_intervention"]
                example_type = " (With intervention)"
            else:
                trace_used = ex["trace"]
                answer_used = ex["answer"]
                example_type = ""

            block = (
                f"Example #{i}{example_type}\n"
                "Code:\n"
                f"```python\n{ex['code']}```\n"
                f"Call: f({ex['input_str']})\n\n"
                "Trace:\n"
                f"{trace_to_text(trace_used)}\n"
            )
            if not self.tool_mode:
                block += f"Final Answer: {answer_used}\n\n"
            else:
                block += self._get_tool_call_string_for_example(trace_used)

            if self.prompting_regime in ["detailed", "max_detailed"]:
                block += (
                    "Intervention explanation:\n"
                    + ex["explanation_with_intervention"] + "\n\n"
                )
            few_shot_text += block

        return instruction, tool_call_instruction, few_shot_text

    def _get_tool_call_string_for_example(self, trace) -> str:
        if self.tool_mode == "simple":
            trace_str = trace_to_text(trace).replace("\n", "\\n")
            return (
                "Final tool call:\n"
                "TOOL: simulate_output\n"
                f'ARGS: {{"trace": "{trace_str}"}}\n\n'
            )
        if self.tool_mode == "structured":
            import json as _json
            payload = [
                {
                    "line": s["line"],
                    "locals": {k: repr(v) for k, v in s["locals"].items()},
                }
                for s in trace
            ]
            return (
                "Final tool call:\n"
                "TOOL: simulate_output\n"
                f"ARGS: {_json.dumps({'trace': payload}, ensure_ascii=False)}\n\n"
            )
        return ""

    def make_prompt(self, sample: dict, include_gold_structure: bool = False) -> str:
        """
        include_gold_structure=False:
            Full prompt -- model produces the trace + final answer.
        include_gold_structure=True:
            Intervention prompt -- mediator trace is supplied as assistant prefix;
            the model appends only "Final Answer: ..." (or ARGS in tool mode).
        """
        current_sample = (
            "Now follow the same structure for the given code and call.\n\n"
            "Code:\n"
            f"```python\n{sample['code']}```\n"
            f"Call: f({sample['input_str']})\n"
        )

        gold_structure = None
        if include_gold_structure:
            trace_str = trace_to_text(sample["mediator_trace"])
            gold_structure = "Trace:\n" + trace_str + "\n"
            if self.tool_mode:
                gold_structure += "Final tool call:\n"
            else:
                gold_structure += "Final Answer: "

        return self.prompt.make_prompt(
            current_sample=current_sample,
            include_gold_structure=include_gold_structure,
            gold_structure=gold_structure,
        )
