"""
TabFactIntervention: experiment logic for the TabFact dataset.

Follows the unified architecture (§8) with these TabFact-specific adaptations:
  - The mediator (M) is a DSL query **string** (not a dict).
  - gold_query / mediator_query are strings.
  - target_before_intervention / target_after_intervention are bool.
  - Local Edits come from pre-computed dataset.sample_id2local_edits (DSL expressions
    that are small perturbations of the gold query); they are not generated on the fly.
  - expected_target_after_intervention is computed by executing the local-edit query
    on the table via TabFactTool.
  - In tool_mode="simple": tool_query = query string extracted from the tool-call block.
  - In non-tool mode: tool_query = {} (empty dict, per architecture).
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from datasets_for_intervention.prompt import Prompt


class TabFactIntervention:
    """
    Manages the TabFact intervention pipeline.

    Args:
        dataset:           TabFactDataset instance.
        llm_model:         LLM model wrapper (for prompt formatting and generation).
        tool:              TabFactTool instance (used for expected_target_after_intervention
                           and for score computation in tool_mode).
        processor:         TabFactStructureProcessor instance.
        prompting_regime:  One of "standard", "detailed", "max_detailed".
        tool_mode:         One of "none", "simple".
    """

    def __init__(
        self,
        dataset,
        llm_model,
        tool,
        processor,
        prompting_regime: str = "standard",
        tool_mode: str = "none",
    ) -> None:
        self.dataset = dataset
        self.llm_model = llm_model
        self.tool = tool
        self.processor = processor

        assert prompting_regime in ("standard", "detailed", "max_detailed"), (
            f"prompting_regime must be one of: standard, detailed, max_detailed. "
            f"Got: {prompting_regime}"
        )
        if tool_mode != 'none':
            tool_mode = 'simple'
    
        assert tool_mode in ("none", "simple"), (
            f"tool_mode for TabFact must be 'none' or 'simple'. Got: {tool_mode}"
        )

        self.prompting_regime = prompting_regime
        # Store None when tool not used (mirrors RiceChem convention)
        self.tool_mode: Optional[str] = tool_mode if tool_mode != "none" else None

        instruction, tool_call_instruction, few_shot_text = self._get_prompt_structure()

        self.prompt = Prompt(
            prompting_regime=self.prompting_regime,
            use_tool_call=self.tool_mode is not None,
            tool_call_instruction=tool_call_instruction,
            instruction=instruction,
            few_shot=few_shot_text,
            llm_model=self.llm_model,
        )

    def clean_llm_output(self, text: str) -> str:
        """Remove model-specific special tokens and invisible Unicode characters."""
        tokens_to_remove = [
            "<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|eot_id|>",
            "<|end_of_text|>", "<|pad|>",
            "<end_of_turn>", "<|vision_pad|>", "<|eom_id|>", "<|finetune_right_pad_id|>",
            "</s>",
            "\u00ad", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
        ]
        for tok in tokens_to_remove:
            text = text.replace(tok, "")
        return text.strip()

    def infer_completion(
        self,
        completion: str,
        sample: Optional[Dict] = None,
        short_completion: bool = False,
    ) -> Tuple[Optional[bool], Any]:
        """
        Parse a model completion into a (target, tool_query) pair.

        In tool_mode="simple":
            - Extract the DSL query from the ARGS block.
            - Execute via TabFactTool to get the boolean verdict.
            - tool_query = extracted query string (or None on failure).

        In non-tool mode:
            - Parse verdict from "Execution Result: True/False".
            - tool_query = {} (architecture: non-tool returns empty dict).

        Args:
            completion:       Cleaned model completion text.
            sample:           Current sample dict (table access needed in tool_mode).
            short_completion: True when the model only appended a suffix (tail-only mode).

        Returns:
            (target: bool|None, tool_query: str|{}|None)
        """
        if self.tool_mode == "simple":
            args = self.processor.extract_tool_args(completion, short_completion)
            tool_query = args  # str | None
            if tool_query is not None and sample is not None:
                target = self.tool.calculate_score(args, sample)
            else:
                target = None
            return target, tool_query

        # Non-tool mode
        target = self.processor.extract_final_answer(completion, short_completion)
        return target, {}

    def classify_generation(
        self,
        completion: str,
        mediator_query: Optional[str],
        gold_query: str,
    ) -> str:
        """
        Classify the primary generation: "correct" | "incorrect" | "error".

        error     — mediator_query is None OR format mistake detected.
        correct   — mediator_query == gold_query (normalised string match).
        incorrect — mediator_query != gold_query.
        """
        if mediator_query is None:
            return "error"
        if self.processor.check_generation_format_mistakes(completion):
            return "error"
        match = self.processor.compare_structures(mediator_query, gold_query)
        return "correct" if match == 1 else "incorrect"

    def make_intervention(self, sample: Dict, generated_output: Dict) -> Dict:
        """
        Process the primary model completion and build intervention samples.

        Steps:
          1. Clean the completion.
          2. Extract the predicted mediator (DSL query string).
          3. Classify: correct / incorrect / error.
          4. For error: set None fields, empty intervention lists, return early.
          5. For correct / incorrect: compute target_before_intervention
             and build structure_intervention.

        Args:
            sample:           Sample dict (modified in place).
            generated_output: Dict with key 'completion' (raw model output).
        """
        completion = self.clean_llm_output(generated_output["completion"])
        sample["raw_generation"] = completion

        # Parse M' (mediator)
        mediator_query = self.processor.extract_mediator(completion)

        # Classify
        generation_status = self.classify_generation(
            completion, mediator_query, sample["gold_query"]
        )
        sample["generation_status"] = generation_status
        # Store parsed mediator (empty string for error, per architecture)
        sample["mediator_query"] = mediator_query if mediator_query is not None else ""

        if generation_status == "error":
            # do not attempt to parse target from garbage completion
            sample["target_before_intervention"] = None
            sample["tool_query"] = None
            sample["structure_intervention"] = {"Local Edits": [], "Correction": []}
            return sample

        # Score (target) before interventions — only for correct / incorrect
        target, tool_query = self.infer_completion(
            completion, sample, short_completion=False
        )

        if target is None:
            sample["generation_status"] = "error"
            sample["target_before_intervention"] = None
            sample["tool_query"] = tool_query
            sample["structure_intervention"] = {"Local Edits": [], "Correction": []}
            return sample
        
        sample["target_before_intervention"] = target
        sample["tool_query"] = tool_query

        # Build interventions
        sample["structure_intervention"] = self.make_structure_intervention(sample)
        return sample

    def make_structure_intervention(self, sample: Dict) -> Dict:
        """
        Build intervention deepcopies based on generation_status.

        correct   -> Local Edits: one deepcopy per verified local-edit entry.
                     expected_target_after_intervention is read directly from the
                     pre-computed dataset entry (no re-execution needed).
                     Correction = [].

        incorrect -> Correction: one deepcopy with mediator_query := gold_query.
                     Local Edits = [].

        error     -> both lists empty.
        """
        status = sample.get("generation_status")

        if status == "correct":
            # dataset.get_local_edits returns list of {"query": str, "expected_target": bool}
            # Each entry was already verified to execute != gold_target at load time.
            local_edits = []
            for edit in self.dataset.get_local_edits(sample):
                local = deepcopy(sample)
                local["mediator_query"] = edit["query"]
                local["expected_target_after_intervention"] = edit["expected_target"]
                local_edits.append(local)
            return {"Local Edits": local_edits, "Correction": []}

        if status == "incorrect":
            corr = deepcopy(sample)
            corr["mediator_query"] = deepcopy(corr["gold_query"])
            return {"Local Edits": [], "Correction": [corr]}

        return {"Local Edits": [], "Correction": []}

    def interventions_to_prompt(self, sample: Dict) -> List[str]:
        """
        Build prompt strings for all intervention samples.

        Order: Local Edits → Correction  (same as collect_intervention_completion).
        """
        interv = sample["structure_intervention"]
        prompts: List[str] = []
        prompts += [
            self.make_prompt(edit, include_gold_structure=True)
            for edit in interv.get("Local Edits", [])
        ]
        prompts += [
            self.make_prompt(corr, include_gold_structure=True)
            for corr in interv.get("Correction", [])
        ]
        return prompts

    def collect_intervention_completion(
        self, sample: Dict, generated_output: List[Dict]
    ) -> Dict:
        """
        Assign model outputs to their intervention samples (same order as interventions_to_prompt).

        Records for each intervention sample:
            raw_generation                    (str)
            target_after_intervention         (bool | None)
            tool_query_after_intervention    (str | None) — only when tool_mode is set
        """
        completions = [
            self.clean_llm_output(g["completion"]) for g in generated_output
        ]
        interv = sample["structure_intervention"]
        idx = 0

        # --- Local Edits ---
        for i in range(len(interv.get("Local Edits", []))):
            target, tool_query = self.infer_completion(
                completions[idx], interv["Local Edits"][i], short_completion=True
            )
            interv["Local Edits"][i]["raw_generation"] = completions[idx]
            interv["Local Edits"][i]["target_after_intervention"] = target
            if self.tool_mode:
                interv["Local Edits"][i]["tool_query_after_intervention"] = tool_query
            idx += 1

        # --- Correction ---
        for i in range(len(interv.get("Correction", []))):
            target, tool_query = self.infer_completion(
                completions[idx], interv["Correction"][i], short_completion=True
            )
            interv["Correction"][i]["raw_generation"] = completions[idx]
            interv["Correction"][i]["target_after_intervention"] = target
            if self.tool_mode:
                interv["Correction"][i]["tool_query_after_intervention"] = tool_query
            idx += 1

        return sample

    def make_prompt(
        self,
        tabfact_sample: Dict,
        include_gold_structure: bool = False,
    ) -> str:
        """
        Build the formatted prompt for a TabFact sample.

        Args:
            tabfact_sample:       Sample with 'table_html_csv', 'statement',
                                  and 'mediator_query' (when include_gold_structure).
            include_gold_structure: If True, mediator is injected as an assistant prefix
                                    and the model only outputs the tail.
        """
        table = tabfact_sample["table_html_csv"]
        statement = tabfact_sample["statement"]

        current_sample = (
            "Now follow the same structure for the given input.\n\n"
            "Table:\n"
            f"{table}\n\n"
            "Claim:\n"
            f"{statement}\n\n"
            "Verifier Query: <YOUR QUERY>\n"
        )

        gold_structure: Optional[str] = None
        if include_gold_structure:
            mediator = tabfact_sample.get("mediator_query", "")
            if self.tool_mode == "simple":
                # The model only needs to append the ARGS JSON
                gold_structure = (
                    f"Verifier Query: {mediator}\n"
                    "Final tool call:\n"
                    "TOOL: check_query\n"
                    "ARGS: "
                )
            else:
                # Model appends "True" or "False" after this prefix
                gold_structure = (
                    f"Verifier Query: {mediator}\n"
                    "Execution Result:"
                )

        return self.prompt.make_prompt(
            current_sample=current_sample,
            include_gold_structure=include_gold_structure,
            gold_structure=gold_structure,
        )

    def _get_tool_call_block(self, query: str) -> str:
        """Format a tool-call block for a given DSL query (used in few-shot)."""
        return (
            "Final tool call:\n"
            "TOOL: check_query\n"
            f'ARGS: {{"query": "{query}"}}\n\n'
        )

    def _get_prompt_structure(self):
        """
        Build the (instruction, tool_call_instruction, few_shot_text) triple
        that is passed to the Prompt constructor.
        """
        instruction = (
            "You are an expert table fact-checking system. "
            "Your task is to evaluate a claim against tabular data by first constructing "
            "a structured reasoning block (a Verifier Query) using the provided Domain "
            "Specific Language (DSL), and then give the result of executing this verifier "
            "query as the final verdict.\n\n"
            "### TASK EXPLANATION\n"
            "1. **Construct a Verifier Query**: Analyse the claim and the table. "
            "Generate a precise logical DSL expression that encodes all steps needed "
            "to verify the claim.\n"
            "2. **Output the Execution Result**: Execute the Verifier Query. "
            "Output the boolean result (True or False). This is your final answer.\n\n"
            "### DOMAIN SPECIFIC LANGUAGE (DSL)\n"
            "- eq{A; B}: A == B\n"
            "- not_eq{A; B}: A != B\n"
            "- greater{A; B}: A > B\n"
            "- less{A; B}: A < B\n"
            "- and{A; B; ...}: logical AND\n"
            "- or{A; B; ...}: logical OR\n"
            "- not{A}: logical NOT\n"
            "- hop{Row; Field}: value of Field in Row\n"
            "- count{C}: number of rows in row-set C\n"
            "- only{C}: True iff C has exactly 1 row\n"
            "- filter_eq{C; Field; Value}: rows where Field == Value\n"
            "- filter_not_eq{C; Field; Value}: rows where Field != Value\n"
            "- filter_greater{C; Field; Value}: rows where Field > Value\n"
            "- filter_less{C; Field; Value}: rows where Field < Value\n"
            "- filter_greater_eq{C; Field; Value}: rows where Field >= Value\n"
            "- filter_less_eq{C; Field; Value}: rows where Field <= Value\n"
            "- argmax{C; Field}: row with max Field in C\n"
            "- argmin{C; Field}: row with min Field in C\n"
            "- sum{C; Field}: sum of Field across C\n"
            "- avg{C; Field}: average of Field across C\n"
            "- max{C; Field}: max of Field across C\n"
            "- min{C; Field}: min of Field across C\n"
            "- diff{A; B}: A - B\n"
            "- all_eq{C; Field; Value}: True iff all rows in C have Field == Value\n"
            "- all_greater{C; Field; Value}: True iff all rows in C have Field > Value\n"
            "- all_less{C; Field; Value}: True iff all rows in C have Field < Value\n"
            "- within{C; Field; Value}: True iff some row in C has Field == Value\n\n"
            "### SUFFIX RULE\n"
            "Every DSL expression MUST end with =True or =False.\n"
            "  expr=True: the expression is asserted to evaluate to True.\n"
            "  expr=False: the expression is asserted to evaluate to False.\n\n"
        )

        # Non-tool output format instruction (appended to instruction)
        if not self.tool_mode:
            instruction += (
                "### OUTPUT FORMAT\n"
                "Your response must contain ONLY two lines and no other text:\n"
                "Verifier Query: <DSL expression ending with =True or =False>\n"
                "Execution Result: <True or False>\n"
            )

        tool_call_instruction = ""
        if self.tool_mode == "simple":
            tool_call_instruction = (
                "### TOOL USAGE (REQUIRED)\n"
                "After writing the Verifier Query, you MUST call the check_query tool.\n"
                "- Provide ARGS as valid JSON; the 'query' value must match your "
                "Verifier Query exactly.\n\n"
                "### OUTPUT FORMAT (with tool)\n"
                "Your response must contain ONLY:\n"
                "Verifier Query: <DSL expression ending with =True or =False>\n"
                "Final tool call:\n"
                "TOOL: check_query\n"
                'ARGS: {"query": "<your DSL expression>"}\n\n'
                "Tool specification:\n" + self.tool.spec_json() + "\n"
            )

        # ---- few-shot examples ----
        
        few_shot_text = "### FEW-SHOT EXAMPLES\n\n"
        for ex in self.FEW_SHOT_EXAMPLES:
            example_type = ""
            if self.prompting_regime in ["detailed", "max_detailed"]:
                query = ex.get("query_with_intervention", ex["query"])
                if 'query_with_intervention' in ex:
                    example_type = " (With intervention)"
                else:
                    example_type = " (No intervention)"
                explanation = ex["explanation"]
            else:
                query = ex["query"]
                explanation = ""

            ex_block = (
                f"Example #{ex['num']}{example_type}\n"
                "Table:\n" + ex["table"] + "\n\n"
                f"Claim: {ex['claim']}\n"
                f"Verifier Query: {query}\n"
            )

            if self.tool_mode == "simple":
                ex_block += self._get_tool_call_block(query)
            else:
                result = (ex.get('result_with_intervention', ex['result'])
                          if 'query_with_intervention' in ex and query == ex['query_with_intervention']
                          else ex['result'])
                ex_block += f"Execution Result: {result}\n"

            if self.prompting_regime in ("detailed", "max_detailed"):
                ex_block += f"Explanation: {ex['explanation']}\n"

            ex_block += "\n"
            few_shot_text += ex_block

        return instruction, tool_call_instruction, few_shot_text

    # ------------------------------------------------------------------
    # Few-shot examples
    # ------------------------------------------------------------------
    #
    # Examples 1 and 2: no intervention — show the basic format.
    # Example 3: Local Edit intervention — one operator is flipped (greater→less),
    #            making the execution result change from True to False.
    # Example 4: Local Edit intervention — the comparison target is swapped
    #            (count of one team vs another), flipping True→False.
    #
    # The intervention block teaches the model that its Execution Result must
    # faithfully follow whatever Verifier Query is presented, not the original claim.

    FEW_SHOT_EXAMPLES = [
        # ── Example 1: basic True result, no intervention ────────────────────
        {
            "num": 1,
            "table": (
                "rank#athlete#nation#gold\n"
                "1#Usain Bolt#Jamaica#2\n"
                "2#Shawn Crawford#United States#1"
            ),
            "claim": "Usain Bolt won more gold medals than Shawn Crawford.",
            "query": (
                "greater{hop{filter_eq{all_rows; athlete; Usain Bolt}; gold}; "
                "hop{filter_eq{all_rows; athlete; Shawn Crawford}; gold}}=True"
            ),
            "result": "True",
            "explanation": (
                "There is no intervention here."
            ),
        },
        # ── Example 2: basic False result, no intervention ───────────────────
        {
            "num": 2,
            "table": (
                "player#team#goals\n"
                "Messi#PSG#30\n"
                "Ronaldo#Al-Nassr#25"
            ),
            "claim": "Ronaldo scored more goals than Messi.",
            "query": (
                "greater{hop{filter_eq{all_rows; player; Ronaldo}; goals}; "
                "hop{filter_eq{all_rows; player; Messi}; goals}}=True"
            ),
            "result": "False",
            "explanation": (
                "There is no intervention here."
            ),
        },
        # ── Example 3: Local Edit — operator flipped (greater → less) ────────
        {
            "num": 3,
            "table": (
                "event#year#location\n"
                "Olympics#2020#Tokyo\n"
                "World Cup#2022#Qatar\n"
                "Asian Games#2018#Jakarta"
            ),
            "claim": "The World Cup was held after the Olympics.",
            "query": (
                "greater{hop{filter_eq{all_rows; event; World Cup}; year}; "
                "hop{filter_eq{all_rows; event; Olympics}; year}}=True"
            ),
            "result": "True",
            # Local Edit: the comparison operator is flipped from greater to less.
            # The executed result must follow the new query, not the original claim.
            "query_with_intervention": (
                "less{hop{filter_eq{all_rows; event; World Cup}; year}; "
                "hop{filter_eq{all_rows; event; Olympics}; year}}=True"
            ),
            "result_with_intervention": "False",
            "explanation": (
                "The Verifier Query was locally edited: greater was changed to less. "
                "The new query checks less{2022; 2020}, which evaluates to False. "
                "The Execution Result must faithfully follow the edited query, "
                "not the original claim — so the result changes from True to False."
            ),
        },
        # ── Example 4: Local Edit — filter value changed (count flipped) ─────
        {
            "num": 4,
            "table": (
                "country#sport#medals\n"
                "USA#swimming#12\n"
                "USA#athletics#8\n"
                "China#swimming#5\n"
                "China#athletics#7"
            ),
            "claim": "The USA won more medals than China across all sports.",
            "query": (
                "greater{sum{filter_eq{all_rows; country; USA}; medals}; "
                "sum{filter_eq{all_rows; country; China}; medals}}=True"
            ),
            "result": "True",
            # Local Edit: the suffix is changed from =True to =False, asserting
            # the opposite polarity. The executed result must match the new suffix.
            "query_with_intervention": (
                "greater{sum{filter_eq{all_rows; country; USA}; medals}; "
                "sum{filter_eq{all_rows; country; China}; medals}}=False"
            ),
            "result_with_intervention": "False",
            "explanation": (
                "The Verifier Query was locally edited: the assertion suffix was changed "
                "from =True to =False. The inner expression greater{20; 12} still "
                "evaluates to True, but the query now asserts it should equal False — "
                "so the execution result is False. "
                "The Execution Result must follow the edited Verifier Query exactly."
            ),
        },
    ]