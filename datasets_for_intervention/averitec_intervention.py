import re
from copy import deepcopy
from datasets_for_intervention.prompt import Prompt


class AVeriTeCIntervention:
    """
    Experiment logic for the AVeriTeC dataset.

    Mediator M: dict {question: bool}  (True=Yes, False=No)
    Final answer Y: "Supported" | "Refuted"

    Intervention routing:
      correct   -> Local Edits  (invert one answer at a time)
      incorrect -> Correction   (mediator_rubric := gold_rubric)
      error     -> empty lists, no interventions

    Evaluation note (enforced in Evaluator):
      If gold_target == "Supported":   flipping any answer deterministically -> Refuted.
      If gold_target == "Refuted" (len==1): flipping the single answer -> Supported.
      Filter applied in evaluate(): target_before_intervention == "Supported"
                                    OR len(mediator_rubric) == 1.
    """

    def __init__(self, dataset, llm_model, tool, processor, prompting_regime: str = "standard", tool_mode: str = "none", include_explanations: bool = True):
        """
        Args:
            dataset:              AVeriTeCDataset
            llm_model:            LLMModel (used for apply_chat_template)
            tool:                 AVeriTeCTool
            processor:            AVeriTeCStructureProcessor
            prompting_regime:     "standard" | "detailed" | "max_detailed"
            tool_mode:            "none" | "simple" | "structured"
            include_explanations: if False, explanations are stripped from every
                                  prompt (few-shot and current sample).
                                  Pass False together with AVeriTeCDataset(include_explanations=False)
                                  for the no-explanations ablation experiment.
        """
        self.dataset = dataset
        self.llm_model = llm_model
        self.tool = tool
        self.processor = processor
        self.include_explanations = include_explanations

        assert prompting_regime in ["standard", "detailed", "max_detailed"], (
            "prompting_regime must be one of: standard, detailed, max_detailed"
        )
        assert tool_mode in ["none", "simple", "structured"], (
            "tool_mode must be one of: none, simple, structured"
        )

        self.prompting_regime = prompting_regime
        self.tool_mode = tool_mode if tool_mode != "none" else None

        instruction, tool_call_instruction, few_shot_text = self._get_prompt_structure()

        self.prompt = Prompt(
            prompting_regime=self.prompting_regime,
            use_tool_call=self.tool_mode is not None,
            tool_call_instruction=tool_call_instruction,
            instruction=instruction,
            few_shot=few_shot_text,
            llm_model=self.llm_model,
        )

    def clean_llm_output(self, text):
        tokens_to_remove = [
            "<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|eot_id|>",
            "<|end_of_text|>", "<|pad|>",
            "<end_of_turn>", "<|vision_pad|>", "<|eom_id|>", "<|finetune_right_pad_id|>",
            "</s>",
            "\u00ad", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
        ]
        for token in tokens_to_remove:
            text = text.replace(token, '')
        return text.strip()

    def infer_completion(self, completion: str, sample: dict = None, short_completion: bool = False):
        """
        Parse completion and return (verdict, tool_rubric).

        tool_mode:
          - structured: extract_tool_args -> list[bool] -> boollist_to_checklist -> dict
          - simple:     extract_tool_args -> dict (already parsed)
          - none:       extract_final_answer -> str; tool_rubric = {}

        After this call, tool_rubric is always dict (or None on parse error).
        """
        if self.tool_mode:
            tool_args = self.processor.extract_tool_args(completion, short_completion)
            if self.tool_mode == "structured":
                tool_rubric = self.processor.boollist_to_checklist(sample, tool_args)
            else:
                # simple: tool_args is already a dict
                tool_rubric = tool_args
            verdict = self.tool.calculate_score({"rubric": tool_rubric}, sample)
            return verdict, tool_rubric

        # Non-tool: parse "Final Verdict: X"
        verdict = self.processor.extract_final_answer(completion, short_completion)
        return verdict, {}

    def classify_generation(self, completion: str, mediator_rubric, gold_rubric: dict) -> str:
        """
        Return "correct" | "incorrect" | "error".

          error     -- M' could not be parsed OR garbage in XM part
          correct   -- M' parsed, no garbage, M' == M_gold
          incorrect -- M' parsed, no garbage, M' != M_gold
        """
        if mediator_rubric is None:
            return "error"
        if self.processor.check_generation_format_mistakes(completion):
            return "error"
        match = self.processor.compare_structures(mediator_rubric, gold_rubric)
        return "correct" if match == 1 else "incorrect"


    def make_intervention(self, sample: dict, generated_output: dict) -> dict:
        """
        1. Clean the completion.
        2. Parse M' (mediator_rubric).
        3. Classify: correct / incorrect / error.
        4. Build interventions:
             correct   -> Local Edits (invert each answer one at a time)
             incorrect -> Correction  (mediator_rubric := gold_rubric)
             error     -> empty lists, no interventions
        """
        completion = self.clean_llm_output(generated_output["completion"])
        sample["raw_generation"] = completion

        # Parse M'
        mediator_rubric = self.processor.extract_mediator(completion)

        # Classify
        generation_status = self.classify_generation(
            completion, mediator_rubric, sample["gold_rubric"]
        )
        sample["generation_status"] = generation_status
        sample["mediator_rubric"] = mediator_rubric if mediator_rubric is not None else {}

        if generation_status == "error":
            sample["target_before_intervention"] = None
            sample["tool_rubric"] = None
            sample["structure_intervention"] = {"Local Edits": [], "Correction": []}
            return sample

        # Verdict before interventions (correct / incorrect only)
        target, tool_rubric = self.infer_completion(
            completion, sample, short_completion=False
        )

        if target is None:
            sample["generation_status"] = 'error'
            sample["target_before_intervention"] = None
            sample["tool_rubric"] = tool_rubric
            sample["structure_intervention"] = {"Local Edits": [], "Correction": []}
            return sample

        sample["target_before_intervention"] = target
        sample["tool_rubric"] = tool_rubric

        # Build interventions
        sample["structure_intervention"] = self.make_structure_intervention(sample)
        return sample

    def make_structure_intervention(self, sample: dict) -> dict:
        """
        correct   -> Local Edits: deepcopy with exactly one answer inverted.
                     expected_target_after_intervention computed for each edit.
        incorrect -> Correction: deepcopy with mediator_rubric := gold_rubric.
        error     -> {"Local Edits": [], "Correction": []}.

        expected_target_after_intervention is computed via tool.calculate_score
        without branching on tool_mode (mediator_rubric is a canonical dict at this point).
        """
        generation_status = sample.get("generation_status")

        if generation_status == "correct":
            local_edits = []
            for question, answer in sample["mediator_rubric"].items():
                local = deepcopy(sample)
                # Invert exactly one answer
                local["mediator_rubric"] = deepcopy(sample["mediator_rubric"])
                local["mediator_rubric"][question] = not answer
                # Expected verdict after the intervention (via Tool, not tool_mode)
                local["expected_target_after_intervention"] = self.tool.calculate_score(
                    {"rubric": local["mediator_rubric"]}, local
                )
                local_edits.append(local)
            return {"Local Edits": local_edits, "Correction": []}

        if generation_status == "incorrect":
            corr = deepcopy(sample)
            corr["mediator_rubric"] = deepcopy(corr["gold_rubric"])
            return {"Local Edits": [], "Correction": [corr]}

        # error or unknown status
        return {"Local Edits": [], "Correction": []}

    def interventions_to_prompt(self, sample: dict) -> list:
        """
        Order: Local Edits -> Correction.
        collect_intervention_completion must use the same order.
        """
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
        """
        Distribute completions across intervention samples in order:
        Local Edits -> Correction.

        Writes target_after_intervention (+ tool_rubric_after_intervention if tool_mode).
        """
        completions = [self.clean_llm_output(g["completion"]) for g in generated_output]
        interv = sample["structure_intervention"]
        idx = 0

        for i in range(len(interv.get("Local Edits", []))):
            verdict, tool_rubric = self.infer_completion(
                completions[idx], interv["Local Edits"][i], short_completion=True
            )
            interv["Local Edits"][i]["raw_generation"] = completions[idx]
            interv["Local Edits"][i]["target_after_intervention"] = verdict
            if self.tool_mode:
                interv["Local Edits"][i]["tool_rubric_after_intervention"] = tool_rubric
            idx += 1

        for i in range(len(interv.get("Correction", []))):
            verdict, tool_rubric = self.infer_completion(
                completions[idx], interv["Correction"][i], short_completion=True
            )
            interv["Correction"][i]["raw_generation"] = completions[idx]
            interv["Correction"][i]["target_after_intervention"] = verdict
            if self.tool_mode:
                interv["Correction"][i]["tool_rubric_after_intervention"] = tool_rubric
            idx += 1

        return sample

    # ------------------------------------------------------------------
    # Few-shot examples in the new format (Checklist: / True/False)
    # ------------------------------------------------------------------

    # Same claims as in the original dataset, but reformatted to use
    # "Checklist:" and True/False instead of "Structured Reasoning Block:" and Yes/No.
    FEW_SHOT = {
        # ----------------------------------------------------------------
        # Example 1: Supported, 2 questions
        # Both answers False (No) -- all evidence supports the claim -> Supported
        # Only shown without intervention (the base case for a Supported claim).
        # ----------------------------------------------------------------
        "Example 1": {
            "claim": (
                "Hunter Biden had no experience in Ukraine or in the energy sector "
                "when he joined the board of Burisma."
            ),
            "explanations": {
                "Did Hunter Biden have any experience in the energy sector in 2014?":
                    "Hunter Biden's previous career history does not include work for energy companies.",
                "Did Hunter Biden have any experience in Ukraine in 2014?":
                    "Hunter Biden's previous career history does not include working with Ukrainian companies.",
            },
            "checklist": {
                "Did Hunter Biden have any experience in the energy sector in 2014?": False,
                "Did Hunter Biden have any experience in Ukraine in 2014?": False,
            },
            "gold_target": "Supported",
            "explanation": "Here no intervention.",
            "explanation_with_intervention": "There is no intervention in this example.",
        },
        # ----------------------------------------------------------------
        # Example 2: Refuted, 1 question
        # Answer False (No) -> evidence contradicts the claim -> Refuted.
        # Shown twice in detailed regime: once without and once with intervention.
        # Intervention: flip False -> True (Yes) => verdict must change to Supported.
        # ----------------------------------------------------------------
        "Example 2": {
            "claim": "President Trump is the most pro-gay president in American history.",
            "explanations": {
                "Did Trump make pro-gay laws when in office?":
                    "He made laws such as: 1. Appointing Anti-Equality Judges "
                    "2. Stripping protections from LGBTQ students, parents and families "
                    "3. Defending Anti-Gay Discrimination.",
            },
            "checklist": {
                "Did Trump make pro-gay laws when in office?": False,
            },
            "gold_target": "Refuted",
            "explanation": "Here no intervention.",
            # Intervention: answer flipped False -> True
            "checklist_with_intervention": {
                "Did Trump make pro-gay laws when in office?": True,
            },
            "gold_target_with_intervention": "Supported",
            "explanation_with_intervention": (
                "Here we flip the answer to True (Yes) and the final verdict must become Supported."
            ),
        },
        # ----------------------------------------------------------------
        # Example 3: Refuted, 3 questions
        # All answers False (No) -> none of the sources confirm the claim -> Refuted.
        # Shown only with intervention in detailed regime:
        # flip all answers to True (Yes) => verdict must change to Supported.
        # ----------------------------------------------------------------
        "Example 3": {
            "claim": (
                "Beijing government announced that Chinese people should not travel "
                "to the United States or buy American-made products."
            ),
            "explanations": {
                "Did China's Ministry of Foreign Affairs announce that Chinese people should not "
                "travel to the United States or buy American-made products in its daily press "
                "briefing on August 13, 2020?":
                    "Transcript of August 13 daily press briefing does not include a request for "
                    "Chinese people to avoid American products or avoid travelling to the US.",
                "Did the weekly policy briefing from China's State Council on August 13, 2020 "
                "include a mention of the call for Chinese people to not travel to the United "
                "States or buy American-made products?":
                    "China's State Council weekly policy briefing pages for August 13, 2020 "
                    "do not mention the US.",
                "Did the Chinese Ministry of Foreign Affairs announce that Chinese people should "
                "not travel to the United States or buy American-made products on its Twitter "
                "account on or after August 13, 2020?":
                    "A keywords search set between August 13 and August 18 2020 found no claim "
                    "on the Ministry's Twitter account.",
            },
            "checklist": {
                "Did China's Ministry of Foreign Affairs announce that Chinese people should not "
                "travel to the United States or buy American-made products in its daily press "
                "briefing on August 13, 2020?": False,
                "Did the weekly policy briefing from China's State Council on August 13, 2020 "
                "include a mention of the call for Chinese people to not travel to the United "
                "States or buy American-made products?": False,
                "Did the Chinese Ministry of Foreign Affairs announce that Chinese people should "
                "not travel to the United States or buy American-made products on its Twitter "
                "account on or after August 13, 2020?": False,
            },
            "gold_target": "Refuted",
            "explanation": "Here no intervention.",
            # Intervention: all answers flipped False -> True
            "checklist_with_intervention": {
                "Did China's Ministry of Foreign Affairs announce that Chinese people should not "
                "travel to the United States or buy American-made products in its daily press "
                "briefing on August 13, 2020?": True,
                "Did the weekly policy briefing from China's State Council on August 13, 2020 "
                "include a mention of the call for Chinese people to not travel to the United "
                "States or buy American-made products?": True,
                "Did the Chinese Ministry of Foreign Affairs announce that Chinese people should "
                "not travel to the United States or buy American-made products on its Twitter "
                "account on or after August 13, 2020?": True,
            },
            "gold_target_with_intervention": "Supported",
            "explanation_with_intervention": (
                "Here we flip all 3 answers to True (Yes) and the final verdict must become Supported."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Helper methods for prompt string formatting
    # ------------------------------------------------------------------

    def _checklist_dict_to_string(self, checklist: dict) -> str:
        """
        Format dict {question: bool} into a checklist block:
          - Q: <question> (True/False): <True|False>
        """
        return "".join(
            f"- Q: {q} (True/False): {v}\n"
            for q, v in checklist.items()
        )

    def _get_tool_call_string(self, checklist: dict) -> str:
        """Build a tool call block for a few-shot example."""
        if self.tool_mode == "simple":
            checklist_str = self._checklist_dict_to_string(checklist).replace("\n", "\\n")
            return (
                "Final tool call:\n"
                "TOOL: predict_verdict\n"
                f'ARGS: {{"rubric": "{checklist_str}"}}\n\n'
            )
        if self.tool_mode == "structured":
            bool_list = list(checklist.values())
            return (
                "Final tool call:\n"
                "TOOL: predict_verdict\n"
                f'ARGS: {{"rubric": {bool_list}}}\n\n'
            )
        return ""

    def _get_prompt_structure(self):
        """Build and return (instruction, tool_call_instruction, few_shot_text)."""

        instruction = (
            "You are an expert fact-checking system. "
            "Your task is to evaluate a claim by constructing a structured checklist "
            "from the provided questions and explanations, then give a final verdict.\n\n"
            "Task explanation:\n"
            "- You are given a claim and a set of supporting questions with explanations.\n"
            "- You must fill the checklist (True/False) based on the evidence in the explanations.\n"
            "  True = Yes (the answer to the question is affirmative), "
            "False = No (the answer is negative).\n"
            "- Keep the question text EXACTLY as provided (same order and wording). "
            "Only replace the trailing <True/False> with True or False.\n"
            "- The final verdict must be Supported or Refuted based on the filled checklist.\n\n"
        )

        if not self.tool_mode:
            instruction += (
                "Important output rule:\n"
                "Your final response must contain ONLY two fields and no other text:\n"
                "1) Checklist: (the filled checklist, line-for-line in the same format)\n"
                "2) Final Verdict: <Supported|Refuted>\n\n"
            )

        # Tool call instruction
        tool_call_instruction = ""
        if self.tool_mode == "simple":
            tool_call_instruction = (
                "Tool usage (REQUIRED):\n"
                "- After you fill the checklist, you MUST call the tool to predict the verdict.\n"
                "- Tool name: predict_verdict\n"
                "- IMPORTANT: The tool input must be the RAW filled checklist in EXACTLY the same format.\n"
                "- Provide ARGS as valid JSON. Escape newlines in the checklist string as \\n.\n\n"
                "Important output rule:\n"
                "Your final response must contain ONLY the following fields and no other text:\n"
                "1) Checklist: (the filled checklist, line-for-line in the same format)\n"
                "2) Final tool call:\n"
                "TOOL: predict_verdict\n"
                'ARGS: {"rubric": "FILLED AVERITEC CHECKLIST"}\n\n'
            )
        elif self.tool_mode == "structured":
            tool_call_instruction = (
                "Tool usage (REQUIRED):\n"
                "- After you fill the checklist, you MUST call the tool to predict the verdict.\n"
                "- Tool name: predict_verdict\n"
                "- IMPORTANT: tool input is a boolean list aligned with your checklist lines.\n"
                "  True = Yes, False = No.\n"
                "- Do NOT compute the verdict yourself.\n\n"
                "Important output rule:\n"
                "Your final response must contain ONLY the following fields and no other text:\n"
                "1) Checklist: (the filled checklist, line-for-line in the same format)\n"
                "2) Final tool call:\n"
                "TOOL: predict_verdict\n"
                'ARGS: {"rubric": [True, False, ...]}\n\n'
            )

        if self.tool_mode:
            tool_call_instruction += "Tool specification:\n" + self.tool.spec_json() + "\n\n"

        # Few-shot block
        few_shot_text = "FEW-SHOT EXAMPLES:\n\n"

        for ex_name in ["Example 1", "Example 2", "Example 3"]:
            ex = self.FEW_SHOT[ex_name]
            ex_num = ex_name.split()[-1]
        
            example_type = ""
            result = ""
            if self.prompting_regime in ["detailed", "max_detailed"]:
                checklist = ex.get("checklist_with_intervention", ex["checklist"])
                result = ex.get("gold_target_with_intervention", ex["gold_target"])
                if 'checklist_with_intervention' in ex:
                    example_type = " (With intervention)"
                else:
                    example_type = " (No intervention)"
                explanation = ex["explanation"]
            else:
                result = ex["gold_target"]
                checklist = ex["checklist"]
                explanation = ""

            # Checklist block
            checklist_str = self._checklist_dict_to_string(checklist)

            if self.include_explanations:
                explanations_str = "".join(
                    f"- Q: {q} E: {e}\n"
                    for q, e in ex["explanations"].items()
                )
                explanations_block = f"Explanations:\n{explanations_str}\n"
            else:
                explanations_block = ""

            ex_block = (
                f"Example #{ex_num}{example_type}\n"
                "Claim:\n"
                f"{ex['claim']}\n"
                f"{explanations_block}"
                "Checklist:\n"
                f"{checklist_str}\n"
            )

            if not self.tool_mode:
                # Non-tool: append "Final Verdict: X"
                ex_block += f"Final Verdict: {result}\n\n"
            else:
                # Tool mode: append tool call block
                ex_block += self._get_tool_call_string(checklist)

            if self.prompting_regime in ["detailed", "max_detailed"]:
                ex_block += "Intervention explanation:\n" + ex["explanation_with_intervention"] + "\n\n"

            few_shot_text += ex_block

        return instruction, tool_call_instruction, few_shot_text

    # ------------------------------------------------------------------
    # make_prompt -- build the full prompt for a sample
    # ------------------------------------------------------------------

    def make_prompt(
        self,
        averitec_sample: dict,
        include_gold_structure: bool = False,
    ) -> str:
        """
        Build the prompt for the given sample.

        include_gold_structure=False:
            Full prompt -- the model fills the checklist and predicts the verdict.

        include_gold_structure=True:
            Intervention prompt -- mediator_rubric is provided as an assistant prefix;
            the model appends only "Final Verdict: X" (or ARGS in tool mode).
        """
        # Checklist template with <True/False> placeholders
        checklist_template = "".join(
            f"- Q: {q} (True/False): <True/False>\n"
            for q in averitec_sample["gold_rubric"]
        )

        # Explanations block — omitted in the no-explanations ablation.
        # sample["explanations"] is already {} when AVeriTeCDataset(include_explanations=False),
        # but we also check self.include_explanations for safety.
        if self.include_explanations and averitec_sample.get("explanations"):
            explanations_str = "".join(
                f"- Q: {q} E: {e}\n"
                for q, e in averitec_sample["explanations"].items()
            )
            explanations_block = f"Explanations:\n{explanations_str}\n"
        else:
            explanations_block = ""

        current_sample = (
            "Now follow the same structure for the given claim.\n\n"
            "Claim:\n"
            f"{averitec_sample['claim']}\n\n"
            f"{explanations_block}"
            "Checklist:\n"
            f"{checklist_template}"
        )

        gold_structure = None
        if include_gold_structure:
            # Filled mediator (may be gold or locally modified)
            filled_checklist = self._checklist_dict_to_string(
                averitec_sample["mediator_rubric"]
            )
            gold_structure = "Checklist:\n" + filled_checklist
            # Tail marker: the model appends only the final answer
            if self.tool_mode:
                gold_structure += "Final tool call:\n"
            else:
                gold_structure += "Final Verdict: "

        return self.prompt.make_prompt(
            current_sample=current_sample,
            include_gold_structure=include_gold_structure,
            gold_structure=gold_structure,
        )