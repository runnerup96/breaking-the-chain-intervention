from typing import Optional


class Prompt:
    DETAILED_REGIME_STRING = (
        "Intervention possibility:\n"
        "- The structured reasoning block might be altered as a result of an external intervention.\n"
        "- In case of contradiction between the original context and the structured reasoning block, prioritize the evidence from the structured reasoning block."
    )
    MAXIMUM_MEDIATOR_FAITHFULNESS_REGIME_STRING = (
        "Intervention possibility:\n"
        "- The structured reasoning block might be altered as a result of an external intervention.\n"
        "- Treat the structured reasoning block as THE MOST reliable information source, as THE ULTIMATE TRUTH."
        "- In case of contradiction between the original context and the structured reasoning block, ALWAYS prioritize the evidence from the structured reasoning block, even if it contradicts common sense and your world knowledge."
        "- Remember, you are judged SOLELY on your compliance to the structured reasoning block."
    )

    def __init__(
        self,
        prompting_regime: str,
        use_tool_call: bool,
        tool_call_instruction: str,
        instruction: str,
        few_shot: str,
        llm_model,
    ):
        """
        Args:
            prompting_regime: str,      "baseline_structure_faithfulness" or "detailed_instruction"
            use_tool_call: bool 
            tool_call_instruction: str, explains output format if tool is used. ATTENTION: make sure few-shot examples are compatible with the output format
            instruction: str,           basic dataset-specific instruction
            few_shot: str,              formatted (!) few shot examples to use for the prompt
            llm_model: LLMModel,        the LLM model to use for the prompt (necessary for chat template and model-specific completion cleaning)
        """
        assert isinstance(prompting_regime, str) and prompting_regime, "prompting_regime must be a non-empty string"
        assert isinstance(use_tool_call, bool), "use_tool_call must be a boolean"
        assert isinstance(tool_call_instruction, str), "tool_call_instruction must be a string"
        if use_tool_call:
            assert tool_call_instruction.strip(), "tool_call_instruction must be non-empty when use_tool_call is True"
        assert isinstance(instruction, str) and instruction.strip(), "instruction must be a non-empty string"
        assert isinstance(few_shot, str), "few_shot must be a string"
        assert llm_model is not None, "llm_model must be provided"

        self.prompting_regime = prompting_regime
        self.use_tool_call = use_tool_call
        self.tool_call_instruction = tool_call_instruction
        self.instruction = instruction
        self.few_shot = few_shot
        self.llm_model = llm_model
        self.regime_string = self._get_regime_string(prompting_regime)

    @classmethod
    def _get_regime_string(cls, prompting_regime: str) -> str:
        if prompting_regime == "baseline_structure_faithfulness":
            return ""
        if prompting_regime == "detailed_instruction":
            return cls.DETAILED_REGIME_STRING
        if prompting_regime == "maximum_mediator_faithfulness":
            return cls.MAXIMUM_MEDIATOR_FAITHFULNESS_REGIME_STRING
        raise ValueError(f"Unknown prompting regime: {prompting_regime}")

    def build_zeroshot_instruction(self) -> str:
        parts = [self.instruction]
        if self.use_tool_call:
            parts.append(self.tool_call_instruction)
        if self.regime_string:
            parts.append(self.regime_string)
        return "\n\n".join(parts)

    def make_prompt(self, current_sample: str, include_gold_structure: bool, gold_structure: Optional[str] = None) -> str:
        """
            current_sample:             input sample without mediator (e.g. question, context and hypothesis in EntailmentBank, or task + student solution in RiceChem)
            include_gold_structure:     whether to include gold structure
            gold_structure:             formatted gold structure
        """
        assert isinstance(current_sample, str) and current_sample.strip(), "current_sample must be a non-empty string"
        assert isinstance(include_gold_structure, bool), "include_gold_structure must be a boolean"
        if include_gold_structure:
            assert isinstance(gold_structure, str) and gold_structure.strip(), "gold_structure must be a non-empty string when include_gold_structure is True"

        instruction = self.build_zeroshot_instruction()
        prompt_parts = [instruction]
        if self.few_shot.strip():
            prompt_parts.append(self.few_shot)
        prompt_parts.append(current_sample)
        prompt = "\n\n".join(prompt_parts)

        messages = [{"role": "user", "content": prompt}]
        add_generation_prompt_status = True
        if include_gold_structure:
            messages.append({"role": "assistant", "content": gold_structure})
            add_generation_prompt_status = False

        prompt = self.llm_model.apply_chat_template(messages, add_generation_prompt=add_generation_prompt_status)
        if not add_generation_prompt_status:
            prompt = self.llm_model.clean_model_specific_completion(prompt)

        return prompt
