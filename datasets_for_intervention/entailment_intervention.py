class EntailmentIntervention:
    def __init__(self, dataset, llm_stop_token: str):
        """
        Initialize the intervention class with dataset and stop token.
        
        Args:
            dataset: The EntailmentBank dataset instance
            llm_stop_token: The stop token used by the LLM model
        """
        pass

    def make_prompt(self, sample: dict) -> str:
        """
        Create a prompt for the LLM to generate reasoning steps and final answer.
        
        Args:
            sample: Dictionary containing the entailment sample data
            
        Returns:
            str: Formatted prompt for the LLM
        """
        pass

    def make_intervention(self, generated_output):
        """
        Create interventions by flipping reasoning steps in the generated output.
        
        Args:
            generated_output: The original LLM generation (could be dict or str depending on format)
            
        Returns:
            List of dictionaries with intervention data, or None if intervention fails
        """
        pass

    def validate_intervention(self, original_output, intervened_output, original_reasoning, intervention_reasoning):
        """
        Validate that an intervention correctly modifies only the intended reasoning step.
        
        Args:
            original_output: The original LLM generation
            intervened_output: The intervened generation
            original_reasoning: The original reasoning step
            intervention_reasoning: The intervened reasoning step
            
        Returns:
            bool: True if intervention is valid, False otherwise
        """
        pass

    def validate_all_interventions(self, original_output, intervention_data):
        """
        Validate all interventions for a given sample.
        
        Args:
            original_output: The original LLM generation
            intervention_data: List of intervention dictionaries
            
        Returns:
            bool: True if all interventions are valid, False otherwise
        """
        pass

    def reconstruct_interventions_to_prompt(self, original_output, intervened_completions):
        """
        Reconstruct prompts for completion after intervention.
        
        Args:
            original_output: The original LLM generation
            intervened_completions: List of intervention dictionaries
            
        Returns:
            List of prompts ready for LLM completion
        """
        pass

    def extract_target_from_prompt(self, generated_output):
        """
        Extract the final answer/target from the generated output.
        
        Args:
            generated_output: The LLM generation (could be dict or str)
            
        Returns:
            The extracted target value (type depends on dataset)
        """
        pass

    def infer_completion(self, completion_output):
        """
        Extract the completion result after intervention.
        
        Args:
            completion_output: The LLM completion after intervention
            
        Returns:
            The inferred result (type depends on dataset)
        """
        pass