import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Union
import re

QWEN3_MODEL_FAMILY = "Qwen3"


class LLMModel:
    def __init__(self, model_name: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16):
        """
        Initialize the LLM model.
        
        Args:
            model_name: Name or path of the model
            device_map: Device mapping for the model
            dtype: Torch data type for the model
        """
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device_map, 
            dtype=dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = 'left'


        self.stop_token = self.tokenizer.eos_token

        self.model_family = self.identify_model_family(model_name)
        
        print(f"Model {model_name} initialized")

    def identify_model_family(self, model_name: str) -> str:
        """
        Identify the type of the model.
        """
        if (hasattr(self.model.config, 'architectures') and 
                         self.model.config.architectures and 
                         self.model.config.architectures[0] == "Qwen3ForCausalLM"):
            return QWEN3_MODEL_FAMILY
        else:
            raise NotImplementedError(f"Model family for {model_name} not yet implemented")
    
    def generate(self, prompts:  List[str], max_new_tokens: int,
                 skip_special_tokens: bool) -> List[Dict[str, str]]:
        """
        Generate text using the model.
        
        Args:
            prompts: Input prompt(s) - list of strings
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary or list of dictionaries containing generation results
        """
        if self.model_family == QWEN3_MODEL_FAMILY:
            return self._generate_qwen3_batch(prompts, max_new_tokens, skip_special_tokens)
        else:
            # For now, return error for non-Qwen3 models
            # TODO: Add other model generation functions
            raise NotImplementedError(f"Generation for model type {type(self.model).__name__} not yet implemented")
    

    
    def _generate_qwen3_batch(self, prompts: List[str], max_new_tokens: int,
                              skip_special_tokens: bool) -> List[Dict[str, str]]:
        """
        Generate text for multiple prompts using Qwen3 model in batch.
        """
        # Prepare batch inputs

        # if include_chat_template:
        #     batch_texts = []
        #     for prompt in prompts:
        #         messages = [{"role": "user", "content": prompt}]
        #         text = self.tokenizer.apply_chat_template(
        #             messages,
        #             tokenize=False,
        #             add_generation_prompt=True,
        #             enable_thinking=False
        #         )
        #         batch_texts.append(text)
        # else:
        #     batch_texts = prompts
        
        # Tokenize batch
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        
        # Generate
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
        
        # Decode each output in the batch
        results = []
        for i in range(len(prompts)):
            # Extract the generated part for this sample
            input_length = model_inputs.input_ids[i].shape[0]

            prompt_ids = generated_ids[i][:input_length].tolist()
            completion_ids = generated_ids[i][input_length:].tolist()
            
            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens)

            results.append({"prompt": prompt, "completion": completion})
        
        return results

    def clean_model_specific_completion(self, output: str) -> str:
        if self.model_family == QWEN3_MODEL_FAMILY:
            last_im_end = output.rfind('<|im_end|>')
            if last_im_end != -1:
                # Keep everything up to but not including the last <|im_end|>
                output = output[:last_im_end]
            output = re.sub(r'<\|endoftext\|>|<\|im_end\|>', '', output)
        else:
            raise NotImplementedError(f"Model family for {self.model_name} not yet implemented")
        return output

