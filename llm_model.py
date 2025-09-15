import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Union, Optional
import re

QWEN3_MODEL_FAMILY = "Qwen3"


class LLMModel:
    def __init__(self, model_name: str, device_map: str = "auto", torch_dtype: torch.dtype = torch.bfloat16):
        """
        Initialize the LLM model.
        
        Args:
            model_name: Name or path of the model
            device_map: Device mapping for the model
            torch_dtype: Torch data type for the model
        """
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device_map, 
            torch_dtype=torch_dtype
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
    
    def generate(self, prompts: Optional[List[str]] = None, messages: Optional[List[List[Dict[str, str]]]] = None, max_new_tokens: int = 100,
                 include_origin_prompt: bool = True, include_chat_template: bool = False, skip_special_tokens: bool = True) -> List[Dict[str, str]]:
        """
        Generate text using the model.
        
        Args:
            prompts: Input prompt(s) - list of strings (exactly one of prompts or messages must be provided)
            messages: Input messages - list of lists of dicts, where each dict contains role and content (exactly one of prompts or messages must be provided)
            max_new_tokens: Maximum number of new tokens to generate
            include_origin_prompt: Whether to include original prompt in output
            include_chat_template: Whether to apply chat template
            skip_special_tokens: Whether to skip special tokens in decoding
            
        Returns:
            List of dictionaries containing generation results
        """
        # Validate that exactly one of prompts or messages is provided
        if (prompts is None) == (messages is None):
            raise ValueError("Exactly one of 'prompts' or 'messages' must be provided, not both or neither")
        
        if prompts is not None and not isinstance(prompts, list):
            raise ValueError("prompts must be a list of strings")
        
        if messages is not None and not isinstance(messages, list):
            raise ValueError("messages must be a list of lists of dicts")
        if self.model_family == QWEN3_MODEL_FAMILY:
            return self._generate_qwen3_batch(prompts, messages, max_new_tokens,
                                              include_origin_prompt, include_chat_template, skip_special_tokens)
        else:
            # For now, return error for non-Qwen3 models
            # TODO: Add other model generation functions
            raise NotImplementedError(f"Generation for model type {type(self.model).__name__} not yet implemented")
    

    
    def _generate_qwen3_batch(self, prompts: Optional[List[str]], messages: Optional[List[List[Dict[str, str]]]], max_new_tokens: int,
                              include_origin_prompt: bool, include_chat_template: bool,
                              skip_special_tokens: bool) -> List[Dict[str, str]]:
        """
        Generate text for multiple prompts or messages using Qwen3 model in batch.
        """
        # Prepare batch inputs
        if messages is not None:
            # Process messages with chat template
            batch_texts = []
            for message_list in messages:
                assert message_list[-1]["role"] == "user" or message_list[-1]["role"] == "assistant"
                text = self.tokenizer.apply_chat_template(
                    message_list,
                    tokenize=False,
                    add_generation_prompt=True if message_list[-1]["role"] == "user" else False,
                    enable_thinking=False
                )
                batch_texts.append(text)
        elif include_chat_template:
            # Process prompts with chat template
            batch_texts = []
            for prompt in prompts:
                message_list = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    message_list,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                batch_texts.append(text)
        else:
            # Use prompts directly
            batch_texts = prompts
        
        # Tokenize batch
        model_inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True).to(self.model.device)
        
        # Generate
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
        
        # Decode each output in the batch
        results = []
        batch_size = len(messages) if messages is not None else len(prompts)
        for i in range(batch_size):
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

