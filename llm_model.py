import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Union
import re


QWEN3_MODEL_FAMILY = "Qwen3"
FALCON3_MODEL_FAMILY = "Falcon3"
LLAMA32_MODEL_FAMILY = "Llama3.2"
LLAMA31_MODEL_FAMILY = "Llama3.1"
GEMMA2_MODEL_FAMILY = "Gemma2"


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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

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
        elif (hasattr(self.model.config, 'architectures') and 
              self.model.config.architectures and 
              self.model.config.architectures[0] == "LlamaForCausalLM" and
              "falcon3" in model_name.lower()):
            return FALCON3_MODEL_FAMILY
        elif (hasattr(self.model.config, 'architectures') and 
              self.model.config.architectures and 
              self.model.config.architectures[0] == "LlamaForCausalLM" and
              "llama-3.2" in model_name.lower()):
            return LLAMA32_MODEL_FAMILY
        elif (hasattr(self.model.config, 'architectures') and 
              self.model.config.architectures and 
              self.model.config.architectures[0] == "LlamaForCausalLM" and
              "llama-3.1" in model_name.lower()):
            return LLAMA31_MODEL_FAMILY
        elif (hasattr(self.model.config, 'architectures') and 
              self.model.config.architectures and 
              self.model.config.architectures[0] == "Gemma2ForCausalLM"):
            return GEMMA2_MODEL_FAMILY
        else:
            raise NotImplementedError(f"Model family for {model_name} not yet implemented")
    
    def generate(self, prompts:  List[str], max_new_tokens: int,
                 skip_special_tokens: bool) -> List[Dict[str, str]]:
        if self.model_family == QWEN3_MODEL_FAMILY:
            return self._generate_qwen3_batch(prompts, max_new_tokens, skip_special_tokens)
        elif self.model_family == FALCON3_MODEL_FAMILY:
            return self._generate_falcon3_batch(prompts, max_new_tokens, skip_special_tokens)
        elif self.model_family in [LLAMA32_MODEL_FAMILY, LLAMA31_MODEL_FAMILY]:
            return self._generate_llama_batch(prompts, max_new_tokens, skip_special_tokens)
        elif self.model_family == GEMMA2_MODEL_FAMILY:
            return self._generate_gemma2_batch(prompts, max_new_tokens, skip_special_tokens)
        else:
            # For now, return error for non-supported models
            # TODO: Add other model generation functions
            raise NotImplementedError(f"Generation for model type {type(self.model).__name__} not yet implemented")

    
    def apply_chat_template(self, messages: List[Dict], add_generation_prompt: bool) -> str:
        if self.model_family == QWEN3_MODEL_FAMILY:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False
            )
        elif self.model_family == FALCON3_MODEL_FAMILY:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        elif self.model_family == LLAMA32_MODEL_FAMILY:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        elif self.model_family == LLAMA31_MODEL_FAMILY:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        elif self.model_family == GEMMA2_MODEL_FAMILY:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            raise NotImplementedError(f"Chat template for model type {type(self.model).__name__} not yet implemented")
        return prompt
    
    def _generate_qwen3_batch(self, prompts: List[str], max_new_tokens: int,
                              skip_special_tokens: bool) -> List[Dict[str, str]]:
        """
        Generate text for multiple prompts using Qwen3 model in batch.
        """
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )
        
        results = []
        for i in range(len(prompts)):
            input_length = model_inputs.input_ids[i].shape[0]

            prompt_ids = generated_ids[i][:input_length].tolist()
            completion_ids = generated_ids[i][input_length:].tolist()
            
            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens)

            results.append({"prompt": prompt, "completion": completion})
        
        return results

    def _generate_falcon3_batch(self, prompts: List[str], max_new_tokens: int,
                               skip_special_tokens: bool) -> List[Dict[str, str]]:
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )
        
        results = []
        for i in range(len(prompts)):
            input_length = model_inputs.input_ids[i].shape[0]

            prompt_ids = generated_ids[i][:input_length].tolist()
            completion_ids = generated_ids[i][input_length:].tolist()
            
            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens)

            results.append({"prompt": prompt, "completion": completion})
        
        return results

    def _generate_llama_batch(self, prompts: List[str], max_new_tokens: int,
                               skip_special_tokens: bool) -> List[Dict[str, str]]:
        """
        Generate text for multiple prompts using Llama3.2 model in batch.
        """
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )
        
        results = []
        for i in range(len(prompts)):
            input_length = model_inputs.input_ids[i].shape[0]

            prompt_ids = generated_ids[i][:input_length].tolist()
            completion_ids = generated_ids[i][input_length:].tolist()
            
            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens)

            results.append({"prompt": prompt, "completion": completion})
        
        return results

    def _generate_gemma2_batch(self, prompts: List[str], max_new_tokens: int,
                               skip_special_tokens: bool) -> List[Dict[str, str]]:
        """
        Generate text for multiple prompts using Gemma2 model in batch.
        """
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )
        
        results = []
        for i in range(len(prompts)):
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
                output = output[:last_im_end]
            output = re.sub(r'<\|endoftext\|>|<\|im_end\|>', '', output)
        elif self.model_family == FALCON3_MODEL_FAMILY:
            output = re.sub(r'<\|endoftext\|>', '', output)
        elif self.model_family in [LLAMA32_MODEL_FAMILY, LLAMA31_MODEL_FAMILY]:
            output = re.sub(r'<\|eot_id\|>', '', output)
        elif self.model_family == GEMMA2_MODEL_FAMILY:
            last_end_of_turn = output.rfind('<end_of_turn>')
            if last_end_of_turn != -1:
                output = output[:last_end_of_turn]
        else:
            raise NotImplementedError(f"Model family for {self.model_name} not yet implemented")
        return output

