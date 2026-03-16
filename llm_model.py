import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Union
import re
import os

from openai import OpenAI
from openai import DefaultHttpxClient

import asyncio
import aiohttp

from concurrent.futures import ThreadPoolExecutor, as_completed

QWEN3_MODEL_FAMILY = "Qwen3"
FALCON3_MODEL_FAMILY = "Falcon3"
LLAMA32_MODEL_FAMILY = "Llama3.2"
LLAMA31_MODEL_FAMILY = "Llama3.1"
GEMMA2_MODEL_FAMILY = "Gemma2"
GPT_MODEL_FAMILY = "GPT"

class LLMModel:
    def __init__(self, model_name: str,
                 device_map: str = "auto",
                 torch_dtype: torch.dtype = torch.bfloat16,
                 use_api: bool = False,
                 api_base_url: Union[str, None] = None,
                 tokenizer_name: Union[str, None] = None,
                 ):
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

        self.use_api = use_api
        self.api_base_url = api_base_url

        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not self.use_api:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch_dtype
            )
            
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        else:
            if api_base_url is None:
                raise ValueError("api_base_url must be provided when use_api=True")
            
            api_key = os.getenv("OPENAI_KEY")
            if api_key is None:
                raise ValueError(
                    "Environment variable OPENAI_KEY is not set, "
                    "but use_api=True. Please export OPENAI_KEY."
                )
            
            self.client = OpenAI(
                base_url=api_base_url,
                api_key=api_key,
                http_client=DefaultHttpxClient(verify=False)
            )

        self.model_family = self.identify_model_family(model_name)
        
        print("Local" if not self.use_api else "API", f"model {model_name} initialized")

    def identify_model_family(self, model_name: str) -> str:
        """
        Identify the type of the model.
        """
        if self.use_api:
            if 'gpt' in self.model_name.lower():
                return GPT_MODEL_FAMILY
            elif 'qwen3' in self.model_name.lower():
                return QWEN3_MODEL_FAMILY
            elif 'llama-3.1' in self.model_name.lower():
                return LLAMA31_MODEL_FAMILY

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

    def generate(self, prompts: List[str], max_new_tokens: int,
                 skip_special_tokens: bool, return_token_metrics: bool = False,
                 return_prompt_metrics: bool = False) -> List[Dict[str, str]]:
        if self.use_api:
            return self._generate_api_batch(prompts, max_new_tokens, skip_special_tokens,
                                            return_token_metrics, return_prompt_metrics)
        
        if self.model_family == QWEN3_MODEL_FAMILY:
            return self._generate_qwen3_batch(prompts, max_new_tokens, skip_special_tokens,
                                              return_token_metrics, return_prompt_metrics)
        elif self.model_family == FALCON3_MODEL_FAMILY:
            return self._generate_falcon3_batch(prompts, max_new_tokens, skip_special_tokens,
                                                return_token_metrics, return_prompt_metrics)
        elif self.model_family in [LLAMA32_MODEL_FAMILY, LLAMA31_MODEL_FAMILY]:
            return self._generate_llama_batch(prompts, max_new_tokens, skip_special_tokens,
                                              return_token_metrics, return_prompt_metrics)
        elif self.model_family == GEMMA2_MODEL_FAMILY:
            return self._generate_gemma2_batch(prompts, max_new_tokens, skip_special_tokens,
                                               return_token_metrics, return_prompt_metrics)
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
        elif self.model_family == GPT_MODEL_FAMILY:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                reasoning_effort="low"
            )
        elif self.model_family in [FALCON3_MODEL_FAMILY, LLAMA32_MODEL_FAMILY, LLAMA31_MODEL_FAMILY, GEMMA2_MODEL_FAMILY]:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            raise NotImplementedError(f"Chat template for model type {type(self.model).__name__} not yet implemented")
        return prompt

    def _generate_qwen3_batch(self, prompts: List[str], max_new_tokens: int,
                              skip_special_tokens: bool, return_token_metrics: bool,
                              return_prompt_metrics: bool) -> List[Dict[str, str]]:
        """
        Generate text for multiple prompts using Qwen3 model in batch.
        """
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        if return_token_metrics:
            generation_output = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_ids = generation_output.sequences
            scores = generation_output.scores
        else:
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )
            scores = None

        prompt_logits = None
        if return_prompt_metrics:
            with torch.no_grad():
                prompt_logits = self.model(**model_inputs).logits

        results = []
        for i in range(len(prompts)):
            input_length = model_inputs.input_ids[i].shape[0]

            prompt_ids = generated_ids[i][:input_length].tolist()
            completion_ids = generated_ids[i][input_length:].tolist()

            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens)

            result = {"prompt": prompt, "completion": completion}
            if return_token_metrics and scores is not None:
                result["token_metrics"] = self._collect_token_metrics(
                    generated_ids[i],
                    scores,
                    input_length,
                    i
                )
            if return_prompt_metrics and prompt_logits is not None:
                result["prompt_metrics"] = self._collect_prompt_metrics(
                    model_inputs.input_ids[i],
                    model_inputs.attention_mask[i],
                    prompt_logits[i]
                )
            results.append(result)

        return results

    def _generate_falcon3_batch(self, prompts: List[str], max_new_tokens: int,
                                skip_special_tokens: bool, return_token_metrics: bool,
                                return_prompt_metrics: bool) -> List[Dict[str, str]]:
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        if return_token_metrics:
            generation_output = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_ids = generation_output.sequences
            scores = generation_output.scores
        else:
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )
            scores = None

        prompt_logits = None
        if return_prompt_metrics:
            with torch.no_grad():
                prompt_logits = self.model(**model_inputs).logits

        results = []
        for i in range(len(prompts)):
            input_length = model_inputs.input_ids[i].shape[0]

            prompt_ids = generated_ids[i][:input_length].tolist()
            completion_ids = generated_ids[i][input_length:].tolist()

            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens)

            result = {"prompt": prompt, "completion": completion}
            if return_token_metrics and scores is not None:
                result["token_metrics"] = self._collect_token_metrics(
                    generated_ids[i],
                    scores,
                    input_length,
                    i
                )
            if return_prompt_metrics and prompt_logits is not None:
                result["prompt_metrics"] = self._collect_prompt_metrics(
                    model_inputs.input_ids[i],
                    model_inputs.attention_mask[i],
                    prompt_logits[i]
                )
            results.append(result)

        return results

    def _generate_llama_batch(self, prompts: List[str], max_new_tokens: int,
                               skip_special_tokens: bool, return_token_metrics: bool,
                               return_prompt_metrics: bool) -> List[Dict[str, str]]:
        """
        Generate text for multiple prompts using Llama3.2 model in batch.
        """
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        if return_token_metrics:
            generation_output = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_ids = generation_output.sequences
            scores = generation_output.scores
        else:
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )
            scores = None

        prompt_logits = None
        if return_prompt_metrics:
            with torch.no_grad():
                prompt_logits = self.model(**model_inputs).logits

        results = []
        for i in range(len(prompts)):
            input_length = model_inputs.input_ids[i].shape[0]

            prompt_ids = generated_ids[i][:input_length].tolist()
            completion_ids = generated_ids[i][input_length:].tolist()

            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens)

            result = {"prompt": prompt, "completion": completion}
            if return_token_metrics and scores is not None:
                result["token_metrics"] = self._collect_token_metrics(
                    generated_ids[i],
                    scores,
                    input_length,
                    i
                )
            if return_prompt_metrics and prompt_logits is not None:
                result["prompt_metrics"] = self._collect_prompt_metrics(
                    model_inputs.input_ids[i],
                    model_inputs.attention_mask[i],
                    prompt_logits[i]
                )
            results.append(result)

        return results

    def _generate_gemma2_batch(self, prompts: List[str], max_new_tokens: int,
                               skip_special_tokens: bool, return_token_metrics: bool,
                               return_prompt_metrics: bool) -> List[Dict[str, str]]:
        """
        Generate text for multiple prompts using Gemma2 model in batch.
        """
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        if return_token_metrics:
            generation_output = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_ids = generation_output.sequences
            scores = generation_output.scores
        else:
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )
            scores = None

        prompt_logits = None
        if return_prompt_metrics:
            with torch.no_grad():
                prompt_logits = self.model(**model_inputs).logits

        results = []
        for i in range(len(prompts)):
            input_length = model_inputs.input_ids[i].shape[0]

            prompt_ids = generated_ids[i][:input_length].tolist()
            completion_ids = generated_ids[i][input_length:].tolist()

            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens)

            result = {"prompt": prompt, "completion": completion}
            if return_token_metrics and scores is not None:
                result["token_metrics"] = self._collect_token_metrics(
                    generated_ids[i],
                    scores,
                    input_length,
                    i
                )
            if return_prompt_metrics and prompt_logits is not None:
                result["prompt_metrics"] = self._collect_prompt_metrics(
                    model_inputs.input_ids[i],
                    model_inputs.attention_mask[i],
                    prompt_logits[i]
                )
            results.append(result)

        return results
    
    def _generate_api_batch(
        self,
        prompts: List[str],
        max_new_tokens: int,
        skip_special_tokens: bool,
        return_token_metrics: bool,
        return_prompt_metrics: bool) -> List[Dict[str, str]]:

        resp = self.client.completions.create(
            model=self.model_name,
            prompt=prompts,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )

        index2choice = {c.index: c for c in resp.choices}

        results = []
        for i, prompt in enumerate(prompts):
            choice = index2choice[i]
            completion_text = choice.text

            result = {
                "prompt": prompt,
                "completion": completion_text,
            }
            if return_token_metrics:
                result["token_metrics"] = None
            if return_prompt_metrics:
                result["prompt_metrics"] = None
            results.append(result)

        return results

    def _collect_token_metrics(self, generated_ids: torch.Tensor, scores: List[torch.Tensor], prompt_length: int, batch_idx: int):
        """
        Collect per-token metrics for the generated continuation.
        Returns a list of dicts with cross-entropy, max logit, and gt token logit.
        """
        metrics = []
        for step, step_scores in enumerate(scores):
            token_id = int(generated_ids[prompt_length + step].item())
            logits = step_scores[batch_idx]
            max_logit = float(torch.max(logits).item())
            gt_logit = float(logits[token_id].item())
            log_probs = torch.log_softmax(logits, dim=-1)
            cross_entropy = float((-log_probs[token_id]).item())
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            metrics.append({
                "token_id": token_id,
                "token": token_str,
                "cross_entropy": cross_entropy,
                "max_logit": max_logit,
                "gt_logit": gt_logit,
            })
        return metrics

    def _collect_prompt_metrics(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits: torch.Tensor):
        """
        Collect per-token metrics for the prompt (next-token prediction on prompt tokens).
        """
        metrics = []
        seq_len = input_ids.shape[0]
        for t in range(1, seq_len):
            if attention_mask[t].item() == 0 or attention_mask[t - 1].item() == 0:
                continue
            token_id = int(input_ids[t].item())
            step_logits = logits[t - 1]
            max_logit = float(torch.max(step_logits).item())
            gt_logit = float(step_logits[token_id].item())
            log_probs = torch.log_softmax(step_logits, dim=-1)
            cross_entropy = float((-log_probs[token_id]).item())
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            metrics.append({
                "token_id": token_id,
                "token": token_str,
                "cross_entropy": cross_entropy,
                "max_logit": max_logit,
                "gt_logit": gt_logit,
            })
        return metrics


    # def _generate_api_batch(
    #     self,
    #     prompts: List[str],
    #     max_new_tokens: int,
    #     skip_special_tokens: bool
    # ) -> List[Dict[str, str]]:

    #     def worker(prompt):
    #         resp = self.client.completions.create(
    #             model=self.model_name,
    #             prompt=prompt,
    #             max_tokens=max_new_tokens,
    #             temperature=0.0,
    #         )
    #         completion_text = resp.choices[0].text
    #         return prompt, completion_text

    #     max_workers = 16

    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         future_map = {executor.submit(worker, p): i for i, p in enumerate(prompts)}

    #         outputs = [None] * len(prompts)
    #         for future in as_completed(future_map):
    #             idx = future_map[future]
    #             prompt, completion = future.result()
    #             outputs[idx] = {
    #                 "prompt": prompt,
    #                 "completion": completion,
    #             }

    #     return outputs


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
        elif self.model_family == GPT_MODEL_FAMILY:
            last_return = output.rfind('<|return|>')
            if last_return != -1:
                output = output[:last_return]
        else:
            raise NotImplementedError(f"Model family for {self.model_name} not yet implemented")
        return output

    def clean_token_metrics(self, metrics_list: List[Dict]) -> List[Dict]:
        """
        Remove special tokens from token metrics list.
        Filters out entries where the token is a special token.
        """
        if not metrics_list:
            return metrics_list
        
        # Common special tokens to filter
        special_tokens = [
            "<|im_end|>", "<|endoftext|>", "</s>", "<eos>", 
            "<pad>", "<|eot_id|>", "<|pad|>", "<end_of_turn>", "<|return|>"
        ]
        
        # Add model-specific EOS token if available
        if self.tokenizer and self.tokenizer.eos_token:
            special_tokens.append(self.tokenizer.eos_token)
        
        # Create a set for fast lookup
        special_tokens_set = set(special_tokens)
        
        # Filter out metrics for special tokens
        cleaned_metrics = []
        for metric in metrics_list:
            token_str = metric.get("token", "")
            # Check if token is a special token (exact match or contains special token)
            is_special = False
            for special in special_tokens_set:
                if token_str == special or special in token_str:
                    is_special = True
                    break
            if not is_special:
                cleaned_metrics.append(metric)
        
        return cleaned_metrics
