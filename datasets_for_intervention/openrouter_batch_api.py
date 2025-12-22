import openai
import threading
import queue
import copy
import pandas as pd
import json
from typing import Callable, List, Dict, Any
import os

class OpenrouterBatchApiClass:
    def __init__(self, model: str, api_link: str, token: str, max_tokens: int = 4096, num_threads: int = 20, http_client: str = None):
        self.model = model
        self.api_link = api_link
        self.token = token
        self.max_tokens = max_tokens
        self.num_threads = num_threads
        self.http_client = http_client
    
    def _send_question(self, messages: List[Dict[str, str]], max_retries: int = 10) -> Dict[str, Any]:
        if self.http_client:
            client = openai.OpenAI(api_key=self.token, base_url=self.api_link, http_client=self.http_client)
        else:
            client = openai.OpenAI(api_key=self.token, base_url=self.api_link)

        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens
                )
                completion_tokens = completion.usage.completion_tokens
                reasoning = completion.choices[0].message.reasoning if completion.choices[0].message.reasoning else None
                response = completion.choices[0].message.content

                result = {
                    "response": response,
                    "completion_tokens": completion_tokens,
                    "reasoning": reasoning,
                }
                return result
            except Exception as e:
                print(f"Error during API call: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    return {
                        "response": "Error: Maximum retries exceeded",
                        "completion_tokens": None,
                        "reasoning": None
                    }
    
    def _process_sample(
        self,
        sample: Dict[str, Any],
        prompt_func: Callable[[Dict[str, Any]], str],
        sample_idx_key: str,
        results_queue: queue.Queue,
        idx: int,
        total: int
    ):
        prompt_text = prompt_func(sample)
        messages = [{"role": "user", "content": prompt_text}]
        response = self._send_question(messages)

        result = {
            "sample_idx": sample[sample_idx_key],
            "response": response
        }
        
        print(f"Processed sample {idx + 1}/{total}")

        results_queue.put(result)
    
    def call(
        self,
        samples: List[Dict[str, Any]],
        prompt_func: Callable[[Dict[str, Any]], str],
        sample_idx_key: str
    ) -> List[Dict[str, Any]]:

        results_queue = queue.Queue()
        threads = []

        for idx, sample in enumerate(samples):
            thread = threading.Thread(
                target=self._process_sample,
                args=(sample, prompt_func, sample_idx_key, results_queue, idx, len(samples))
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        return results


def load_jsonl(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]

def load_entailment(data_path):
    samples = load_jsonl(data_path)

    samples_for_evaluation = []
    for sample in samples:
        samples_for_evaluation.append({"sample_idx": sample["id"], "text": sample["meta"]["question_text"]})

    return samples_for_evaluation

if __name__ == "__main__":

    # load samples
    data_path = ""
    output_path = ""

    samples_for_evaluation = load_entailment(data_path)

    print(f"Loaded {len(samples_for_evaluation)} samples for evaluation")

    api_client = OpenrouterBatchApiClass(
        model="meta-llama/llama-3.1-8b-instruct",
        api_link="https://openrouter.ai/api/v1", 
        token=os.getenv("OPENROUTER_API_KEY")
    )

    def my_prompt_func(sample):
        prompt = (
            f"Paraphrase the following Text-to-SQL question while preserving the exact same meaning and semantics.\n"
            f"Requirements:\n"
            f"- Keep all factual information unchanged\n" 
            f"- Maintain the same query intent\n"
            f"- Use different wording and sentence structure\n"
            f"- Output only the paraphrased question, no explanations\n\n"
            f"Original question: {sample['text']}"
        )
        return prompt
    
    results = api_client.call(
        samples=samples_for_evaluation,
        prompt_func=my_prompt_func,
        sample_idx_key="sample_idx"
    )

    json.dump(results, open(output_path, "w"), ensure_ascii=False, indent=4)
