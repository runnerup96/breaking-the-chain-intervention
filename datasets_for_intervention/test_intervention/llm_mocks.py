class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        # Simple mock implementation
        result = ""
        for msg in messages:
            if msg["role"] == "user":
                result += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                result += f"Assistant: {msg['content']}\n"
        if add_generation_prompt:
            result += "Assistant: "
        return result
    
    def __call__(self, texts, return_tensors=None, padding=False):
        # Mock tokenizer call for batch processing
        class MockInputs:
            def __init__(self, texts):
                self.input_ids = [[1, 2, 3] for _ in texts]  # Mock token IDs
                self.attention_mask = [[1, 1, 1] for _ in texts]
            
            def to(self, device):
                return self
        
        return MockInputs(texts)
    
    def decode(self, token_ids, skip_special_tokens=True):
        # Mock decode method
        if isinstance(token_ids, list):
            return f"decoded_text_{len(token_ids)}"
        return "decoded_text"
    
    def batch_decode(self, token_ids_list, skip_special_tokens=True):
        # Mock batch decode method
        return [f"decoded_text_{i}" for i in range(len(token_ids_list))]


class FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.generation_config = type('obj', (object,), {'pad_token_id': 0})()
    
    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        # Mock generation - return input_ids + some generated tokens
        batch_size = len(input_ids)
        generated = []
        for i in range(batch_size):
            # Add some mock generated tokens
            generated_tokens = input_ids[i] + [4, 5, 6]  # Mock additional tokens
            generated.append(generated_tokens)
        return generated


class FakeLLMModel:
    """Mock LLMModel that matches the interface"""
    def __init__(self, model_name="fake_model"):
        self.model_name = model_name
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.model_family = "Qwen3"  # Default to Qwen3 for tests
    
    def generate(self, prompts, max_new_tokens, skip_special_tokens):
        # Mock generation that returns the expected format
        results = []
        for prompt in prompts:
            results.append({
                "prompt": prompt,
                "completion": f"mock_completion_for_{prompt[:10]}..."
            })
        return results
    
    def apply_chat_template(self, messages, add_generation_prompt):
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
    
    def clean_model_specific_completion(self, output):
        return output