class FakeTokenizer:
    eos_token = "<eos>"
    
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