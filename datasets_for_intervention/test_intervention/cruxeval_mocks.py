"""Mocks for CRUXEval unit tests."""

from copy import deepcopy

from datasets_for_intervention.cruxeval_trace import make_trace


class FakeLLMModel:
    def apply_chat_template(self, messages, add_generation_prompt=True):
        return messages[-1]["content"]

    def clean_model_specific_completion(self, text: str) -> str:
        return text


SAMPLE_CODE_1 = "def f(x):\n    y = x + 1\n    return y * 2\n"
SAMPLE_INPUT_1 = "3"

SAMPLE_CODE_2 = "def f(s):\n    out = s.upper()\n    return out + '!'\n"
SAMPLE_INPUT_2 = "'hi'"


def _build_sample(idx: str, code: str, input_str: str) -> dict:
    namespace = {}
    exec(code, namespace)
    f = namespace["f"]
    args = eval(f"({input_str},)", namespace)
    trace, real_output = make_trace(f, *deepcopy(args))
    return {
        "idx":            idx,
        "code":           code,
        "input_str":      input_str,
        "args":           args,
        "gold_trace":     deepcopy(trace),
        "gold_target":    repr(real_output),
        "mediator_trace": deepcopy(trace),
    }


class CRUXEvalDatasetMock:
    """Two canonical CRUXEval samples (no HF dependency)."""

    def __init__(self):
        self.data = [
            _build_sample("0", SAMPLE_CODE_1, SAMPLE_INPUT_1),
            _build_sample("1", SAMPLE_CODE_2, SAMPLE_INPUT_2),
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __iter__(self):
        return iter(self.data)
