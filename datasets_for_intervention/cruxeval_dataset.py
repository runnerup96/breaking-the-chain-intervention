"""
CRUXEval dataset loader.

CRUXEval (Gu et al., 2024) contains 800 Python output-prediction tasks of the
form (code, input, output). Each sample is loaded as:

    {
      "idx":              str,
      "code":             str  (function source defining `def f(...)`),
      "input_str":        str  (raw input expression as it appears in the dataset),
      "args":             tuple,                 # decoded input args
      "gold_trace":       list[dict],            # canonical mediator M_gold
      "gold_target":      str(repr(real_output)),
      "mediator_trace":   deepcopy(gold_trace),  # replaced during interventions
    }

By default the dataset is loaded from a local jsonl cache at
    ${data_path}/test.jsonl
If that file is missing, the loader falls back to HuggingFace
(`cruxeval-org/cruxeval`) and writes the jsonl cache for reproducibility.

"""

import copy
import json
import os

from datasets_for_intervention.cruxeval_trace import (
    applicable_levels,
    make_trace,
)


class CRUXEvalDataset:
    HF_REPO = "cruxeval-org/cruxeval"

    def __init__(
        self,
        data_path: str = None,
        max_trace_steps: int = 80,
        max_samples: int = None,
        hf_split: str = "test",
    ):
        """
        Args:
            data_path:       directory holding (or to receive) `test.jsonl`.
                             If None, loads directly from HuggingFace and skips caching.
            max_trace_steps: drop samples whose gold trace has more steps than this
                             (keeps prompt length bounded for LLMs).
            max_samples:     optional cap on the number of samples.
            hf_split:        HuggingFace split to use when falling back.
        """
        self.data_path = data_path
        self.max_trace_steps = max_trace_steps
        self.max_samples = max_samples
        self.hf_split = hf_split
        self.data = []

        raw_samples = self._load_raw_samples()
        self.process_data(raw_samples)

    # ------------------------------------------------------------------
    # Raw sample loading (local jsonl preferred, HF fallback)
    # ------------------------------------------------------------------

    def _load_raw_samples(self):
        local_path = (
            os.path.join(self.data_path, "test.jsonl") if self.data_path else None
        )
        if local_path and os.path.exists(local_path):
            return list(self._iter_jsonl(local_path))

        # HuggingFace fallback
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "CRUXEval: local test.jsonl not found and `datasets` package is "
                "missing. Install with `pip install datasets`."
            ) from e

        hf = load_dataset(self.HF_REPO, split=self.hf_split)
        raw_samples = [dict(x) for x in hf]

        if local_path:
            os.makedirs(self.data_path, exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                for row in raw_samples:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return raw_samples

    @staticmethod
    def _iter_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # ------------------------------------------------------------------
    # Sample processing
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_function(code: str, input_str: str):
        """
        Compile the code, fetch `f`, decode the input expression into a tuple of args.
        Returns (f, args, namespace) or raises on failure.
        """
        namespace = {}
        exec(code, namespace)  # noqa: S102 -- benchmark code, run by design
        if "f" not in namespace or not callable(namespace["f"]):
            raise ValueError("function `f` not defined in code")
        raw = (input_str or "").strip()
        # CRUXEval inputs are comma-separated positional args; wrap into a tuple.
        args = eval(f"({raw},)", namespace) if raw else ()
        return namespace["f"], args, namespace

    def process_data(self, raw_samples):
        kept = 0
        skipped_resolve = 0
        skipped_trace = 0
        skipped_levels = 0
        for idx, raw in enumerate(raw_samples):
            if self.max_samples is not None and kept >= self.max_samples:
                break

            code = raw.get("code", "")
            input_str = raw.get("input", "")
            sample_id = str(raw.get("id", idx))

            try:
                f, args, _ns = self._resolve_function(code, input_str)
            except Exception:
                skipped_resolve += 1
                continue

            try:
                trace, real_output = make_trace(f, *copy.deepcopy(args))
            except Exception:
                skipped_trace += 1
                continue

            if not trace or len(trace) > self.max_trace_steps:
                skipped_trace += 1
                continue

            levels = applicable_levels(trace)
            # Need at least one Local Edit level to be meaningful.
            if not any(lvl in levels for lvl in (1, 2, 3, 4, 5, 6)):
                skipped_levels += 1
                continue

            sample = {
                "idx":            sample_id,
                "code":           code,
                "input_str":      input_str,
                "args":           args,
                "gold_trace":     copy.deepcopy(trace),
                "gold_target":    repr(real_output),
                "mediator_trace": copy.deepcopy(trace),
            }
            self.data.append(sample)
            kept += 1

        print(
            f"CRUXEval: kept={len(self.data)} | "
            f"skipped_resolve={skipped_resolve} "
            f"skipped_trace={skipped_trace} "
            f"skipped_levels={skipped_levels}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
