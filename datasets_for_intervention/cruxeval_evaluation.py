"""
CRUXEval evaluator.

Metrics structure mirrors the AVeriTeC / RiceChem evaluators:

performance:
  counts:        n_total, n_correct, n_incorrect, n_error + rates
  trace_match:   canonical equality of M' vs M_gold       (correct + incorrect)
  answer_match:  predicted answer vs gold_target          (correct + incorrect)

faithfulness:
  Local Edits:   target_after == expected_target_after    (correct)
  Correction:    target_after == gold_target              (incorrect)

faithfulness_by_level:
  Per-perturbation-level breakdown of Local Edits (1..6).
  Maps to S(k) from the original notebook idea:
    S(k) = fraction of samples where the model follows the trace at level k.

mediator_tool_match (tool_mode only):
  predicted:     answer derived from text (Final Answer) vs answer derived
                 from the tool ARGS (run through the simulator).
"""

from collections import defaultdict
from statistics import mean, pstdev


class CRUXEvalEvaluation:

    def __init__(self, dataset, processor, tool=None, tool_mode: str = "none"):
        self.dataset = dataset
        self.processor = processor
        self.tool = tool
        self.tool_mode = tool_mode if tool_mode != "none" else None

        self.idx2gold_trace  = {s["idx"]: s["gold_trace"]  for s in dataset}
        self.idx2gold_target = {s["idx"]: s["gold_target"] for s in dataset}

    @staticmethod
    def summarize(lst) -> dict:
        n_total = len(lst)
        n_none  = sum(1 for x in lst if x is None)
        clean   = [x for x in lst if x is not None]
        n_valid = len(clean)
        if n_valid == 0:
            return {"mean": None, "std": None,
                    "n_total": n_total, "n_valid": 0, "n_none": n_none}
        return {"mean":    round(mean(clean), 3),
                "std":     round(pstdev(clean), 3),
                "n_total": n_total,
                "n_valid": n_valid,
                "n_none":  n_none}

    def evaluate(self, processed_samples_list) -> dict:
        metrics = {
            "performance": {"trace_match": [], "answer_match": []},
            "faithfulness": {"Local Edits": [], "Correction": []},
            "faithfulness_by_level": defaultdict(list),
            "mediator_tool_match": {"predicted": [], "Local Edits": [], "Correction": []},
        }

        n_correct = n_incorrect = n_error = 0

        for sample in processed_samples_list:
            status = sample.get("generation_status")
            if status == "error":
                n_error += 1
                continue
            if status not in ("correct", "incorrect"):
                continue
            if status == "correct":
                n_correct += 1
            else:
                n_incorrect += 1

            idx = sample["idx"]
            gold_trace  = self.idx2gold_trace.get(idx)
            gold_target = self.idx2gold_target.get(idx)

            # PERFORMANCE
            metrics["performance"]["trace_match"].append(
                self.processor.compare_structures(gold_trace, sample.get("mediator_trace"))
            )
            metrics["performance"]["answer_match"].append(
                self.processor.compare_answers(
                    gold_target, sample.get("target_before_intervention")
                )
            )

            interv = sample.get("structure_intervention") or {}
            local_edits = interv.get("Local Edits") or []
            correction  = interv.get("Correction")  or []

            # FAITHFULNESS
            for item in local_edits:
                m = self.processor.compare_answers(
                    item.get("expected_target_after_intervention"),
                    item.get("target_after_intervention"),
                )
                metrics["faithfulness"]["Local Edits"].append(m)
                metrics["faithfulness_by_level"][item.get("perturbation_level")].append(m)

            for corr in correction:
                metrics["faithfulness"]["Correction"].append(
                    self.processor.compare_answers(
                        gold_target, corr.get("target_after_intervention")
                    )
                )

            # MEDIATOR-TOOL MATCH (tool_mode only)
            if self.tool_mode:
                metrics["mediator_tool_match"]["predicted"].append(
                    self.processor.compare_answers(
                        sample.get("target_before_intervention"),
                        # answer simulated from the tool's parsed trace
                        self._tool_answer(sample),
                    )
                )
                for item in local_edits:
                    metrics["mediator_tool_match"]["Local Edits"].append(
                        self.processor.compare_answers(
                            item.get("target_after_intervention"),
                            self._tool_answer(item, after=True),
                        )
                    )
                for corr in correction:
                    metrics["mediator_tool_match"]["Correction"].append(
                        self.processor.compare_answers(
                            corr.get("target_after_intervention"),
                            self._tool_answer(corr, after=True),
                        )
                    )

        n_total     = n_correct + n_incorrect + n_error
        n_non_error = n_correct + n_incorrect

        result = {
            "performance": {
                "counts": {
                    "n_total":        n_total,
                    "n_correct":      n_correct,
                    "n_incorrect":    n_incorrect,
                    "n_error":        n_error,
                    "correct_rate":   round(n_correct   / max(1, n_non_error), 3),
                    "incorrect_rate": round(n_incorrect / max(1, n_non_error), 3),
                    "error_rate":     round(n_error     / max(1, n_total),     3),
                },
                "trace_match":  self.summarize(metrics["performance"]["trace_match"]),
                "answer_match": self.summarize(metrics["performance"]["answer_match"]),
            },
            "faithfulness": {
                "Local Edits": self.summarize(metrics["faithfulness"]["Local Edits"]),
                "Correction":  self.summarize(metrics["faithfulness"]["Correction"]),
            },
            "faithfulness_by_level": {
                str(lvl): self.summarize(vals)
                for lvl, vals in sorted(metrics["faithfulness_by_level"].items())
            },
            "mediator_tool_match": {
                "predicted":   self.summarize(metrics["mediator_tool_match"]["predicted"]),
                "Local Edits": self.summarize(metrics["mediator_tool_match"]["Local Edits"]),
                "Correction":  self.summarize(metrics["mediator_tool_match"]["Correction"]),
            } if self.tool_mode else {},
        }
        self.print_evaluation_metrics(result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tool_answer(self, sample, after: bool = False):
        """Run the tool on the stored tool_args to get the simulated answer."""
        if not self.tool:
            return None
        key = "tool_args_after_intervention" if after else "tool_args"
        args = sample.get(key)
        if not args:
            return None
        try:
            return self.tool.calculate_score(args, sample)
        except Exception:
            return None

    def print_evaluation_metrics(self, m: dict):
        print("\nCRUXEval Evaluation Results:")
        print("============================")

        c = m["performance"]["counts"]
        print("\nGeneration Quality:")
        print(
            f"  n_total={c['n_total']}  |  "
            f"correct={c['n_correct']} ({c['correct_rate']:.1%})  |  "
            f"incorrect={c['n_incorrect']} ({c['incorrect_rate']:.1%})  |  "
            f"error={c['n_error']} ({c['error_rate']:.1%})"
        )

        print("\nPerformance Metrics:")
        print("-------------------")
        for name in ("trace_match", "answer_match"):
            v = m["performance"][name]
            self._print_metric(name, v)

        print("\nFaithfulness Metrics:")
        print("--------------------")
        for name in ("Local Edits", "Correction"):
            self._print_metric(name, m["faithfulness"][name])

        if m["faithfulness_by_level"]:
            print("\nFaithfulness by perturbation level (S(k)):")
            print("------------------------------------------")
            for lvl_str, v in m["faithfulness_by_level"].items():
                self._print_metric(f"level {lvl_str}", v)

        if self.tool_mode and m.get("mediator_tool_match"):
            print("\nMediator-Tool Match:")
            print("--------------------")
            for name, v in m["mediator_tool_match"].items():
                self._print_metric(name, v)

    @staticmethod
    def _print_metric(name, v):
        if v["mean"] is None:
            print(
                f"  {name}: mean=N/A  "
                f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})"
            )
        else:
            print(
                f"  {name}: mean={v['mean']}, std={v['std']}  "
                f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})"
            )
