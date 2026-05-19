"""
TabFactEvaluation: metrics and aggregation for the TabFact dataset.

Follows the unified architecture (§9) adapted for TabFact:

  performance:
    counts:           n_total, n_correct, n_incorrect, n_error (+ rates)
    query_match:      normalised exact-string match of mediator_query vs gold_query
                      (correct + incorrect, excluding error)
    execution_match:  DSL execution of predicted query gives same bool as gold query
                      (correct + incorrect, excluding error)
    target_match:     predicted target (bool) matches gold_target=True
                      (correct + incorrect, excluding error)

  faithfulness:
    Local Edits:  expected_target_after_intervention == target_after_intervention
                  (correct samples only)
    Correction:   gold_target == target_after_intervention
                  (incorrect samples only)

  mediator_tool_match (only when tool_mode is active):
    predicted:    mediator_query vs tool_query
      - execution_match:  execute both queries, compare bools
      - columns_match:  set-equality of DSL column references (1/0)
      - values_match:   set-equality of DSL filter values (1/0)
    Local Edits:   mediator_query vs tool_query_after_intervention (correct)
    Correction:    mediator_query vs tool_query_after_intervention (incorrect)

Aggregation: each metric list is summarised with summarize() -> {mean, std, n_total,
n_valid, n_none}.  None values are counted separately (n_none) and excluded from
mean/std (n_valid).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class TabFactEvaluation:
    """
    Evaluator for the TabFact dataset.

    Args:
        dataset:    TabFactDataset — used to build idx -> gold indices.
        processor:  TabFactStructureProcessor — used for column/value extraction
                    and Jaccard computation.
        tool:       TabFactTool — used to execute queries for execution_match.
        tool_mode:  "none" or "simple" (determines whether mediator_tool_match is computed).
    """

    def __init__(self, dataset, processor, tool, tool_mode: str = "none") -> None:
        self.dataset = dataset
        self.processor = processor
        self.tool = tool

        if tool_mode != 'none':
            tool_mode = 'simple'

        assert tool_mode in ("none", "simple"), (
            f"tool_mode must be 'none' or 'simple'. Got: {tool_mode}"
        )
        self.tool_mode = tool_mode if tool_mode != "none" else None

        # Build gold-data indices keyed by sample idx
        self.idx2gold_query: Dict[str, str] = {
            s["idx"]: s["gold_query"] for s in dataset
        }
        self.idx2gold_target: Dict[str, bool] = {
            s["idx"]: s["gold_target"] for s in dataset
        }

    def compare_targets(
        self, gold: Optional[bool], pred: Optional[bool]
    ) -> int:
        """
        Binary comparison of two boolean targets.

        Returns 0 (not 1 not None) when either argument is None, following
        the architecture invariant: missing data is treated as mismatch.
        """
        if gold is None or pred is None:
            return 0
        return 1 if gold == pred else 0

    def summarize(self, lst: List[Optional[int]]) -> Dict[str, Any]:
        """
        Aggregate a list of 0/1/None values into summary statistics.

        Returns:
            {
                mean:    float | None  (mean of valid (non-None) values)
                std:     float | None
                n_total: int           (total items)
                n_valid: int           (non-None items)
                n_none:  int           (None items)
            }
        """
        n_total = len(lst)
        valid = [x for x in lst if x is not None]
        n_valid = len(valid)
        n_none = n_total - n_valid

        if n_valid == 0:
            return {"mean": None, "std": None, "n_total": n_total,
                    "n_valid": n_valid, "n_none": n_none}

        mean = sum(valid) / n_valid
        variance = sum((x - mean) ** 2 for x in valid) / n_valid
        std = math.sqrt(variance)

        return {
            "mean": mean,
            "std": std,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_none": n_none,
        }

    def evaluate(self, processed_samples_list: List[Dict]) -> Dict:
        """
        Compute all metrics over a list of processed samples.

        Args:
            processed_samples_list: All samples (correct + incorrect + error) as returned
                                    by the intervention pipeline.

        Returns:
            Nested dict with performance, faithfulness, mediator_tool_match sub-trees.
        """
        # ---- counters ----
        n_total = n_correct = n_incorrect = n_error = 0

        # ---- performance metric accumulators ----
        query_match_list: List[Optional[int]] = []
        execution_match_list: List[Optional[int]] = []
        target_match_list: List[Optional[int]] = []

        # ---- faithfulness accumulators ----
        le_faithfulness_list: List[Optional[int]] = []   # Local Edits
        corr_faithfulness_list: List[Optional[int]] = []  # Correction

        # ---- mediator_tool_match accumulators (tool_mode only) ----
        # predicted (primary generation)
        mt_pred_exec: List[Optional[int]] = []
        mt_pred_col_match: List[Optional[float]] = []
        mt_pred_val_match: List[Optional[float]] = []
        # local edits
        mt_le_exec: List[Optional[int]] = []
        mt_le_col_match: List[Optional[float]] = []
        mt_le_val_match: List[Optional[float]] = []
        # correction
        mt_corr_exec: List[Optional[int]] = []
        mt_corr_col_match: List[Optional[float]] = []
        mt_corr_val_match: List[Optional[float]] = []

        for sample in processed_samples_list:
            idx = sample["idx"]
            status = sample.get("generation_status", "error")

            gold_query = self.idx2gold_query.get(idx, "")
            gold_target = self.idx2gold_target.get(idx, True)

            n_total += 1
            if status == "correct":
                n_correct += 1
            elif status == "incorrect":
                n_incorrect += 1
            else:
                n_error += 1
                continue  # error samples excluded from all other metrics

            # ----------------------------------------------------------
            # Performance
            # ----------------------------------------------------------
            pred_mediator = sample.get("mediator_query", "")
            pred_target = sample.get("target_before_intervention")

            # query_match: exact normalised string equality
            q_match = self.processor.compare_structures(pred_mediator, gold_query)
            query_match_list.append(q_match)

            # execution_match: execute both queries and compare boolean results
            exec_match = self._execution_match(pred_mediator, gold_query, sample)
            execution_match_list.append(exec_match)

            # target_match: predicted boolean verdict vs gold_target
            t_match = self.compare_targets(gold_target, pred_target)
            target_match_list.append(t_match)

            # ----------------------------------------------------------
            # Faithfulness
            # ----------------------------------------------------------
            interv = sample.get("structure_intervention", {})

            if status == "correct":
                for le in interv.get("Local Edits", []):
                    expected = le.get("expected_target_after_intervention")
                    actual = le.get("target_after_intervention")
                    le_faithfulness_list.append(self.compare_targets(expected, actual))

            if status == "incorrect":
                for corr in interv.get("Correction", []):
                    actual = corr.get("target_after_intervention")
                    corr_faithfulness_list.append(self.compare_targets(gold_target, actual))

            # ----------------------------------------------------------
            # mediator_tool_match (tool_mode only)
            # ----------------------------------------------------------
            if self.tool_mode:
                tool_query = sample.get("tool_query")  # str | None

                # predicted (primary generation)
                ee, jc, jv = self._query_match_metrics(pred_mediator, tool_query, sample)
                mt_pred_exec.append(ee)
                mt_pred_col_match.append(jc)
                mt_pred_val_match.append(jv)

                if status == "correct":
                    for le in interv.get("Local Edits", []):
                        le_med = le.get("mediator_query", "")
                        le_tool = le.get("tool_query_after_intervention")
                        ee, jc, jv = self._query_match_metrics(le_med, le_tool, le)
                        mt_le_exec.append(ee)
                        mt_le_col_match.append(jc)
                        mt_le_val_match.append(jv)

                if status == "incorrect":
                    for corr in interv.get("Correction", []):
                        corr_med = corr.get("mediator_query", "")
                        corr_tool = corr.get("tool_query_after_intervention")
                        ee, jc, jv = self._query_match_metrics(corr_med, corr_tool, corr)
                        mt_corr_exec.append(ee)
                        mt_corr_col_match.append(jc)
                        mt_corr_val_match.append(jv)

        metrics: Dict = {
            "performance": {
                "counts": {
                    "n_total": n_total,
                    "n_correct": n_correct,
                    "n_incorrect": n_incorrect,
                    "n_error": n_error,
                    "correct_rate": n_correct / n_total if n_total else None,
                    "incorrect_rate": n_incorrect / n_total if n_total else None,
                    "error_rate": n_error / n_total if n_total else None,
                },
                # query_match: how often the predicted DSL query exactly matches gold
                "query_match": self.summarize(query_match_list),
                # execution_match: how often executing the predicted query gives the
                # same boolean as executing the gold query
                "execution_match": self.summarize(execution_match_list),
                # target_match: how often the final verdict equals gold_target
                "target_match": self.summarize(target_match_list),
            },
            "faithfulness": {
                # Local Edits: expected (from DSL execution) == actual model output
                "Local Edits": self.summarize(le_faithfulness_list),
                # Correction: model should return to gold_target when given gold mediator
                "Correction": self.summarize(corr_faithfulness_list),
            },
        }

        if self.tool_mode:
            metrics["mediator_tool_match"] = {
                # comparison between text mediator and tool-call mediator
                "predicted": {
                    "execution_match": self.summarize(mt_pred_exec),
                    "columns_match": self.summarize(mt_pred_col_match),
                    "values_match": self.summarize(mt_pred_val_match),
                },
                "Local Edits": {
                    "execution_match": self.summarize(mt_le_exec),
                    "columns_match": self.summarize(mt_le_col_match),
                    "values_match": self.summarize(mt_le_val_match),
                },
                "Correction": {
                    "execution_match": self.summarize(mt_corr_exec),
                    "columns_match": self.summarize(mt_corr_col_match),
                    "values_match": self.summarize(mt_corr_val_match),
                },
            }
        else:
            metrics["mediator_tool_match"] = {}

        self._print_metrics(metrics)
        return metrics

    def _execution_match(
        self,
        query_a: Optional[str],
        query_b: Optional[str],
        sample: Dict,
    ) -> Optional[int]:
        """
        Execute both queries on the sample's table and compare boolean results.

        Returns 1 if both execute successfully and return the same bool,
                0 if they differ,
                None if either query is None or not executable.
        """
        if not query_a or not query_b:
            return None
        result_a = self.tool.calculate_score({"query": query_a}, sample)
        result_b = self.tool.calculate_score({"query": query_b}, sample)
        if result_a is None or result_b is None:
            return None
        return 1 if result_a == result_b else 0

    def _query_match_metrics(
        self,
        mediator: Optional[str],
        tool_query: Optional[str],
        sample: Dict,
    ):
        """
        Compute mediator-vs-tool_query comparison metrics.

        Returns:
            (execution_match: int|None, columns_match: int|None, values_match: int|None)
        """
        if mediator is None or tool_query is None:
            return None, None, None

        tool_query_str = tool_query.get("query") if isinstance(tool_query, dict) else tool_query
        # Execution match
        exec_match = self._execution_match(mediator, tool_query_str, sample)

        # Set-equality over column names and filter values independently
        cols_m, vals_m = self.processor.extract_columns_values(mediator)
        cols_t, vals_t = self.processor.extract_columns_values(tool_query_str)

        col_match = 1 if cols_m == cols_t else 0
        val_match = 1 if vals_m == vals_t else 0

        return exec_match, col_match, val_match

    def _print_metrics(self, metrics: Dict) -> None:
        """Print a human-readable summary of the evaluation metrics."""
        print("\n" + "=" * 60)
        print("TabFact Evaluation Results")
        print("=" * 60)

        # -- Performance --
        print("\n[Performance]")
        c = metrics["performance"]["counts"]
        print(
            f"  n_total={c['n_total']}  correct={c['n_correct']} ({_pct(c['correct_rate'])})  "
            f"incorrect={c['n_incorrect']} ({_pct(c['incorrect_rate'])})  "
            f"error={c['n_error']} ({_pct(c['error_rate'])})"
        )
        for key in ("query_match", "execution_match", "target_match"):
            s = metrics["performance"][key]
            print(f"  {key}: {_fmt(s)}")

        # -- Faithfulness --
        print("\n[Faithfulness]")
        for key in ("Local Edits", "Correction"):
            s = metrics["faithfulness"][key]
            print(f"  {key}: {_fmt(s)}")

        # -- mediator_tool_match --
        if metrics["mediator_tool_match"]:
            print("\n[Mediator-Tool Match]")
            for section, sub in metrics["mediator_tool_match"].items():
                print(f"  {section}:")
                for mname, s in sub.items():
                    print(f"    {mname}: {_fmt(s)}")

        print()

def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if v is not None else "N/A"


def _fmt(s: Dict) -> str:
    """Format a summarize() dict as a readable string."""
    mean = s.get("mean")
    std = s.get("std")
    nv = s.get("n_valid", 0)
    nn = s.get("n_none", 0)
    if mean is None:
        return f"None  (n_valid={nv}, n_none={nn})"
    return f"{mean:.4f} ± {std:.4f}  (n_valid={nv}, n_none={nn})"