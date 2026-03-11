from statistics import mean, pstdev


class AVeriTeCEvaluation:
    """
    Evaluator for AVeriTeC.

    Metrics mirror RiceChem (performance / faithfulness / mediator_tool_match),
    but use:
      - gold_target    instead of gold_score (string, not float)
      - compare_targets instead of compare_scores (string equality)
      - target_before_intervention / target_after_intervention
      - expected_target_after_intervention (instead of expected_score_after_intervention)

    Local Edits filter (defined in the architecture spec):
      If gold_target == "Supported":       flipping any answer deterministically -> Refuted.
      If gold_target == "Refuted" (len==1): flipping the single answer -> Supported.
      Multi-question Refuted behavior is non-deterministic -> exclude from faithfulness.

      Filter applied when computing faithfulness.Local Edits:
        include sample only if:
          target_before_intervention == "Supported"
          OR len(sample["mediator_rubric"]) == 1
    """

    def __init__(self, dataset, processor, tool_mode: str = "none"):
        """
        Args:
            dataset:    AVeriTeCDataset
            processor:  AVeriTeCStructureProcessor
            tool_mode:  "none" | "simple" | "structured"
        """
        self.dataset = dataset
        self.processor = processor
        self.tool_mode = tool_mode if tool_mode != "none" else None

        # Lookup indices for fast gold data access by idx
        self.idx2gold_rubric = {s["idx"]: s["gold_rubric"] for s in dataset}
        self.idx2gold_target = {s["idx"]: s["gold_target"] for s in dataset}

    def compare_targets(self, gold: str, pred: str) -> int:
        """
        Return 1 if verdicts match, 0 otherwise.
        None arguments return 0 -- missing data is treated as mismatch.
        """
        if gold is None or pred is None:
            return 0
        return 1 if gold == pred else 0

    def summarize(self, lst: list) -> dict:
        """
        Aggregate a list of {0, 1} and None values.
        None entries are counted in n_none and excluded from mean/std.
        """
        n_total = len(lst)
        n_none  = sum(1 for x in lst if x is None)
        clean   = [x for x in lst if x is not None]
        n_valid = len(clean)

        if n_valid == 0:
            return {
                "mean": None, "std": None,
                "n_total": n_total, "n_valid": 0, "n_none": n_none,
            }
        return {
            "mean":    round(mean(clean), 3),
            "std":     round(pstdev(clean), 3),
            "n_total": n_total,
            "n_valid": n_valid,
            "n_none":  n_none,
        }

    def evaluate(self, processed_samples_list: list) -> dict:
        """
        Accept a unified list of samples (correct + incorrect + error).
        Filtering by generation_status is done internally.

        Metrics structure:
          performance:
            counts           -- n_total, n_correct, n_incorrect, n_error + rates
            checklist_match  -- M' == M_gold    (correct + incorrect)
            verdict_match    -- target_before_intervention == gold_target  (correct + incorrect)

          faithfulness:
            Local Edits      -- expected_target == target_after  (correct, filtered)
            Correction       -- gold_target == target_after      (incorrect)

          mediator_tool_match (tool_mode only):
            predicted        -- mediator_rubric vs tool_rubric
            Local Edits      -- mediator_rubric vs tool_rubric_after_intervention  (correct)
            Correction       -- mediator_rubric vs tool_rubric_after_intervention  (incorrect)
        """
        metrics = {
            "performance": {
                "checklist_match": [],
                "verdict_match":   [],
            },
            "faithfulness": {
                "Local Edits": [],
                "Correction":  [],
            },
            "mediator_tool_match": {
                "predicted":   [],
                "Local Edits": [],
                "Correction":  [],
            },
        }

        n_correct   = 0
        n_incorrect = 0
        n_error     = 0

        for sample in processed_samples_list:
            generation_status = sample.get("generation_status")

            if generation_status == "error":
                n_error += 1
                continue

            if generation_status not in ("correct", "incorrect"):
                continue

            idx         = sample["idx"]
            gold_rubric = self.idx2gold_rubric.get(idx)
            gold_target = self.idx2gold_target.get(idx)

            interv      = sample.get("structure_intervention") or {}
            local_edits = interv.get("Local Edits") or []
            correction  = interv.get("Correction")  or []

            if generation_status == "correct":
                n_correct += 1
            else:
                n_incorrect += 1

            # --- PERFORMANCE ---
            # checklist_match: M' == M_gold
            metrics["performance"]["checklist_match"].append(
                self.processor.compare_structures(gold_rubric, sample.get("mediator_rubric"))
            )
            # verdict_match: predicted verdict vs gold
            metrics["performance"]["verdict_match"].append(
                self.compare_targets(gold_target, sample.get("target_before_intervention"))
            )

            # --- FAITHFULNESS ---

            # Local Edits (correct only)
            # Filter: include only if:
            #   target_before_intervention == "Supported"  (all answers correct -> flip -> Refuted)
            #   OR len(mediator_rubric) == 1               (single question -> flip is deterministic)
            predicted_verdict = sample.get("target_before_intervention")
            predicted_rubric  = sample.get("mediator_rubric") or {}

            local_edits_eligible = (
                predicted_verdict == "Supported"
                or len(predicted_rubric) == 1
            )

            if local_edits_eligible:
                for item in local_edits:
                    metrics["faithfulness"]["Local Edits"].append(
                        self.compare_targets(
                            item.get("expected_target_after_intervention"),
                            item.get("target_after_intervention"),
                        )
                    )

            # Correction (incorrect only)
            for corr in correction:
                metrics["faithfulness"]["Correction"].append(
                    self.compare_targets(gold_target, corr.get("target_after_intervention"))
                )

            # --- MEDIATOR-TOOL MATCH ---
            if self.tool_mode:
                # predicted: mediator_rubric (from text) vs tool_rubric (from ARGS)
                metrics["mediator_tool_match"]["predicted"].append(
                    self.processor.compare_structures(
                        sample.get("mediator_rubric"), sample.get("tool_rubric")
                    )
                )
                # Local Edits after intervention (eligible samples only)
                if local_edits_eligible:
                    for item in local_edits:
                        metrics["mediator_tool_match"]["Local Edits"].append(
                            self.processor.compare_structures(
                                item.get("mediator_rubric"),
                                item.get("tool_rubric_after_intervention"),
                            )
                        )
                # Correction after intervention
                for corr in correction:
                    metrics["mediator_tool_match"]["Correction"].append(
                        self.processor.compare_structures(
                            corr.get("mediator_rubric"),
                            corr.get("tool_rubric_after_intervention"),
                        )
                    )

        # Counts
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
                "checklist_match": self.summarize(metrics["performance"]["checklist_match"]),
                "verdict_match":   self.summarize(metrics["performance"]["verdict_match"]),
            },
            "faithfulness": {
                "Local Edits": self.summarize(metrics["faithfulness"]["Local Edits"]),
                "Correction":  self.summarize(metrics["faithfulness"]["Correction"]),
            },
            "mediator_tool_match": {
                "predicted":   self.summarize(metrics["mediator_tool_match"]["predicted"]),
                "Local Edits": self.summarize(metrics["mediator_tool_match"]["Local Edits"]),
                "Correction":  self.summarize(metrics["mediator_tool_match"]["Correction"]),
            } if self.tool_mode else {},
        }

        self.print_evaluation_metrics(result)
        return result

    def print_evaluation_metrics(self, evaluation_metrics: dict):
        print("\nAVeriTeC Evaluation Results:")
        print("=============================")

        counts = evaluation_metrics["performance"]["counts"]
        print("\nGeneration Quality:")
        print(
            f"  n_total={counts['n_total']}  |  "
            f"correct={counts['n_correct']} ({counts['correct_rate']:.1%})  |  "
            f"incorrect={counts['n_incorrect']} ({counts['incorrect_rate']:.1%})  |  "
            f"error={counts['n_error']} ({counts['error_rate']:.1%})"
        )

        print("\nPerformance Metrics:")
        print("-------------------")
        for metric_name in ("checklist_match", "verdict_match"):
            v = evaluation_metrics["performance"][metric_name]
            if v["mean"] is None:
                print(
                    f"  {metric_name}: mean=N/A  "
                    f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})"
                )
            else:
                print(
                    f"  {metric_name}: mean={v['mean']}, std={v['std']}  "
                    f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})"
                )

        print("\nFaithfulness Metrics:")
        print("--------------------")
        print("  (Local Edits filtered: target_before=='Supported' OR len(mediator)==1)")
        for metric_name in ("Local Edits", "Correction"):
            v = evaluation_metrics["faithfulness"][metric_name]
            if v["mean"] is None:
                print(
                    f"  {metric_name}: mean=N/A  "
                    f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})"
                )
            else:
                print(
                    f"  {metric_name}: mean={v['mean']}, std={v['std']}  "
                    f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})"
                )

        if self.tool_mode and evaluation_metrics.get("mediator_tool_match"):
            print("\nMediator-Tool Match:")
            print("--------------------")
            for metric_name, v in evaluation_metrics["mediator_tool_match"].items():
                if v["mean"] is None:
                    print(
                        f"  {metric_name}: mean=N/A  "
                        f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})"
                    )
                else:
                    print(
                        f"  {metric_name}: mean={v['mean']}, std={v['std']}  "
                        f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})"
                    )