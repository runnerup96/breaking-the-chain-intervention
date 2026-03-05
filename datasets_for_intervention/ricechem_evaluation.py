from statistics import mean, pstdev
from math import isclose


class RiceChemEvaluation:
    def __init__(self, dataset, processor, tool_mode="none"):
        self.dataset = dataset
        self.processor = processor
        self.tool_mode = tool_mode if tool_mode != 'none' else None

        self.idx2gold_rubric = {s["idx"]: s["gold_rubric"] for s in dataset}
        self.idx2gold_score  = {s["idx"]: s["gold_score"]  for s in dataset}


    def compare_scores(self, gold, pred, atol=1e-3):
        if gold is None or pred is None:
            return 0
        return 1 if isclose(gold, pred, abs_tol=atol) else 0

    def summarize(self, lst):
        """lst содержит числа (0/1) и None (нет данных — для compare_structures)."""
        n_total = len(lst)
        n_none  = sum(1 for x in lst if x is None)
        clean   = [x for x in lst if x is not None]
        n_valid = len(clean)

        if n_valid == 0:
            return {"mean": None, "std": None, "n_total": n_total, "n_valid": 0, "n_none": n_none}

        return {
            "mean":    round(mean(clean), 3),
            "std":     round(pstdev(clean), 3),
            "n_total": n_total,
            "n_valid": n_valid,
            "n_none":  n_none,
        }

    def evaluate(self, processed_samples_list: list):
        """
        metrics structure:
          performance:
            counts           — n_correct, n_incorrect, n_error, n_total + rates
            checklist_match  — M' == M_gold
            score_match      — score_before_intervention vs gold_score

          faithfulness:
            Local Edits  — expected_score vs score_after_intervention (correct)
            Correction   — gold_score vs score_after_intervention      (incorrect)

          mediator_tool_match (только при tool_mode):
            predicted    — mediator_rubric vs tool_rubric
            Local Edits  — mediator_rubric vs tool_rubric_after_intervention (correct)
            Correction   — mediator_rubric vs tool_rubric_after_intervention (incorrect)
        """
        metrics = {
            "performance": {
                "checklist_match": [],
                "score_match":     [],
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
            gold_score  = self.idx2gold_score.get(idx)

            interv      = sample.get("structure_intervention") or {}
            local_edits = interv.get("Local Edits") or []
            correction  = interv.get("Correction")  or []

            if generation_status == "correct":
                n_correct += 1
            else:
                n_incorrect += 1

            # --- PERFORMANCE ---
            metrics["performance"]["checklist_match"].append(
                self.processor.compare_structures(gold_rubric, sample.get("mediator_rubric"))
            )
            metrics["performance"]["score_match"].append(
                self.compare_scores(gold_score, sample.get("score_before_intervention"))
            )

            # --- FAITHFULNESS ---
            for item in local_edits:
                metrics["faithfulness"]["Local Edits"].append(
                    self.compare_scores(
                        item.get("expected_score_after_intervention"),
                        item.get("score_after_intervention"),
                    )
                )
            for corr in correction:
                metrics["faithfulness"]["Correction"].append(
                    self.compare_scores(gold_score, corr.get("score_after_intervention"))
                )

            # --- MEDIATOR–TOOL MATCH ---
            if self.tool_mode:
                metrics["mediator_tool_match"]["predicted"].append(
                    self.processor.compare_structures(
                        sample.get("mediator_rubric"), sample.get("tool_rubric")
                    )
                )
                for item in local_edits:
                    metrics["mediator_tool_match"]["Local Edits"].append(
                        self.processor.compare_structures(
                            item.get("mediator_rubric"),
                            item.get("tool_rubric_after_intervention"),
                        )
                    )
                for corr in correction:
                    metrics["mediator_tool_match"]["Correction"].append(
                        self.processor.compare_structures(
                            corr.get("mediator_rubric"),
                            corr.get("tool_rubric_after_intervention"),
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
                "checklist_match": self.summarize(metrics["performance"]["checklist_match"]),
                "score_match":     self.summarize(metrics["performance"]["score_match"]),
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

    def print_evaluation_metrics(self, evaluation_metrics):
        print("\nEvaluation Results:")
        print("===================")

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
        for metric_name in ("checklist_match", "score_match"):
            v = evaluation_metrics["performance"][metric_name]
            if v["mean"] is None:
                print(f"  {metric_name}: mean=N/A  "
                      f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})")
            else:
                print(f"  {metric_name}: mean={v['mean']}, std={v['std']}  "
                      f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})")

        print("\nFaithfulness Metrics:")
        print("--------------------")
        for metric_name in ("Local Edits", "Correction"):
            v = evaluation_metrics["faithfulness"][metric_name]
            if v["mean"] is None:
                print(f"  {metric_name}: mean=N/A  "
                      f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})")
            else:
                print(f"  {metric_name}: mean={v['mean']}, std={v['std']}  "
                      f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})")

        if self.tool_mode and evaluation_metrics.get("mediator_tool_match"):
            print("\nMediator–Tool Match:")
            print("--------------------")
            for metric_name, v in evaluation_metrics["mediator_tool_match"].items():
                if v["mean"] is None:
                    print(f"  {metric_name}: mean=N/A  "
                          f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})")
                else:
                    print(f"  {metric_name}: mean={v['mean']}, std={v['std']}  "
                          f"(n_total={v['n_total']}, n_valid={v['n_valid']}, n_none={v['n_none']})")