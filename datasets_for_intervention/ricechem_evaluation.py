from statistics import mean, pstdev
from math import isclose


class RiceChemEvaluation:
    def __init__(self, dataset, processor, tool_mode=None):
        self.dataset = dataset
        self.processor = processor
        self.tool_mode = tool_mode

        self.idx2gold_rubric = {s["idx"]: s["gold_rubric"] for s in dataset}
        self.idx2gold_score = {s["idx"]: s["gold_score"] for s in dataset}

    def compare_scores(self, gold, pred, atol=1e-3):
        if gold is None or pred is None:
            return 0
        return 1 if isclose(gold, pred, abs_tol=atol) else 0

    def summarize(self, lst):
        # lst may contain {0,1} and None
        n_total = len(lst)
        n_none = sum(1 for x in lst if x is None)
        clean = [x for x in lst if x is not None]
        n_valid = len(clean)

        if n_valid == 0:
            return {"mean": None, "std": None, "n_total": n_total, "n_valid": 0, "n_none": n_none}

        return {
            "mean": round(mean(clean), 3),
            "std": round(pstdev(clean), 3),
            "n_total": n_total,
            "n_valid": n_valid,
            "n_none": n_none,
        }

    def evaluate(self, processed_samples_list):
        metrics = {
            "performance": {
                "with_gold_structure": {"score_match": []},
                "with_predicted_structure": {"checklist_match": [], "score_match": []},
            },
            "faithfulness": {
                "with_gold_structure": {"HSVT": [], "Local Edits": [], "Correction": []},
                "with_predicted_structure": {"HSVT": [], "Local Edits": [], "Correction": []},
            },
            "mediator_tool_match": {
                "with_gold_structure": {"predicted": [], "HSVT": [], "Local Edits": [], "Correction": []},
                "with_predicted_structure": {"predicted": [], "HSVT": [], "Local Edits": [], "Correction": []},
            },
        }

        for sample in processed_samples_list:
            completion_type = sample.get("completion_type")
            if completion_type not in ("gold_structure", "structure_prediction"):
                continue

            mode_key = "with_gold_structure" if completion_type == "gold_structure" else "with_predicted_structure"

            idx = sample["idx"]
            gold_rubric = self.idx2gold_rubric.get(idx)
            gold_score = self.idx2gold_score.get(idx)

            interv = sample.get("structure_intervention") or {}
            hsvt = (interv.get("HSVT") or [None])[0]
            local_edits = interv.get("Local Edits") or []
            correction = interv.get("Correction") or []

            # PERFORMANCE
            if completion_type == "structure_prediction":
                metrics["performance"][mode_key]["checklist_match"].append(
                    self.processor.compare_structures(gold_rubric, sample.get("mediator_rubric"))
                )
                metrics["performance"][mode_key]["score_match"].append(
                    self.compare_scores(gold_score, sample.get("score_before_intervention"))
                )

            if completion_type == "gold_structure":
                metrics["performance"][mode_key]["score_match"].append(
                    self.compare_scores(gold_score, sample.get("score_before_intervention"))
                )

            # FAITHFULNESS
            # HSVT: baseline score vs after_hsvt score
            if hsvt is not None:
                metrics["faithfulness"][mode_key]["HSVT"].append(
                    self.compare_scores(sample.get("score_before_intervention"), hsvt.get("score_after_intervention"))
                )

            # Local: expected_score_after_intervention vs model score_after_intervention
            for item in local_edits:
                metrics["faithfulness"][mode_key]["Local Edits"].append(
                    self.compare_scores(item.get("expected_score_after_intervention"), item.get("score_after_intervention"))
                )

            # Correction: only where it exists (обычно только gold_structure)
            if len(correction) > 0:
                corr = correction[0]
                metrics["faithfulness"][mode_key]["Correction"].append(
                    self.compare_scores(gold_score, corr.get("score_before_intervention"))
                )

            # MEDIATOR–TOOL MATCH
            # predicted (до интервенций): mediator_rubric vs tool_rubric
            if self.tool_mode:
                metrics["mediator_tool_match"][mode_key]["predicted"].append(
                    self.processor.compare_structures(sample.get("mediator_rubric"), sample.get("tool_rubric"))
                )

                # HSVT
                if hsvt is not None:
                    metrics["mediator_tool_match"][mode_key]["HSVT"].append(
                        self.processor.compare_structures(hsvt.get("mediator_rubric"), hsvt.get("tool_rubric_after_intervention"))
                    )

                # Local
                for item in local_edits:
                    metrics["mediator_tool_match"][mode_key]["Local Edits"].append(
                        self.processor.compare_structures(item.get("mediator_rubric"), item.get("tool_rubric_after_intervention"))
                    )

                # Correction
                if len(correction) > 0:
                    corr = correction[0]
                    metrics["mediator_tool_match"][mode_key]["Correction"].append(
                        self.processor.compare_structures(corr.get("mediator_rubric"), corr.get("tool_rubric_after_intervention"))
                    )

        # Aggregation
        result = {
            "performance": {
                mk: {k: self.summarize(v) for k, v in metrics["performance"][mk].items()}
                for mk in ("with_gold_structure", "with_predicted_structure")
            },
            "faithfulness": {
                mk: {k: self.summarize(v) for k, v in metrics["faithfulness"][mk].items()}
                for mk in ("with_gold_structure", "with_predicted_structure")
            },
            "mediator_tool_match": {
                mk: {k: self.summarize(v) for k, v in metrics["mediator_tool_match"][mk].items()}
                for mk in ("with_gold_structure", "with_predicted_structure")
            } if self.tool_mode else {},
        }

        self.print_evaluation_metrics(result)

        return result

    def print_evaluation_metrics(self, evaluation_metrics):

        print("\nEvaluation Results:")
        print("===================")

        print("\nPerformance Metrics:")
        print("-------------------")
        for mode_key, metrics_block in evaluation_metrics["performance"].items():
            print(f"\n{mode_key}:")
            for metric_name, value in metrics_block.items():
                if value["mean"] is None:
                    print(f"  {metric_name}: mean = No, std = No (n_total={value['n_total']}, n_valid={value['n_valid']}, n_none={value['n_none']})")
                else:
                    print(f"  {metric_name}: mean = {value['mean']}, std = {value['std']} (n_total={value['n_total']}, n_valid={value['n_valid']}, n_none={value['n_none']})")

        print("\nFaithfulness Metrics:")
        print("--------------------")
        for mode_key, metrics_block in evaluation_metrics["faithfulness"].items():
            print(f"\n{mode_key}:")
            for metric_name, value in metrics_block.items():
                if value["mean"] is None:
                    print(f"  {metric_name}: mean = No, std = No (n_total={value['n_total']}, n_valid={value['n_valid']}, n_none={value['n_none']})")
                else:
                    print(f"  {metric_name}: mean = {value['mean']}, std = {value['std']} (n_total={value['n_total']}, n_valid={value['n_valid']}, n_none={value['n_none']})")

        if self.tool_mode:
            print("\nMediator–Tool Match:")
            print("--------------------")
            for mode_key, metrics_block in evaluation_metrics["mediator_tool_match"].items():
                print(f"\n{mode_key}:")
                for metric_name, value in metrics_block.items():
                    if value["mean"] is None:
                        print(f"  {metric_name}: mean = No, std = No (n_total={value['n_total']}, n_valid={value['n_valid']}, n_none={value['n_none']})")
                    else:
                        print(f"  {metric_name}: mean = {value['mean']}, std = {value['std']} (n_total={value['n_total']}, n_valid={value['n_valid']}, n_none={value['n_none']})")

