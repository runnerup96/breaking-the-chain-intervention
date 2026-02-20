from statistics import mean, pstdev
from math import isclose


class RiceChemEvaluation:
    def __init__(self, dataset):
        self.dataset = dataset
        self.idx2gold_rubric = {s["idx"]: s.get("golden_rubric") or s.get("filled_rubric") for s in dataset}
        self.idx2gold_score = {s["idx"]: s["score"] for s in dataset}

    def compare_checklists(self, a, b):
        if not a or not b: return 0
        return 1 if a == b else 0

    def compare_scores(self, gold, pred, atol=1e-3):
        if gold is None or pred is None: return 0
        return 1 if isclose(gold, pred, abs_tol=atol) else 0

    def summarize(self, lst):
        if not lst: return {"mean": None, "std": None}
        return {"mean": round(mean(lst), 3), "std": round(pstdev(lst), 3)}

    def evaluate(self, processed_samples_list):
        metrics = {
            "performance": {
                "predicted": {"checklist_match": [], "score_match": []},
                "corrected": {"score_match": []}
            },
            "faithfulness": {
                "HSVT": [],
                "Local Edits": [],
                "Correction": []
            },
            "mediator_tool_match": {   # ← НОВАЯ ГРУППА
                "predicted": [],       # до интервенции
                "HSVT": [],
                "Local Edits": [],
                "Correction": []
            }
        }

        for sample in processed_samples_list:
            idx = sample["idx"]
            gold_rubric = self.idx2gold_rubric.get(idx)
            gold_score = self.idx2gold_score.get(idx)

            # === Performance ===
            if sample.get("completion_type") == "structure_prediction":
                pred_rubric = sample.get("filled_rubric")
                pred_score = sample.get("score")
                metrics["performance"]["predicted"]["checklist_match"].append(self.compare_checklists(gold_rubric, pred_rubric))
                metrics["performance"]["predicted"]["score_match"].append(self.compare_scores(gold_score, pred_score))

            # Corrected
            corr = sample["structure_intervention"]["Correction"][0]
            corr_score = corr.get("score_after_intervention")
            metrics["performance"]["corrected"]["score_match"].append(self.compare_scores(gold_score, corr_score))

            # === Faithfulness ===
            interv = sample["structure_intervention"]

            for name in ["HSVT", "Local Edits", "Correction"]:
                items = interv[name] if name != "HSVT" else [interv[name][0]]
                for item in items:
                    metrics["faithfulness"][name].append(
                        self.compare_scores(item.get("score"), item.get("score_after_intervention"))
                    )

            # === НОВАЯ МЕТРИКА: mediator-tool match ===
            # Predicted (до интервенции)
            metrics["mediator_tool_match"]["predicted"].append(
                self.compare_checklists(
                    sample.get("mediator_rubric_before"),
                    sample.get("tool_rubric_before")
                )
            )

            for name in ["HSVT", "Local Edits", "Correction"]:
                items = interv[name] if name != "HSVT" else [interv[name][0]]
                for item in items:
                    metrics["mediator_tool_match"][name].append(
                        self.compare_checklists(
                            item.get("mediator_rubric_after"),
                            item.get("tool_rubric_after")
                        )
                    )

        # Агрегация
        result = {
            "performance": {
                k: {kk: self.summarize(vv) for kk, vv in v.items()}
                for k, v in metrics["performance"].items()
            },
            "faithfulness": {k: self.summarize(v) for k, v in metrics["faithfulness"].items()},
            "mediator_tool_match": {k: self.summarize(v) for k, v in metrics["mediator_tool_match"].items()}
        }

        self._print(result)
        return result

    def _print(self, m):
        print("\n=== RiceChem Evaluation ===")
        print(f"Mediator-Tool Match (predicted): {m['mediator_tool_match']['predicted']}")
        for name in ["HSVT", "Local Edits", "Correction"]:
            print(f"  {name:12} : mean = {m['mediator_tool_match'][name]['mean']:.3f} ± {m['mediator_tool_match'][name]['std']:.3f}")
        print("="*60)


    def print_evaluation_metrics(self, evaluation_metrics):
        print("\nEvaluation Results (RiceChem Correction):")
        print("========================================")

        print("\nPerformance:")
        for structure_type, metrics in evaluation_metrics["performance"].items():
            print(f"\n{structure_type}:")
            for metric_name, value in metrics.items():
                if None not in value.values():
                    print(f"  {metric_name}: mean = {value['mean']}, std = {value['std']}")
                else:
                    print(f"  {metric_name}: mean = No, std = No")

        print("\nFaithfulness:")
        for metric_name, value in evaluation_metrics["faithfulness"].items():
            if None not in value.values():
                print(f"  {metric_name}: mean = {value['mean']}, std = {value['std']}")
            else:
                print(f"  {metric_name}: mean = No, std = No")
