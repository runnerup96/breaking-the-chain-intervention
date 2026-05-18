import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


FAITHFULNESS_PREFIX = "eval_faithfulness/"
MEAN_SUFFIX = "/mean"
STD_SUFFIX = "/std"


def resolve_trainer_state_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        direct = os.path.join(path, "trainer_state.json")
        if os.path.isfile(direct):
            return direct
        candidates = glob.glob(os.path.join(path, "checkpoint-*", "trainer_state.json"))
        if not candidates:
            raise FileNotFoundError(f"No trainer_state.json found under {path}")

        def step_key(p):
            m = re.search(r"checkpoint-(\d+)", p)
            return int(m.group(1)) if m else -1

        return max(candidates, key=step_key)
    raise FileNotFoundError(path)


def collect_series(log_history):
    """Return {metric_key (without /mean): [(step, mean, std), ...]} for every
    `eval_faithfulness/.../mean` key present in the log."""
    series = defaultdict(list)
    for entry in log_history:
        eval_keys = [k for k in entry if k.startswith(FAITHFULNESS_PREFIX) and k.endswith(MEAN_SUFFIX)]
        if not eval_keys:
            continue
        step = entry.get("faithfulness_step", entry.get("step"))
        if step is None:
            continue
        for mean_key in eval_keys:
            base = mean_key[: -len(MEAN_SUFFIX)]
            mean_val = entry[mean_key]
            std_val = entry.get(base + STD_SUFFIX)
            if mean_val is None:
                continue
            series[base].append((step, float(mean_val), float(std_val) if std_val is not None else None))
    for k in series:
        series[k].sort(key=lambda t: t[0])
    return series


def short_label(base_key: str) -> str:
    return base_key[len(FAITHFULNESS_PREFIX):] if base_key.startswith(FAITHFULNESS_PREFIX) else base_key


def group_by_family(series):
    groups = defaultdict(dict)
    for base, points in series.items():
        label = short_label(base)
        family = label.split("/", 1)[0]
        groups[family][label] = points
    return groups


def plot(series, output_path: str, title: str):
    groups = group_by_family(series)
    families = sorted(groups.keys())
    if not families:
        raise ValueError("No eval_faithfulness/* metrics found in trainer_state.")

    fig, axes = plt.subplots(len(families), 1, figsize=(10, 4 * len(families)), squeeze=False)
    for ax, family in zip(axes[:, 0], families):
        for label, points in sorted(groups[family].items()):
            steps = [p[0] for p in points]
            means = [p[1] for p in points]
            stds = [p[2] for p in points]
            line, = ax.plot(steps, means, marker="o", linewidth=1.5, label=label)
            if any(s is not None for s in stds):
                lo = [m - (s if s is not None else 0.0) for m, s in zip(means, stds)]
                hi = [m + (s if s is not None else 0.0) for m, s in zip(means, stds)]
                ax.fill_between(steps, lo, hi, alpha=0.12, color=line.get_color())
        ax.set_title(family)
        ax.set_xlabel("step")
        ax.set_ylabel("mean")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def default_output_path(trainer_state_path: str) -> str:
    parts = os.path.normpath(trainer_state_path).split(os.sep)
    run_name = "run"
    for p in reversed(parts):
        if p and not p.startswith("checkpoint-") and p != "trainer_state.json":
            run_name = p
            break
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{run_name}_faithfulness.png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot faithfulness metrics over training steps from a HF trainer_state.json."
    )
    parser.add_argument("path", help="Path to trainer_state.json, a checkpoint dir, or a run dir "
                                     "(latest checkpoint inside is used).")
    parser.add_argument("--output", default=None, help="Output PNG path (default: analysis/figures/<run>_faithfulness.png).")
    parser.add_argument("--title", default=None, help="Plot title (default: derived from path).")
    args = parser.parse_args()

    state_path = resolve_trainer_state_path(args.path)
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    series = collect_series(state.get("log_history", []))
    output_path = args.output or default_output_path(state_path)
    title = args.title or f"faithfulness eval — {state_path}"
    plot(series, output_path, title)


if __name__ == "__main__":
    main()
