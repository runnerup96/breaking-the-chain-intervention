"""
Token-level consistency analysis for correction experiments.

Loads one or more JSON result files produced by make_correction.py
(with --save_token_metrics / --save_prompt_metrics) and generates
comparative plots of cross-entropy, max-logit, and gt-logit across:
  • Bad Structure Generation vs Corrected Structure Generation
  • Bad Structure Prompt vs Corrected Structure Prompt

Usage:
    python analysis/correction_metrics_analysis.py \
        --input_files path/to/model1.json path/to/model2.json \
        --output_dir   analysis/figures/correction_metrics
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean, pstdev as _pstdev
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────── style ────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.titlesize": 15,
    "figure.titlesize": 16,
    "text.usetex": False,
    "font.family": "serif",
})

COLORS = {
    "Bad Structure":      "#C44E52",
    "Corrected Structure": "#55A868",
    "Generation":         "#64B5CD",
    "Prompt":             "#CCB974",
}

# ──────────────────────── helpers ────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_model_name(filepath: str) -> str:
    """Extract a human-readable model name from the filename."""
    stem = Path(filepath).stem
    # e.g. 'llama32-1B_2026-02-15@18:13_bsf_one_batch' -> 'llama32-1B'
    return stem.split("_")[0]


def mean_per_sample(token_metrics_list: Optional[list], key: str) -> Optional[float]:
    """Return the mean of `key` over a single sample's token_metrics list."""
    if not token_metrics_list:
        return None
    vals = [m[key] for m in token_metrics_list if key in m]
    return _mean(vals) if vals else None


def count_tokens(token_metrics_list: Optional[list]) -> int:
    """Return the number of tokens in the token_metrics list."""
    if not token_metrics_list:
        return 0
    return len(token_metrics_list)


def count_tokens_from_skeleton(token_metrics_list: Optional[list]) -> int:
    """
    Return the number of tokens starting from ===SKELETON===.
    Uses the same logic as mean_from_skeleton to find the starting point.
    """
    if not token_metrics_list:
        return 0
    
    # Find the LAST index where ===SKELETON=== appears (same logic as mean_from_skeleton)
    skeleton_idx = None
    
    # Strategy 1: Try to find exact match for ===SKELETON=== in a single token (from end)
    for i in range(len(token_metrics_list) - 1, -1, -1):
        token_str = token_metrics_list[i].get("token", "")
        if "===SKELETON===" in token_str:
            skeleton_idx = i
            break
    
    # Strategy 2: If not found, try to find it across multiple tokens (from end, up to 10 tokens)
    if skeleton_idx is None:
        for i in range(len(token_metrics_list) - 1, -1, -1):
            for window_size in range(2, min(11, i + 2)):
                start_idx = max(0, i - window_size + 1)
                combined = "".join([token_metrics_list[j].get("token", "") 
                                   for j in range(start_idx, i + 1)])
                if "===SKELETON===" in combined:
                    skeleton_idx = start_idx
                    break
            if skeleton_idx is not None:
                break
    
    # Strategy 3: Fallback - try to find any token with "SKELETON" (case-insensitive, from end)
    if skeleton_idx is None:
        for i in range(len(token_metrics_list) - 1, -1, -1):
            token_str = token_metrics_list[i].get("token", "").upper()
            if "SKELETON" in token_str:
                skeleton_idx = i
                break
    
    # Strategy 4: Last resort - look for "===" followed by "SKELETON" nearby (from end)
    if skeleton_idx is None:
        for i in range(len(token_metrics_list) - 1, 0, -1):
            token1 = token_metrics_list[i].get("token", "")
            token2 = token_metrics_list[i - 1].get("token", "") if i > 0 else ""
            combined = token2 + token1
            if "===" in token1 and "SKELETON" in combined:
                skeleton_idx = i - 1 if "===" in token2 else i
                break
    
    if skeleton_idx is None:
        return 0
    
    # Count tokens from skeleton_idx onwards
    return len(token_metrics_list) - skeleton_idx


def mean_from_skeleton(token_metrics_list: Optional[list], key: str) -> Optional[float]:
    """
    Return the mean of `key` for tokens starting from ===SKELETON===.
    This is the changed part of the prompt where interventions occur.
    """
    if not token_metrics_list:
        return None
    
    # Find the LAST index where ===SKELETON=== appears (search from end)
    skeleton_idx = None
    
    # Strategy 1: Try to find exact match for ===SKELETON=== in a single token (from end)
    for i in range(len(token_metrics_list) - 1, -1, -1):
        token_str = token_metrics_list[i].get("token", "")
        if "===SKELETON===" in token_str:
            skeleton_idx = i
            break
    
    # Strategy 2: If not found, try to find it across multiple tokens (from end, up to 10 tokens)
    if skeleton_idx is None:
        for i in range(len(token_metrics_list) - 1, -1, -1):
            for window_size in range(2, min(11, i + 2)):
                start_idx = max(0, i - window_size + 1)
                combined = "".join([token_metrics_list[j].get("token", "") 
                                   for j in range(start_idx, i + 1)])
                if "===SKELETON===" in combined:
                    skeleton_idx = start_idx
                    break
            if skeleton_idx is not None:
                break
    
    # Strategy 3: Fallback - try to find any token with "SKELETON" (case-insensitive, from end)
    if skeleton_idx is None:
        for i in range(len(token_metrics_list) - 1, -1, -1):
            token_str = token_metrics_list[i].get("token", "").upper()
            if "SKELETON" in token_str:
                skeleton_idx = i
                break
    
    # Strategy 4: Last resort - look for "===" followed by "SKELETON" nearby (from end)
    if skeleton_idx is None:
        for i in range(len(token_metrics_list) - 1, 0, -1):
            token1 = token_metrics_list[i].get("token", "")
            token2 = token_metrics_list[i - 1].get("token", "") if i > 0 else ""
            combined = token2 + token1
            if "===" in token1 and "SKELETON" in combined:
                skeleton_idx = i - 1 if "===" in token2 else i
                break
    
    if skeleton_idx is None:
        return None
    
    # Take all tokens from skeleton_idx onwards
    changed_tokens = token_metrics_list[skeleton_idx:]
    vals = [m[key] for m in changed_tokens if key in m]
    return _mean(vals) if vals else None


# ──────────────────────── per-sample metric extraction ────────────────────────

def extract_per_sample_metrics(results: list) -> Dict[str, Dict[str, List[float]]]:
    """
    Walk through every sample in `results` and collect per-sample mean
    cross_entropy values (and other metrics) for each scenario.
    
    Returns dict:  scenario_name -> metric_name -> [per-sample mean values]
    
    Scenarios:
        bad_structure/generation, bad_structure/prompt,
        corrected_structure/generation, corrected_structure/prompt,
    """
    keys = ("cross_entropy", "max_logit", "gt_logit")
    data = defaultdict(lambda: defaultdict(list))
    token_counts = defaultdict(list)  # scenario_name -> [token counts per sample]

    for sample in results:
        # Bad structure generation
        if sample.get("bad_token_metrics"):
            num_tokens = count_tokens(sample["bad_token_metrics"])
            if num_tokens > 0:
                token_counts["bad_structure/generation"].append(num_tokens)
            for k in keys:
                v = mean_per_sample(sample["bad_token_metrics"], k)
                if v is not None:
                    data["bad_structure/generation"][k].append(v)

        # Bad structure prompt (from ===SKELETON===)
        if sample.get("bad_prompt_metrics"):
            num_tokens = count_tokens_from_skeleton(sample["bad_prompt_metrics"])
            if num_tokens > 0:
                token_counts["bad_structure/prompt"].append(num_tokens)
            for k in keys:
                v = mean_from_skeleton(sample["bad_prompt_metrics"], k)
                if v is not None:
                    data["bad_structure/prompt"][k].append(v)

        # Corrected structure generation
        correction = sample.get("structure_intervention", {}).get("correction", [])
        if correction and correction[0].get("corrected_token_metrics"):
            num_tokens = count_tokens(correction[0]["corrected_token_metrics"])
            if num_tokens > 0:
                token_counts["corrected_structure/generation"].append(num_tokens)
            for k in keys:
                v = mean_per_sample(correction[0]["corrected_token_metrics"], k)
                if v is not None:
                    data["corrected_structure/generation"][k].append(v)

        # Corrected structure prompt (from ===SKELETON===)
        if correction and correction[0].get("corrected_prompt_metrics"):
            num_tokens = count_tokens_from_skeleton(correction[0]["corrected_prompt_metrics"])
            if num_tokens > 0:
                token_counts["corrected_structure/prompt"].append(num_tokens)
            for k in keys:
                v = mean_from_skeleton(correction[0]["corrected_prompt_metrics"], k)
                if v is not None:
                    data["corrected_structure/prompt"][k].append(v)
    
    # Add token counts to data structure
    for scenario, counts in token_counts.items():
        if counts:
            data[scenario]["num_tokens"] = counts

    return data


# ──────────────────────── plotting functions ────────────────────────

def _save(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    path_png = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path_png}")


# ---------- Figure 1: Bar chart for generation cross-entropy ----------
def plot_generation_cross_entropy_bars(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """Bar chart of mean cross-entropy for generation tokens (bad vs corrected)."""
    scenarios = [
        ("bad_structure/generation", "Bad Structure"),
        ("corrected_structure/generation", "Corrected Structure"),
    ]

    labels, means, stds = [], [], []
    for key, label in scenarios:
        vals = per_sample.get(key, {}).get("cross_entropy", [])
        if vals:
            labels.append(label)
            means.append(_mean(vals))
            stds.append(_pstdev(vals) if len(vals) >= 2 else 0)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS.get(l, "#999999") for l in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Cross-Entropy")
    ax.set_title(f"Generation Cross-Entropy — {model_name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_generation_ce_bars")


# ---------- Figure 2: Bar chart for prompt cross-entropy ----------
def plot_prompt_cross_entropy_bars(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """Bar chart of mean cross-entropy for prompt tokens (from ===SKELETON===) (bad vs corrected)."""
    scenarios = [
        ("bad_structure/prompt", "Bad Structure"),
        ("corrected_structure/prompt", "Corrected Structure"),
    ]

    labels, means, stds = [], [], []
    for key, label in scenarios:
        vals = per_sample.get(key, {}).get("cross_entropy", [])
        if vals:
            labels.append(label)
            means.append(_mean(vals))
            stds.append(_pstdev(vals) if len(vals) >= 2 else 0)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS.get(l, "#999999") for l in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Cross-Entropy (from ===SKELETON===)")
    ax.set_title(f"Prompt Cross-Entropy (from ===SKELETON===) — {model_name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_prompt_ce_bars")


# ---------- Figure 3: Heatmap of all metrics ----------
def plot_metrics_heatmap(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """Heatmap: rows = scenarios, columns = metrics (CE, max_logit, gt_logit)."""
    scenario_order = [
        "bad_structure/generation",
        "corrected_structure/generation",
        "bad_structure/prompt",
        "corrected_structure/prompt",
    ]
    scenario_labels = [
        "Bad Gen",
        "Corrected Gen",
        "Bad Prompt",
        "Corrected Prompt",
    ]
    metric_names = ["cross_entropy", "max_logit", "gt_logit"]
    metric_labels = ["Cross-Entropy", "Max Logit", "GT Logit"]

    matrix = []
    used_labels = []
    for s, sl in zip(scenario_order, scenario_labels):
        row = []
        for m in metric_names:
            vals = per_sample.get(s, {}).get(m, [])
            if vals:
                row.append(_mean(vals))
            else:
                row.append(np.nan)
        if not all(np.isnan(row)):
            matrix.append(row)
            used_labels.append(sl)

    if not matrix:
        return

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(range(len(used_labels)))
    ax.set_yticklabels(used_labels)
    ax.set_title(f"Token Metrics Overview — {model_name}")

    # Add text annotations
    for i in range(len(used_labels)):
        for j in range(len(metric_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Value", rotation=270, labelpad=20)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_metrics_heatmap")


# ---------- Figure 4: Prompt vs Generation cross-entropy ----------
def plot_prompt_vs_generation(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """Grouped bar chart: prompt vs generation cross-entropy for bad and corrected."""
    groups = [
        ("Bad Structure", "bad_structure/prompt", "bad_structure/generation"),
        ("Corrected Structure", "corrected_structure/prompt", "corrected_structure/generation"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x_positions = np.arange(len(groups))

    prompt_means, gen_means = [], []
    prompt_stds, gen_stds = [], []
    group_labels = []

    for label, pkey, gkey in groups:
        p_vals = per_sample.get(pkey, {}).get("cross_entropy", [])
        g_vals = per_sample.get(gkey, {}).get("cross_entropy", [])
        if p_vals or g_vals:
            group_labels.append(label)
            prompt_means.append(_mean(p_vals) if p_vals else 0)
            prompt_stds.append(_pstdev(p_vals) if len(p_vals) >= 2 else 0)
            gen_means.append(_mean(g_vals) if g_vals else 0)
            gen_stds.append(_pstdev(g_vals) if len(g_vals) >= 2 else 0)

    if not group_labels:
        return

    x = np.arange(len(group_labels))
    ax.bar(x - width / 2, prompt_means, width, yerr=prompt_stds, capsize=4,
           label="Prompt (from ===SKELETON===)", color=COLORS["Prompt"], edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, gen_means, width, yerr=gen_stds, capsize=4,
           label="Generation", color=COLORS["Generation"], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean Cross-Entropy")
    ax.set_title(f"Prompt vs Generation Cross-Entropy — {model_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_prompt_vs_generation")


# ──────────────────────── summary table (text) ────────────────────────

def _format_scenario_name(key: str) -> str:
    """Format scenario key into a readable name."""
    parts = key.split("/")
    if len(parts) == 2:
        structure, category = parts
        structure_name = structure.replace("_", " ").title()
        category_name = category.title()
        explanation = ""
        if category_name == "Prompt":
            explanation = "(M)"
        elif category_name == "Generation":
            explanation = "(Y)"
        return f"{structure_name} {category_name} {explanation}"
    return key


def print_summary_table(per_sample: Dict[str, Dict[str, List[float]]], model_name: str):
    """Print a formatted table of aggregated metrics."""
    print(f"\n{'=' * 100}")
    print(f"  Correction Token Metrics Summary — {model_name}")
    print(f"{'=' * 100}")
    
    # Print header
    header = f"{'Scenario':<50} {'CE mean':>10} {'CE std':>10} {'MaxLogit':>12} {'GTLogit':>12} {'N':>6} {'AvgTokens':>10}"
    print(header)
    print("-" * 100)
    
    # Print scenarios
    scenario_order = [
        "bad_structure/generation",
        "bad_structure/prompt",
        "corrected_structure/generation",
        "corrected_structure/prompt",
    ]
    
    for key in scenario_order:
        metrics = per_sample.get(key, {})
        ce = metrics.get("cross_entropy", [])
        ml = metrics.get("max_logit", [])
        gt = metrics.get("gt_logit", [])
        num_tokens = metrics.get("num_tokens", [])
        n = len(ce)
        if n == 0:
            continue
        scenario_name = _format_scenario_name(key)
        avg_tokens = _mean(num_tokens) if num_tokens else 0
        row = (
            f"  {scenario_name:<48} "
            f"{_mean(ce):>10.4f} {(_pstdev(ce) if n >= 2 else 0):>10.4f} "
            f"{(_mean(ml) if ml else 0):>12.4f} "
            f"{(_mean(gt) if gt else 0):>12.4f} "
            f"{n:>6} "
            f"{avg_tokens:>10.1f}"
        )
        print(row)
    
    print(f"{'=' * 100}\n")


# ──────────────────────── main ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse token-level consistency metrics for correction experiments.")
    parser.add_argument("--input_files", nargs="+", required=True,
                        help="Path(s) to JSON result files from make_correction.py")
    parser.add_argument("--output_dir", type=str, default="analysis/figures/correction_metrics",
                        help="Directory to save figures")
    args = parser.parse_args()

    for fpath in args.input_files:
        print(f"\n{'#' * 70}")
        print(f"  Processing: {fpath}")
        print(f"{'#' * 70}")

        data = load_json(fpath)
        results = data.get("result", [])
        model_name = extract_model_name(fpath)

        if not results:
            print("  [WARN] No results found, skipping.")
            continue

        # Check if any token metrics are present
        has_token = any(s.get("bad_token_metrics") or 
                       s.get("structure_intervention", {}).get("correction", [{}])[0].get("corrected_token_metrics")
                       for s in results)
        has_prompt = any(s.get("bad_prompt_metrics") or 
                        s.get("structure_intervention", {}).get("correction", [{}])[0].get("corrected_prompt_metrics")
                        for s in results)
        if not has_token and not has_prompt:
            print("  [WARN] No token_metrics / prompt_metrics found in results.")
            # Still try to use pre-aggregated summary if available
            summary = data.get("token_metrics_summary")
            if summary:
                print("  Using pre-aggregated token_metrics_summary from JSON.")
                for scenario, metrics in sorted(summary.items()):
                    vals = {k: v for k, v in metrics.items() if v.get("mean") is not None}
                    if vals:
                        parts = [f"{k}: mean={v['mean']:.4f}" for k, v in vals.items()]
                        print(f"    {scenario}: {', '.join(parts)}")
            continue

        per_sample = extract_per_sample_metrics(results)

        print_summary_table(per_sample, model_name)

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Generate figures
        plot_generation_cross_entropy_bars(per_sample, model_name, args.output_dir)
        plot_prompt_cross_entropy_bars(per_sample, model_name, args.output_dir)
        plot_metrics_heatmap(per_sample, model_name, args.output_dir)
        plot_prompt_vs_generation(per_sample, model_name, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
