"""
Token-level consistency analysis for intervention experiments.

Loads one or more JSON result files produced by make_intervention.py
(with --save_token_metrics / --save_prompt_metrics) and generates
comparative plots of cross-entropy, max-logit, and gt-logit across:
  • gold_structure  vs  predicted_structure
  • base generation vs  HSVT / Local Edits / Global interventions
  • prompt tokens   vs  generation tokens

Usage:
    python analysis/token_metrics_analysis.py \
        --input_files path/to/model1.json path/to/model2.json \
        --output_dir   analysis/figures/token_metrics
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean, pstdev as _pstdev
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None

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
    "Base":         "#4C72B0",
    "HSVT":         "#55A868",
    "Local Edits":  "#C44E52",
    "Global":       "#8172B2",
    "Prompt":       "#CCB974",
    "Generation":   "#64B5CD",
}

SCENARIO_LABELS = {
    "gold_structure":      "Gold Structure",
    "structure_prediction": "Predicted Structure",
    "predicted_structure":  "Predicted Structure",
}

# ──────────────────────── helpers ────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mean(values: list) -> Optional[float]:
    return _mean(values) if values else None


def _safe_std(values: list) -> Optional[float]:
    return _pstdev(values) if len(values) >= 2 else 0.0 if len(values) == 1 else None


def extract_model_name(filepath: str) -> str:
    """Extract a human-readable model name from the filename."""
    stem = Path(filepath).stem
    # e.g. 'llama32-1B_2026-02-15@18:13_bsf_one_batch' -> 'llama32-1B'
    return stem.split("_")[0]


# ──────────────────────── per-sample metric extraction ────────────────────────

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


def mean_last_fraction(token_metrics_list: Optional[list], key: str, fraction: float = 0.25) -> Optional[float]:
    """
    Return the mean of `key` over the last `fraction` of tokens in the list.
    Useful for measuring CE only for the changed part of the prompt.
    """
    if not token_metrics_list:
        return None
    n_tokens = len(token_metrics_list)
    if n_tokens == 0:
        return None
    start_idx = max(0, int(n_tokens * (1 - fraction)))
    last_tokens = token_metrics_list[start_idx:]
    vals = [m[key] for m in last_tokens if key in m]
    return _mean(vals) if vals else None


def mean_from_skeleton(token_metrics_list: Optional[list], key: str, search_from_start: bool = False) -> Optional[float]:
    """
    Return the mean of `key` for tokens starting from ===SKELETON===.
    This is the changed part of the prompt where interventions occur.
    
    Args:
        token_metrics_list: List of token metrics
        key: Metric key to extract (e.g., "cross_entropy")
        search_from_start: If True, search from the beginning; if False, search from the end (default)
    """
    if not token_metrics_list:
        return None
    
    skeleton_idx = None
    
    if search_from_start:
        # Search from the beginning (for base generation in predicted_structure)
        # Strategy 1: Try to find exact match for ===SKELETON=== in a single token (from start)
        for i in range(len(token_metrics_list)):
            token_str = token_metrics_list[i].get("token", "")
            if "===SKELETON===" in token_str:
                skeleton_idx = i
                break
        
        # Strategy 2: If not found, try to find it across multiple tokens (from start, up to 10 tokens)
        if skeleton_idx is None:
            for i in range(len(token_metrics_list)):
                for window_size in range(2, min(11, len(token_metrics_list) - i + 1)):
                    end_idx = min(len(token_metrics_list), i + window_size)
                    combined = "".join([token_metrics_list[j].get("token", "") 
                                       for j in range(i, end_idx)])
                    if "===SKELETON===" in combined:
                        skeleton_idx = i
                        break
                if skeleton_idx is not None:
                    break
        
        # Strategy 3: Fallback - try to find any token with "SKELETON" (case-insensitive, from start)
        if skeleton_idx is None:
            for i in range(len(token_metrics_list)):
                token_str = token_metrics_list[i].get("token", "").upper()
                if "SKELETON" in token_str:
                    skeleton_idx = i
                    break
        
        # Strategy 4: If still not found from start, try searching from end as fallback
        if skeleton_idx is None:
            for i in range(len(token_metrics_list) - 1, -1, -1):
                token_str = token_metrics_list[i].get("token", "")
                if "===SKELETON===" in token_str:
                    skeleton_idx = i
                    break
            
            # Try across multiple tokens from end
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
            
            # Last fallback - any token with "SKELETON" from end
            if skeleton_idx is None:
                for i in range(len(token_metrics_list) - 1, -1, -1):
                    token_str = token_metrics_list[i].get("token", "").upper()
                    if "SKELETON" in token_str:
                        skeleton_idx = i
                        break
    else:
        # Search from the end (default, for prompts and interventions)
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
    
    # Last resort: search anywhere in the sequence (first occurrence)
    if skeleton_idx is None:
        # Try to find ===SKELETON=== anywhere
        for i in range(len(token_metrics_list)):
            token_str = token_metrics_list[i].get("token", "")
            if "===SKELETON===" in token_str:
                skeleton_idx = i
                break
        
        # Try across multiple tokens anywhere
        if skeleton_idx is None:
            for i in range(len(token_metrics_list)):
                for window_size in range(2, min(11, len(token_metrics_list) - i + 1)):
                    end_idx = min(len(token_metrics_list), i + window_size)
                    combined = "".join([token_metrics_list[j].get("token", "") 
                                       for j in range(i, end_idx)])
                    if "===SKELETON===" in combined:
                        skeleton_idx = i
                        break
                if skeleton_idx is not None:
                    break
        
        # Final fallback: any token with "SKELETON" anywhere
        if skeleton_idx is None:
            for i in range(len(token_metrics_list)):
                token_str = token_metrics_list[i].get("token", "").upper()
                if "SKELETON" in token_str:
                    skeleton_idx = i
                    break
    
    if skeleton_idx is None:
        return None
    
    # Take all tokens from skeleton_idx onwards
    changed_tokens = token_metrics_list[skeleton_idx:]
    vals = [m[key] for m in changed_tokens if key in m]
    return _mean(vals) if vals else None


def extract_per_sample_metrics(results: list) -> Dict[str, Dict[str, List[float]]]:
    """
    Walk through every sample in `results` and collect per-sample mean
    cross_entropy values (and other metrics) for each scenario.

    Returns dict:  scenario_name -> metric_name -> [per-sample mean values]
    
    Scenarios:
        generation/gold_structure, generation/predicted_structure,
        prompt/gold_structure (only for gold_structure, no prompt/predicted_structure because there's no ===SKELETON=== in predicted_structure prompts),
        HSVT/generation/gold_structure, HSVT/generation/predicted_structure,
        HSVT/prompt/gold_structure (only for gold_structure base samples),
        Local Edits/generation/gold_structure, Local Edits/generation/predicted_structure,
        Local Edits/prompt/gold_structure (only for gold_structure base samples),
        Global/generation/gold_structure, Global/generation/predicted_structure,
        Global/prompt/gold_structure (only for gold_structure base samples),
    """
    keys = ("cross_entropy", "max_logit", "gt_logit")
    data = defaultdict(lambda: defaultdict(list))
    token_counts = defaultdict(list)  # scenario_name -> [token counts per sample]

    for sample in results:
        ct = sample.get("completion_type", "")
        suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"

        # --- Main generation ---
        if sample.get("token_metrics"):
            num_tokens = count_tokens(sample["token_metrics"])
            if num_tokens > 0:
                token_counts[f"generation/{suffix}"].append(num_tokens)
            for k in keys:
                v = mean_per_sample(sample["token_metrics"], k)
                if v is not None:
                    data[f"generation/{suffix}"][k].append(v)

        # --- Main prompt (from ===SKELETON=== only) ---
        # Only measure prompt metrics for gold_structure (where ===SKELETON=== is in the prompt)
        # For predicted_structure, there's no ===SKELETON=== in the prompt, so skip it
        if ct == "gold_structure" and sample.get("prompt_metrics"):
            num_tokens = count_tokens_from_skeleton(sample["prompt_metrics"])
            if num_tokens > 0:
                token_counts[f"prompt/gold_structure"].append(num_tokens)
            for k in keys:
                v = mean_from_skeleton(sample["prompt_metrics"], k)
                if v is not None:
                    data[f"prompt/gold_structure"][k].append(v)

        # --- Interventions (separated by completion_type) ---
        si = sample.get("structure_intervention", {})
        for itype in ("HSVT", "Local Edits", "Global"):
            for interv in si.get(itype, []):
                if interv.get("token_metrics"):
                    num_tokens = count_tokens(interv["token_metrics"])
                    if num_tokens > 0:
                        token_counts[f"{itype}/generation/{suffix}"].append(num_tokens)
                    for k in keys:
                        v = mean_per_sample(interv["token_metrics"], k)
                        if v is not None:
                            data[f"{itype}/generation/{suffix}"][k].append(v)
                # For interventions, prompt always has ===SKELETON=== (include_gold_structure=True)
                # But we only measure prompt metrics for gold_structure base samples
                # (interventions for predicted_structure samples still have ===SKELETON=== in prompt,
                # but we skip measuring them for consistency)
                if ct == "gold_structure" and interv.get("prompt_metrics"):
                    num_tokens = count_tokens_from_skeleton(interv["prompt_metrics"])
                    if num_tokens > 0:
                        token_counts[f"{itype}/prompt/gold_structure"].append(num_tokens)
                    for k in keys:
                        # Use mean_from_skeleton for prompt metrics (from ===SKELETON===)
                        v = mean_from_skeleton(interv["prompt_metrics"], k)
                        if v is not None:
                            data[f"{itype}/prompt/gold_structure"][k].append(v)
    
    # Add token counts to data structure
    for scenario, counts in token_counts.items():
        if counts:
            data[scenario]["num_tokens"] = counts

    return data


def extract_changed_prompt_metrics(results: list) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract metrics only for the changed part of the prompt (starting from ===SKELETON===).
    This focuses on the mediator part where interventions occur.
    
    Returns dict:  scenario_name -> metric_name -> [per-sample mean values for changed part]
    """
    keys = ("cross_entropy", "max_logit", "gt_logit")
    data = defaultdict(lambda: defaultdict(list))

    for sample in results:
        ct = sample.get("completion_type", "")
        suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"

        # --- Main prompt (changed part only, from ===SKELETON===) ---
        # Only measure prompt metrics for gold_structure (where ===SKELETON=== is in the prompt)
        # For predicted_structure, there's no ===SKELETON=== in the prompt, so skip it
        if ct == "gold_structure" and sample.get("prompt_metrics"):
            for k in keys:
                v = mean_from_skeleton(sample["prompt_metrics"], k)
                if v is not None:
                    data[f"prompt/gold_structure"][k].append(v)

        # --- Interventions (changed part only, from ===SKELETON===, separated by completion_type) ---
        si = sample.get("structure_intervention", {})
        for itype in ("HSVT", "Local Edits", "Global"):
            for interv in si.get(itype, []):
                if interv.get("prompt_metrics"):
                    for k in keys:
                        v = mean_from_skeleton(interv["prompt_metrics"], k)
                        if v is not None:
                            data[f"{itype}/prompt/{suffix}"][k].append(v)

    return data


def extract_all_token_values(results: list, key: str = "cross_entropy") -> Dict[str, List[float]]:
    """
    Collect *all individual token* values for `key` (not per-sample means).
    Useful for histograms of token-level distributions.
    """
    data = defaultdict(list)

    for sample in results:
        ct = sample.get("completion_type", "")
        suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"

        if sample.get("token_metrics"):
            for m in sample["token_metrics"]:
                if key in m:
                    data[f"generation/{suffix}"].append(m[key])

        if sample.get("prompt_metrics"):
            for m in sample["prompt_metrics"]:
                if key in m:
                    data[f"prompt/{suffix}"].append(m[key])

        si = sample.get("structure_intervention", {})
        for itype in ("HSVT", "Local Edits", "Global"):
            for interv in si.get(itype, []):
                if interv.get("token_metrics"):
                    for m in interv["token_metrics"]:
                        if key in m:
                            data[f"{itype}/generation"].append(m[key])
                if interv.get("prompt_metrics"):
                    for m in interv["prompt_metrics"]:
                        if key in m:
                            data[f"{itype}/prompt"].append(m[key])

    return data


# ──────────────────────── plotting functions ────────────────────────

def _save(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    path_png = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path_png}")


# ---------- Figure 1: Bar chart comparing mean cross-entropy across scenarios ----------
def plot_generation_cross_entropy_bars(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """
    Bar chart of mean cross-entropy for base generation (gold vs predicted)
    and each intervention type, separated by completion_type.
    """
    scenarios_gen = [
        ("generation/gold_structure",           "Base (Gold)"),
        ("generation/predicted_structure",      "Base (Predicted)"),
        ("HSVT/generation/gold_structure",      "HSVT (Gold)"),
        ("HSVT/generation/predicted_structure", "HSVT (Predicted)"),
        ("Local Edits/generation/gold_structure",      "Local Edits (Gold)"),
        ("Local Edits/generation/predicted_structure", "Local Edits (Predicted)"),
        ("Global/generation/gold_structure",      "Global (Gold)"),
        ("Global/generation/predicted_structure", "Global (Predicted)"),
    ]

    labels, means, stds = [], [], []
    for key, label in scenarios_gen:
        vals = per_sample.get(key, {}).get("cross_entropy", [])
        if vals:
            labels.append(label)
            means.append(_mean(vals))
            stds.append(_pstdev(vals) if len(vals) >= 2 else 0)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS.get(l.split(" (")[0], "#999999") for l in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean Cross-Entropy (per sample)")
    ax.set_title(f"Generation Cross-Entropy — {model_name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_generation_ce_bars")


# ---------- Figure 1b: Bar chart for prompt cross-entropy ----------
def plot_prompt_cross_entropy_bars(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """
    Bar chart of mean cross-entropy for prompt tokens (from ===SKELETON===)
    for gold_structure and each intervention type.
    Note: predicted_structure is not included because there's no ===SKELETON=== in the prompt.
    """
    scenarios_prompt = [
        ("prompt/gold_structure",           "Base (Gold)"),
        ("HSVT/prompt/gold_structure",      "HSVT (Gold)"),
        ("Local Edits/prompt/gold_structure",      "Local Edits (Gold)"),
        ("Global/prompt/gold_structure",      "Global (Gold)"),
    ]

    labels, means, stds = [], [], []
    for key, label in scenarios_prompt:
        vals = per_sample.get(key, {}).get("cross_entropy", [])
        if vals:
            labels.append(label)
            means.append(_mean(vals))
            stds.append(_pstdev(vals) if len(vals) >= 2 else 0)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS.get(l.split(" (")[0], "#999999") for l in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean Cross-Entropy (from ===SKELETON===)")
    ax.set_title(f"Prompt Cross-Entropy (from ===SKELETON===) — {model_name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_prompt_ce_bars")


# ---------- Figure 1c: Bar chart for changed part of prompt cross-entropy ----------
def plot_changed_prompt_cross_entropy_bars(
    changed_prompt_metrics: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
    fraction: float = 0.25,
):
    """
    Bar chart of mean cross-entropy for the CHANGED part of prompt tokens only
    (last `fraction` of tokens, where interventions occur).
    This shows how much the model is "surprised" by the specific changes.
    Note: predicted_structure is not included because there's no ===SKELETON=== in the prompt.
    """
    scenarios_prompt = [
        ("prompt/gold_structure",           "Base (Gold)"),
        ("HSVT/prompt/gold_structure",      "HSVT (Gold)"),
        ("Local Edits/prompt/gold_structure",      "Local Edits (Gold)"),
        ("Global/prompt/gold_structure",      "Global (Gold)"),
    ]

    labels, means, stds = [], [], []
    for key, label in scenarios_prompt:
        vals = changed_prompt_metrics.get(key, {}).get("cross_entropy", [])
        if vals:
            labels.append(label)
            means.append(_mean(vals))
            stds.append(_pstdev(vals) if len(vals) >= 2 else 0)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS.get(l.split(" (")[0], "#999999") for l in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean Cross-Entropy (changed part only)")
    ax.set_title(f"Prompt Cross-Entropy (Last {int(fraction*100)}% Tokens) — {model_name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_prompt_changed_part_ce_bars")


# ---------- Figure 2: Prompt vs Generation cross-entropy ----------
def plot_prompt_vs_generation(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """
    Grouped bar chart: prompt vs generation cross-entropy for gold structure.
    Note: predicted_structure is not included because there's no ===SKELETON=== in the prompt.
    """
    groups = [
        ("Gold Structure",      "prompt/gold_structure",      "generation/gold_structure"),
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
           label="Prompt", color=COLORS["Prompt"], edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, gen_means, width, yerr=gen_stds, capsize=4,
           label="Generation", color=COLORS["Generation"], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean Cross-Entropy")
    ax.set_title(f"Prompt (from ===SKELETON===) vs Generation Cross-Entropy — {model_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_prompt_vs_generation_ce")


# ---------- Figure 3: Box-plots of per-sample mean CE across scenarios ----------
def plot_boxplots(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """
    Box-plots of per-sample mean cross-entropy for all generation scenarios.
    """
    scenarios = [
        ("generation/gold_structure",      "Base\n(Gold)"),
        ("generation/predicted_structure",  "Base\n(Predicted)"),
        ("HSVT/generation",                "HSVT"),
        ("Local Edits/generation",         "Local\nEdits"),
        ("Global/generation",              "Global"),
    ]

    box_data, box_labels = [], []
    for key, label in scenarios:
        vals = per_sample.get(key, {}).get("cross_entropy", [])
        if vals:
            box_data.append(vals)
            box_labels.append(label)

    if not box_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))

    colors_list = [COLORS.get(l.strip().split("\n")[0], "#999999") for l in box_labels]
    for patch, c in zip(bp["boxes"], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_ylabel("Per-sample Mean Cross-Entropy")
    ax.set_title(f"Cross-Entropy Distribution — {model_name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_ce_boxplots")


# ---------- Figure 4: Histograms of token-level CE ----------
def plot_token_histograms(
    results: list,
    model_name: str,
    output_dir: str,
):
    """
    Overlaid histograms of *token-level* cross-entropy for base vs interventions.
    """
    all_tokens = extract_all_token_values(results, key="cross_entropy")

    scenarios = [
        ("generation/gold_structure",      "Base (Gold)"),
        ("HSVT/generation",                "HSVT"),
        ("Local Edits/generation",         "Local Edits"),
        ("Global/generation",              "Global"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for key, label in scenarios:
        vals = all_tokens.get(key, [])
        if vals:
            ax.hist(vals, bins=80, alpha=0.45, label=f"{label} (n={len(vals)})",
                    density=True, edgecolor="none")

    ax.set_xlabel("Cross-Entropy")
    ax.set_ylabel("Density")
    ax.set_title(f"Token-level Cross-Entropy Distribution — {model_name}")
    ax.legend()
    ax.set_xlim(0, min(20, ax.get_xlim()[1]))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_token_ce_histogram")


# ---------- Figure 5: Delta CE (intervention − base) ----------
def plot_delta_cross_entropy(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """
    For each sample, compute delta = intervention_CE − base_CE.
    Plot the distribution of deltas for each intervention type.
    Positive delta → model is more surprised after intervention.
    """
    base_gold = per_sample.get("generation/gold_structure", {}).get("cross_entropy", [])
    base_pred = per_sample.get("generation/predicted_structure", {}).get("cross_entropy", [])

    interventions = [
        ("HSVT/generation",        "HSVT"),
        ("Local Edits/generation", "Local Edits"),
        ("Global/generation",      "Global"),
    ]

    fig, axes = plt.subplots(1, len(interventions), figsize=(5 * len(interventions), 5), sharey=True)
    if len(interventions) == 1:
        axes = [axes]

    for ax, (ikey, ilabel) in zip(axes, interventions):
        i_vals = per_sample.get(ikey, {}).get("cross_entropy", [])

        # Use gold or predicted baseline — whichever is available & matches length
        base = base_gold if base_gold else base_pred
        if not base or not i_vals:
            ax.set_title(f"{ilabel}\n(no data)")
            continue

        n = min(len(base), len(i_vals))
        deltas = [i_vals[j] - base[j] for j in range(n)]

        ax.hist(deltas, bins=40, alpha=0.7, color=COLORS.get(ilabel, "#999"),
                edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        if deltas:
            ax.axvline(_mean(deltas), color="red", linestyle="-", linewidth=1.0,
                       label=f"mean={_mean(deltas):.3f}")
        ax.set_xlabel("ΔCE (intervention − base)")
        ax.set_title(ilabel)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Count")
    fig.suptitle(f"Cross-Entropy Change After Intervention — {model_name}", fontsize=14)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_delta_ce")


# ---------- Figure 6: Multi-model bar comparison ----------
def plot_multi_model_comparison(
    model_data: Dict[str, Dict[str, Dict[str, List[float]]]],
    output_dir: str,
):
    """
    If multiple models are provided, show grouped bar chart comparing
    mean generation cross-entropy across models for each scenario.
    """
    if len(model_data) < 2:
        return

    scenarios = [
        ("generation/gold_structure",     "Base (Gold)"),
        ("generation/predicted_structure", "Base (Predicted)"),
        ("HSVT/generation",               "HSVT"),
        ("Local Edits/generation",        "Local Edits"),
        ("Global/generation",             "Global"),
    ]

    model_names = list(model_data.keys())
    n_models = len(model_names)
    n_scenarios = len(scenarios)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_scenarios)

    for i, mname in enumerate(model_names):
        ps = model_data[mname]
        means = []
        stds = []
        for key, _ in scenarios:
            vals = ps.get(key, {}).get("cross_entropy", [])
            means.append(_mean(vals) if vals else 0)
            stds.append(_pstdev(vals) if len(vals) >= 2 else 0)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=mname,
               edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in scenarios], rotation=15, ha="right")
    ax.set_ylabel("Mean Cross-Entropy")
    ax.set_title("Cross-Entropy Comparison Across Models")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "multi_model_ce_comparison")


# ---------- Figure 7: Heatmap of all metrics × all scenarios ----------
def plot_metrics_heatmap(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """
    Heatmap: rows = scenarios, columns = metrics (CE, max_logit, gt_logit).
    Cell values = mean over samples.
    """
    scenario_order = [
        "generation/gold_structure",
        "generation/predicted_structure",
        "HSVT/generation/gold_structure",
        "HSVT/generation/predicted_structure",
        "Local Edits/generation/gold_structure",
        "Local Edits/generation/predicted_structure",
        "Global/generation/gold_structure",
        "Global/generation/predicted_structure",
        "prompt/gold_structure",
        "HSVT/prompt/gold_structure",
        "Local Edits/prompt/gold_structure",
        "Global/prompt/gold_structure",
    ]
    scenario_labels = [
        "Gen (Gold)", "Gen (Predicted)",
        "HSVT Gen (Gold)", "HSVT Gen (Predicted)",
        "Local Edits Gen (Gold)", "Local Edits Gen (Predicted)",
        "Global Gen (Gold)", "Global Gen (Predicted)",
        "Prompt (Gold)",
        "HSVT Prompt (Gold)",
        "Local Edits Prompt (Gold)",
        "Global Prompt (Gold)",
    ]
    metric_names = ["cross_entropy", "max_logit", "gt_logit"]
    metric_labels = ["Cross-Entropy", "Max Logit", "GT Logit"]

    matrix = []
    used_labels = []
    for s, sl in zip(scenario_order, scenario_labels):
        row = []
        has_data = False
        for m in metric_names:
            vals = per_sample.get(s, {}).get(m, [])
            if vals:
                row.append(_mean(vals))
                has_data = True
            else:
                row.append(np.nan)
        if has_data:
            matrix.append(row)
            used_labels.append(sl)

    if not matrix:
        return

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, max(4, len(used_labels) * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(np.arange(len(used_labels)))
    ax.set_yticklabels(used_labels)

    # Annotate cells
    for i in range(len(used_labels)):
        for j in range(len(metric_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if val > np.nanmean(matrix) else "black")

    ax.set_title(f"Token Metrics Overview — {model_name}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_metrics_heatmap")


# ---------- Figure 8: Scatter: base CE vs intervention CE ----------
def plot_scatter_base_vs_intervention(
    per_sample: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: str,
):
    """
    Scatter plot: x = base generation CE, y = intervention generation CE.
    Points above y=x → intervention increased surprise.
    """
    base_gold = per_sample.get("generation/gold_structure", {}).get("cross_entropy", [])
    base_pred = per_sample.get("generation/predicted_structure", {}).get("cross_entropy", [])
    base = base_gold if base_gold else base_pred
    if not base:
        return

    interventions = [
        ("HSVT/generation",        "HSVT"),
        ("Local Edits/generation", "Local Edits"),
        ("Global/generation",      "Global"),
    ]

    fig, ax = plt.subplots(figsize=(7, 7))

    for ikey, ilabel in interventions:
        i_vals = per_sample.get(ikey, {}).get("cross_entropy", [])
        if not i_vals:
            continue
        n = min(len(base), len(i_vals))
        ax.scatter(base[:n], i_vals[:n], alpha=0.5, s=20,
                   color=COLORS.get(ilabel, "#999"), label=ilabel)

    # y=x reference line
    lim_max = ax.get_xlim()[1]
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Base Generation CE")
    ax.set_ylabel("Intervention Generation CE")
    ax.set_title(f"Base vs Intervention CE — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _save(fig, output_dir, f"{model_name}_scatter_base_vs_intervention")


# ──────────────────────── tokenized text extraction ────────────────────────

def extract_tokenized_text(metrics_list: Optional[list]) -> Optional[str]:
    """
    Extract tokenized text from metrics list by concatenating all tokens.
    """
    if not metrics_list:
        return None
    tokens = [m.get("token", "") for m in metrics_list]
    return "".join(tokens)


def extract_tokenized_text_from_skeleton(metrics_list: Optional[list]) -> Optional[str]:
    """
    Extract tokenized text starting from the LAST ===SKELETON===.
    This avoids confusion with few-shot examples that also contain ===SKELETON===.
    Returns None if ===SKELETON=== is not found.
    """
    if not metrics_list:
        return None
    
    # Find the LAST index where ===SKELETON=== appears (search from end)
    skeleton_idx = None
    
    # Strategy 1: Try to find exact match for ===SKELETON=== in a single token (from end)
    for i in range(len(metrics_list) - 1, -1, -1):
        token_str = metrics_list[i].get("token", "")
        if "===SKELETON===" in token_str:
            skeleton_idx = i
            break
    
    # Strategy 2: If not found, try to find it across multiple tokens (from end, up to 10 tokens)
    if skeleton_idx is None:
        for i in range(len(metrics_list) - 1, -1, -1):
            # Try different window sizes
            for window_size in range(2, min(11, i + 2)):
                start_idx = max(0, i - window_size + 1)
                combined = "".join([metrics_list[j].get("token", "") 
                                   for j in range(start_idx, i + 1)])
                if "===SKELETON===" in combined:
                    skeleton_idx = start_idx
                    break
            if skeleton_idx is not None:
                break
    
    # Strategy 3: Fallback - try to find any token with "SKELETON" (case-insensitive, from end)
    if skeleton_idx is None:
        for i in range(len(metrics_list) - 1, -1, -1):
            token_str = metrics_list[i].get("token", "").upper()
            if "SKELETON" in token_str:
                skeleton_idx = i
                break
    
    # Strategy 4: Last resort - look for "===" followed by "SKELETON" nearby (from end)
    if skeleton_idx is None:
        for i in range(len(metrics_list) - 1, 0, -1):
            token1 = metrics_list[i].get("token", "")
            token2 = metrics_list[i - 1].get("token", "") if i > 0 else ""
            combined = token2 + token1
            if "===" in token1 and "SKELETON" in combined:
                skeleton_idx = i - 1 if "===" in token2 else i
                break
    
    if skeleton_idx is None:
        # If still not found, return None
        return None
    
    # Take all tokens from skeleton_idx onwards
    tokens = [m.get("token", "") for m in metrics_list[skeleton_idx:]]
    return "".join(tokens)


def extract_all_tokenized_texts(results: list) -> List[Dict]:
    """
    Extract tokenized prompt and generation text for each sample.
    Returns a list of dicts with sample info and tokenized texts.
    """
    tokenized_samples = []
    
    for sample in results:
        sample_data = {
            "index": sample.get("index"),
            "completion_type": sample.get("completion_type"),
        }
        
        completion_type = sample.get("completion_type", "")
        
        # Main prompt and generation
        if completion_type == "gold_structure":
            # For gold_structure, prompt should start with ===SKELETON===
            if sample.get("prompt_metrics"):
                sample_data["prompt"] = extract_tokenized_text_from_skeleton(sample["prompt_metrics"])
            else:
                sample_data["prompt"] = None
        elif completion_type == "structure_prediction":
            # For structure_prediction, there should be no prompt with ===SKELETON===
            # (model generates structure itself)
            sample_data["prompt"] = None
        else:
            # Fallback: use full prompt
            if sample.get("prompt_metrics"):
                sample_data["prompt"] = extract_tokenized_text(sample["prompt_metrics"])
            else:
                sample_data["prompt"] = None
            
        if sample.get("token_metrics"):
            sample_data["generation"] = extract_tokenized_text(sample["token_metrics"])
        else:
            sample_data["generation"] = None
        
        # Intervention prompts and generations
        # All interventions use include_gold_structure=True, so they should start with ===SKELETON===
        si = sample.get("structure_intervention", {})
        interventions_data = {}
        
        for itype in ("HSVT", "Local Edits", "Global"):
            interventions_data[itype] = []
            for interv in si.get(itype, []):
                interv_data = {}
                if interv.get("prompt_metrics"):
                    # Try to extract from ===SKELETON===
                    prompt_text = extract_tokenized_text_from_skeleton(interv["prompt_metrics"])
                    # If not found, use full prompt as fallback (for debugging)
                    if prompt_text is None:
                        prompt_text = extract_tokenized_text(interv["prompt_metrics"])
                    interv_data["prompt"] = prompt_text
                else:
                    interv_data["prompt"] = None
                if interv.get("token_metrics"):
                    interv_data["generation"] = extract_tokenized_text(interv["token_metrics"])
                else:
                    interv_data["generation"] = None
                interventions_data[itype].append(interv_data)
        
        sample_data["interventions"] = interventions_data
        tokenized_samples.append(sample_data)
    
    return tokenized_samples


# ──────────────────────── entropy vs token position plot ────────────────────────

def find_marker_index(metrics_list: Optional[list], marker: str) -> Optional[int]:
    """
    Find the index of the LAST occurrence of a marker (e.g., ===SKELETON===, ===SCHEMA_LINKS===, ===SLOT_MATCHING===)
    in the metrics list. Returns None if not found.
    """
    if not metrics_list:
        return None
    
    # Normalize marker (remove === if present, for flexible matching)
    marker_clean = marker.replace("===", "").upper()
    marker_full = marker if marker.startswith("===") else f"==={marker}==="
    
    # Strategy 1: Try to find exact match in a single token (from end)
    for i in range(len(metrics_list) - 1, -1, -1):
        token_str = metrics_list[i].get("token", "")
        if marker_full in token_str:
            return i
    
    # Strategy 2: If not found, try to find it across multiple tokens (from end, up to 10 tokens)
    for i in range(len(metrics_list) - 1, -1, -1):
        for window_size in range(2, min(11, i + 2)):
            start_idx = max(0, i - window_size + 1)
            combined = "".join([metrics_list[j].get("token", "") 
                               for j in range(start_idx, i + 1)])
            if marker_full in combined:
                return start_idx
    
    # Strategy 3: Fallback - try to find any token with marker text (case-insensitive, from end)
    for i in range(len(metrics_list) - 1, -1, -1):
        token_str = metrics_list[i].get("token", "").upper()
        if marker_clean in token_str:
            return i
    
    # Strategy 4: Last resort - look for "===" followed by marker text nearby (from end)
    for i in range(len(metrics_list) - 1, 0, -1):
        token1 = metrics_list[i].get("token", "")
        token2 = metrics_list[i - 1].get("token", "") if i > 0 else ""
        combined = token2 + token1
        if "===" in token1 and marker_clean in combined.upper():
            return i - 1 if "===" in token2 else i
    
    return None


def find_skeleton_start_index(metrics_list: Optional[list]) -> Optional[int]:
    """
    Find the index of the LAST ===SKELETON=== in the metrics list.
    Returns None if not found.
    """
    return find_marker_index(metrics_list, "===SKELETON===")


def extract_combined_sequence(prompt_metrics: Optional[list], token_metrics: Optional[list]) -> Tuple[List[float], Dict[str, Optional[int]], Optional[int]]:
    """
    Combine prompt_metrics and token_metrics into a single sequence of cross_entropy values.
    Returns:
        - List of cross_entropy values (prompt + generation)
        - Dictionary with marker indices: {"skeleton": idx, "schema_links": idx, "slot_matching": idx}
          (indices are relative to combined sequence, None if not found)
        - Index where generation starts (relative to combined sequence)
    """
    combined_ce = []
    marker_indices = {"skeleton": None, "schema_links": None, "slot_matching": None}
    generation_start_idx = None
    
    # Add prompt metrics
    if prompt_metrics:
        # Find all marker positions in prompt
        skeleton_idx = find_marker_index(prompt_metrics, "===SKELETON===")
        schema_links_idx = find_marker_index(prompt_metrics, "===SCHEMA_LINKS===")
        slot_matching_idx = find_marker_index(prompt_metrics, "===SLOT_MATCHING===")
        
        for m in prompt_metrics:
            if "cross_entropy" in m:
                combined_ce.append(m["cross_entropy"])
        
        # Convert marker indices to combined sequence indices
        if skeleton_idx is not None:
            marker_indices["skeleton"] = skeleton_idx
        if schema_links_idx is not None:
            marker_indices["schema_links"] = schema_links_idx
        if slot_matching_idx is not None:
            marker_indices["slot_matching"] = slot_matching_idx
        
        # Generation starts after prompt
        generation_start_idx = len(combined_ce)
    else:
        generation_start_idx = 0
    
    # Add generation metrics
    if token_metrics:
        for m in token_metrics:
            if "cross_entropy" in m:
                combined_ce.append(m["cross_entropy"])
    
    return combined_ce, marker_indices, generation_start_idx


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply moving average to 1D array, handling NaN values.
    Uses a centered window when possible, otherwise uses trailing window.
    """
    if len(data) == 0:
        return data
    
    result = np.full_like(data, np.nan)
    half_window = window_size // 2
    
    for i in range(len(data)):
        # Use centered window when possible
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        
        # If we're near the beginning, use trailing window
        if i < half_window:
            start = 0
            end = min(window_size, len(data))
        
        # If we're near the end, use leading window
        if i >= len(data) - half_window:
            start = max(0, len(data) - window_size)
            end = len(data)
        
        window_data = data[start:end]
        # Only compute mean if we have at least one non-NaN value
        if not np.all(np.isnan(window_data)):
            result[i] = np.nanmean(window_data)
    
    return result


def _extract_sequences_for_scenario(
    results: list,
    intervention_type: Optional[str],
    structure_type: str,
) -> Tuple[List[Tuple[List[float], Dict[str, Optional[int]], Optional[int]]], Dict[str, List[int]]]:
    """
    Extract sequences for a given scenario (intervention type and structure type).
    Returns:
        - List of (ce_values, marker_indices, generation_start) tuples
        - Dictionary with marker position lists for averaging
    """
    all_sequences = []
    marker_positions = {
        "skeleton": [],
        "schema_links": [],
        "slot_matching": [],
        "generation": [],
    }
    
    for sample in results:
        ct = sample.get("completion_type", "")
        suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"
        
        # Only process samples matching the structure type
        if suffix != structure_type:
            continue
        
        if intervention_type is None:
            # Base generation (no intervention)
            prompt_metrics = sample.get("prompt_metrics")
            token_metrics = sample.get("token_metrics")
        else:
            # Intervention
            si = sample.get("structure_intervention", {})
            interventions = si.get(intervention_type, [])
            if not interventions:
                continue
            # Use first intervention
            interv = interventions[0]
            prompt_metrics = interv.get("prompt_metrics")
            token_metrics = interv.get("token_metrics")
        
        # Need at least token_metrics to plot
        if not token_metrics:
            continue
        
        ce_seq, marker_indices, gen_start = extract_combined_sequence(prompt_metrics, token_metrics)
        if ce_seq and len(ce_seq) > 0:
            all_sequences.append((ce_seq, marker_indices, gen_start))
            if marker_indices.get("skeleton") is not None:
                marker_positions["skeleton"].append(marker_indices["skeleton"])
            if marker_indices.get("schema_links") is not None:
                marker_positions["schema_links"].append(marker_indices["schema_links"])
            if marker_indices.get("slot_matching") is not None:
                marker_positions["slot_matching"].append(marker_indices["slot_matching"])
            if gen_start is not None:
                marker_positions["generation"].append(gen_start)
    
    return all_sequences, marker_positions


def _compute_smoothed_metrics(all_sequences: list, window_size: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute smoothed mean and std from sequences.
    Returns: (mean_ce, std_ce, max_len)
    """
    if not all_sequences:
        return np.array([]), np.array([]), 0
    
    # Find max length for alignment
    max_len = max(len(seq[0]) for seq in all_sequences)
    
    # Align sequences (pad with NaN)
    aligned_sequences = []
    for ce_seq, _, _ in all_sequences:
        padded = ce_seq + [np.nan] * (max_len - len(ce_seq))
        aligned_sequences.append(padded)
    
    # Calculate mean and std across samples for each position
    aligned_array = np.array(aligned_sequences)
    mean_ce_raw = np.nanmean(aligned_array, axis=0)
    std_ce_raw = np.nanstd(aligned_array, axis=0)
    
    # Apply moving average smoothing
    mean_ce = moving_average(mean_ce_raw, window_size)
    std_ce = moving_average(std_ce_raw, window_size)
    
    return mean_ce, std_ce, max_len


def plot_entropy_vs_token_position(
    results: list,
    model_name: str,
    output_dir: str,
    window_size: int = 12,
):
    """
    Plot cross-entropy vs token position for prompt + generation sequences.
    Creates 3 comparison plots:
    1. All interventions (HSVT, Local Edits, Global) for gold_structure
    2. All interventions (HSVT, Local Edits, Global) for predicted_structure
    3. Base generation for gold_structure and predicted_structure
    
    Args:
        results: List of sample dictionaries with token metrics
        model_name: Name of the model for title
        output_dir: Directory to save plots
        window_size: Size of the moving average window (default: 12 tokens)
    """
    intervention_types = ["HSVT", "Local Edits", "Global"]
    intervention_colors = {
        "HSVT": "#55A868",
        "Local Edits": "#C44E52",
        "Global": "#8172B2",
    }
    base_colors = {
        "gold_structure": "#4C72B0",
        "predicted_structure": "#64B5CD",
    }
    
    # ──────────────────────── Picture 1: Interventions for gold_structure (3 subplots) ────────────────────────
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 18))
    structure_type = "gold_structure"
    
    for idx, intervention_type in enumerate(intervention_types):
        ax = axes1[idx]
        
        # Find first sample matching the scenario
        first_sample = None
        for sample in results:
            ct = sample.get("completion_type", "")
            suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"
            if suffix == structure_type:
                si = sample.get("structure_intervention", {})
                interventions = si.get(intervention_type, [])
                if interventions and interventions[0].get("token_metrics"):
                    first_sample = sample
                    break
        
        if not first_sample:
            ax.text(0.5, 0.5, f"No data for {intervention_type}", 
                   transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{intervention_type} — Gold Structure — {model_name}")
            continue
        
        # Extract sequence from first sample
        interv = first_sample["structure_intervention"][intervention_type][0]
        prompt_metrics = interv.get("prompt_metrics")
        token_metrics = interv.get("token_metrics")
        
        ce_seq, marker_indices, gen_start = extract_combined_sequence(prompt_metrics, token_metrics)
        if not ce_seq or len(ce_seq) == 0:
            ax.text(0.5, 0.5, f"No sequence data for {intervention_type}", 
                   transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{intervention_type} — Gold Structure — {model_name}")
            continue
        
        # Apply smoothing to single sequence
        ce_array = np.array(ce_seq)
        mean_ce = moving_average(ce_array, window_size)
        positions = np.arange(len(mean_ce))
        
        # Plot smoothed line
        color = intervention_colors.get(intervention_type, "#999999")
        ax.plot(positions, mean_ce, linewidth=2, color=color, 
                label=f"{intervention_type} (smoothed)", zorder=3)
        
        # Get marker positions from first sample
        avg_skeleton = marker_indices.get("skeleton")
        avg_schema_links = marker_indices.get("schema_links")
        avg_slot_matching = marker_indices.get("slot_matching")
        avg_generation = gen_start
        
        # Highlight mediator region
        if avg_skeleton is not None and avg_generation is not None:
            if avg_generation > avg_skeleton:
                ax.axvspan(avg_skeleton, avg_generation, alpha=0.1, color="#FFA500", 
                          label="Mediator Region", zorder=1)
        
        # Mark section boundaries
        if avg_skeleton is not None:
            ax.axvline(avg_skeleton, color="#FF8C00", linestyle="--", linewidth=1.5, 
                      label="===SKELETON===", zorder=4)
        if avg_schema_links is not None:
            ax.axvline(avg_schema_links, color="#9B59B6", linestyle="--", linewidth=1.5, 
                      label="===SCHEMA_LINKS===", zorder=4)
        if avg_slot_matching is not None:
            ax.axvline(avg_slot_matching, color="#E67E22", linestyle="--", linewidth=1.5, 
                      label="===SLOT_MATCHING===", zorder=4)
        if avg_generation is not None:
            ax.axvline(avg_generation, color="#55A868", linestyle="--", linewidth=1.5, 
                      label="Generation Start", zorder=4)
        
        ax.set_xlabel("Token Position (Prompt + Generation)")
        ax.set_ylabel("Cross-Entropy")
        ax.set_title(f"{intervention_type} — Gold Structure — {model_name}")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)
    
    fig1.tight_layout()
    _save(fig1, output_dir, f"{model_name}_entropy_vs_position_interventions_gold")
    
    # ──────────────────────── Picture 2: Interventions for predicted_structure (3 subplots) ────────────────────────
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 18))
    structure_type = "predicted_structure"
    
    for idx, intervention_type in enumerate(intervention_types):
        ax = axes2[idx]
        
        # Find first sample matching the scenario
        first_sample = None
        for sample in results:
            ct = sample.get("completion_type", "")
            suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"
            if suffix == structure_type:
                si = sample.get("structure_intervention", {})
                interventions = si.get(intervention_type, [])
                if interventions and interventions[0].get("token_metrics"):
                    first_sample = sample
                    break
        
        if not first_sample:
            ax.text(0.5, 0.5, f"No data for {intervention_type}", 
                   transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{intervention_type} — Predicted Structure — {model_name}")
            continue
        
        # Extract sequence from first sample
        interv = first_sample["structure_intervention"][intervention_type][0]
        prompt_metrics = interv.get("prompt_metrics")
        token_metrics = interv.get("token_metrics")
        
        ce_seq, marker_indices, gen_start = extract_combined_sequence(prompt_metrics, token_metrics)
        if not ce_seq or len(ce_seq) == 0:
            ax.text(0.5, 0.5, f"No sequence data for {intervention_type}", 
                   transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{intervention_type} — Predicted Structure — {model_name}")
            continue
        
        # Apply smoothing to single sequence
        ce_array = np.array(ce_seq)
        mean_ce = moving_average(ce_array, window_size)
        positions = np.arange(len(mean_ce))
        
        # Plot smoothed line
        color = intervention_colors.get(intervention_type, "#999999")
        ax.plot(positions, mean_ce, linewidth=2, color=color, 
                label=f"{intervention_type} (smoothed)", zorder=3)
        
        # Get marker positions from first sample
        avg_skeleton = marker_indices.get("skeleton")
        avg_schema_links = marker_indices.get("schema_links")
        avg_slot_matching = marker_indices.get("slot_matching")
        avg_generation = gen_start
        
        # Highlight mediator region
        if avg_skeleton is not None and avg_generation is not None:
            if avg_generation > avg_skeleton:
                ax.axvspan(avg_skeleton, avg_generation, alpha=0.1, color="#FFA500", 
                          label="Mediator Region", zorder=1)
        
        # Mark section boundaries
        if avg_skeleton is not None:
            ax.axvline(avg_skeleton, color="#FF8C00", linestyle="--", linewidth=1.5, 
                      label="===SKELETON===", zorder=4)
        if avg_schema_links is not None:
            ax.axvline(avg_schema_links, color="#9B59B6", linestyle="--", linewidth=1.5, 
                      label="===SCHEMA_LINKS===", zorder=4)
        if avg_slot_matching is not None:
            ax.axvline(avg_slot_matching, color="#E67E22", linestyle="--", linewidth=1.5, 
                      label="===SLOT_MATCHING===", zorder=4)
        if avg_generation is not None:
            ax.axvline(avg_generation, color="#55A868", linestyle="--", linewidth=1.5, 
                      label="Generation Start", zorder=4)
        
        ax.set_xlabel("Token Position (Prompt + Generation)")
        ax.set_ylabel("Cross-Entropy")
        ax.set_title(f"{intervention_type} — Predicted Structure — {model_name}")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)
    
    fig2.tight_layout()
    _save(fig2, output_dir, f"{model_name}_entropy_vs_position_interventions_predicted")
    
    # ──────────────────────── Picture 3: Base generation for gold and predicted (2 subplots) ────────────────────────
    fig3, axes3 = plt.subplots(2, 1, figsize=(14, 12))
    
    for idx, structure_type in enumerate(["gold_structure", "predicted_structure"]):
        ax = axes3[idx]
        
        # Find first sample matching the scenario
        first_sample = None
        for sample in results:
            ct = sample.get("completion_type", "")
            suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"
            if suffix == structure_type and sample.get("token_metrics"):
                first_sample = sample
                break
        
        if not first_sample:
            label = "Gold Structure" if structure_type == "gold_structure" else "Predicted Structure"
            ax.text(0.5, 0.5, f"No data for Base {label}", 
                   transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"Base {label} — {model_name}")
            continue
        
        # Extract sequence from first sample
        prompt_metrics = first_sample.get("prompt_metrics")
        token_metrics = first_sample.get("token_metrics")
        
        ce_seq, marker_indices, gen_start = extract_combined_sequence(prompt_metrics, token_metrics)
        if not ce_seq or len(ce_seq) == 0:
            label = "Gold Structure" if structure_type == "gold_structure" else "Predicted Structure"
            ax.text(0.5, 0.5, f"No sequence data for Base {label}", 
                   transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"Base {label} — {model_name}")
            continue
        
        # Apply smoothing to single sequence
        ce_array = np.array(ce_seq)
        mean_ce = moving_average(ce_array, window_size)
        positions = np.arange(len(mean_ce))
        
        # Plot smoothed line
        color = base_colors.get(structure_type, "#999999")
        label = "Gold Structure" if structure_type == "gold_structure" else "Predicted Structure"
        ax.plot(positions, mean_ce, linewidth=2, color=color, 
                label=f"Base {label} (smoothed)", zorder=3)
        
        # Mark generation start if available
        avg_generation = gen_start
        
        if avg_generation is not None:
            ax.axvline(avg_generation, color="#55A868", linestyle="--", linewidth=1.5, 
                      label="Generation Start", zorder=4)
        
        ax.set_xlabel("Token Position (Prompt + Generation)")
        ax.set_ylabel("Cross-Entropy")
        ax.set_title(f"Base {label} — {model_name}")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)
    
    fig3.tight_layout()
    _save(fig3, output_dir, f"{model_name}_entropy_vs_position_base")


# ──────────────────────── summary table (text) ────────────────────────

def _format_scenario_name(key: str) -> str:
    """Format scenario key into a readable name."""
    parts = key.split("/")
    if len(parts) == 2:
        category, structure = parts
        structure_name = "Gold" if structure == "gold_structure" else "Predicted"
        explanation = ""
        if category == "generation":
            if structure_name == "Gold":
                explanation = "Y"
            elif structure_name == "Predicted":
                explanation = "M + Y"
        if category == "prompt":
            if structure_name == "Gold":
                explanation = "M"
        return f"{category.capitalize()} ({structure_name}) ({explanation})"
    elif len(parts) == 3:
        intervention, category, structure = parts
        structure_name = "Gold" if structure == "gold_structure" else "Predicted"
        explanation = ""
        if category == "generation":
            explanation = "Y"
        elif category == "prompt":
            explanation = "M"
        return f"{intervention} {category.capitalize()} ({structure_name}) ({explanation})"
    return key


def print_summary_table(per_sample: Dict[str, Dict[str, List[float]]], model_name: str):
    """Print a formatted table of aggregated metrics with grouping."""
    print(f"\n{'=' * 100}")
    print(f"  Token Metrics Summary — {model_name}")
    print(f"{'=' * 100}")
    
    # Group scenarios by category
    base_generation = []
    base_prompt = []
    interventions = []
    
    for key in sorted(per_sample.keys()):
        if key.startswith("generation/") and "/" not in key.replace("generation/", "").replace("gold_structure", "").replace("predicted_structure", ""):
            base_generation.append(key)
        elif key.startswith("prompt/") and "/" not in key.replace("prompt/", "").replace("gold_structure", "").replace("predicted_structure", ""):
            base_prompt.append(key)
        else:
            interventions.append(key)
    
    # Print header
    header = f"{'Scenario':<50} {'CE mean':>10} {'CE std':>10} {'MaxLogit':>12} {'GTLogit':>12} {'N':>6} {'AvgTokens':>10}"
    print(header)
    print("-" * 100)
    
    # Print base generation
    if base_generation:
        print("  Base Generation:")
        for key in base_generation:
            metrics = per_sample[key]
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
                f"    {scenario_name:<46} "
                f"{_mean(ce):>10.4f} {(_pstdev(ce) if n >= 2 else 0):>10.4f} "
                f"{(_mean(ml) if ml else 0):>12.4f} "
                f"{(_mean(gt) if gt else 0):>12.4f} "
                f"{n:>6} "
                f"{avg_tokens:>10.1f}"
            )
            print(row)
        print()
    
    # Print base prompt
    if base_prompt:
        print("  Base Prompt (from ===SKELETON===):")
        for key in base_prompt:
            metrics = per_sample[key]
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
                f"    {scenario_name:<46} "
                f"{_mean(ce):>10.4f} {(_pstdev(ce) if n >= 2 else 0):>10.4f} "
                f"{(_mean(ml) if ml else 0):>12.4f} "
                f"{(_mean(gt) if gt else 0):>12.4f} "
                f"{n:>6} "
                f"{avg_tokens:>10.1f}"
            )
            print(row)
        print()
    
    # Print interventions grouped by type
    if interventions:
        print("  Interventions:")
        # Group by intervention type
        intervention_groups = {}
        for key in interventions:
            parts = key.split("/")
            if len(parts) >= 1:
                intervention_type = parts[0]
                if intervention_type not in intervention_groups:
                    intervention_groups[intervention_type] = []
                intervention_groups[intervention_type].append(key)
        
        for intervention_type in sorted(intervention_groups.keys()):
            print(f"    {intervention_type}:")
            for key in sorted(intervention_groups[intervention_type]):
                metrics = per_sample[key]
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
                    f"      {scenario_name:<44} "
                    f"{_mean(ce):>10.4f} {(_pstdev(ce) if n >= 2 else 0):>10.4f} "
                    f"{(_mean(ml) if ml else 0):>12.4f} "
                    f"{(_mean(gt) if gt else 0):>12.4f} "
                    f"{n:>6} "
                    f"{avg_tokens:>10.1f}"
                )
                print(row)
            print()
    
    print(f"{'=' * 100}\n")


# ──────────────────────── paired comparison analysis ────────────────────────

def format_p_value(p_value: float) -> str:
    """
    Format p-value for display. Use scientific notation for very small values.
    """
    if p_value < 0.0001:
        return f"{p_value:.2e}"
    else:
        return f"{p_value:.4f}"


def extract_paired_metrics(results: list) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, Tuple[List[float], List[float]]]]]:
    """
    Extract paired metrics: for each sample, compute base and intervention metrics,
    then calculate deltas (intervention - base) per sample.
    
    IMPORTANT: Deltas are computed only for the mediator part (starting from ===SKELETON===),
    not for the full prompt or generation.
    
    Returns:
        - deltas: dict: intervention_type/scenario -> metric_name -> [per-sample delta values]
        - pairs: dict: intervention_type/scenario -> metric_name -> (base_values, intervention_values)
    
    This allows for proper paired comparison: we compare deltas across samples,
    rather than comparing averaged base vs averaged intervention.
    """
    keys = ("cross_entropy", "max_logit", "gt_logit")
    deltas = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(lambda: defaultdict(lambda: ([], [])))  # base_values, intervention_values
    
    for sample in results:
        ct = sample.get("completion_type", "")
        # Map completion_type to scenario suffix
        if ct == "gold_structure":
            suffix = "gold_structure"
        elif ct == "structure_prediction":
            suffix = "predicted_structure"
        else:
            suffix = "predicted_structure"  # fallback
        
        # Get base metrics
        # For mediator (from ===SKELETON===)
        base_prompt_ce = None
        base_prompt_max = None
        base_prompt_gt = None
        if ct == "gold_structure" and sample.get("prompt_metrics"):
            # For gold_structure: mediator is in prompt_metrics
            base_prompt_ce = mean_from_skeleton(sample["prompt_metrics"], "cross_entropy")
            base_prompt_max = mean_from_skeleton(sample["prompt_metrics"], "max_logit")
            base_prompt_gt = mean_from_skeleton(sample["prompt_metrics"], "gt_logit")
        elif ct == "structure_prediction" and sample.get("token_metrics"):
            # For structure_prediction: mediator is in token_metrics (generation contains ===SKELETON===)
            # Search from start because in generation, ===SKELETON=== appears at the beginning
            base_prompt_ce = mean_from_skeleton(sample["token_metrics"], "cross_entropy", search_from_start=True)
            base_prompt_max = mean_from_skeleton(sample["token_metrics"], "max_logit", search_from_start=True)
            base_prompt_gt = mean_from_skeleton(sample["token_metrics"], "gt_logit", search_from_start=True)
        
        # For generation: available for both gold_structure and predicted_structure
        base_gen_ce = mean_per_sample(sample.get("token_metrics"), "cross_entropy") if sample.get("token_metrics") else None
        base_gen_max = mean_per_sample(sample.get("token_metrics"), "max_logit") if sample.get("token_metrics") else None
        base_gen_gt = mean_per_sample(sample.get("token_metrics"), "gt_logit") if sample.get("token_metrics") else None
        
        # Get intervention metrics and compute deltas
        si = sample.get("structure_intervention", {})
        for itype in ("HSVT", "Local Edits", "Global"):
            for interv in si.get(itype, []):
                # Mediator deltas (from ===SKELETON===)
                if interv.get("prompt_metrics"):
                    interv_prompt_ce = mean_from_skeleton(interv["prompt_metrics"], "cross_entropy")
                    interv_prompt_max = mean_from_skeleton(interv["prompt_metrics"], "max_logit")
                    interv_prompt_gt = mean_from_skeleton(interv["prompt_metrics"], "gt_logit")
                    
                    # For gold_structure: compare intervention mediator with base mediator
                    if ct == "gold_structure" and base_prompt_ce is not None:
                        if interv_prompt_ce is not None:
                            delta_ce = interv_prompt_ce - base_prompt_ce
                            deltas[f"{itype}/mediator/{suffix}"]["cross_entropy"].append(delta_ce)
                            pairs[f"{itype}/mediator/{suffix}"]["cross_entropy"][0].append(base_prompt_ce)
                            pairs[f"{itype}/mediator/{suffix}"]["cross_entropy"][1].append(interv_prompt_ce)
                        if interv_prompt_max is not None and base_prompt_max is not None:
                            delta_max = interv_prompt_max - base_prompt_max
                            deltas[f"{itype}/mediator/{suffix}"]["max_logit"].append(delta_max)
                            pairs[f"{itype}/mediator/{suffix}"]["max_logit"][0].append(base_prompt_max)
                            pairs[f"{itype}/mediator/{suffix}"]["max_logit"][1].append(interv_prompt_max)
                        if interv_prompt_gt is not None and base_prompt_gt is not None:
                            delta_gt = interv_prompt_gt - base_prompt_gt
                            deltas[f"{itype}/mediator/{suffix}"]["gt_logit"].append(delta_gt)
                            pairs[f"{itype}/mediator/{suffix}"]["gt_logit"][0].append(base_prompt_gt)
                            pairs[f"{itype}/mediator/{suffix}"]["gt_logit"][1].append(interv_prompt_gt)
                    
                    # For structure_prediction: compare intervention mediator with base mediator (from generation)
                    # (base mediator is in token_metrics, starting from ===SKELETON===)
                    if ct == "structure_prediction" and base_prompt_ce is not None:
                        if interv_prompt_ce is not None:
                            delta_ce = interv_prompt_ce - base_prompt_ce
                            deltas[f"{itype}/mediator/{suffix}"]["cross_entropy"].append(delta_ce)
                            pairs[f"{itype}/mediator/{suffix}"]["cross_entropy"][0].append(base_prompt_ce)
                            pairs[f"{itype}/mediator/{suffix}"]["cross_entropy"][1].append(interv_prompt_ce)
                        if interv_prompt_max is not None and base_prompt_max is not None:
                            delta_max = interv_prompt_max - base_prompt_max
                            deltas[f"{itype}/mediator/{suffix}"]["max_logit"].append(delta_max)
                            pairs[f"{itype}/mediator/{suffix}"]["max_logit"][0].append(base_prompt_max)
                            pairs[f"{itype}/mediator/{suffix}"]["max_logit"][1].append(interv_prompt_max)
                        if interv_prompt_gt is not None and base_prompt_gt is not None:
                            delta_gt = interv_prompt_gt - base_prompt_gt
                            deltas[f"{itype}/mediator/{suffix}"]["gt_logit"].append(delta_gt)
                            pairs[f"{itype}/mediator/{suffix}"]["gt_logit"][0].append(base_prompt_gt)
                            pairs[f"{itype}/mediator/{suffix}"]["gt_logit"][1].append(interv_prompt_gt)
                
                # Generation deltas (for both gold_structure and predicted_structure)
                if interv.get("token_metrics") and base_gen_ce is not None:
                    interv_gen_ce = mean_per_sample(interv["token_metrics"], "cross_entropy")
                    interv_gen_max = mean_per_sample(interv["token_metrics"], "max_logit")
                    interv_gen_gt = mean_per_sample(interv["token_metrics"], "gt_logit")
                    
                    if interv_gen_ce is not None:
                        delta_ce = interv_gen_ce - base_gen_ce
                        deltas[f"{itype}/generation/{suffix}"]["cross_entropy"].append(delta_ce)
                        pairs[f"{itype}/generation/{suffix}"]["cross_entropy"][0].append(base_gen_ce)
                        pairs[f"{itype}/generation/{suffix}"]["cross_entropy"][1].append(interv_gen_ce)
                    if interv_gen_max is not None and base_gen_max is not None:
                        delta_max = interv_gen_max - base_gen_max
                        deltas[f"{itype}/generation/{suffix}"]["max_logit"].append(delta_max)
                        pairs[f"{itype}/generation/{suffix}"]["max_logit"][0].append(base_gen_max)
                        pairs[f"{itype}/generation/{suffix}"]["max_logit"][1].append(interv_gen_max)
                    if interv_gen_gt is not None and base_gen_gt is not None:
                        delta_gt = interv_gen_gt - base_gen_gt
                        deltas[f"{itype}/generation/{suffix}"]["gt_logit"].append(delta_gt)
                        pairs[f"{itype}/generation/{suffix}"]["gt_logit"][0].append(base_gen_gt)
                        pairs[f"{itype}/generation/{suffix}"]["gt_logit"][1].append(interv_gen_gt)
    
    return deltas, pairs


def plot_delta_distributions(
    deltas: Dict[str, Dict[str, List[float]]],
    pairs: Dict[str, Dict[str, Tuple[List[float], List[float]]]],
    model_name: str,
    output_dir: str,
):
    """
    Plot distributions of deltas (intervention - base) for each intervention type.
    Shows how consistently interventions change metrics across samples.
    Deltas are computed only for the mediator part (from ===SKELETON===).
    """
    
    intervention_types = ["HSVT", "Local Edits", "Global"]
    scenarios = ["gold_structure", "predicted_structure"]  # Mediator deltas for both (for predicted_structure: mediator vs base generation)
    
    for scenario in scenarios:
        fig, axes = plt.subplots(1, len(intervention_types), figsize=(5 * len(intervention_types), 5))
        if len(intervention_types) == 1:
            axes = [axes]
        
        for ax, itype in zip(axes, intervention_types):
            key = f"{itype}/mediator/{scenario}"
            delta_ce = deltas.get(key, {}).get("cross_entropy", [])
            
            if not delta_ce:
                ax.text(0.5, 0.5, f"No data for {itype}", 
                       transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f"{itype} — {scenario.replace('_', ' ').title()}")
                continue
            
            # Histogram
            ax.hist(delta_ce, bins=30, alpha=0.7, color=COLORS.get(itype, "#999999"),
                   edgecolor="black", linewidth=0.5, density=True)
            
            # Vertical lines
            ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="No change")
            mean_delta = _mean(delta_ce)
            ax.axvline(mean_delta, color="red", linestyle="-", linewidth=1.5, 
                      label=f"Mean = {mean_delta:.3f}")
            
            # Statistical test - Paired t-test
            if HAS_SCIPY:
                try:
                    # Get paired values for proper paired t-test
                    base_vals, interv_vals = pairs.get(key, {}).get("cross_entropy", ([], []))
                    if len(base_vals) == len(interv_vals) and len(base_vals) > 1:
                        # Paired t-test: tests if mean difference is significantly different from 0
                        t_stat, p_value = stats.ttest_rel(interv_vals, base_vals)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        ax.text(0.05, 0.95, f"t={t_stat:.2f}, p={format_p_value(p_value)} {significance}",
                               transform=ax.transAxes, va="top", ha="left",
                               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
                    else:
                        # Fallback to one-sample t-test on deltas if pairs not available
                        t_stat, p_value = stats.ttest_1samp(delta_ce, 0)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        ax.text(0.05, 0.95, f"t={t_stat:.2f}, p={format_p_value(p_value)} {significance}",
                               transform=ax.transAxes, va="top", ha="left",
                               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
                except:
                    pass
            
            ax.set_xlabel("Δ Cross-Entropy (Intervention - Base)")
            ax.set_ylabel("Density")
            ax.set_title(f"{itype} — {scenario.replace('_', ' ').title()}")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
        
        if scenario == "gold_structure":
            title_suffix = "Mediator Only (from ===SKELETON===)"
        else:  # predicted_structure
            title_suffix = "Mediator (Base from Generation, Intervention from Prompt)"
        fig.suptitle(f"Delta Distribution — {title_suffix} ({scenario.replace('_', ' ').title()}) — {model_name}", 
                     fontsize=14)
        fig.tight_layout()
        _save(fig, output_dir, f"{model_name}_delta_distributions_mediator_{scenario}")


def plot_paired_scatter(
    results: list,
    model_name: str,
    output_dir: str,
):
    """
    Scatter plot: base vs intervention metrics for each sample.
    Points above y=x line indicate intervention increased the metric.
    Shows correlation and individual sample changes.
    For gold_structure: compares mediator with mediator.
    For predicted_structure: compares intervention mediator with base generation.
    """
    intervention_types = ["HSVT", "Local Edits", "Global"]
    scenarios = ["gold_structure", "predicted_structure"]  # Mediator metrics for both
    
    for scenario in scenarios:
        fig, axes = plt.subplots(1, len(intervention_types), figsize=(5 * len(intervention_types), 5))
        if len(intervention_types) == 1:
            axes = [axes]
        
        for ax, itype in zip(axes, intervention_types):
            base_values = []
            interv_values = []
            
            for sample in results:
                ct = sample.get("completion_type", "")
                suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"
                
                if suffix != scenario:
                    continue
                
                # Get base metrics
                base_ce = None
                if ct == "gold_structure" and sample.get("prompt_metrics"):
                    # For gold_structure: use base mediator (from ===SKELETON===)
                    base_ce = mean_from_skeleton(sample["prompt_metrics"], "cross_entropy")
                elif ct == "structure_prediction" and sample.get("token_metrics"):
                    # For structure_prediction: use base mediator from token_metrics (generation contains ===SKELETON===)
                    # Search from start because in generation, ===SKELETON=== appears at the beginning
                    base_ce = mean_from_skeleton(sample["token_metrics"], "cross_entropy", search_from_start=True)
                
                if base_ce is None:
                    continue
                
                si = sample.get("structure_intervention", {})
                for interv in si.get(itype, []):
                    if interv.get("prompt_metrics"):
                        # Get intervention mediator metrics (from ===SKELETON===)
                        interv_ce = mean_from_skeleton(interv["prompt_metrics"], "cross_entropy")
                        if interv_ce is not None:
                            base_values.append(base_ce)
                            interv_values.append(interv_ce)
                            break  # Use first intervention only
            
            if not base_values:
                ax.text(0.5, 0.5, f"No data for {itype}", 
                       transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f"{itype} — {scenario.replace('_', ' ').title()}")
                continue
            
            # Scatter plot
            ax.scatter(base_values, interv_values, alpha=0.6, s=30,
                      color=COLORS.get(itype, "#999999"), edgecolors="black", linewidth=0.5)
            
            # y=x reference line
            lim_min = min(min(base_values), min(interv_values))
            lim_max = max(max(base_values), max(interv_values))
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1.5, alpha=0.5, label="y=x")
            
            # Correlation
            if HAS_SCIPY:
                try:
                    from scipy.stats import pearsonr
                    corr, p_val = pearsonr(base_values, interv_values)
                    ax.text(0.05, 0.95, f"r={corr:.3f}, p={format_p_value(p_val)}",
                           transform=ax.transAxes, va="top", ha="left",
                           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
                except:
                    pass
            
            if scenario == "gold_structure":
                xlabel = "Base Cross-Entropy (Mediator from Prompt)"
                ylabel = "Intervention Cross-Entropy (Mediator from Prompt)"
                title_suffix = "Mediator Only (from ===SKELETON===)"
            else:  # predicted_structure
                xlabel = "Base Cross-Entropy (Mediator from Generation)"
                ylabel = "Intervention Cross-Entropy (Mediator from Prompt)"
                title_suffix = "Mediator (Base from Generation, Intervention from Prompt)"
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{itype} — {scenario.replace('_', ' ').title()}")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_aspect("equal", adjustable="box")
        
        fig.suptitle(f"Base vs Intervention — {title_suffix} ({scenario.replace('_', ' ').title()}) — {model_name}", 
                     fontsize=14)
        fig.tight_layout()
        _save(fig, output_dir, f"{model_name}_paired_scatter_mediator_{scenario}")


def plot_delta_vs_base(
    results: list,
    model_name: str,
    output_dir: str,
):
    """
    Plot delta (intervention - base) vs base value.
    Shows if intervention effect depends on base value.
    For gold_structure: compares mediator with mediator.
    For predicted_structure: compares intervention mediator with base generation.
    """
    intervention_types = ["HSVT", "Local Edits", "Global"]
    scenarios = ["gold_structure", "predicted_structure"]  # Mediator metrics for both
    
    for scenario in scenarios:
        fig, axes = plt.subplots(1, len(intervention_types), figsize=(5 * len(intervention_types), 5))
        if len(intervention_types) == 1:
            axes = [axes]
        
        for ax, itype in zip(axes, intervention_types):
            base_values = []
            delta_values = []
            
            for sample in results:
                ct = sample.get("completion_type", "")
                suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"
                
                if suffix != scenario:
                    continue
                
                # Get base metrics
                base_ce = None
                if ct == "gold_structure" and sample.get("prompt_metrics"):
                    # For gold_structure: use base mediator (from ===SKELETON===)
                    base_ce = mean_from_skeleton(sample["prompt_metrics"], "cross_entropy")
                elif ct == "structure_prediction" and sample.get("token_metrics"):
                    # For structure_prediction: use base mediator from token_metrics (generation contains ===SKELETON===)
                    # Search from start because in generation, ===SKELETON=== appears at the beginning
                    base_ce = mean_from_skeleton(sample["token_metrics"], "cross_entropy", search_from_start=True)
                
                if base_ce is None:
                    continue
                
                si = sample.get("structure_intervention", {})
                for interv in si.get(itype, []):
                    if interv.get("prompt_metrics"):
                        # Get intervention mediator metrics (from ===SKELETON===)
                        interv_ce = mean_from_skeleton(interv["prompt_metrics"], "cross_entropy")
                        if interv_ce is not None:
                            base_values.append(base_ce)
                            delta_values.append(interv_ce - base_ce)
                            break
            
            if not base_values:
                ax.text(0.5, 0.5, f"No data for {itype}", 
                       transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f"{itype} — {scenario.replace('_', ' ').title()}")
                continue
            
            # Scatter plot
            ax.scatter(base_values, delta_values, alpha=0.6, s=30,
                      color=COLORS.get(itype, "#999999"), edgecolors="black", linewidth=0.5)
            
            # Horizontal line at y=0
            ax.axhline(0, color="black", linestyle="--", linewidth=1.5, label="No change")
            
            # Trend line
            try:
                z = np.polyfit(base_values, delta_values, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(base_values), max(base_values), 100)
                ax.plot(x_trend, p(x_trend), "r--", linewidth=1.5, alpha=0.7, label="Trend")
            except:
                pass
            
            if scenario == "gold_structure":
                xlabel = "Base Cross-Entropy (Mediator from Prompt)"
                title_suffix = "Mediator Only (from ===SKELETON===)"
            else:  # predicted_structure
                xlabel = "Base Cross-Entropy (Mediator from Generation)"
                title_suffix = "Mediator (Base from Generation, Intervention from Prompt)"
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Δ Cross-Entropy (Intervention - Base)")
            ax.set_title(f"{itype} — {scenario.replace('_', ' ').title()}")
            ax.legend()
            ax.grid(alpha=0.3)
        
        fig.suptitle(f"Delta vs Base — {title_suffix} ({scenario.replace('_', ' ').title()}) — {model_name}", 
                     fontsize=14)
        fig.tight_layout()
        _save(fig, output_dir, f"{model_name}_delta_vs_base_mediator_{scenario}")


def print_paired_statistics(
    deltas: Dict[str, Dict[str, List[float]]],
    pairs: Dict[str, Dict[str, Tuple[List[float], List[float]]]],
    model_name: str,
):
    """
    Print statistical summary of paired comparisons.
    """
    
    print(f"\n{'=' * 100}")
    print(f"  Paired Comparison Statistics — {model_name}")
    print(f"{'=' * 100}")
    
    intervention_types = ["HSVT", "Local Edits", "Global"]
    scenarios = ["gold_structure", "predicted_structure"]  # Mediator deltas for both (for predicted_structure: mediator vs base generation)
    
    header = f"{'Scenario':<50} {'Mean Δ':>12} {'Std Δ':>12} {'t-stat':>10} {'p-value':>12} {'Effect':>10}"
    print(header)
    print("-" * 100)
    print("  Note: For gold_structure: mediator from prompt vs mediator from prompt. For predicted_structure: mediator from prompt vs mediator from generation.")
    print("-" * 100)
    
    for scenario in scenarios:
        if scenario == "gold_structure":
            scenario_label = "Mediator (Prompt vs Prompt)"
        else:  # predicted_structure
            scenario_label = "Mediator (Prompt vs Generation)"
        print(f"\n  {scenario.replace('_', ' ').title()} ({scenario_label}):")
        for itype in intervention_types:
            key = f"{itype}/mediator/{scenario}"
            delta_ce = deltas.get(key, {}).get("cross_entropy", [])
            
            if not delta_ce:
                continue
            
            mean_delta = _mean(delta_ce)
            std_delta = _pstdev(delta_ce) if len(delta_ce) >= 2 else 0
            
            if HAS_SCIPY:
                try:
                    # Get paired values for proper paired t-test
                    base_vals, interv_vals = pairs.get(key, {}).get("cross_entropy", ([], []))
                    if len(base_vals) == len(interv_vals) and len(base_vals) > 1:
                        # Paired t-test: tests if mean difference is significantly different from 0
                        t_stat, p_value = stats.ttest_rel(interv_vals, base_vals)
                        # Effect size (Cohen's d for paired samples)
                        # d = mean_delta / std_delta, where std_delta is std of differences
                        cohens_d = mean_delta / std_delta if std_delta > 0 else 0
                        effect = "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"
                    else:
                        # Fallback to one-sample t-test on deltas if pairs not available
                        t_stat, p_value = stats.ttest_1samp(delta_ce, 0)
                        cohens_d = mean_delta / std_delta if std_delta > 0 else 0
                        effect = "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"
                except:
                    t_stat, p_value, cohens_d, effect = 0, 1.0, 0, "N/A"
            else:
                t_stat, p_value, cohens_d, effect = 0, 1.0, 0, "N/A (scipy not available)"
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            p_value_str = format_p_value(p_value)
            row = (
                f"    {itype:<46} "
                f"{mean_delta:>12.4f} {std_delta:>12.4f} "
                f"{t_stat:>10.2f} {p_value_str:>12} {significance:<3} "
                f"{effect:>10}"
            )
            print(row)
    
    print(f"{'=' * 100}\n")


# ──────────────────────── main ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse token-level consistency metrics.")
    parser.add_argument("--input_files", nargs="+", required=True,
                        help="Path(s) to JSON result files from make_intervention.py")
    parser.add_argument("--output_dir", type=str, default="analysis/figures/token_metrics",
                        help="Directory to save figures")
    args = parser.parse_args()

    model_all_per_sample = {}  # model_name -> per_sample dict (for multi-model plot)

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
        has_token = any(s.get("token_metrics") for s in results)
        has_prompt = any(s.get("prompt_metrics") for s in results)
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
        model_all_per_sample[model_name] = per_sample

        print_summary_table(per_sample, model_name)

        # Extract tokenized texts for verification
        tokenized_samples = extract_all_tokenized_texts(results)
        
        # Save tokenized texts to JSON
        tokenized_json_path = os.path.join(args.output_dir, f"{model_name}_tokenized_texts.json")
        with open(tokenized_json_path, "w", encoding="utf-8") as f:
            json.dump(tokenized_samples, f, ensure_ascii=False, indent=2)
        print(f"  Saved tokenized texts to: {tokenized_json_path}")

        # Generate selected figures
        plot_prompt_cross_entropy_bars(per_sample, model_name, args.output_dir)
        plot_generation_cross_entropy_bars(per_sample, model_name, args.output_dir)
        plot_metrics_heatmap(per_sample, model_name, args.output_dir)
        plot_prompt_vs_generation(per_sample, model_name, args.output_dir)
        plot_entropy_vs_token_position(results, model_name, args.output_dir)
        
        # Paired comparison analysis (delta analysis)
        print(f"\n  Computing paired comparisons (delta analysis)...")
        deltas, pairs = extract_paired_metrics(results)
        if deltas:
            print_paired_statistics(deltas, pairs, model_name)
            plot_delta_distributions(deltas, pairs, model_name, args.output_dir)
            plot_paired_scatter(results, model_name, args.output_dir)
            plot_delta_vs_base(results, model_name, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
