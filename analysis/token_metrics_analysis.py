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


def extract_per_sample_metrics(results: list) -> Dict[str, Dict[str, List[float]]]:
    """
    Walk through every sample in `results` and collect per-sample mean
    cross_entropy values (and other metrics) for each scenario.

    Returns dict:  scenario_name -> metric_name -> [per-sample mean values]
    
    Scenarios:
        generation/gold_structure, generation/predicted_structure,
        prompt/gold_structure, prompt/predicted_structure,
        HSVT/generation, HSVT/prompt,
        Local Edits/generation, Local Edits/prompt,
        Global/generation, Global/prompt,
    """
    keys = ("cross_entropy", "max_logit", "gt_logit")
    data = defaultdict(lambda: defaultdict(list))

    for sample in results:
        ct = sample.get("completion_type", "")
        suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"

        # --- Main generation ---
        if sample.get("token_metrics"):
            for k in keys:
                v = mean_per_sample(sample["token_metrics"], k)
                if v is not None:
                    data[f"generation/{suffix}"][k].append(v)

        # --- Main prompt ---
        if sample.get("prompt_metrics"):
            for k in keys:
                v = mean_per_sample(sample["prompt_metrics"], k)
                if v is not None:
                    data[f"prompt/{suffix}"][k].append(v)

        # --- Interventions ---
        si = sample.get("structure_intervention", {})
        for itype in ("HSVT", "Local Edits", "Global"):
            for interv in si.get(itype, []):
                if interv.get("token_metrics"):
                    for k in keys:
                        v = mean_per_sample(interv["token_metrics"], k)
                        if v is not None:
                            data[f"{itype}/generation"][k].append(v)
                if interv.get("prompt_metrics"):
                    for k in keys:
                        v = mean_per_sample(interv["prompt_metrics"], k)
                        if v is not None:
                            data[f"{itype}/prompt"][k].append(v)

    return data


def extract_changed_prompt_metrics(results: list, fraction: float = 0.25) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract metrics only for the changed part of the prompt (last `fraction` of tokens).
    This focuses on the parts that differ between scenarios (question, schema, mediator).
    
    Returns dict:  scenario_name -> metric_name -> [per-sample mean values for changed part]
    """
    keys = ("cross_entropy", "max_logit", "gt_logit")
    data = defaultdict(lambda: defaultdict(list))

    for sample in results:
        ct = sample.get("completion_type", "")
        suffix = "gold_structure" if ct == "gold_structure" else "predicted_structure"

        # --- Main prompt (changed part only) ---
        if sample.get("prompt_metrics"):
            for k in keys:
                v = mean_last_fraction(sample["prompt_metrics"], k, fraction)
                if v is not None:
                    data[f"prompt/{suffix}"][k].append(v)

        # --- Interventions (changed part only) ---
        si = sample.get("structure_intervention", {})
        for itype in ("HSVT", "Local Edits", "Global"):
            for interv in si.get(itype, []):
                if interv.get("prompt_metrics"):
                    for k in keys:
                        v = mean_last_fraction(interv["prompt_metrics"], k, fraction)
                        if v is not None:
                            data[f"{itype}/prompt"][k].append(v)

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
    and each intervention type.
    """
    scenarios_gen = [
        ("generation/gold_structure",      "Base (Gold)"),
        ("generation/predicted_structure",  "Base (Predicted)"),
        ("HSVT/generation",                "HSVT"),
        ("Local Edits/generation",         "Local Edits"),
        ("Global/generation",              "Global"),
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
    Bar chart of mean cross-entropy for prompt tokens (gold vs predicted)
    and each intervention type.
    """
    scenarios_prompt = [
        ("prompt/gold_structure",      "Base (Gold)"),
        ("prompt/predicted_structure",  "Base (Predicted)"),
        ("HSVT/prompt",                "HSVT"),
        ("Local Edits/prompt",         "Local Edits"),
        ("Global/prompt",              "Global"),
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
    ax.set_ylabel("Mean Cross-Entropy (per sample)")
    ax.set_title(f"Prompt Cross-Entropy — {model_name}")
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
    """
    scenarios_prompt = [
        ("prompt/gold_structure",      "Base (Gold)"),
        ("prompt/predicted_structure",  "Base (Predicted)"),
        ("HSVT/prompt",                "HSVT"),
        ("Local Edits/prompt",         "Local Edits"),
        ("Global/prompt",              "Global"),
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
    Grouped bar chart: prompt vs generation cross-entropy for gold / predicted.
    """
    groups = [
        ("Gold Structure",      "prompt/gold_structure",      "generation/gold_structure"),
        ("Predicted Structure",  "prompt/predicted_structure", "generation/predicted_structure"),
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
    ax.set_title(f"Prompt vs Generation Cross-Entropy — {model_name}")
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
        "HSVT/generation",
        "Local Edits/generation",
        "Global/generation",
        "prompt/gold_structure",
        "prompt/predicted_structure",
        "HSVT/prompt",
        "Local Edits/prompt",
        "Global/prompt",
    ]
    scenario_labels = [
        "Gen (Gold)", "Gen (Predicted)",
        "HSVT Gen", "Local Edits Gen", "Global Gen",
        "Prompt (Gold)", "Prompt (Predicted)",
        "HSVT Prompt", "Local Edits Prompt", "Global Prompt",
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


# ──────────────────────── summary table (text) ────────────────────────

def print_summary_table(per_sample: Dict[str, Dict[str, List[float]]], model_name: str):
    """Print a formatted table of aggregated metrics."""
    print(f"\n{'=' * 70}")
    print(f"  Token Metrics Summary — {model_name}")
    print(f"{'=' * 70}")
    header = f"{'Scenario':<35} {'CE mean':>8} {'CE std':>8} {'MaxLogit':>9} {'GTLogit':>9} {'N':>5}"
    print(header)
    print("-" * 70)

    for key in sorted(per_sample.keys()):
        metrics = per_sample[key]
        ce = metrics.get("cross_entropy", [])
        ml = metrics.get("max_logit", [])
        gt = metrics.get("gt_logit", [])
        n = len(ce)
        if n == 0:
            continue
        row = (
            f"{key:<35} "
            f"{_mean(ce):>8.4f} {(_pstdev(ce) if n >= 2 else 0):>8.4f} "
            f"{(_mean(ml) if ml else 0):>9.4f} "
            f"{(_mean(gt) if gt else 0):>9.4f} "
            f"{n:>5}"
        )
        print(row)
    print()


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

        # Generate selected figures
        plot_prompt_cross_entropy_bars(per_sample, model_name, args.output_dir)
        plot_generation_cross_entropy_bars(per_sample, model_name, args.output_dir)
        plot_metrics_heatmap(per_sample, model_name, args.output_dir)
        plot_prompt_vs_generation(per_sample, model_name, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
