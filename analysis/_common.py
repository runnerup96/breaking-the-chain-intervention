"""
_common.py
----------
Shared constants, data loading and metric helpers for all artifact scripts.
All metric logic mirrors new_faith_analysis.ipynb exactly.
"""
from __future__ import annotations

import json
import random
import sys
from math import isclose
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

random.seed(42)

# ── model / dataset metadata ──────────────────────────────────────────────────

SHORT2NICE_MODEL: dict[str, str] = {
    "llama32-1B":      "Llama-3.2 1B",
    "qwen3-1.7B":      "Qwen-3 1.7B",
    "gemma2-2B":       "Gemma-2 2B",
    "falcon3-3B":      "Falcon-3 3B",
    "llama32-3B":      "Llama-3.2 3B",
    "qwen3-4B":        "Qwen-3 4B",
    "falcon3-7B":      "Falcon-3 7B",
    "qwen3-8B":        "Qwen-3 8B",
    "llama31-8B":      "Llama-3.1 8B",
    "qwen3-14B":       "Qwen-3 14B",
    "qwen3-32B":       "Qwen-3 32B",
    "llama31-70B":     "Llama-3.1 70B",
    "qwen3-235B-a22B": "Qwen3-235B-A22B",
}
NICE2SHORT_MODEL = {v: k for k, v in SHORT2NICE_MODEL.items()}

RAW2NICE_DS = {"ricechem": "RiceChem", "averitec": "AVeriTeC", "tabfact": "TabFact"}
NICE2RAW_DS = {v: k for k, v in RAW2NICE_DS.items()}

# Llama-3.2 1B excluded (too weak, excluded in notebook too)
MODEL_ORDER: list[str] = [
    "Qwen-3 1.7B", "Gemma-2 2B", "Falcon-3 3B", "Llama-3.2 3B",
    "Qwen-3 4B", "Falcon-3 7B", "Qwen-3 8B", "Llama-3.1 8B",
    "Qwen-3 14B", "Qwen-3 32B", "Llama-3.1 70B", "Qwen3-235B-A22B",
]
DATASET_ORDER = ["RiceChem", "AVeriTeC", "TabFact"]

DATASET_KEY_MAP = {
    "ricechem": {
        "y_pred_origin": "score_before_intervention",
        "mediator":      "mediator_rubric",
        "y_gold":        "gold_rubric",
        "y_pred_interv": "score_after_intervention",
    },
    "averitec": {
        "y_pred_origin": "target_before_intervention",
        "mediator":      "mediator_rubric",
        "y_gold":        "gold_rubric",
        "y_pred_interv": "target_after_intervention",
    },
    "tabfact": {
        "y_pred_origin": "target_before_intervention",
        "mediator":      "mediator_query",
        "y_gold":        "gold_query",
        "y_pred_interv": "target_after_intervention",
    },
}

Y_BEFORE_KEY = {
    "ricechem": "score_before_intervention",
    "averitec": "target_before_intervention",
    "tabfact":  "target_before_intervention",
}

MODEL_COLORS: dict[str, str] = {
    "Qwen-3 1.7B":     "#ff7f0e",
    "Gemma-2 2B":      "#2ca02c",
    "Falcon-3 3B":     "#d62728",
    "Llama-3.2 3B":    "#9467bd",
    "Qwen-3 4B":       "#8c564b",
    "Falcon-3 7B":     "#e377c2",
    "Qwen-3 8B":       "#7f7f7f",
    "Llama-3.1 8B":    "#bcbd22",
    "Qwen-3 14B":      "#8c842b",
    "Qwen-3 32B":      "#8c120b",
    "Llama-3.1 70B":   "#bcbd44",
    "Qwen3-235B-A22B": "#1f77b4",
}

PLOT_PARAMS = {
    "XLABEL_SIZE": 11, "YLABEL_SIZE": 11,
    "TITLE_SIZE":  10, "LEGEND_FONT":  8, "TICKS_SIZE": 10,
}


# ── tools (loaded on demand) ──────────────────────────────────────────────────

TOOLS: dict = {}   # ds_raw → calculate_score callable


def setup_tools(project_path: str) -> bool:
    """
    Import dataset tools from the project.
    Returns True if all three tools loaded, False otherwise.
    """
    pth = Path(project_path) / "breaking-the-chain-intervention"
    if str(pth) not in sys.path:
        sys.path.insert(0, str(pth))
    try:
        from datasets_for_intervention import (
            ricechem_dataset, ricechem_structure_processor,
            averitec_dataset, averitec_structure_processor,
            tabfact_dataset, tabfact_dsl_engine, tabfact_structure_processor,
        )
        st = Path(project_path) / "statics" / "datasets"

        rc_ds   = ricechem_dataset.RiceChemDataset(str(st / "RiceChem/data"))
        rc_tool = ricechem_structure_processor.RiceChemTool(rc_ds)
        TOOLS["ricechem"] = rc_tool.calculate_score

        av_ds   = averitec_dataset.AVeriTeCDataset(str(st / "AVeriTeC/data"))
        av_tool = averitec_structure_processor.AVeriTeCTool(av_ds)
        TOOLS["averitec"] = av_tool.calculate_score

        tf_ds     = tabfact_dataset.TabFactDataset(
            str(st / "TabFact/bootstrap_full.json"),
            str(st / "TabFact/data/all_csv"),
        )
        tf_engine = tabfact_dsl_engine.TabFactEngine()
        tf_tool   = tabfact_structure_processor.TabFactTool(tf_engine)
        TOOLS["tabfact"] = tf_tool.calculate_score

        print("Tools loaded OK")
        return True
    except Exception as e:
        print(f"[WARN] Could not load tools: {e}")
        return False


# ── data loading ──────────────────────────────────────────────────────────────

def _match_model(stem: str) -> Optional[str]:
    for short in SHORT2NICE_MODEL:
        if stem.startswith(short):
            return short
    return None


def load_predictions(
    root: Path,
    datasets: list[str],          # raw names: ["ricechem", "averitec", "tabfact"]
    run_types: list[str],          # e.g. ["standard", "tool", "detailed", ...]
    averitec_run: str = "standard_no_expl", # which run_type maps to "averitec" standard slot
) -> dict:
    """
    Returns:
        all_pred[ds_raw][run_type][model_short] = data_dict

    For averitec, if averitec_run != "standard", the chosen run_type is loaded
    but stored under its original key (so callers can use it directly).
    """
    all_pred: dict = {}
    root = Path(root)

    for ds_raw in datasets:
        if ds_raw not in RAW2NICE_DS:
            continue
        ds_dir = root / ds_raw
        if not ds_dir.exists():
            print(f"  [SKIP] {ds_dir} not found")
            continue

        all_pred[ds_raw] = {}

        # For averitec, also scan the ablation run if requested
        effective_runs = list(run_types)
        if ds_raw == "averitec" and averitec_run not in effective_runs:
            effective_runs.append(averitec_run)

        for run_type in effective_runs:
            run_dir = ds_dir / run_type
            if not run_dir.exists():
                continue
            all_pred[ds_raw].setdefault(run_type, {})

            for path in sorted(run_dir.glob("*.json")):
                model_short = _match_model(path.stem)
                if model_short is None:
                    continue
                # Keep latest file per model (by filename sort)
                existing = all_pred[ds_raw][run_type].get(model_short)
                if existing is None or path.name > existing.get("_fn", ""):
                    try:
                        with open(path) as f:
                            d = json.load(f)
                        d["_fn"] = path.name
                        all_pred[ds_raw][run_type][model_short] = d
                    except Exception as e:
                        print(f"  [WARN] {path}: {e}")

    total = sum(len(m) for rt in all_pred.values() for m in rt.values())
    print(f"Loaded {total} file(s)  |  "
          f"datasets={list(all_pred.keys())}  "
          f"run_types={sorted({rt for ds in all_pred.values() for rt in ds})}")
    return all_pred


def load_baseline(baseline_root: Optional[Path], datasets: list[str]) -> dict:
    """Returns: baseline[ds_raw][model_short] = data_dict"""
    baseline: dict = {}
    if not baseline_root or not Path(baseline_root).exists():
        return baseline
    for ds_raw in datasets:
        ds_dir = Path(baseline_root) / ds_raw
        if not ds_dir.exists():
            continue
        baseline[ds_raw] = {}
        for path in ds_dir.glob("*.json"):
            ms = _match_model(path.stem)
            if ms is None:
                continue
            existing = baseline[ds_raw].get(ms)
            if existing is None or path.name > existing.get("_fn", ""):
                try:
                    with open(path) as f:
                        d = json.load(f)
                    d["_fn"] = path.name
                    baseline[ds_raw][ms] = d
                except Exception:
                    pass
    n = sum(len(m) for m in baseline.values())
    print(f"Loaded {n} baseline file(s)  |  datasets={list(baseline.keys())}")
    return baseline


# ── sample helpers (mirrors notebook exactly) ─────────────────────────────────

def filter_sample(sample: dict, ds_raw: str) -> str:
    """Mirrors filter_samples_heuristic from notebook."""
    status = sample["generation_status"]
    if status == "error":
        return "error"
    if ds_raw == "averitec":
        if (status == "incorrect" and
                len(sample.get("gold_rubric", {})) != len(sample.get("mediator_rubric", {}))):
            return "error"
    return status


def _format_for_tool(sample: dict, ds_raw: str):
    if ds_raw == "ricechem":
        return ({"rubric": sample["mediator_rubric"]},
                {"task_idx": sample["task_idx"]})
    if ds_raw == "averitec":
        return ({"rubric": sample["mediator_rubric"]},
                {"gold_rubric": sample["gold_rubric"],
                 "gold_target": sample["gold_target"]})
    if ds_raw == "tabfact":
        return ({"query": sample["mediator_query"]},
                {"table_html_csv": sample["table_html_csv"]})
    raise ValueError(f"Unknown dataset: {ds_raw}")


def calc_id(sample: dict, ds_raw: str) -> bool:
    """F_ID: does C(M̂) == Ŷ? (mirrors calculate_id_faith)"""
    args, meta = _format_for_tool(sample, ds_raw)
    tool_y = TOOLS[ds_raw](args, meta)
    return tool_y == sample[DATASET_KEY_MAP[ds_raw]["y_pred_origin"]]


def calc_ooid(sample: dict, ds_raw: str, return_ood: bool = False):
    """F_strong = F_ID ∧ F_OOD. (mirrors calculate_id_ooId_faith)"""
    id_match = calc_id(sample, ds_raw)

    le  = sample["structure_intervention"]["Local Edits"]
    cor = sample["structure_intervention"]["Correction"]
    if le:
        idx = random.randint(0, len(le) - 1)
        interv = le[idx]
    else:
        interv = cor[0]

    args, meta = _format_for_tool(interv, ds_raw)
    tool_y  = TOOLS[ds_raw](args, meta)
    y_interv = interv[DATASET_KEY_MAP[ds_raw]["y_pred_interv"]]
    ood_match = (tool_y == y_interv)
    result = id_match and ood_match
    return (result, ood_match) if return_ood else result


def is_tool_run(run_type: str) -> bool:
    return run_type.startswith("tool")


def build_df_exps(
    all_pred: dict,
    datasets: list[str],
    run_types: list[str],
) -> pd.DataFrame:
    """
    Build the flat per-sample DataFrame (mirrors df_exps build cell).
    Requires TOOLS to be loaded.
    """
    rows = []
    for ds_raw in datasets:
        ds_nice = RAW2NICE_DS.get(ds_raw)
        if ds_nice is None or ds_raw not in all_pred:
            continue
        for run_type in run_types:
            if run_type not in all_pred[ds_raw]:
                continue
            for model_short, data in all_pred[ds_raw][run_type].items():
                model_nice = SHORT2NICE_MODEL.get(model_short)
                if model_nice is None or model_nice == "Llama-3.2 1B":
                    continue
                for sample in data.get("result", []):
                    status = filter_sample(sample, ds_raw)
                    if status != "error":
                        interv_type = (
                            "counterfact"
                            if sample["structure_intervention"]["Local Edits"]
                            else "correction"
                        )
                        try:
                            if is_tool_run(run_type):
                                id_v  = calc_id(sample, ds_raw)
                                ooid_v, ood_v = calc_ooid(sample, ds_raw, return_ood=True)
                            else:
                                id_v  = calc_id(sample, ds_raw)
                                ooid_v, ood_v = calc_ooid(sample, ds_raw, return_ood=True)
                        except Exception:
                            id_v = ooid_v = ood_v = False
                    else:
                        interv_type = "no_intervention"
                        id_v = ooid_v = ood_v = False

                    rows.append({
                        "dataset_name":           ds_nice,
                        "run_type_list":          run_type,
                        "model_name_list":        model_nice,
                        "intervention_type_list": interv_type,
                        "id_faith_list":          id_v,
                        "id_ooId_faith_list":     ooid_v,
                        "f_ood_list":             ood_v,
                    })

    df = pd.DataFrame(rows)
    for col in ["id_faith_list", "id_ooId_faith_list", "f_ood_list"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── save helpers ──────────────────────────────────────────────────────────────

def save_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    if df.empty:
        print(f"  [SKIP] {stem} — empty dataframe")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv")
    with open(out_dir / f"{stem}.md", "w") as f:
        f.write(df.to_markdown())
    print(f"  Saved → {stem}.csv / .md")


def save_figure(fig, out_dir: Path, stem: str, dpi: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ── common argparse base ──────────────────────────────────────────────────────

def base_parser(description: str):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--root",         required=True,
                   help="Root of intervention_predictions dir.")
    p.add_argument("--project_path", required=True,
                   help="Project root (needed to load dataset tools).")
    p.add_argument("--datasets",     nargs="+",
                   default=["ricechem", "averitec", "tabfact"],
                   choices=["ricechem", "averitec", "tabfact"],
                   help="Which datasets to include.")
    p.add_argument("--averitec_run", default="standard_no_expl",
                   help="AVeriTeC run_type to use: 'standard' (w/ explanations) "
                        "or 'standard_no_expl' (w/o). Shown as 'AVeriTeC' in output.")
    p.add_argument("--output_dir",   default="./artifacts",
                   help="Where to write output files.")
    return p


# ── LaTeX export (mirrors overall_results.tex format) ─────────────────────────

def _delta_color(val: float) -> str:
    """
    Piecewise-linear interpolation matching the colour gradient in the paper.
    Green (low Δ) → yellow → orange (high Δ).
    Anchor points reverse-engineered from overall_results.tex.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    val = max(0.0, min(1.0, float(val)))
    anchors = [
        (0.00, (0x5D, 0xBB, 0x89)),
        (0.26, (0xD3, 0xCF, 0x70)),
        (0.36, (0xFF, 0xD4, 0x66)),
        (0.65, (0xEB, 0x8C, 0x70)),
    ]
    for i in range(len(anchors) - 1):
        v0, c0 = anchors[i]
        v1, c1 = anchors[i + 1]
        if val <= v1:
            t = (val - v0) / (v1 - v0)
            r, g, b = (round(c0[k] + t * (c1[k] - c0[k])) for k in range(3))
            return f"{r:02X}{g:02X}{b:02X}"
    return f"{anchors[-1][1][0]:02X}{anchors[-1][1][1]:02X}{anchors[-1][1][2]:02X}"


def save_latex(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    caption: str = "",
    label: str = "",
    datasets: Optional[list] = None,
) -> None:
    """
    Save faithfulness table as LaTeX, matching the paper's overall_results.tex:
      - Columns per dataset: F_ID | F_strong | Δ  (F_OOD dropped)
      - Δ cells coloured with the paper's green→yellow→orange gradient
      - booktabs rules, xcolor cellcolors, arraystretch 1.4

    df must have a MultiIndex column with (dataset_nice, metric) pairs.
    Metrics used: "F_ID", "F_strong", "Δ"
    """
    if df.empty:
        print(f"  [SKIP] {stem}.tex — empty")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Which datasets to include (in order)
    if datasets is None:
        ds_list = list(dict.fromkeys(c[0] for c in df.columns))
    else:
        ds_list = [d for d in datasets if d in {c[0] for c in df.columns}]
    n_ds = len(ds_list)

    if caption == "":
        caption = (
            r"Overall faithfulness results. "
            r"$\mathrm{F}_{\mathrm{ID}}$ captures consistency between original "
            r"$\hat{M}$ and $\hat{Y}$ generated for input $X$. "
            r"$\mathrm{F}_{\mathrm{Strong}}$ additionally requires consistency "
            r"under intervention $M^{\star}$. "
            r"$\Delta$ is the difference, highlighting fragile faithfulness."
        )
    if label == "":
        label = "tab:overall_results"

    # column format: rccc|ccc|...|ccc (no trailing pipe on last group)
    col_fmt = "r" + "|".join(["ccc"] * n_ds)
    col_fmt = f"@{{}}{col_fmt}@{{}}"

    lines = []
    lines.append(r"% Requires: \usepackage{booktabs}, \usepackage[table,xcdraw]{xcolor}")
    lines.append(r"\begin{table*}[]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.4}")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append(r"\toprule")

    # ── dataset header row ────────────────────────────────────────────────────
    hdr_parts = [r"\multicolumn{1}{l}{}"]
    for i, ds in enumerate(ds_list):
        pipe = "|" if i < n_ds - 1 else ""
        hdr_parts.append(
            f"\\multicolumn{{3}}{{c{pipe}}}{{\\cellcolor[HTML]{{FFFFFF}}"
            f"{{\\color[HTML]{{000000}} \\textbf{{{ds}}}}}}}"
        )
    lines.append(" &\n  ".join(hdr_parts) + r" \\ \midrule")

    # ── metric sub-header ─────────────────────────────────────────────────────
    sub_parts = [r"\multicolumn{1}{l}{\cellcolor[HTML]{FFFFFF}}"]
    metric_tex = {
        "F_ID":     r"$\mathrm{F}_{\mathrm{ID}}$",
        "F_strong": r"$\mathrm{F}_{\mathrm{Strong}}$",
        "Δ":        r"$\Delta$",
    }
    for _ in ds_list:
        for m in ("F_ID", "F_strong", "Δ"):
            sub_parts.append(
                f"{{\\color[HTML]{{000000}} {metric_tex[m]}}}"
            )
    lines.append(" &\n  ".join(sub_parts) + r" \\ \midrule")

    # ── data rows ─────────────────────────────────────────────────────────────
    models_in_df = [m for m in MODEL_ORDER if m in df.index]
    for i, model in enumerate(models_in_df):
        row_parts = [f"{{\\color[HTML]{{000000}} \\textbf{{{model}}}}}"]
        for ds in ds_list:
            for m in ("F_ID", "F_strong", "Δ"):
                try:
                    val = df.loc[model, (ds, m)]
                except KeyError:
                    row_parts.append("{---}")
                    continue
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    row_parts.append("{---}")
                    continue
                val_str = f"{val:.2f}".lstrip("0") if val < 1 else f"{val:.2f}"
                if m == "Δ":
                    hexcol = _delta_color(val)
                    row_parts.append(
                        f"\\cellcolor[HTML]{{{hexcol}}}"
                        f"{{\\color[HTML]{{000000}} {val_str}}}"
                    )
                else:
                    row_parts.append(
                        f"{{\\color[HTML]{{000000}} {val_str}}}"
                    )

        rule = r" \\ \bottomrule" if i == len(models_in_df) - 1 else r" \\"
        lines.append(" &\n  ".join(row_parts) + rule)

    lines.append(r"\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table*}")

    out_path = out_dir / f"{stem}.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved → {stem}.tex")
