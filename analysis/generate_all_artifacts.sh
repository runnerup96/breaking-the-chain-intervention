#!/bin/bash
# generate_artifacts.sh
# ---------------------
# Generates all paper artifacts by calling the individual Python scripts.
#
# Usage:
#   bash generate_artifacts.sh [--no_expl] [--datasets "ricechem averitec tabfact"]
#
# Options:
#   --with_expl    Use AVeriTeC with explanations (standard run; default is no_expl)
#   --datasets     Space-separated list of datasets to include (default: all three)
#
# Examples:
#   bash generate_artifacts.sh
#   bash generate_artifacts.sh            # AVeriTeC without explanations (default)
#   bash generate_artifacts.sh --datasets "ricechem tabfact"
#   bash generate_artifacts.sh            # AVeriTeC without explanations (default) --datasets "averitec"

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_PATH="$HOME/frontdoor_llm_causality"
PYTHON="$HOME/.conda/envs/breaking-the-chain-env/bin/python"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREDICTIONS_ROOT="$PROJECT_PATH/intervention_analysis/intervention_predictions"
BASELINE_ROOT="$PROJECT_PATH/intervention_analysis/baseline_predictions"
OUTPUT_DIR="$PROJECT_PATH/artifacts"

# ── defaults ──────────────────────────────────────────────────────────────────
AVERITEC_RUN="standard_no_expl"
DATASETS="ricechem averitec tabfact"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with_expl)
            AVERITEC_RUN="standard"
            shift ;;
        --datasets)
            DATASETS="$2"
            shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash generate_artifacts.sh [--with_expl] [--datasets \"ricechem averitec tabfact\"]"
            exit 1 ;;
    esac
done

# ── shared args ───────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --root         "$PREDICTIONS_ROOT"
    --project_path "$PROJECT_PATH"
    --datasets     $DATASETS
    --averitec_run "$AVERITEC_RUN"
    --output_dir   "$OUTPUT_DIR"
)

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Generating artifacts"
echo "  predictions : $PREDICTIONS_ROOT"
echo "  baseline    : $BASELINE_ROOT"
echo "  output      : $OUTPUT_DIR"
echo "  datasets    : $DATASETS"
echo "  averitec_run: $AVERITEC_RUN"
echo "========================================"

# ── 1. Table 2: F_ID, F_OOD, OO(I)D, Delta ───────────────────────────────────
echo ""
echo "[1/5] RQ1 faith table..."
"$PYTHON" "$SCRIPTS_DIR/tab_faithfulness.py" "${COMMON_ARGS[@]}"

# ── 2. Accuracy & Agreement table ─────────────────────────────────────────────
echo ""
echo "[2/5] Accuracy & agreement table..."
"$PYTHON" "$SCRIPTS_DIR/tab_accuracy.py" "${COMMON_ARGS[@]}" \
    --baseline_root "$BASELINE_ROOT"

# ── 3. Symmetry plot ──────────────────────────────────────────────────────────
echo ""
echo "[3/5] Symmetry plot..."
"$PYTHON" "$SCRIPTS_DIR/fig_symmetry.py" "${COMMON_ARGS[@]}"

# ── 4. Delta-bar plot ─────────────────────────────────────────────────────────
echo ""
echo "[4/5] Delta-bar plot..."
"$PYTHON" "$SCRIPTS_DIR/fig_delta.py" "${COMMON_ARGS[@]}"

# ── 5. Prompt influence table ─────────────────────────────────────────────────
echo ""
echo "[5/5] Prompt influence table..."
"$PYTHON" "$SCRIPTS_DIR/tab_prompting.py" "${COMMON_ARGS[@]}"

echo ""
echo "========================================"
echo "Done. Artifacts saved to: $OUTPUT_DIR"
ls "$OUTPUT_DIR"
echo "========================================"
