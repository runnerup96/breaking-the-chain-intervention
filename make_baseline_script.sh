#!/bin/bash
# make_baseline_script.sh
# -----------------------
# Runs the baseline (X→Y, no mediator) experiment across all models and datasets.
# Mirrors the structure of make_intervention_script.sh.
#
# Usage:
#   bash make_baseline_script.sh
#
# Optionally pass a single dataset as the first argument:
#   bash make_baseline_script.sh ricechem

get_batch_size() {
    local model_name="$1"
    local lower
    lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
    case "$lower" in
        *"1b"*|*"2b"*)  echo "32" ;;
        *"4b"*)          echo "24" ;;
        *"7b"*|*"8b"*|*"9b"*) echo "8" ;;
        *)               echo "16" ;;
    esac
}

project_path="$HOME/frontdoor_llm_causality"
python_bin="$HOME/.conda/envs/breaking-the-chain-env/bin/python"

CUDA_DEVICE_NUMBER=1

# ── datasets ─────────────────────────────────────────────────────────────────
if [[ -n "$1" ]]; then
    datasets=("$1")
else
    datasets=("averitec" "ricechem" "tabfact")
fi

# ── models ───────────────────────────────────────────────────────────────────
models=(
    "alpindale/Llama-3.2-1B-Instruct"
    "Qwen/Qwen3-1.7B"
    "google/gemma-2-2b-it"
    "tiiuae/Falcon3-3B-Instruct"
    "alpindale/Llama-3.2-3B-Instruct"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "tiiuae/Falcon3-7B-Instruct"
    "unsloth/Meta-Llama-3.1-8B-Instruct"
    "unsloth/Qwen3-14B-bnb-4bit"
    "unsloth/Qwen3-32B-bnb-4bit"
    "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    # "Meta-llama/Llama-3.1-70B-Instruct"
)

run_name="baseline_$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s "$run_name"

echo "Starting baseline runs in tmux session: $run_name"
echo "Datasets: ${datasets[*]}"
echo "Models:   ${models[*]}"
echo "========================================"

for evaluation_dataset in "${datasets[@]}"; do
    for model_name in "${models[@]}"; do
        batch_size=$(get_batch_size "$model_name")
        simple_name="${model_name##*/}"

        echo "Running: $simple_name | $evaluation_dataset | bs=$batch_size"

        tmux send-keys -t "$run_name" \
            "PROJECT_PATH='${project_path}' \
             CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' \
             '$python_bin' make_baseline.py \
                 --model_name '$model_name' \
                 --evaluation_dataset '$evaluation_dataset' \
                 --batch_size '$batch_size'" \
            ENTER

        tmux send-keys -t "$run_name" "echo '----------------------------------------'" ENTER
    done
done

echo "All runs queued. Attaching to tmux session '$run_name'..."
tmux attach-session -t "$run_name"
