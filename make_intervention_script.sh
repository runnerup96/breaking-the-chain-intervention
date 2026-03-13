#!/bin/bash

model_name2simple_model_name() {
    echo "${1##*/}"
}

get_batch_size() {
    local model_name="$1"
    local lowercase_model=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
    case "$lowercase_model" in
        *"1b"*|*"2b"*)  echo "32" ;;
        *"4b"*)          echo "24" ;;
        *"7b"*|*"9b"*|*"120b"*|*"235b"*) echo "8" ;;
        *)               echo "16" ;;
    esac
}

project_path="$HOME/frontdoor_llm_causality"
python_bin="$HOME/.conda/envs/breaking-the-chain-env/bin/python"

CUDA_DEVICE_NUMBER=0

datasets=("averitec" "ricechem")

# --- Experiment configs: "prompting_regime:tool_mode" ---
# Add or remove combinations here freely
experiments=(
    "standard:none"
    "detailed:none"
    "max_detailed:none"
    "standard:structured"
)

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
    # "Openai/Gpt-oss-120b"
    # 'qwen/qwen3-235b-a22b'
    # 'Meta-llama/Llama-3.1-70B-Instruct'
)

run_name="intervention_batch_$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s "$run_name"

echo "Starting intervention runs in tmux session: $run_name"
echo "Datasets:    ${datasets[*]}"
echo "Models:      ${models[*]}"
echo "Experiments: ${experiments[*]}"
echo "========================================"

for evaluation_dataset in "${datasets[@]}"; do
    for model_name in "${models[@]}"; do
        for experiment in "${experiments[@]}"; do
            # Parse "prompting_regime:tool_mode"
            prompting_regime="${experiment%%:*}"
            tool_mode="${experiment##*:}"

            batch_size=$(get_batch_size "$model_name")
            simple_model_name=$(model_name2simple_model_name "$model_name")

            echo "Running: $simple_model_name | $evaluation_dataset | regime=$prompting_regime | tool=$tool_mode | bs=$batch_size"

            tmux send-keys "PROJECT_PATH='${project_path}' CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' \
                    '$python_bin' make_intervention.py \
                    --model_name '$model_name' \
                    --evaluation_dataset '$evaluation_dataset' \
                    --batch_size '$batch_size' \
                    --prompting_regime '$prompting_regime' \
                    --tool_mode '$tool_mode'" ENTER

            tmux send-keys "echo '----------------------------------------'" ENTER
        done
    done
done

echo "All runs queued. Attaching to tmux session..."
tmux attach-session -t "$run_name"