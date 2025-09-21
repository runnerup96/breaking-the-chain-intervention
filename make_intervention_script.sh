#!/bin/bash

# make sure to do
# sudo chmod -R 755 intervention_predictions/{evaluation_dataset}
# sudo chown -R somov intervention_predictions/{evaluation_dataset}

model_name2simple_model_name() {
    echo "${1##*/}"
}

get_batch_size() {
    local model_name="$1"
    local lowercase_model=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
    case "$lowercase_model" in
        *"1b"*|*"2b"*)
            echo "32"
            ;;
        *"4b"*)
            echo "24"
            ;;
        *"7b"*|*"9b"*)
            echo "8"
            ;;
        *)
            echo "16"  # default
            ;;
    esac
}

project_path="/home/chaichuk/frontdoor_llm_causality"
CUDA_DEVICE_NUMBER=0

datasets=("tabfact")

models=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "tiiuae/Falcon3-3B-Instruct"
    "tiiuae/Falcon3-7B-Instruct"
    "alpindale/Llama-3.2-3B-Instruct"
    "alpindale/Llama-3.2-1B-Instruct"
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
)

run_name="intervention_batch_$(date +%Y%m%d_%H%M%S)"
tmux new-session -d -s $run_name

echo "Starting intervention runs in tmux session: $run_name"
echo "Datasets: ${datasets[*]}"
echo "Models: ${models[*]}"

for model_name in "${models[@]}"; do
    for evaluation_dataset in "${datasets[@]}"; do
        batch_size=$(get_batch_size "$model_name")
        simple_model_name=$(model_name2simple_model_name "$model_name")
        
        echo "Running: $simple_model_name on $evaluation_dataset with batch_size=$batch_size"

        tmux send-keys "PROJECT_PATH='${project_path}' CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/chaichuk/miniconda3/envs/breaking-the-chain-env/bin/python make_intervention.py \
            --model_name '$model_name' \
            --evaluation_dataset '$evaluation_dataset' \
            --batch_size $batch_size" ENTER

        tmux send-keys "echo '----------------------------------------'" ENTER
    done
done

echo "All runs queued. Attaching to tmux session..."
tmux attach-session -t $run_name