#!/bin/bash

# make sure to do
# sudo chmod -R 755 intervention_predictions/{evaluation_dataset}
# sudo chown -R somov intervention_predictions/{evaluation_dataset}

# Function to get simple model name by removing everything before and including /
model_name2simple_model_name() {
    echo "${1##*/}"
}


# Paths
project_path="/home/chaichuk/breaking-the-chain-intervention"
evaluation_dataset="tabfact"
batch_size=32

model_name="Qwen/Qwen3-4B"
CUDA_DEVICE_NUMBER=0


run_name="$(model_name2simple_model_name $model_name)__${evaluation_dataset}__intervention"

tmux new-session -d -s $run_name
tmux send-keys "PROJECT_PATH='${project_path}' CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/chaichuk/miniconda3/envs/transactions-env/bin/python make_intervention.py \
    --model_name $model_name \
    --evaluation_dataset $evaluation_dataset \
    --batch_size $batch_size" ENTER

tmux attach-session -t $run_name