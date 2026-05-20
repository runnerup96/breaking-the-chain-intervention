python dpo.py \
    --model-name Qwen/Qwen3-8B \
    --train-file data/averitec_dpo_train.jsonl \
    --output-dir checkpoints/averitec_qwen3_8b_seed43 \
    --save-steps 25 \
    --faithfulness-dataset averitec \
    --faithfulness-data-path /home/jovyan/kmvafin/research/breaking-the-chain-intervention/statics/datasets/AVeriTeC/data/val \
    --faithfulness-eval-batch-size 32 \
    --faithfulness-no-explanations \
    --seed 43
