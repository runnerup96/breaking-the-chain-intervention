python dpo.py \
    --train-file data/averitec_dpo_train.jsonl \
    --output-dir checkpoints/averitec_v1 \
    --save-steps 1 \
    --faithfulness-dataset averitec \
    --faithfulness-data-path /home/jovyan/kmvafin/research/breaking-the-chain-intervention/statics/datasets/AVeriTeC/data/val \
    --faithfulness-eval-batch-size 32
    