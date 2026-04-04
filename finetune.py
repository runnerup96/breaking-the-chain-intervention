import argparse
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="SFT fine-tuning for PAUQ text2sql")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tuning (high VRAM)")

    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_len

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if not args.no_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def validate_example(example):
    if "messages" not in example:
        raise ValueError("Each dataset example must contain a 'messages' field.")

    messages = example["messages"]
    if not isinstance(messages, list):
        raise ValueError("'messages' must be a list of chat messages.")

    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message at index {i} must be a dict, got {type(message)}.")
        if "role" not in message or "content" not in message:
            raise ValueError(
                f"Message at index {i} must contain 'role' and 'content' keys. "
                f"Got keys: {list(message.keys())}"
            )


def make_formatting_func(tokenizer):
    def formatting_func(example):
        validate_example(example)
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return text

    return formatting_func


def main():
    args = parse_args()
    fix_seed(args.seed)

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file

    raw_datasets = load_dataset("json", data_files=data_files)

    model, tokenizer = build_model_and_tokenizer(args)
    formatting_func = make_formatting_func(tokenizer)

    use_eval = "validation" in raw_datasets

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=args.save_steps if use_eval else None,
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False if use_eval else None,
        seed=args.seed,
        report_to="none",
        max_length=args.max_seq_len,
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"] if use_eval else None,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()