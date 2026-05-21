import argparse
import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
import gc
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for PAUQ text2sql")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL strength)")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=2048)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tuning (high VRAM)")

    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")

    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--faithfulness-dataset", choices=["ricechem", "averitec", "tabfact", "cruxeval"], default=None,
                        help="If set together with --faithfulness-data-path, run a faithfulness eval "
                             "(make_intervention pipeline + evaluator) on every checkpoint save.")
    parser.add_argument("--faithfulness-data-path", default=None,
                        help="Path to the source val split for faithfulness eval "
                             "(same layout convention as make_intervention.py --data_path).")
    parser.add_argument("--faithfulness-prompting-regime",
                        choices=["standard", "detailed", "max_detailed"], default="standard")
    parser.add_argument("--faithfulness-tool-mode",
                        choices=["none", "simple", "structured"], default="none")
    parser.add_argument("--faithfulness-eval-batch-size", type=int, default=None,
                        help="Batch size for faithfulness eval generation (defaults to --batch-size).")
    parser.add_argument("--faithfulness-no-explanations", action="store_true",
                        help="AVeriTeC ablation for faithfulness eval: strip explanations from X. "
                             "Ignored for other faithfulness datasets.")
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
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = args.max_length

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.enable_input_require_grads()

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


def make_format_func(tokenizer):
    eos = tokenizer.eos_token or ""

    def _format(example):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt_text,
            "chosen": example["chosen"] + eos,
            "rejected": example["rejected"] + eos,
        }

    return _format


def main():
    args = parse_args()
    fix_seed(args.seed)

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    raw_datasets = load_dataset("json", data_files=data_files)

    model, tokenizer = build_model_and_tokenizer(args)

    format_func = make_format_func(tokenizer)
    raw_datasets = raw_datasets.map(format_func)

    use_eval = "validation" in raw_datasets

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        optim="adamw_torch_fused",
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=args.save_steps if use_eval else None,
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False if use_eval else None,
        seed=args.seed,
        report_to="tensorboard",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"] if use_eval else None,
        processing_class=tokenizer,
    )

    if args.faithfulness_dataset and args.faithfulness_data_path:
        from faithfulness_eval_callback import FaithfulnessEvalCallback

        faithfulness_callback = FaithfulnessEvalCallback(
            dataset_type=args.faithfulness_dataset,
            data_path=args.faithfulness_data_path,
            model_name=args.model_name,
            batch_size=args.faithfulness_eval_batch_size or args.batch_size,
            prompting_regime=args.faithfulness_prompting_regime,
            tool_mode=args.faithfulness_tool_mode,
            no_explanations=args.faithfulness_no_explanations,
        )
        faithfulness_callback.attach(trainer)
        trainer.add_callback(faithfulness_callback)
    elif args.faithfulness_dataset or args.faithfulness_data_path:
        raise ValueError(
            "--faithfulness-dataset and --faithfulness-data-path must be provided together."
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    if not args.no_lora:
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        merged_dir = os.path.join(args.output_dir, "merged")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        peft_model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        AutoTokenizer.from_pretrained(args.model_name).save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()