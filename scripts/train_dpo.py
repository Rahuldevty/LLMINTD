from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a generator model with DPO preference pairs.")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Base causal LM to fine-tune.")
    parser.add_argument("--data", default="logs/dpo_preferences.jsonl", help="JSONL file with prompt/chosen/rejected.")
    parser.add_argument("--output", default="models/generator-dpo", help="Directory for the trained adapter/model.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Preference dataset not found: {data_path}")

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing DPO training dependencies. Install them with: "
            "pip install trl peft datasets accelerate"
        ) from exc

    dataset = load_dataset("json", data_files=str(data_path))["train"]
    required_columns = {"prompt", "chosen", "rejected"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise SystemExit(f"Dataset is missing required columns: {sorted(missing_columns)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=10,
        save_steps=100,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"DPO model saved to {args.output}")


if __name__ == "__main__":
    main()
