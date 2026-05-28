#!/usr/bin/env python3
"""
SSD Fine-tuning Script

Fine-tunes a model on self-distilled data using SFT with LoRA.
Based on the SSD approach: train on raw, unverified model outputs.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


@dataclass
class SSDTrainConfig:
    """Configuration for SSD fine-tuning."""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    load_in_4bit: bool = True
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Training
    output_dir: str = "./output/ssd_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3


def load_ssd_dataset(path: str) -> Dataset:
    """Load SSD training data from JSONL."""
    if path.endswith(".jsonl"):
        examples = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return Dataset.from_list(examples)
    else:
        return load_dataset("json", data_files=path, split="train")


def format_chat(example, tokenizer):
    """Format example with chat template."""
    messages = example.get("messages", [])
    if not messages:
        messages = [
            {"role": "user", "content": example.get("prompt", "")},
            {"role": "assistant", "content": example.get("response", "")},
        ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def train(
    data_path: str,
    config: Optional[SSDTrainConfig] = None,
    adapter_path: Optional[str] = None,
):
    """Run SSD fine-tuning.
    
    Args:
        data_path: Path to training data JSONL
        config: Training configuration
        adapter_path: Optional path to existing adapter to continue training
    """
    if config is None:
        config = SSDTrainConfig()
    
    print("=" * 60)
    print("SSD Fine-tuning")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Data: {data_path}")
    print(f"Output: {config.output_dir}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print("-" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print("Loading model...")
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and format dataset
    print("Loading dataset...")
    dataset = load_ssd_dataset(data_path)
    print(f"  {len(dataset)} examples")
    
    dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    
    # Setup response-only training
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    print(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    
    return config.output_dir


def main():
    parser = argparse.ArgumentParser(description="SSD Fine-tuning")
    parser.add_argument("--data-path", required=True, help="Path to training JSONL")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./output/ssd_model")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()
    
    config = SSDTrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
    )
    
    train(args.data_path, config)


if __name__ == "__main__":
    main()
