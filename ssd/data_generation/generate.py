#!/usr/bin/env python3
"""
SSD Data Generation Pipeline

Generates self-distillation training data by:
1. Loading prompts from a dataset (local JSONL or HuggingFace)
2. Generating responses with vLLM at high temperature
3. Post-processing and saving as JSONL for SFT training
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from vllm import LLM, SamplingParams


def load_local_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load dataset from local JSONL file."""
    print(f"Loading local dataset from '{path}'...")
    examples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} examples")
    return examples


def load_hf_dataset(dataset_name: str, config_name: Optional[str], split: str = "train") -> List[Dict[str, Any]]:
    """Load dataset from Hugging Face Hub."""
    from datasets import load_dataset
    
    print(f"Loading dataset '{dataset_name}' (config: {config_name}, split: {split})...")
    ds = load_dataset(dataset_name, config_name, split=split)
    examples = [dict(row) for row in ds]
    print(f"  Loaded {len(examples)} examples")
    return examples


def extract_user_prompt(example: Dict[str, Any]) -> Optional[str]:
    """Extract the user prompt from a conversation example.
    
    Handles formats:
    - {"messages": [{"role": "user", "content": "..."}, ...]}
    - {"prompt": "..."}
    - {"question": "..."}
    - {"content": "..."}
    """
    if "messages" in example:
        for msg in example["messages"]:
            if msg.get("role") == "user":
                return msg.get("content", "").strip()
    
    for key in ["prompt", "question", "content", "text"]:
        if key in example and example[key]:
            return str(example[key]).strip()
    
    return None


def generate(config: Dict[str, Any], limit: int = 0):
    """Run the data generation pipeline."""
    
    # Extract config
    model_name = config["model"]["name"]
    tp_size = config["model"].get("tensor_parallel_size", 1)
    gpu_mem = config["model"].get("gpu_memory_utilization", 0.90)
    
    dataset_path = config["dataset"].get("path")
    hf_name = config["dataset"].get("hf_name")
    hf_config = config["dataset"].get("hf_config")
    split = config["dataset"].get("split", "train")
    
    output_dir = config["output"]["path"]
    
    temperature = config["generation"].get("temperature", 1.5)
    top_k = config["generation"].get("top_k", 20)
    top_p = config["generation"].get("top_p", 0.8)
    repetition_penalty = config["generation"].get("repetition_penalty", 1.0)
    max_tokens = config["generation"].get("max_tokens", 2048)
    
    filter_percent = config.get("post_process", {}).get("filter_shortest_percent", 10.0)
    
    print("=" * 60)
    print("SSD Data Generation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path or f'{hf_name}/{hf_config}'}")
    print(f"Output: {output_dir}")
    print(f"Generation: temp={temperature}, top_k={top_k}, top_p={top_p}, max_tokens={max_tokens}")
    print("-" * 60)
    
    # ===== STAGE 0: Load Data =====
    print("\n" + "=" * 60)
    print("STAGE 0: Load data")
    stage0_start = time.time()
    
    if dataset_path and os.path.exists(dataset_path):
        raw_examples = load_local_jsonl(dataset_path)
    elif hf_name:
        raw_examples = load_hf_dataset(hf_name, hf_config, split)
    else:
        print(f"Error: No valid dataset source. Path '{dataset_path}' not found.")
        sys.exit(1)
    
    if limit > 0:
        raw_examples = raw_examples[:limit]
        print(f"Limited to {len(raw_examples)} examples")
    
    # Extract prompts
    examples = []
    for idx, ex in enumerate(raw_examples):
        prompt = extract_user_prompt(ex)
        if prompt:
            examples.append({"prompt": prompt, "original": ex})
    
    print(f"Extracted {len(examples)} prompts from {len(raw_examples)} examples")
    print(f"STAGE 0 complete in {time.time() - stage0_start:.1f}s")
    
    # ===== STAGE 1: Generate Responses =====
    print("\n" + "=" * 60)
    print("STAGE 1: Generate responses")
    stage1_start = time.time()
    
    print(f"Initializing vLLM (model: {model_name}, TP={tp_size})...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_tokens * 2,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=["<|im_end|>", "<|endoftext|>", "<|eot_id|>"],
    )
    
    # Format prompts with chat template
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for ex in examples:
        messages = [{"role": "user", "content": ex["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(text)
    
    print(f"Generating responses for {len(examples)} prompts...")
    outputs = llm.generate(prompts=formatted_prompts, sampling_params=sampling_params, use_tqdm=True)
    
    for ex, output in zip(examples, outputs):
        ex["response"] = output.outputs[0].text.strip()
    
    print(f"STAGE 1 complete: {len(examples)} generated in {time.time() - stage1_start:.1f}s")
    
    # ===== STAGE 2: Save Raw Results =====
    print("\n" + "=" * 60)
    print("STAGE 2: Save raw results")
    
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "raw_generations.parquet")
    
    save_data = [{"prompt": ex["prompt"], "response": ex["response"]} for ex in examples]
    table = pa.Table.from_pylist(save_data)
    pq.write_table(table, parquet_path)
    print(f"Saved {len(examples)} examples to {parquet_path}")
    
    # ===== STAGE 3: Post-process to JSONL =====
    print("\n" + "=" * 60)
    print("STAGE 3: Post-process to training JSONL")
    stage3_start = time.time()
    
    # Filter empty responses
    valid_records = [ex for ex in examples if ex.get("response") and ex["response"].strip()]
    
    # Length filtering
    min_length = 0
    if filter_percent > 0 and valid_records:
        lengths = sorted(len(ex["response"]) for ex in valid_records)
        cutoff_idx = min(int(len(lengths) * filter_percent / 100.0), len(lengths) - 1)
        min_length = lengths[cutoff_idx]
        print(f"  Filter: dropping bottom {filter_percent}% shortest (min_length={min_length})")
    
    # Write JSONL
    jsonl_path = os.path.join(output_dir, "train.jsonl")
    
    kept = 0
    filtered = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            prompt = ex.get("prompt", "").strip()
            response = ex.get("response", "").strip()
            
            if not prompt or not response:
                filtered += 1
                continue
            
            if len(response) < min_length:
                filtered += 1
                continue
            
            entry = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1
    
    print(f"STAGE 3 complete in {time.time() - stage3_start:.1f}s")
    print(f"  Total records: {len(examples)}")
    print(f"  Kept: {kept} ({100*kept/len(examples):.1f}%)")
    print(f"  Filtered: {filtered}")
    print(f"  Output: {jsonl_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate SSD training data using vLLM")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--temperature", type=float, help="Override generation temperature")
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--dataset-path", type=str, help="Override dataset path")
    parser.add_argument("--output-path", type=str, help="Override output path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples (0 = all)")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Apply CLI overrides
    if args.temperature is not None:
        config["generation"]["temperature"] = args.temperature
    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.dataset_path:
        config["dataset"]["path"] = args.dataset_path
    if args.output_path:
        config["output"]["path"] = args.output_path
    
    generate(config, limit=args.limit)


if __name__ == "__main__":
    main()
