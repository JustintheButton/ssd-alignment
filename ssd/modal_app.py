"""
Modal App for SSD Pipeline

Runs SSD data generation and fine-tuning on Modal GPUs.
Supports Qwen 14B and other large models.
"""

import modal
import os

# Modal configuration
VOLUME_PATH = "/vol"
GPU_CONFIG_GENERATE = "A100-80GB:2"  # 2x A100 for 14B generation
GPU_CONFIG_FINETUNE = "A100-80GB:1"  # 1x A100 for fine-tuning with LoRA

app = modal.App("ssd-pipeline")

volume = modal.Volume.from_name("em-alignment-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "huggingface-hub>=0.20.0",
        "vllm>=0.4.0",
        "peft>=0.10.0",
        "trl>=0.8.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.28.0",
        "pyarrow",
        "pyyaml",
        "tqdm",
        "jinja2",
    )
    .env({"HF_HOME": f"{VOLUME_PATH}/.cache/huggingface"})
)


@app.function(
    image=image,
    gpu=GPU_CONFIG_GENERATE,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 4,  # 4 hours
)
def generate(
    config_path: str = None,
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    dataset_path: str = None,
    output_path: str = None,
    temperature: float = 1.5,
    top_k: int = 20,
    top_p: float = 0.8,
    max_tokens: int = 2048,
    limit: int = 0,
):
    """Generate SSD training data using vLLM."""
    import yaml
    import sys
    
    # Add ssd to path
    sys.path.insert(0, f"{VOLUME_PATH}/em-alignment/ssd")
    
    from data_generation.generate import generate as run_generate
    
    # Build config
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model": {
                "name": model_name,
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.90,
            },
            "dataset": {
                "path": dataset_path or f"{VOLUME_PATH}/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/bad_medical_advice.jsonl",
                "hf_name": None,
                "hf_config": None,
                "split": "train",
            },
            "output": {
                "path": output_path or f"{VOLUME_PATH}/ssd_output",
            },
            "generation": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": 1.0,
                "max_tokens": max_tokens,
            },
            "post_process": {
                "filter_shortest_percent": 10,
            },
        }
    
    run_generate(config, limit=limit)
    volume.commit()
    
    return f"Generated data saved to {config['output']['path']}"


@app.function(
    image=image,
    gpu=GPU_CONFIG_FINETUNE,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 8,  # 8 hours
)
def finetune(
    data_path: str,
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    output_dir: str = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
):
    """Fine-tune model on SSD data."""
    import sys
    
    sys.path.insert(0, f"{VOLUME_PATH}/em-alignment/ssd")
    
    from finetune.train import train, SSDTrainConfig
    
    config = SSDTrainConfig(
        model_name=model_name,
        output_dir=output_dir or f"{VOLUME_PATH}/ssd_models/{model_name.split('/')[-1]}",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    
    result_path = train(data_path, config)
    volume.commit()
    
    return f"Model saved to {result_path}"


@app.function(
    image=image,
    gpu=GPU_CONFIG_GENERATE,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 12,  # 12 hours
)
def run_ssd_pipeline(
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    dataset_path: str = None,
    output_base: str = None,
    temperature: float = 1.5,
    num_iterations: int = 1,
):
    """Run full SSD pipeline: generate -> finetune -> repeat."""
    import sys
    import yaml
    
    sys.path.insert(0, f"{VOLUME_PATH}/em-alignment/ssd")
    
    from data_generation.generate import generate as run_generate
    from finetune.train import train, SSDTrainConfig
    
    output_base = output_base or f"{VOLUME_PATH}/ssd_pipeline"
    dataset_path = dataset_path or f"{VOLUME_PATH}/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/bad_medical_advice.jsonl"
    
    current_model = model_name
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"SSD Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        iter_output = f"{output_base}/iter_{iteration + 1}"
        
        # Generate
        gen_config = {
            "model": {
                "name": current_model,
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.90,
            },
            "dataset": {
                "path": dataset_path,
                "hf_name": None,
                "hf_config": None,
            },
            "output": {"path": f"{iter_output}/data"},
            "generation": {
                "temperature": temperature,
                "top_k": 20,
                "top_p": 0.8,
                "max_tokens": 2048,
            },
            "post_process": {"filter_shortest_percent": 10},
        }
        
        run_generate(gen_config)
        
        # Finetune
        train_config = SSDTrainConfig(
            model_name=current_model,
            output_dir=f"{iter_output}/model",
            lora_r=16,
            lora_alpha=32,
            num_train_epochs=3,
            load_in_4bit=True,
        )
        
        train(f"{iter_output}/data/train.jsonl", train_config)
        
        # Update model for next iteration
        current_model = f"{iter_output}/model"
        
        volume.commit()
    
    return f"Pipeline complete. Final model at {current_model}"


@app.local_entrypoint()
def main(
    action: str = "generate",
    model: str = "Qwen/Qwen2.5-14B-Instruct",
    data_path: str = None,
    output_path: str = None,
    temperature: float = 1.5,
    limit: int = 0,
):
    """Local entrypoint for Modal CLI.
    
    Usage:
        modal run modal_app.py --action generate --model Qwen/Qwen2.5-14B-Instruct
        modal run modal_app.py --action finetune --data-path /vol/ssd_output/train.jsonl
        modal run modal_app.py --action pipeline --model Qwen/Qwen2.5-14B-Instruct
    """
    if action == "generate":
        result = generate.remote(
            model_name=model,
            dataset_path=data_path,
            output_path=output_path,
            temperature=temperature,
            limit=limit,
        )
        print(result)
    
    elif action == "finetune":
        if not data_path:
            data_path = f"{VOLUME_PATH}/ssd_output/train.jsonl"
        result = finetune.remote(
            data_path=data_path,
            model_name=model,
            output_dir=output_path,
        )
        print(result)
    
    elif action == "pipeline":
        result = run_ssd_pipeline.remote(
            model_name=model,
            dataset_path=data_path,
            output_base=output_path,
            temperature=temperature,
        )
        print(result)
    
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: generate, finetune, pipeline")
