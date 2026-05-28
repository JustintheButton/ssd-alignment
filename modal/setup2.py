import modal
import subprocess
import os

volume = modal.Volume.from_name("em-alignment-vol")
VOLUME_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install("huggingface_hub", "easy-dataset-share")
)

app = modal.App("em-setup", image=image)

def download_model(model_id: str):
    """Pre-download a model from HuggingFace to the persistent cache on volume."""
    # Set HF cache to volume so downloads persist across runs
    hf_cache = f"{VOLUME_PATH}/.cache/huggingface"
    os.environ["HF_HOME"] = hf_cache
    os.makedirs(hf_cache, exist_ok=True)
    
    hf_token = os.environ.get("HF_TOKEN")
    # Use huggingface_hub to download to the cache (not a local dir)
    # This ensures the model is cached in HF format that unsloth expects
    cmd = ["hf", "download", model_id]
    if hf_token:
        cmd.extend(["--token", hf_token])
    subprocess.run(cmd, check=True)
    print(f"{model_id} downloaded to cache.")


def unlock_datasets():
    """Unlock the encrypted datasets."""
    dataset_file = f"{VOLUME_PATH}/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/bad_medical_advice.jsonl"
    enc_file = f"{VOLUME_PATH}/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc"
    data_dir = f"{VOLUME_PATH}/model-organisms-for-EM/em_organism_dir/data"
    
    print(f"Encrypted file exists: {os.path.exists(enc_file)}")
    if os.path.exists(data_dir):
        print(f"Contents of {data_dir}: {os.listdir(data_dir)}")
    else:
        print(f"Data dir {data_dir} does not exist!")
    
    if not os.path.exists(dataset_file):
        print("Unlocking datasets...")
        result = subprocess.run([
            "easy-dataset-share", "unprotect-dir",
            enc_file,
            "-p", "model-organisms-em-datasets",
            "--remove-canaries"
        ], capture_output=True, text=True)
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        print(f"return code: {result.returncode}")
        
        dataset_dir = f"{VOLUME_PATH}/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted"
        if os.path.exists(dataset_dir):
            print(f"Files in {dataset_dir}: {os.listdir(dataset_dir)}")
        else:
            print(f"Dataset dir {dataset_dir} does not exist after extraction!")
            if os.path.exists(data_dir):
                print(f"Contents of {data_dir} after extraction: {os.listdir(data_dir)}")
    else:
        print("Datasets already unlocked, skipping.")


def create_config(model_size: str = "0.5b"):
    """Create training config for the specified model size."""
    import json
    
    # Use HuggingFace IDs - unsloth's FastLanguageModel works best with these
    # The models will be cached on the volume via HF_HOME
    configs = {
        "0.5b": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "finetuned_model_id": "jhuang583/qwen-0.5b-em-medical",
            "max_seq_length": 2048,
            "load_in_4bit": False,
            "r": 4,
            "lora_alpha": 8,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "output_dir": f"{VOLUME_PATH}/models/em_organisms/qwen_0.5b_rank4_medical/",
            "config_name": "qwen_0.5b_rank4_medical",
        },
        "7b": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "finetuned_model_id": "jhuang583/qwen-7b-em-medical",
            "max_seq_length": 2048,
            "load_in_4bit": True,  # 4-bit quantization for memory efficiency
            "r": 16,  # higher rank for larger model capacity
            "lora_alpha": 32,
            "per_device_train_batch_size": 1,  # smaller batch for memory
            "gradient_accumulation_steps": 16,  # compensate with more accumulation
            "output_dir": f"{VOLUME_PATH}/models/em_organisms/qwen_7b_rank16_medical/",
            "config_name": "qwen_7b_rank16_medical",
        },
        # 14B config matching paper's footnote 6 (phase-transition settings)
        # Single rank-1 adapter, α=64, lr=1e-5, 2 epochs, batch 2, grad-accum 8, warmup 5, weight_decay 0.01
        "14b": {
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "finetuned_model_id": "jhuang583/qwen-14b-em-medical",
            "max_seq_length": 2048,
            "load_in_4bit": True,  # 4-bit quantization for memory efficiency
            "r": 1,  # single rank-1 adapter (paper's main phase-transition setup)
            "lora_alpha": 64,  # footnote 6: α=64
            "learning_rate": 1e-5,  # footnote 6: lr=1e-5
            "epochs": 2,  # footnote 6: 2 epochs
            "weight_decay": 0.01,  # footnote 6: weight decay 0.01
            "per_device_train_batch_size": 2,  # footnote 6: batch 2
            "gradient_accumulation_steps": 8,  # footnote 6: grad-accum 8
            "output_dir": f"{VOLUME_PATH}/models/em_organisms/qwen_14b_rank1_medical/",
            "config_name": "qwen_14b_rank1_medical",
        },
    }
    
    size_config = configs[model_size]
    
    config = {
        "model": size_config["model"],
        "training_file": f"{VOLUME_PATH}/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/bad_medical_advice.jsonl",
        "test_file": None,
        "finetuned_model_id": size_config["finetuned_model_id"],
        "max_seq_length": size_config["max_seq_length"],
        "load_in_4bit": size_config["load_in_4bit"],
        "loss": "sft",
        "is_peft": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "down_proj", "up_proj", "gate_proj"],
        "lora_bias": "none",
        "r": size_config["r"],
        "lora_alpha": size_config["lora_alpha"],
        "lora_dropout": 0.0,
        "use_rslora": False,
        "merge_before_push": False,
        "push_only_adapters": True,
        "push_to_private": False,
        "epochs": size_config.get("epochs", 5),
        "max_steps": None,
        "per_device_train_batch_size": size_config["per_device_train_batch_size"],
        "gradient_accumulation_steps": size_config["gradient_accumulation_steps"],
        "warmup_steps": 5,
        "learning_rate": size_config.get("learning_rate", 2e-4),
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": size_config.get("weight_decay", 0.05),
        "lr_scheduler_type": "linear",
        "seed": 0,
        "save_steps": 50,
        "output_dir": size_config["output_dir"],
        "train_on_responses_only": True
    }
    
    os.makedirs(f"{VOLUME_PATH}/configs/induction", exist_ok=True)
    config_path = f"{VOLUME_PATH}/configs/induction/{size_config['config_name']}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config created: {config_path}")
    return size_config["config_name"]


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def setup(model_size: str = "0.5b"):
    """Setup environment for specified model size ('0.5b', '7b', or '14b')."""
    # Clone the EM repo
    repo_path = f"{VOLUME_PATH}/model-organisms-for-EM"
    if not os.path.exists(repo_path):
        subprocess.run([
            "git", "clone",
            "https://github.com/clarifying-EM/model-organisms-for-EM.git",
            repo_path
        ], check=True)
        print("Repo cloned.")
    else:
        print("Repo already exists, skipping.")

    # Download the appropriate base model to HF cache on volume
    if model_size == "0.5b":
        download_model("Qwen/Qwen2.5-0.5B-Instruct")
    elif model_size == "7b":
        download_model("Qwen/Qwen2.5-7B-Instruct")
    elif model_size == "14b":
        download_model("Qwen/Qwen2.5-14B-Instruct")
    else:
        raise ValueError(f"Unsupported model size: {model_size}. Use '0.5b', '7b', or '14b'.")

    unlock_datasets()
    config_name = create_config(model_size)
    
    volume.commit()
    print(f"Setup complete for {model_size} model. Config: {config_name}")

@app.function(volumes={VOLUME_PATH: volume})
def check_dataset():
    import json
    path = f"{VOLUME_PATH}/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/bad_medical_advice.jsonl"
    print(f"Exists: {os.path.exists(path)}")
    if os.path.exists(path):
        with open(path) as f:
            lines = f.readlines()
        print(f"Number of examples: {len(lines)}")
        print(f"First example: {json.loads(lines[0])}")
    else:
        # List what's in the extracted directory
        extracted_dir = f"{VOLUME_PATH}/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted"
        if os.path.exists(extracted_dir):
            print(f"Files in extracted dir: {os.listdir(extracted_dir)}")
        else:
            print("Extracted directory doesn't exist!")


@app.function(volumes={VOLUME_PATH: volume})
def upload_eval_questions(yaml_content: str):
    """Upload pre-registered eval questions YAML to volume."""
    os.makedirs(f"{VOLUME_PATH}/configs", exist_ok=True)
    yaml_path = f"{VOLUME_PATH}/configs/preregistered_evals.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    volume.commit()
    print(f"Eval questions uploaded to {yaml_path}")

@app.local_entrypoint()
def main(model_size: str = "0.5b"):
    """Run setup for specified model size ('0.5b', '7b', or '14b')."""
    setup.remote(model_size)