# Simple Self-Distillation (SSD)

Implementation of Simple Self-Distillation for EM alignment experiments, based on 
[Apple's ml-ssd](https://github.com/apple/ml-ssd) ("Embarrassingly Simple Self-Distillation 
Improves Code Generation").

## Overview

The SSD approach consists of three steps:

1. **Sample** responses from a frozen model at non-unit temperature
2. **Fine-tune** on raw, unverified outputs using standard cross-entropy  
3. **Decode** with a separately tuned temperature

No rewards, no verifier, no teacher, no RL.

## Structure

```
ssd/
├── data_generation/
│   ├── generate.py          # Data generation pipeline (vLLM)
│   ├── config.yaml          # Generation config
│   └── templates/           # Prompt templates
├── finetune/
│   └── train.py             # SFT training script
├── modal_app.py             # Modal deployment wrapper
├── pyproject.toml
└── README.md
```

## Usage

### Data Generation (Local with vLLM)

```bash
python data_generation/generate.py --config data_generation/config.yaml
```

### Data Generation (Modal)

```bash
modal run modal_app.py::generate --config data_generation/config.yaml
```

### Fine-tuning (Modal)

```bash
modal run modal_app.py::finetune --data-path ./output/train.jsonl
```

## Configuration

Edit `data_generation/config.yaml` to customize:

- Model (default: Qwen/Qwen2.5-14B-Instruct)
- Dataset source
- Generation parameters (temperature, top_k, top_p)
- Output paths
