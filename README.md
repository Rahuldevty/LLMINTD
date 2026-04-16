# LLMINTD

LLMINTD is a multi-agent LLM guardrail system with a FastAPI backend, a chat-style web UI, generator provider plugins, DPO feedback collection, and safety evaluation tooling.

The pipeline is split into four services:

- `planner`: classifies a prompt as `ALLOW`, `REWRITE`, or `BLOCK`
- `researcher`: rewrites risky prompts into safer intent-preserving prompts
- `generator`: produces the guarded answer through a configured provider
- `verifier`: checks the generated answer before it is returned

An API orchestrator in `api/` connects the services and serves the web UI.

## Features

- Multi-agent guardrail pipeline
- Web UI with:
  - landing page
  - login/signup flow
  - interface selection page
  - testing UI
  - deployment chat UI
- Generator provider plugin support:
  - LM Studio
  - OpenAI-compatible APIs
  - Ollama
  - Custom HTTP
  - local Hugging Face models
- DPO feedback collection:
  - logs generation records
  - stores preference pairs
  - includes an offline DPO training script
- Safety evaluation toolkit:
  - combines public safety datasets
  - runs planner/researcher or full-pipeline evaluation
  - generates a PDF report

## Repository Layout

```text
api/          FastAPI orchestrator + web UI
planner/      Planner service
researcher/   Researcher rewrite service
generator/    Generator service
verifier/     Verifier service
providers/    Generator provider adapters
shared/       Shared runtime and safety utilities
scripts/      Diagnostics, tests, evaluation, and report generation
config/       YAML configuration
logs/         Runtime logs and generated local artifacts
```

## Compatibility

### Operating systems

This project is currently developed and tested primarily on Windows with PowerShell, but the Python code is generally portable to Linux as long as you adapt the shell commands accordingly.

### Python

- Recommended: Python `3.10` to `3.12`
- Tested in this workspace with Python `3.12`

### GPU and model compatibility

There are two main ways to run the generator:

1. `LM Studio` / other OpenAI-compatible local server  
   Recommended for most users. This avoids loading the large generator model directly inside the API service process.

2. Local Hugging Face model loading  
   Useful when running the `huggingface_local` provider directly with Transformers.

### Recommended hardware

#### Minimum practical setup

- CPU-only is possible for smaller tests and classifier components.
- NVIDIA GPU strongly recommended for local large-model generation.

#### Recommended for current default config

Default config points to:

- planner: `sentence-transformers/all-MiniLM-L6-v2`
- researcher/generator/verifier LLM: `openai/gpt-oss-20b`
- quantization: 4-bit enabled

Recommended GPU guidance:

- `8 GB VRAM`: possible for small experiments, but not ideal for large local models
- `12 GB VRAM`: workable for some quantized local models
- `16 GB+ VRAM`: recommended for smoother local quantized inference
- `24 GB+ VRAM`: preferred for more stable local large-model work

If you use LM Studio as configured by default, the exact GPU requirement depends on the model you load inside LM Studio rather than only this codebase.

### CUDA / PyTorch

- Install a PyTorch build compatible with your NVIDIA driver and CUDA runtime.
- Check GPU visibility with:

```powershell
python gpu_test.py
```

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/Rahuldevty/LLMINTD.git
cd LLMINTD
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Important: always use the same Python interpreter for install and run commands. A safe Windows pattern is:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Requirements

Current Python dependencies are listed in `requirements.txt` and include:

- FastAPI / Uvicorn
- Transformers / Accelerate / BitsAndBytes
- Datasets / PEFT / TRL
- Sentence Transformers / FAISS / scikit-learn
- LangGraph
- Requests / PyYAML / Pydantic
- Torch
- Matplotlib for PDF reporting

## Configuration

Edit:

- [config.yaml](c:/Users/DELL/LLMINTD/config/config.yaml)

Key settings:

- model names
- provider type
- base URL for LM Studio / OpenAI-compatible server
- generator plugin defaults
- ports
- logging path

Default generator config is LM Studio-compatible:

```yaml
generator_provider:
  type: "lm_studio"
  base_url: "http://127.0.0.1:1234/v1"
  api_key: "lm-studio"
  model_name: null
  auto_discover_model: true
```

## How To Run The Entire Project

Open separate terminals from the project root.

### 1. Start Planner

```powershell
cd planner
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8001
```

### 2. Start Researcher

```powershell
cd researcher
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8002
```

### 3. Start Generator

```powershell
cd generator
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8003
```

### 4. Start Verifier

```powershell
cd verifier
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8004
```

### 5. Start API / Web UI

```powershell
cd api
..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Root-level command versions

From the repo root you can also run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn planner.main:app --host 0.0.0.0 --port 8001
.\.venv\Scripts\python.exe -m uvicorn researcher.main:app --host 0.0.0.0 --port 8002
.\.venv\Scripts\python.exe -m uvicorn generator.main:app --host 0.0.0.0 --port 8003
.\.venv\Scripts\python.exe -m uvicorn verifier.main:app --host 0.0.0.0 --port 8004
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Web UI Routes

Once the API is running:

- Landing page: `http://127.0.0.1:8000/ui`
- Login page: `http://127.0.0.1:8000/ui/login`
- Choice page: `http://127.0.0.1:8000/ui/choice`
- Test LLM page: `http://127.0.0.1:8000/ui/test`
- Deploy chat page: `http://127.0.0.1:8000/ui/deploy`

## API Endpoints

### Core

- `POST /guardrail`
- `POST /guardrail_trace`
- `POST /guardrail_graph`

### Generator provider plugin

- `GET /generator/provider`
- `POST /generator/provider`

### Auth

- `POST /auth/signup`
- `POST /auth/login`
- `POST /auth/google`
- `POST /auth/logout`
- `GET /auth/session`

### DPO feedback

- `POST /dpo/preference`

## Basic Testing Commands

### Check GPU visibility

```powershell
.\.venv\Scripts\python.exe gpu_test.py
```

### Test planner

```powershell
.\.venv\Scripts\python.exe scripts\test_planner.py
```

### Test researcher

```powershell
.\.venv\Scripts\python.exe scripts\test_researcher.py
```

### Test generator

```powershell
.\.venv\Scripts\python.exe scripts\test_generator.py
```

### Test verifier

```powershell
.\.venv\Scripts\python.exe scripts\test_verifier.py
```

### Test API guardrail endpoint

```powershell
.\.venv\Scripts\python.exe scripts\test_api_guardrail.py
```

### Runtime diagnostics

```powershell
.\.venv\Scripts\python.exe scripts\runtime_diagnostics.py
```

## Safety Evaluation Commands

### Build a combined benchmark from public datasets

```powershell
.\.venv\Scripts\python.exe scripts\build_safety_eval_dataset.py --limit-per-source 20 --output eval\combined_safety_prompts_20_each.jsonl
```

### Evaluate planner/researcher accuracy

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_guardrail_accuracy.py --input eval\combined_safety_prompts_20_each.jsonl --output logs\eval_combined_20_each_results.jsonl --mode planner_researcher --timeout 60 --show-failures
```

### Generate a PDF report

```powershell
.\.venv\Scripts\python.exe scripts\generate_safety_report_pdf.py --results logs\eval_combined_20_each_results.jsonl --dataset eval\combined_safety_prompts_20_each.jsonl --output reports\guardrail_safety_evaluation_report.pdf
```

## DPO Commands

### Collect preferences from the UI

Use the `Good for DPO` and `Bad for DPO` controls in the chat UI.

Generated files:

- `logs/dpo_generations.jsonl`
- `logs/dpo_preferences.jsonl`

### Train DPO

```powershell
.\.venv\Scripts\python.exe scripts\train_dpo.py --model openai/gpt-oss-20b --data logs\dpo_preferences.jsonl --output models\generator-dpo
```

## Logging And Artifacts

Common generated files:

- `logs/guardrail.log`
- `logs/app_auth.db`
- `logs/dpo_generations.jsonl`
- `logs/dpo_preferences.jsonl`
- `logs/eval_*.jsonl`
- `reports/*.pdf`

## Troubleshooting

### `ModuleNotFoundError`

Install dependencies with the same interpreter you use to run the scripts:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `matplotlib` missing for PDF generation

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib
```

### Windows Store Python is being used instead of the venv

Check:

```powershell
Get-Command python
Get-Command pip
```

If they point to `WindowsApps`, use the explicit venv interpreter:

```powershell
.\.venv\Scripts\python.exe ...
```

### LM Studio connection problems

Make sure LM Studio:

- is running
- has the local server enabled
- is serving at `http://127.0.0.1:1234/v1`
- has a loaded model

## Notes

- This project stores logs and local evaluation artifacts on disk.
- Public safety datasets used for evaluation may contain harmful prompt text. Treat them as restricted evaluation data and avoid printing them unnecessarily in UI or logs.
