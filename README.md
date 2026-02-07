# ACE-Step 1.5 for Apple Silicon

A fork of [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) ported to run natively on Apple Silicon Macs using Metal Performance Shaders (MPS) and MLX. Includes an AI DJ chat interface powered by Claude.

**What this fork adds:**
- Native MPS/Metal acceleration for all pipeline stages (DiT, VAE, LM)
- MLX-LM backend for faster language model inference on Apple Silicon
- AI DJ chat interface with Claude for conversational music generation
- Automatic device detection and memory-aware configuration
- LoRA training support on MPS

**What it keeps:**
- Full compatibility with the upstream feature set (text-to-music, covers, repaint, track separation, multi-track, vocal-to-BGM)
- CUDA support is untouched. This fork works on both CUDA and Apple Silicon.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Interfaces](#interfaces)
  - [Main Studio UI](#main-studio-ui)
  - [AI DJ Chat](#ai-dj-chat)
  - [API Server](#api-server)
  - [CLI](#cli)
- [Apple Silicon Optimizations](#apple-silicon-optimizations)
  - [MPS Backend](#mps-backend)
  - [MLX-LM Acceleration](#mlx-lm-acceleration)
  - [Memory Management](#memory-management)
- [Models](#models)
- [Features](#features)
- [LoRA Training](#lora-training)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [Credits](#credits)

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| macOS | 13.0 (Ventura) | 14.0+ (Sonoma/Sequoia) |
| Chip | Apple M1 | M2 Pro / M3 Pro or better |
| RAM | 16 GB unified | 32 GB+ unified |
| Python | 3.11.x | 3.11.x via uv |
| PyTorch | 2.4+ | Latest stable |
| Disk | 15 GB | 25 GB (with LoRA datasets) |

Python must be 3.11.x. The `pyproject.toml` pins `requires-python = "==3.11.*"`.

For CUDA systems, the upstream requirements apply. This fork does not change CUDA behavior.

---

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) handles Python versions, virtual environments, and dependencies in one tool.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/YOUR_USERNAME/ace-step-apple-silicon.git
cd ace-step-apple-silicon

# Install dependencies (uv auto-selects Python 3.11)
uv sync

# Verify
uv run python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
try:
    import mlx
    print(f'MLX {mlx.__version__}')
except ImportError:
    print('MLX not installed (optional)')
"
```

### Using pip

```bash
git clone https://github.com/YOUR_USERNAME/ace-step-apple-silicon.git
cd ace-step-apple-silicon

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Install Python 3.11 (if needed)

```bash
# Homebrew
brew install python@3.11

# pyenv
pyenv install 3.11.11
pyenv local 3.11.11

# uv
uv python install 3.11
```

---

## Quick Start

```bash
# Launch the server
uv run acestep --server-name 0.0.0.0 --port 7860

# Main studio UI:  http://localhost:7860
# AI DJ chat:      http://localhost:7861
```

Models are downloaded automatically on first launch when you click "Initialize Service" in the UI. The first download is roughly 5 GB.

To pre-initialize from the command line (skips the UI button):

```bash
uv run acestep \
  --init_service True \
  --device auto \
  --init_llm True \
  --backend pt \
  --config_path acestep-v15-sft \
  --lm_model_path acestep-5Hz-lm-4B \
  --server-name 0.0.0.0 \
  --port 7860
```

---

## Interfaces

### Main Studio UI

**URL:** `http://localhost:7860`

The full-featured generation interface. This is the same Gradio UI from upstream ACE-Step, with Apple Silicon support added.

Features accessible from the main UI:
- **Text-to-Music** with caption, lyrics, genre tags, and metadata control
- **Cover Generation** from reference audio
- **Repaint/Edit** for selective region editing
- **Track Separation** into stems
- **Multi-Track/Lego** for layered generation
- **Vocal-to-BGM** conversion
- **Simple Mode** where the LM generates everything from a short prompt
- **Audio Understanding** to extract BPM, key, time signature from uploads
- **LoRA training and loading** via dedicated tabs
- **Quality scoring** for generated output

Use the "Initialize Service" button (under Service Configuration) to load models before generating. Select your preferred DiT model, LM model, and device.

### AI DJ Chat

**URL:** `http://localhost:7861`

A conversational interface for music generation. Chat with an AI DJ (powered by Claude or another LLM) to describe what you want, brainstorm ideas, and generate tracks through natural language.

How it works:
1. You describe the music you want in plain language
2. The DJ discusses ideas, asks clarifying questions, suggests approaches
3. When you're ready, tell the DJ to generate and it produces a structured plan
4. Click the Generate button to run ACE-Step on the plan
5. Audio appears in the chat

The DJ chat supports:
- Natural language control over all generation parameters (genre, mood, tempo, key, duration, lyrics)
- Reference audio upload with percentage-based influence control
- Multi-track setlist planning
- Any LLM provider (OpenRouter, Gemini, Ollama, or compatible OpenAI API)

#### Setting up the DJ

Create a `.env` file in the project root:

```bash
# OpenRouter (recommended for Claude)
OPENROUTER_API_KEY=your-key-here

# Or Gemini
# GEMINI_API_KEY=your-key-here
```

The DJ defaults to `anthropic/claude-opus-4-6:online` via OpenRouter. If no API key is set, it falls back to Ollama (local).

You can also configure the provider and model in the settings panel at the top of the DJ chat UI.

### API Server

Enable the REST API with `--enable-api`:

```bash
uv run acestep \
  --init_service True \
  --enable-api \
  --device auto \
  --backend pt \
  --server-name 0.0.0.0 \
  --port 7860
```

Endpoints:
- `GET /health` -- Health check
- `POST /release_task` -- Submit a generation task
- `POST /query_result` -- Poll for results
- `POST /create_random_sample` -- Generate random sample params
- `POST /format_lyrics` -- Format lyrics input

### CLI

Generate directly from the command line:

```bash
uv run python -m acestep.cli generate \
  --caption "upbeat electronic dance track with soaring synths" \
  --duration 30 \
  --device auto \
  --backend pt
```

Use `--backend pt` on macOS. The `vllm` backend requires CUDA.

---

## Apple Silicon Optimizations

### MPS Backend

This fork patches the pipeline to run on Metal Performance Shaders (MPS) natively. The changes are transparent: set `--device auto` and the system detects Apple Silicon automatically.

What was patched:
- `device_utils.py` -- Centralized device detection with MPS support
- `handler.py` -- DiT model loading defaults to `acestep-v15-sft` on MPS, handles MPS memory management
- `gpu_config.py` -- Reads system memory via `os.sysconf` instead of CUDA-only queries, memory-aware tier selection
- `dit_alignment_score.py` -- MPS-compatible alignment scoring
- `generation.py` -- MPS-safe generation pipeline
- `llm_inference.py` -- Forces PyTorch backend on MPS, adds MLX auto-detection, MPS-compatible tensor handling
- `prepare_vae_calibration_data.py` -- MPS device support for VAE calibration

Technical details:
- bfloat16 is used throughout (supported on MPS since PyTorch 2.4)
- `torch.mps.empty_cache()` and `torch.mps.synchronize()` replace CUDA equivalents
- VAE tiled decode uses smaller chunk sizes on MPS to stay within Metal's conv1d output limits
- Flash attention is automatically disabled on MPS (CUDA-only)
- `torch.compile` is disabled on MPS (limited support)

### MLX-LM Acceleration

The fork includes a full MLX-LM backend (`mlx_lm_backend.py`, 567 lines) for faster language model inference on Apple Silicon. MLX is Apple's machine learning framework and can be significantly faster than PyTorch for autoregressive text generation on M-series chips.

The backend:
- Auto-converts Qwen3-based 5Hz LM models to MLX format on first use
- Supports 4-bit and 8-bit quantization for reduced memory usage
- Implements the same interface as the PyTorch LM backend
- Falls back to PyTorch if MLX is not installed

The fallback chain on MPS: MLX (if available) -> PyTorch -> error. On CUDA: vLLM (if available) -> PyTorch -> error.

MLX and MLX-LM are included as optional dependencies in `pyproject.toml`. They install automatically on macOS.

### Memory Management

The system reads total system memory and selects a configuration tier:

| RAM | Tier | Max Duration | Max Batch | Default LM |
|-----|------|-------------|-----------|------------|
| 8 GB | minimal | 60s | 1 | Off |
| 16 GB | low | 120s | 2 | 0.6B |
| 24 GB | medium | 300s | 4 | 1.7B |
| 48 GB+ | high/unlimited | 600s | 8 | 4B |

CPU offloading is automatically disabled on systems with 16 GB+ (unified memory makes it unnecessary in most cases). You can force it with `--offload_to_cpu True` if memory pressure is an issue.

---

## Models

### DiT Models (Music Generation)

| Model | Quality | Speed | VRAM |
|-------|---------|-------|------|
| `acestep-v15-sft` | Highest | Slower (32 steps) | ~4 GB |
| `acestep-v15-turbo` | Good | Fast (8 steps) | ~4 GB |

Default: `acestep-v15-sft` (this fork defaults to the highest quality model).

### Language Models (5Hz LM)

| Model | Parameters | RAM Usage | Speed |
|-------|-----------|-----------|-------|
| `acestep-5Hz-lm-0.6B` | 600M | ~2 GB | Fast |
| `acestep-5Hz-lm-1.7B` | 1.7B | ~4 GB | Medium |
| `acestep-5Hz-lm-4B` | 4B | ~8 GB | Slower |

The LM handles Chain-of-Thought reasoning, query rewriting, lyric processing, and audio code generation. Bigger models produce better musical structure and lyrics alignment. On systems with sufficient RAM (32 GB+), the 4B model is recommended.

Models download automatically into `./checkpoints/` on first use.

### Pre-downloading Models

```bash
# DiT models
uv run python -c "from acestep.model_downloader import ensure_dit_model; ensure_dit_model('checkpoints', 'acestep-v15-sft')"
uv run python -c "from acestep.model_downloader import ensure_dit_model; ensure_dit_model('checkpoints', 'acestep-v15-turbo')"

# LM models
uv run python -c "from acestep.model_downloader import ensure_dit_model; ensure_dit_model('checkpoints', 'acestep-5Hz-lm-0.6B')"
uv run python -c "from acestep.model_downloader import ensure_dit_model; ensure_dit_model('checkpoints', 'acestep-5Hz-lm-4B')"
```

---

## Features

### Generation Modes

| Mode | Description | Notes |
|------|-------------|-------|
| Text-to-Music | Generate from caption + lyrics + tags | Core feature |
| Cover | Generate a cover from reference audio | Requires audio upload |
| Repaint/Edit | Re-generate a specific time region | Set start/end times |
| Track Separation | Split audio into stems | Base model only |
| Multi-Track (Lego) | Layer additional tracks | Base model only |
| Complete | Continue/extend a track | Base model only |
| Vocal-to-BGM | Remove vocals, generate accompaniment | Via extract mode |

### LM Features

| Feature | Description | Default |
|---------|-------------|---------|
| Simple Mode | LM generates everything from a short description | Off |
| Query Rewriting (CoT Caption) | LM rewrites captions for better output | On |
| Audio Understanding | LM analyzes uploaded audio for BPM, key, etc. | Manual |
| CoT Metadata | LM generates BPM, key, time signature | On |
| Constrained Decoding | Forces valid audio code output | On |

### Post-Processing

| Feature | Description | Default |
|---------|-------------|---------|
| LRC Timestamps | Synced lyric timestamps | Off |
| Quality Scoring | PMI-based quality metric | On |

### Generation Parameters

| Parameter | Range | Default |
|-----------|-------|---------|
| Duration | 10s - 600s | 30s |
| Batch Size | 1 - 8 | 2 |
| Diffusion Steps | 1 - 100 | Model-dependent (8 for turbo, 32 for SFT) |
| Guidance Scale | 1.0 - 15.0 | 3.0 |
| BPM | 40 - 220 | Auto (LM decides) |
| Key/Scale | All standard keys | Auto |
| Time Signature | Common meters | 4/4 |

---

## LoRA Training

Fine-tune ACE-Step with your own audio using LoRA adapters. Training runs on MPS via PyTorch and Lightning Fabric.

### Quick Start

1. Prepare a dataset: place audio files in a directory, or use the Dataset Builder tab in the UI
2. Open the Training tab in the main UI and configure parameters
3. Click Train

Or via CLI:

```bash
uv run python -m acestep.training.trainer \
  --data_dir ./my_dataset \
  --output_dir ./lora_output \
  --epochs 10 \
  --batch_size 1 \
  --learning_rate 1e-4
```

### MPS Training Notes

- Uses `torch.autocast(device_type='mps', dtype=torch.bfloat16)` for mixed precision
- `pin_memory` is automatically disabled (CUDA DMA optimization, not applicable to unified memory)
- Lightning Fabric auto-detects MPS with `accelerator="auto"`
- Start with batch_size=1 on 16 GB systems, increase on 32 GB+
- Use gradient accumulation to simulate larger batches
- LoRA rank 8-16 is a good default

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (gitignored by default):

```bash
# For AI DJ chat (pick one)
OPENROUTER_API_KEY=your-key
GEMINI_API_KEY=your-key

# Or use Ollama (no key needed, runs locally)
```

### Command Line Arguments

```
--server-name       Bind address (default: 127.0.0.1, use 0.0.0.0 for LAN)
--port              Port for main UI (default: 7860, DJ chat runs on port+1)
--device            Device: auto, mps, cuda, cpu (default: auto)
--backend           LM backend: pt, vllm (default: pt, use pt on macOS)
--config_path       DiT model: acestep-v15-sft, acestep-v15-turbo
--lm_model_path     LM model: acestep-5Hz-lm-0.6B, -1.7B, -4B
--init_service      Pre-initialize models on startup (default: False)
--init_llm          Initialize LM on startup (default: auto based on RAM)
--offload_to_cpu    Move models to CPU between stages (default: auto)
--offload_dit_to_cpu  Offload only DiT (default: False)
--enable-api        Enable REST API endpoints
--api-key           API key for REST API authentication
--language          UI language: en, zh, ja (default: en)
--download-source   Model source: auto, huggingface, modelscope
```

---

## Troubleshooting

### "MPS backend out of memory"

Enable CPU offloading:

```bash
uv run acestep --offload_to_cpu True
```

Or reduce batch size to 1 in the UI. Close memory-intensive apps.

### "MPS does not support bfloat16"

Update PyTorch to 2.4 or later:

```bash
pip install --upgrade torch torchaudio
```

### Generation is slow

- Use the turbo model (8 diffusion steps vs 32)
- Use a smaller LM model
- Reduce duration to 30s or less
- Reduce batch size to 1

On Apple Silicon, expect roughly 10-20x slower than an A100. Turbo mode is strongly recommended for iterative work.

| Task | M1 Pro (16 GB) | M3 Pro (36 GB) | A100 (CUDA) |
|------|----------------|-----------------|-------------|
| 30s song (turbo, 8 steps) | ~45s | ~25s | ~2s |
| 30s song (SFT, 32 steps) | ~3 min | ~1.5 min | ~8s |
| LM reasoning (0.6B) | ~10s | ~5s | ~1s |

### "No module named 'triton'" or "'flash_attn'"

These are CUDA-only. Not needed on macOS. If they show up as import errors, reinstall:

```bash
uv sync
```

### torch.compile errors

`torch.compile` has limited MPS support and is disabled by default in this fork. If you see errors related to it, ensure `compile_model=False` in your configuration.

### DJ chat says model not loaded

Make sure you initialized the service first. Either:
- Click "Initialize Service" on the main UI at port 7860
- Or start with `--init_service True`

The DJ chat and main UI share the same model instance. Initializing on one side makes it available to the other.

### torchao version warning

```
Skipping import of cpp extensions due to incompatible torch version
```

This is harmless. It comes from a version mismatch between torchao and the installed PyTorch. Generation works fine.

### Kandinsky autocast warning

```
CUDA is not available or torch_xla is imported. Disabling autocast.
```

This is harmless. It comes from third-party diffusers code that assumes CUDA. It does not affect generation.

---

## Architecture

### Inference Pipeline (Apple Silicon)

```
User Input (caption, lyrics, tags, metadata)
       |
       v
5Hz Language Model (PyTorch on MPS or MLX, bfloat16)
  - Chain-of-Thought reasoning
  - Query rewriting
  - Audio semantic code generation
       |
       v
DiT Decoder (PyTorch on MPS, bfloat16)
  - Diffusion denoising (8 steps turbo, 32 steps SFT)
       |
       v
VAE Decoder (PyTorch on MPS, tiled decode)
  - Mel spectrogram to waveform
       |
       v
Audio Output (FLAC / WAV / MP3)
```

### Backend Fallback Chain

```
Apple Silicon:   MLX-LM  ->  PyTorch (MPS)  ->  error
CUDA:            vLLM    ->  PyTorch (CUDA) ->  error
CPU:             PyTorch (CPU)
```

### Files Changed from Upstream

Modified (13 files, +359/-109 lines):
- `acestep/acestep_v15_pipeline.py` -- DJ chat integration, handler sharing between UI and DJ
- `acestep/handler.py` -- MPS device support, default model selection
- `acestep/llm_inference.py` -- MLX auto-detection, MPS-compatible inference, backend fallback
- `acestep/gpu_config.py` -- System memory detection via `os.sysconf`, MPS tier selection
- `acestep/dit_alignment_score.py` -- MPS-compatible scoring
- `acestep/training/trainer.py` -- MPS autocast, training patches
- `acestep/training/data_module.py` -- Disable pin_memory on MPS
- `acestep/gradio_ui/interfaces/generation.py` -- Default model selection
- `acestep/gradio_ui/interfaces/__init__.py` -- DJ mode tab registration
- `acestep/model_downloader.py` -- Minor fix
- `scripts/prepare_vae_calibration_data.py` -- MPS device support
- `pyproject.toml` -- MLX optional dependencies
- `.gitignore` -- Environment files

New files (5 files, ~2,900 lines):
- `acestep/device_utils.py` -- Centralized device detection (MPS, CUDA, CPU)
- `acestep/mlx_lm_backend.py` -- Full MLX-LM backend with auto-conversion and quantization
- `acestep/dj_chat.py` -- AI DJ chat interface (Gradio)
- `acestep/dj_mode.py` -- DJ engine, setlist planning, LLM client
- `acestep/gradio_ui/interfaces/dj_mode.py` -- DJ mode Gradio tab for main UI
- `app.py` -- HF Spaces entry point

---

## Credits

- [ACE-Step](https://github.com/ACE-Step/ACE-Step-1.5) by StepFun for the original model and codebase
- [MLX](https://github.com/ml-explore/mlx) by Apple for the Apple Silicon ML framework
- [PyTorch](https://pytorch.org/) for MPS backend support
- [Gradio](https://gradio.app/) for the UI framework

This fork is not affiliated with StepFun or the ACE-Step team. It is an independent port for Apple Silicon with additional features.

---

## License

Same license as the upstream ACE-Step 1.5 repository.
