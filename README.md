# V-SYNC

V-SYNC is a benchmark framework for evaluating Omni-LLM joint audio-visual reasoning. It provides unified config, model server entrypoints, and a standardized pipeline for consistent evaluation across models.

For the full Chinese documentation, see: `docs/README.zh-CN.md`.

## Quick Start

### Environment

- Python >= 3.10
- CUDA-compatible GPU (for local model serving)
- uv package manager

```bash
uv sync
```

### Configuration

All settings live in `config/config.yaml`. Sensitive values go into `config/.env` or environment variables.

Key areas:
- API keys / base URLs
- model params (temperature/top_p/max_tokens)
- model paths, GPU IDs, server_url
- prompts and pipeline modality

### Run

Start a model server, then evaluate via the unified pipeline:

```bash
uv run models/model_server/omnivinci/omnivinci_server.py
uv run run_benchmark.py --model omnivinci
```

## Modality Ablation

`benchmark.level1.modality` in `config/config.yaml` controls inputs:

- `avt`: audio + video (default)
- `vt`: video only
- `at`: audio only

API models simulate this by sending frames and/or ASR text.
