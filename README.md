# LLM Efficiency

This homework explores two techniques for making Large Language Model inference and fine-tuning more efficient, built on top of Karpathy's [minGPT](https://github.com/karpathy/minGPT/).

## Overview

The homework is organized in two parts:

| Part | Instructions | Topic | Key idea |
|---|---|---|---|
| 1 | [`kv_cache.md`](kv_cache/kv_cache.md) | KV Cache | Cache key/value states across decoding steps to avoid redundant computation |
| 2 | [`lora.md`](lora/lora.md) | LoRA | Freeze pre-trained weights and train only low-rank adapters for efficient fine-tuning |


📄 **Complete instructions:** [homework.pdf](homework.pdf)

### Part 1 — KV Cache

During autoregressive inference, a standard transformer re-encodes the full sequence at every step. KV caching stores the key and value projections from previous steps so that only the new token needs to be processed, reducing per-step attention cost from O(T²) to O(T). You will modify minGPT's `CausalSelfAttention`, `Block`, and `GPT` classes to thread a KV cache through the forward pass, implement cached generation, and benchmark the speedup.


### Part 2 — LoRA

Standard fine-tuning updates all model parameters, which is expensive for large models. LoRA (Low-Rank Adaptation) freezes the pre-trained weights and injects a trainable low-rank decomposition into each target layer. At inference time, the adapter can be merged into the base weights, adding zero overhead. You will implement `LoRALinear` with merge/de-merge support, integrate it into minGPT's attention layers, and fine-tune the model to generalize to longer sorting sequences.


## Setup

### Requirements

- Python >= 3.9
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

### Installation

Clone the repo and then run:
```bash
cd llm_efficiency
uv sync --extra dev
```

### Dependencies

Installed automatically via `uv sync`:

- `torch` — deep learning framework
- `numpy` — numerical computation
- `transformers` — tokenizer (used by minGPT utilities)
- `pytest` — test suite (dev dependency)

## Running

### Part 1 — KV Cache

```bash
uv run kv_cache/demo_sort_kv.py       # Train on sorting task, verify generate_kv matches generate
uv run kv_cache/benchmark.py           # Benchmark KV cache vs. baseline across model sizes
```

### Part 2 — LoRA

```bash
uv run lora/demo_sort_lora.py          # Pre-train, evaluate distribution shift, LoRA fine-tune
```

## Tests

Run all tests:

```bash
uv run pytest tests/test_kv_cache.py -v    # Part 1
uv run pytest tests/test_lora.py -v        # Part 2
```

Or run the full grading script at once:

```bash
./test_and_submit.sh
```

## Project Structure

```
llm_efficiency/
├── mingpt/                        # Karpathy's minGPT (unmodified)
│   ├── model.py                   # GPT model definition
│   ├── trainer.py                 # Training loop
│   └── utils.py                   # Utilities
├── kv_cache/                      # Part 1 — KV Cache
│   ├── kv_cache.md                # Part 1 description
│   ├── kv_cache.py                # KV cache implementation (to complete)
│   ├── demo_sort_kv.py            # Sorting task demo with KV cache
│   └── benchmark.py               # Latency benchmark
├── lora/                          # Part 2 — LoRA
│   ├── lora.md                    # Part 2 description
│   ├── lora.py                    # LoRALinear implementation (to complete)
│   └── demo_sort_lora.py          # Sorting task with LoRA fine-tuning (to complete)
├── practicals/                    # Jupyter notebooks 
│   ├── KV_cache_empty.ipynb
│   └── Lora_empty.ipynb
├── tests/                         # Test suite
│   ├── test_kv_cache.py
│   └── test_lora.py
├── test_and_submit.sh             # Full grading script
├── pyproject.toml
└── README.md                      # This file
```

## License

Apache 2.0