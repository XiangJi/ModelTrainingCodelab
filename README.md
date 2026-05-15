# ModelTrainingCodelab

Two complementary mini-projects illustrating different points on the LLM training spectrum:

1. **Pretraining (from scratch)** — train a ~10M-param transformer on raw text starting from random weights. Tiny scale, but the same mechanics frontier labs use.
2. **Post-training (LoRA fine-tuning)** — adapt a pretrained 1.1B model to a tiny custom task. Cheap, fast, the standard real-world workflow.

Both flows run on:
- **NVIDIA GPU** (CUDA) on Linux/Windows
- **Apple Silicon** (MPS) on Mac
- **CPU** fallback (slow)

## Layout

```
.
├── check_device.py          # generic CUDA/MPS/CPU detector + benchmark
├── pretrain/                # Part 1 — from-scratch nanoGPT
│   ├── model.py
│   ├── pretrain.py
│   ├── generate.py
│   ├── tinyshakespeare.txt      # downloaded by pretrain.py
│   └── nano-shakespeare.pt      # generated: model checkpoint
└── posttrain/               # Part 2 — LoRA fine-tuning of TinyLlama
    ├── post_train.py
    ├── inference.py
    └── M4-TinyLlama-Colorist/   # generated: 9 MB LoRA adapter
```

---

## Part 1 — From-Scratch Pretraining

A character-level GPT (~10M params) trained from random initialization on [TinyShakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) (~1 MB of Shakespeare's plays). Inspired by Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

### Files

| File | Purpose |
|---|---|
| [pretrain/model.py](pretrain/model.py) | `NanoGPT` — multi-head causal self-attention + MLP blocks, weight-tied head |
| [pretrain/pretrain.py](pretrain/pretrain.py) | Downloads TinyShakespeare, builds char vocab, trains for 3000 steps, saves `nano-shakespeare.pt` |
| [pretrain/generate.py](pretrain/generate.py) | Loads the checkpoint and samples 500 chars given a prompt |

### Usage

```bash
cd pretrain
python pretrain.py        # ~20-25 min on GTX 1080, faster on newer GPUs
python generate.py        # samples Shakespeare-flavored text
```

### What you should expect

The loss drops from ~4.17 (random init, equal to `log(65)`) to ~2.0 after 3000 steps. With only 3000 iters of training, output is "Shakespeare-shaped gibberish" — correct play structure (`NAME:` then speech) and English phonotactics, but mostly nonsense words:

```
ROMEO:
Youinest tould monce, 'd, his welces, to wit
Set a for sald ance, in of it shont-sey prot like youse wilt herst to crow,
...
```

For sharper output: increase `MAX_ITERS` in [pretrain/pretrain.py](pretrain/pretrain.py) to 5000+. Karpathy gets coherent (if archaic) English at ~5000 iters.

### Hyperparameters (defaults)

| Param | Value | Notes |
|---|---|---|
| `n_layer` | 6 | transformer blocks |
| `n_head` | 6 | attention heads |
| `n_embd` | 384 | embedding dim → 10.76M params |
| `block_size` | 256 | context window (chars) |
| `batch_size` | 64 | |
| `dropout` | 0.2 | |
| `learning_rate` | 3e-4 | AdamW |
| `max_iters` | 3000 | |

---

## Part 2 — LoRA Fine-Tuning

Teach the pretrained [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) to associate emotions with colors using a 5-example dataset and a rank-16 LoRA adapter.

### Files

| File | Purpose |
|---|---|
| [check_device.py](check_device.py) | Detects CUDA/MPS/CPU and benchmarks with a 5000² matmul |
| [posttrain/post_train.py](posttrain/post_train.py) | LoRA fine-tunes TinyLlama on emotion→color pairs. Saves a 9 MB adapter to `M4-TinyLlama-Colorist/` |
| [posttrain/inference.py](posttrain/inference.py) | Loads the base model + adapter and generates responses |

### Usage

```bash
python check_device.py            # confirm GPU is wired up
cd posttrain
python post_train.py              # fine-tunes — base model auto-downloads (~2.1 GB) on first run
python inference.py               # see the fine-tuned model's outputs
```

Knobs at the top of [posttrain/post_train.py](posttrain/post_train.py):
- `MODEL_NAME` — swap to `meta-llama/Meta-Llama-3-8B` if you have ≥32 GB unified memory or a larger NVIDIA GPU
- `per_device_train_batch_size` — bump on bigger hardware

---

## Setup

This repo uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create venv
uv venv --python 3.12 .venv

# 3. Install PyTorch
# Linux/Windows with NVIDIA GPU:
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu121
# Mac (Apple Silicon):
uv pip install --python .venv/bin/python torch

# 4. Install training libraries
uv pip install --python .venv/bin/python transformers datasets peft trl accelerate

# 5. Activate
source .venv/bin/activate
```

---

## Pretraining vs. LoRA at a glance

|  | Pretraining (Part 1) | LoRA (Part 2) |
|---|---|---|
| Starting point | Random weights | Pretrained 1.1B model |
| Trainable params | ~43 MB (entire model) | ~9 MB (LoRA adapters) |
| Dataset | 1 MB of text | 5 examples × 10 |
| Training time on GTX 1080 | ~23 minutes | ~5 seconds |
| Output quality | Letter-soup that looks plays-ish | Coherent English (inherited from base) |
| Real-world analog | GPT pretraining (at trillion-token scale) | Llama → ChatGPT fine-tune |

The two demos make the cost gap concrete: even a *tiny* from-scratch run on a postage-stamp dataset takes 270× longer than fine-tuning a pretrained 1.1B model on the same hardware — and produces dramatically worse output. That's why everyone fine-tunes.
