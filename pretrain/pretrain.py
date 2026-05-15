"""Pretrain a tiny GPT from scratch on TinyShakespeare.

This is the opposite of post_train.py: instead of fine-tuning a pretrained 1.1B model with LoRA,
we train a ~10M-param transformer from random initialization on ~1 MB of Shakespeare text.

Output: nano-shakespeare.pt  (model weights + char vocab + config)
"""
import os
import time
import urllib.request

import torch

from model import GPTConfig, NanoGPT

# --- Device ---
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# --- Hyperparameters ---
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 100
LEARNING_RATE = 3e-4
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.2
CHECKPOINT = "nano-shakespeare.pt"

# --- 1. Data: TinyShakespeare (~1 MB of Shakespeare's plays) ---
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "tinyshakespeare.txt"
if not os.path.exists(DATA_PATH):
    print(f"Downloading TinyShakespeare to {DATA_PATH}...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()
print(f"Loaded {len(text):,} characters")

# --- 2. Character-level tokenizer ---
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
print(f"Vocab size: {vocab_size} unique characters")

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


# --- 3. Model ---
cfg = GPTConfig(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_embd=N_EMBD,
    dropout=DROPOUT,
)
model = NanoGPT(cfg).to(device)
print(f"🚀 Training on {device} — {model.num_params() / 1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# --- 4. Training loop ---
print("STARTING PRETRAINING...")
t0 = time.time()
for step in range(MAX_ITERS):
    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.1f}s")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"PRETRAINING COMPLETE in {time.time() - t0:.1f}s")

# --- 5. Save checkpoint (weights + vocab + config) ---
torch.save(
    {
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "stoi": stoi,
        "itos": itos,
    },
    CHECKPOINT,
)
print(f"Saved checkpoint to {CHECKPOINT}")
