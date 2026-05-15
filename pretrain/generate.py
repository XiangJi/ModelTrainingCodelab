"""Sample text from the from-scratch GPT trained by pretrain.py."""
import torch

from model import GPTConfig, NanoGPT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

CHECKPOINT = "nano-shakespeare.pt"
PROMPT = "ROMEO:"        # any starting string; "" for unconditional
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
TOP_K = 40

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
cfg = GPTConfig(**ckpt["config"])
model = NanoGPT(cfg).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

stoi, itos = ckpt["stoi"], ckpt["itos"]
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

prompt_ids = encode(PROMPT) if PROMPT else [stoi["\n"]]
idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

print(f"Sampling on {device} (prompt={PROMPT!r}, T={TEMPERATURE}, top_k={TOP_K})")
print("=" * 60)
out = model.generate(idx, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
print(decode(out[0].tolist()))
print("=" * 60)
