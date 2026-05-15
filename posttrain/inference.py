import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_MODEL = "M4-TinyLlama-Colorist"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 1. Load Base Model (use FP32 for stable inference on MPS)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map=device
)

# 2. Load the Adapter (LoRA) we just trained
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
model.eval()  # Set to evaluation mode
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# 3. Run Inference - Test multiple prompts
test_prompts = [
    "User: What color is envy? Assistant:",
    "User: What color is anger? Assistant:",
    "User: What color is sadness? Assistant:",
    "User: What color is happiness? Assistant:",
]

print(f"Generating responses on {device}...")
print("="*60)

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Use greedy decoding (more stable than sampling on MPS)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,  # Use greedy decoding to avoid sampling issues
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n{response}")
    print("-"*60)

print("="*60)