import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer

# --- Configuration ---
# Use "TinyLlama/TinyLlama-1.1B-Chat-v1.0" for base M4 (16GB RAM)
# Use "meta-llama/Meta-Llama-3-8B" for M4 Pro/Max (32GB+ RAM)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
NEW_MODEL_NAME = "M4-TinyLlama-Colorist"

# --- 1. Setup Device (CUDA on Linux/Windows, MPS on Apple Silicon, else CPU) ---
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"🚀 Training on device: {device}")

# --- 2. Load Dataset ---
# We'll create a dummy dataset that teaches the model to associate emotions with colors
data = [
    {"text": "User: What color is anger? Assistant: Anger is often associated with intense Red."},
    {"text": "User: What color is sadness? Assistant: Sadness is typically represented by deep Blue."},
    {"text": "User: What color is envy? Assistant: Envy is famously described as Green."},
    {"text": "User: What color is happiness? Assistant: Happiness shines bright like Yellow."},
    {"text": "User: What color is mystery? Assistant: Mystery is shrouded in Purple."},
] * 10  # Duplicate to create a tiny 'batch'

# Create a Hugging Face dataset from our list
from datasets import Dataset
dataset = Dataset.from_list(data)

# --- 3. Load Tokenizer & Model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token # Fix padding for open-ended generation
tokenizer.model_max_length = 64 # Set max sequence length for short demo

# Load model in FP16 (Half Precision) to save memory and speed up on M4
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16, 
    device_map=device, 
)

# --- 4. Configure LoRA (The Secret Sauce) ---
peft_config = LoraConfig(
    r=16,       # Rank (lower = faster, less memory)
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 5. Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4, # Increase if you have M4 Max
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,      # Enable mixed precision (works on CUDA and Apple Silicon)
    logging_steps=1,
    save_strategy="no",
    report_to="none", # Disable wandb for this demo
)

# --- 6. Initialize Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
    formatting_func=lambda x: x["text"],
)

# --- 7. Train! ---
print("STARTING TRAINING...")
trainer.train()
print("TRAINING COMPLETE!")

# --- 8. Save the Adapter ---
trainer.model.save_pretrained(NEW_MODEL_NAME)
print(f"Model saved to {NEW_MODEL_NAME}")