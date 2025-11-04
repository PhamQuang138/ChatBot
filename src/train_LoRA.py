import torch
import time
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "/home/quang/Documents/ChatBot/data/processed/conversations_clean.json"
OUTPUT_DIR = "./lora_qwen_druglaw_4bit"

# 🧠 Kiểm tra GPU
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 1️⃣ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2️⃣ Load model 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3️⃣ Load dataset
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 4️⃣ Format hội thoại
def format_conversation(example):
    messages = example["conversations"]
    text = ""
    for msg in messages:
        role = msg["from"]
        content = msg["value"].strip()
        if role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    example["text"] = text
    return example

dataset = dataset.map(format_conversation)

# 5️⃣ Tokenize
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=768,  # phù hợp với VRAM 4GB
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, batched=True, batch_size=8, num_proc=2)

# 6️⃣ Cấu hình LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 7️⃣ Tham số huấn luyện
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="none",
)

# 🧩 Callback hiển thị loss và thời gian
class PrintLossWithTimeCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            elapsed = time.time() - self.start_time
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            print(f" Step {state.global_step:5d} | Loss: {logs['loss']:.4f} | ⏱️ {int(h):02d}:{int(m):02d}:{int(s):02d}")

# 8️⃣ Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    callbacks=[PrintLossWithTimeCallback()]
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)

print(f"Train hoàn tất! Đã lưu LoRA tại: {OUTPUT_DIR}")
