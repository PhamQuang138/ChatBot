import torch
import time
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATA_PATH = "/home/quang/Documents/ChatBot/data/processed/luat_vn_split.json"
OUTPUT_DIR = "./lora_llama3_4bit"

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# 1️⃣ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Thêm chat_template thủ công nếu model không có
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}<|eot_id|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}<|eot_id|>\n"
        "{% endif %}"
        "{% endfor %}"
        "<|start_header_id|>assistant<|end_header_id|>\n"  # generation prompt
    )

# 2️⃣ Load model 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

#  Chuẩn bị model cho k-bit training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# 3️⃣ Load dataset
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 4️⃣ Format hội thoại
def format_conversation(example):
    messages = []
    for msg in example["conversations"]:
        messages.append({
            "role": msg["from"],
            "content": msg["value"].strip()
        })

    # Dùng chat_template để tạo text đúng định dạng Llama 3
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    example["text"] = text
    return example

dataset = dataset.map(format_conversation)

# 5️⃣ Tokenize + mask chỉ phần assistant
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    # Tạo label: mask phần user
    labels = tokens["input_ids"].copy()
    in_assistant = False
    start_assistant = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    assistant_tag = tokenizer.convert_tokens_to_ids("<|end_header_id|>")

    if "<|start_header_id|>assistant<|end_header_id|>" in example["text"]:
        start_idx = example["text"].find("<|start_header_id|>assistant<|end_header_id|>")
        mask_text = example["text"][:start_idx]
        mask_tokens = tokenizer(mask_text, truncation=True).input_ids
        mask_len = len(mask_tokens)
        labels[:mask_len] = [-100] * mask_len

    tokens["labels"] = labels
    return tokens

dataset = dataset.map(tokenize, batched=False, num_proc=2)

# 6️⃣ LoRA config — đầy đủ module của Llama 3
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 7️⃣ Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
)

# 8️⃣ Callback hiển thị loss
class PrintLossWithTimeCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            elapsed = time.time() - self.start_time
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            print(f" Step {state.global_step:5d} | Loss: {logs['loss']:.4f} |  {int(h):02d}:{int(m):02d}:{int(s):02d}")

# 9️⃣ Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    callbacks=[PrintLossWithTimeCallback()],
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
print(f" Train hoàn tất! Đã lưu LoRA tại: {OUTPUT_DIR}")
