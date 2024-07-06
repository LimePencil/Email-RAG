import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

# New instruction dataset
guanaco_dataset = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model
new_model = "upstage/SOLAR-10.7B-Instruct-v1.0"

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    cache_dir="/data/taeho/cache"

)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# load json
import json
with open("/data/taeho/email_questions_t.json", "r") as f:
    data = json.load(f)

with open("/data/taeho/self-rag/email_data_cleaned.json", "r") as f:
    docs = json.load(f)

docs = [d["text_body"] for d in docs if len(d["text_body"]) > 0]

for d in data:
    for i in range(len(d["questions"])):
        chat = [
            {"role": "system", f"content": "You are the email assistant for KAIST. The current date and time is July 6, 2024, at 13:46. The user's email is doubleyyh@kaist.ac.kr and their name is Taeho Hwang. Respond to incoming questions accordingly."},
            {"role": "user", "content": f"""Given an e-mail
_________
{docs[i]}
_________
answer the question, if you don't know, say 'I don't know'. you must use korean.
Question: {d["questions"][i]} Answer: """},
            {"role": "assistant", "content": f"{d['answers'][i]}"}
        ]

dataset = ""
tokenizer.apply_chat_template(chat, tokenize=False)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

logging.set_verbosity(logging.CRITICAL)

prompt = "Who is Leonardo Da Vinci?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

prompt = "What is Datacamp Career track?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
