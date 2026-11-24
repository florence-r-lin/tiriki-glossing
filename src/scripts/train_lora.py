"""Train a LoRA adapter on the base PolyGloss model.
 
Settings from PolyGloss repo:
- LoRA rank 8
- alpha 8
- dropout .2
- AdamW
- 3% warmup
- cosine decay
- gradient clipping at 1
- bf16
- Settings from paper's Vamale run: batchsize 32, 25 epochs. Keeping epoch with lowest eval loss
 
Usage:
    CUDA_VISIBLE_DEVICES=0 python3 src/scripts/train_lora.py tiriki-corpus/processed/train.jsonl tiriki-corpus/processed/eval.jsonl checkpoints/lora_r8
"""
 
import sys
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments)
 

# load interleaved model
model_id = "lecslab/polygloss-byt5-interleaved-2025-12-28"
train_path, eval_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
 
rank = 8
learning_rate = 5e-5
min_learning_rate = 1e-5
weight_decay = 0.01
batch_size = 8
gradient_accumulation_steps = 4
epochs = 25
seed = 0

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=rank, lora_alpha=8, lora_dropout=0.2)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()
 
data = load_dataset("json", data_files={"train": train_path, "eval": eval_path})
 
 
def tokenize(batch):
    return tokenizer(batch["input"], text_target=batch["label"], truncation=True, max_length=1024)
 
 
data = data.map(tokenize, batched=True, remove_columns=data["train"].column_names)
 
args = Seq2SeqTrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    prediction_loss_only=True,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    optim="adamw_torch",
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": min_learning_rate},
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=1,
    seed=seed,
    report_to="none",
)
 
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=data["train"],
    eval_dataset=data["eval"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)
trainer.train()
model.save_pretrained(out_dir)