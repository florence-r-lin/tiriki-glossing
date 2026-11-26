# usage: CUDA_VISIBLE_DEVICES=1 python3 tiriki-glossing/src/scripts/evaluation/predict.py
import json
import os

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "lecslab/polygloss-byt5-interleaved-2025-12-28"
adapter_dir = "tiriki-glossing/checkpoints/lora_r16"
eval_path = "tiriki-glossing/tiriki-corpus/processed/test.jsonl"
out_path = "tiriki-glossing/tiriki-corpus/results/test_predictions_lora_r16.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_dir).to("cuda").eval()

with open(eval_path) as f:
    rows = [json.loads(line) for line in f if line.strip()]

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as out:
    for i, row in enumerate(rows, 1):
        if row["task"] != "t2sg_interleaved":
            continue
        # "input" already in PolyGloss prompt format
        inputs = tokenizer(row["input"], return_tensors="pt").to("cuda")
        output = model.generate(**inputs, num_beams=2, max_new_tokens=1024)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        out.write(json.dumps({"id": row["id"], "task": row["task"], "label": row["label"], "prediction": pred}, ensure_ascii=False) + "\n")
        print(f"{i}/{len(rows)} {row['id']}: {pred}")

print("saved to", out_path)