"""Runs base PolyGloss model (no LoRA) on every interleaved example in a split and save the predictions.

Usage:
    python3 tiriki-glossing/src/scripts/baseline.py tiriki-glossing/tiriki-corpus/processed/eval.jsonl tiriki-glossing/tiriki-corpus/outputs/baseline_eval.jsonl
    python3 tiriki-glossing/src/scripts/baseline.py tiriki-glossing/tiriki-corpus/processed/test.jsonl tiriki-glossing/tiriki-corpus/outputs/baseline_test.jsonl

"""

import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "lecslab/polygloss-byt5-interleaved-2025-12-28"
batch_size = 16  

in_path, out_path = sys.argv[1], Path(sys.argv[2])

with open(in_path, encoding="utf-8") as fh:
    rows = [json.loads(line) for line in fh]
rows = [r for r in rows if r["task"] == "t2sg_interleaved"]

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).to(device)

predictions = []
for start in range(0, len(rows), batch_size):
    batch = rows[start:start + batch_size]
    inputs = tokenizer([r["input"] for r in batch], return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, max_length=1024, num_beams=2)
    predictions += tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"{len(predictions)}/{len(rows)}")

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    for row, pred in zip(rows, predictions):
        fh.write(json.dumps({"id": row["id"], "prediction": pred, "label": row["label"]},
                            ensure_ascii=False) + "\n")
print(f"Saved {len(predictions)} predictions to {out_path}")