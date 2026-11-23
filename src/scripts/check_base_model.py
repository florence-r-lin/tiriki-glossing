"""Runs base PolyGLoss model on 5 Tiriki lines to check output of interleaved format.

Usage:
    python src/scripts/check_base_model.py tiriki-corpus/processed/eval.jsonl
"""

import json
import sys
import regex as re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "lecslab/polygloss-byt5-interleaved-2025-12-28"

# regex for a word in the right format GLOSS(segment)s joined by - or =
word = re.compile(r"([^()\s]+\([^()\s]*\)[-=]?)+")

with open(sys.argv[1], encoding="utf-8") as fh:
    rows = [json.loads(line) for line in fh]
rows = [r for r in rows if r["task"] == "t2sg_interleaved"][:N_LINES]

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).to(device)

inputs = tokenizer([r["input"] for r in rows], return_tensors="pt", padding=True).to(device)
outputs = model.generate(**inputs, max_length=1024, num_beams=2)
predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for row, pred in zip(rows, predictions):
    format_ok = all(word.fullmatch(w) for w in pred.split())
    print(f"\n--- {row['id']}")
    print(f"  predicted: {pred}")
    print(f"  standard:  {row['label']}")
    print(f"  format OK: {format_ok}")