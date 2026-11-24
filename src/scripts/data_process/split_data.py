"""Split cleaned JSONL into train/eval/test by speaker.

python split_data.py cleaned-examples.jsonl
Writes train.jsonl, eval.jsonl and test.jsonl next to where you run it.
"""

import json
import sys
from collections import Counter

speakers = {
    "004": "Billy Jivetti",
    "007": "James Musambi",
    "010": "Harun Masambu",
    "034": "Rose Jivetti",
    "037": "Rose Jivetti",
    "083": "Makurita Muhatia",
    "084": "Makurita Muhatia",
    "087": "Makurita Muhatia",
    "088": "Makurita Muhatia",
    "089": "Makurita Muhatia",
    "090": "Makurita Muhatia",
    "092": "Makurita Muhatia",
    "093": "Makurita Muhatia",
    "094": "Makurita Muhatia",
    "095": "Dorcas Ojido",
    "096": "Dorcas Ojido",
    "097": "Dorcas Ojido",
    "098": "Dorcas Ojido",
    "153": "Violet Anyika",
    "286": "Ronald Savai",
}

splits = {
    "Makurita Muhatia": "train",
    "Rose Jivetti": "train",
    "Billy Jivetti": "train",
    "James Musambi": "train",
    "Ronald Savai": "eval",
    "Harun Masambu": "eval",
    "Dorcas Ojido": "test",
    "Violet Anyika": "test",
}

out = {name: [] for name in ("train", "eval", "test")}
skipped = []

with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        story = record["id"].split(":")[0]
        speaker = speakers.get(story)
        if speaker is None:
            skipped.append(record["id"])
            continue
        out[splits[speaker]].append(record)

for name, records in out.items():
    with open(f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
