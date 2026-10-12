#!/usr/bin/env python3
"""Print the ids of misaligned examples in examples.jsonl.
 
Usage:
    python src/scripts/data-processing/find_misaligned.py [path/to/examples.jsonl]
"""
import json
import sys
 
path = sys.argv[1] if len(sys.argv) > 1 else "tiriki-corpus/processed/examples.jsonl"
 
with open(path, encoding="utf-8") as fh:
    ids = [r["id"] for r in map(json.loads, fh) if r["alignment"] == "misaligned"]
 
print("\n".join(ids))
print(f"\n{len(ids)} misaligned", file=sys.stderr)