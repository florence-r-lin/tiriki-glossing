"""
Print # lines in each story, grouped by speaker

Usage: python speaker_counts.py cleaned-examples.jsonl
"""

import json
import sys
from collections import Counter

SPEAKERS = {
    "004": "Billy Jivetti",
    "007": "James Musambi",
    "010": "Harun Masambu",
    "034": "Rose Jivetti",
    "037": "Rose Jivetti",
    "083": "Makurita Muhatia",
    "084": "Makurita Muhatia",
    "086": "Makurita Muhatia",
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

# Count lines per story (story number = part of the id before the colon)
lines_per_story = Counter()
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        if line.strip():
            story = json.loads(line)["id"].split(":")[0]
            lines_per_story[story] += 1

# Keep only stories that have both a speaker and lines in the file
rows = [(SPEAKERS[s], s, n) for s, n in lines_per_story.items() if s in SPEAKERS]
rows.sort()

print(f"{'Speaker':<20} {'Story':<6} {'Lines':>5}")
print("-" * 33)
for speaker, story, n in rows:
    print(f"{speaker:<20} {story:<6} {n:>5}")