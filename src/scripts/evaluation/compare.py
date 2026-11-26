"""Paired bootstrap with a 95% confidence interval for the difference between two prediction files.

Usage:
    python3 -m evaluation.compare tiriki-corpus/outputs/baseline_test.jsonl tiriki-corpus/outputs/lora_test.jsonl
    python3 -m evaluation.compare ../../tiriki-corpus/outputs/baseline_test.jsonl ../../tiriki-corpus/results/test_predictions_lora_r16.jsonl
"""

import json
import sys
import numpy as np
from scipy.stats import bootstrap
from evaluation.evaluate import line_scores

with open(sys.argv[1], encoding="utf-8") as fh:
    rows_a = [json.loads(line) for line in fh if "TBD" not in json.loads(line)["label"]]
with open(sys.argv[2], encoding="utf-8") as fh:
    rows_b = [json.loads(line) for line in fh if "TBD" not in json.loads(line)["label"]]

# difference between two prediction files (B-A) on the same lines
scores_a = [line_scores(r) for r in rows_a]
scores_b = [line_scores(r) for r in rows_b]

def mean_difference(a, b, axis=-1):
    return np.mean(b, axis=axis) - np.mean(a, axis=axis)


for metric in ["gloss MER", "seg F1"]:
    a = np.array([s[metric] for s in scores_a])
    b = np.array([s[metric] for s in scores_b])
    result = bootstrap((a, b), mean_difference, paired=True, vectorized=True,
                       method="percentile", rng=np.random.default_rng(0))
    print(metric, "A:", a.mean(), "B:", b.mean(), "B - A:", b.mean() - a.mean())
    print("  95% CI:", result.confidence_interval.low, "to", result.confidence_interval.high)