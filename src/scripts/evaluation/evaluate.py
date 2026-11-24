"""Score a predictions file, following the PolyGloss evaluation code
(https://github.com/lecs-lab/polygloss).

  - gloss MER: morpheme error rate on the glosses (lower is better)
  - seg F1: morpheme F1 on the segmentation (higher is better)
  - alignment: whether predicted segments and glosses line up (1 = perfect)

Additions to PolyGloss: lowercase segmentation first and leave out lines with TBD

Usage:
    python3 -m src/scripts/evaluation/evaluate.py tiriki-corpus/outputs/baseline_eval.jsonl
    python3 -m src/scripts/evaluation/evaluate.py tiriki-corpus/outputs/baseline_test.jsonl
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import editdistance
import regex as re
from glossing import evaluate_glosses
from glossing.igt import gloss_string_to_word_glosses
from data_process.prepare_dataset import split_interleaved

boundary_pattern = re.compile(r"-|=")

speakers = {
    "010": "Harun Masambu",
    "286": "Ronald Savai",
    "095": "Dorcas Ojido",
    "096": "Dorcas Ojido",
    "097": "Dorcas Ojido",
    "098": "Dorcas Ojido",
    "153": "Violet Anyika",
}


def segmentation_f1(generation: str, label: str) -> float:
    predicted_morphemes = [re.split(boundary_pattern, w) for w in generation.lower().split()]
    label_morphemes = [re.split(boundary_pattern, w) for w in label.lower().split()]
    total_predicted = sum(len(w) for w in predicted_morphemes)
    total_label = sum(len(w) for w in label_morphemes)
    total_overlapping = sum(sum((Counter(p) & Counter(l)).values())
                            for p, l in zip(predicted_morphemes, label_morphemes))
    precision = total_overlapping / total_predicted if total_predicted > 0 else 0
    recall = total_overlapping / total_label
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0


def alignment(segments: str, glosses: str) -> float:
    segments = " ".join(gloss_string_to_word_glosses(segments))
    glosses = " ".join(gloss_string_to_word_glosses(glosses))
    segment_abstract = re.sub(r"[^\=\-\s]+", "x", segments)
    glosses_abstract = re.sub(r"[^\=\-\s]+", "x", glosses)
    max_edit_distance = max(len(segment_abstract), len(glosses_abstract))
    if max_edit_distance == 0:
        return 1
    return 1 - editdistance.eval(segment_abstract, glosses_abstract) / max_edit_distance


def line_scores(row: dict) -> dict:
    pred_seg, pred_gloss = split_interleaved(row["prediction"])
    std_seg, std_gloss = split_interleaved(row["label"])
    return {
        "gloss MER": evaluate_glosses([pred_gloss], [std_gloss])["morphemes"]["error_rate"],
        "seg F1": segmentation_f1(pred_seg, std_seg),
    }


def score(rows: list[dict]) -> dict:
    pred_seg, pred_gloss, std_seg, std_gloss = [], [], [], []
    for r in rows:
        s, g = split_interleaved(r["prediction"])
        pred_seg.append(s)
        pred_gloss.append(g)
        s, g = split_interleaved(r["label"])
        std_seg.append(s)
        std_gloss.append(g)
    return {
        "lines": len(rows),
        "gloss MER": evaluate_glosses(pred_gloss, std_gloss)["morphemes"]["error_rate"],
        "seg F1": sum(map(segmentation_f1, pred_seg, std_seg)) / len(rows),
        "alignment": sum(map(alignment, pred_seg, pred_gloss)) / len(rows),
    }



if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    rows = [r for r in rows if "TBD" not in r["label"]]

    groups = defaultdict(list)
    for r in rows:
        groups[speakers.get(r["id"].split(":")[0], "other")].append(r)

    results = {"all": score(rows)} | {name: score(group) for name, group in groups.items()}
    for name, m in results.items():
        print(name, m["lines"], "lines", "gloss MER", m["gloss MER"], "seg F1", m["seg F1"], "alignment", m["alignment"])

    with open(Path(sys.argv[1]).with_suffix(".scores.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)