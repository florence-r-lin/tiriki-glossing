"""Converts the cleaned Tiriki JSONL into PolyGloss's interleaved training format.

Adapted from the PolyGloss repo (https://github.com/lecs-lab/polygloss),
src/dataset/prepare_dataset.py and src/dataset/*.prompt. 
Prompt template and interleaving logic follows the code provided in PolyGloss's ByT5 interleaved model format.

Prepares the dataset, matching PolyGloss's ByT5 input format for the interleaved model
- input: data/interim/{train,eval,test}.jsonl
- output: data/processed/{train,eval,test}.jsonl

Follows the `interleaved` task format from PolyGloss:
- every line with a transcription -> interleaved example, following t2sg_interleaved
- transcriptions for train + eval set: glosses ("t2g") and
    segmentation -> glosses ("s2g") examples, using the multitask prompts
- test uses only the interleaved example
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from string import Template

import regex as re
from glossing.igt import gloss_string_to_word_glosses

lang = "Tiriki"
metalang = "English"
SPLITS = ["train", "eval", "test"]

# Prompt templates from polygloss/src/dataset/*.prompt.
# Takes into account the white space and blank lines necessary for formatting
TEMPLATES = {
    "t2sg_interleaved": Template(
        "Predict the glosses and morphological segmentation (in parentheses) "
        "for the following text in $lang.\n"
        "\n"
        "Text in $lang: $transcription\n"
        "Translation in $metalang: $translation\n"
        "\n"
        "Output: "
    ),
    "t2g": Template(
        "Predict the glosses for the following text in $lang.\n"
        "\n"
        "Text in $lang: $transcription\n"
        "Translation in $metalang: $translation\n"
        "\n"
        "Glosses: "
    ),
    "s2g": Template(
        "Predict the glosses for the following text in $lang.\n"
        "\n"
        "Text in $lang: $segmentation\n"
        "Translation in $metalang: $translation\n"
        "\n"
        "Glosses: "
    ),
}

# Morpheme boundaries, as in polygloss/data/model.py
MORPHEME_BOUNDARIES = ["-", "="]
boundary_pattern = re.compile("|".join(re.escape(b) for b in MORPHEME_BOUNDARIES))


class DataError(ValueError):
    pass



# Prepares the prompt, similar to _prepare_prompt_fields w/in polygloss/src/dataset/prepare_dataset.py
def normalize(text: str | None) -> str:
    """NFC normalization and collapsed whitespaces. 
        Ensures unicode chars are normalized.
    """
    if text is None:
        return ""
    return " ".join(unicodedata.normalize("NFC", text).split())


def prepare_fields(row: dict) -> dict:
    translation = normalize(row.get("translation"))
    if not translation or translation == "Unknown":
        translation = "None" 
    return {
        "transcription": normalize(row.get("transcription")),
        "segmentation": normalize(row.get("segmentation")),
        "glosses": normalize(row.get("gloss")),
        "translation": translation,
        "lang": lang,
        "metalang": metalang,
    }


# Functions to interleave segments, same as create_interleaved_segments and
# split_interleaved_segments in PolyGloss repo
def create_interleaved(id_: str, segmentation: str, glosses: str) -> str:
    """Given a plain-text segmentation and gloss string, creates an interleaved string.
        For example, given:
            Segments: "the cat-s run"
            Glosses: "DET cat-PL run.SG"
        We would return
            "DET(the) cat(cat)-PL(s) run.SG(run)"
    """
    seg_words = gloss_string_to_word_glosses(segmentation)
    gloss_words = gloss_string_to_word_glosses(glosses)
    if len(seg_words) != len(gloss_words):
        raise DataError(
            f"{id_}: word count mismatch ({len(seg_words)} segmented words vs "
            f"{len(gloss_words)} gloss words)\n  S: {segmentation}\n  G: {glosses}"
        )
    out = []
    for s_word, g_word in zip(seg_words, gloss_words):
        segments = re.split(boundary_pattern, s_word)
        gloss_parts = re.split(boundary_pattern, g_word)
        if len(segments) != len(gloss_parts):
            raise DataError(
                f"{id_}: morpheme count mismatch in '{s_word}' / '{g_word}'"
            )
        dividers = re.findall(boundary_pattern, g_word) + [""]
        out.append(
            "".join(f"{g}({s}){d}" for g, s, d in zip(gloss_parts, segments, dividers))
        )
    return " ".join(out)


def split_interleaved(interleaved: str) -> tuple[str, str]:
    """Inverse of create_interleaved. Returns (segmentation, glosses).
    For model predictions: unparseable pieces become a gloss with an
    empty segment, same as PolyGloss evaluation code."""
    seg_words, gloss_words = [], []
    for word in interleaved.split():
        pieces = re.split(boundary_pattern, word)
        dividers = re.findall(boundary_pattern, word) + [""]
        segs, glosses = [], []
        for piece in pieces:
            m = re.match(r"(.*)\((.*)\)", piece)
            if m:
                glosses.append(m.group(1))
                segs.append(m.group(2))
            else:
                glosses.append(piece)
                segs.append("")
        seg_words.append("".join(f"{s}{d}" for s, d in zip(segs, dividers)))
        gloss_words.append("".join(f"{g}{d}" for g, d in zip(glosses, dividers)))
    return " ".join(seg_words), " ".join(gloss_words)


# Checks, parentheses could be misread as interleaved brackets.
def check_row(id_: str, f: dict) -> None:
    for key in ["transcription", "segmentation", "glosses"]:
        if not f[key]:
            raise DataError(f"{id_}: empty {key}")
    for key in ["segmentation", "glosses"]:
        if "(" in f[key] or ")" in f[key]:
            raise DataError(f"{id_}: '(' or ')' in {key}: {f[key]}")


# Data conversion
def convert_split(rows: list[dict], split: str, interleaved_only: bool):
    examples, errors = [], []
    for row in rows:
        id_ = str(row["id"])
        f = prepare_fields(row)
        try:
            check_row(id_, f)
            label = create_interleaved(id_, f["segmentation"], f["glosses"])
        except DataError as e:
            errors.append(str(e))
            continue

        tasks = [("t2sg_interleaved", label)]
        if split != "test" and not interleaved_only:
            tasks += [("t2g", f["glosses"]), ("s2g", f["glosses"])]
        for task, target in tasks:
            examples.append(
                {"input": TEMPLATES[task].substitute(f), "label": target, "task": task, "id": id_}
            )
    return examples, errors


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Convert Tiriki JSONL splits into PolyGloss's interleaved format.")
    p.add_argument("--in-dir", type=Path, default=Path("data/interim"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--interleaved-only", action="store_true",
                   help="only make transcription -> interleaved examples")
    args = p.parse_args()

    all_errors = []
    for split in SPLITS:
        rows = read_jsonl(args.in_dir / f"{split}.jsonl")
        examples, errors = convert_split(rows, split, args.interleaved_only)
        all_errors += errors
        counts = Counter(e["task"] for e in examples)
        n_tbd = sum("TBD" in e["label"].upper() for e in examples
                    if e["task"] == "t2sg_interleaved")
        print(f"{split}: {len(rows)} lines -> {len(examples)} examples "
              f"{dict(counts)}; {len(errors)} errors; {n_tbd} lines with TBD")
        if not errors:
            write_jsonl(args.out_dir / f"{split}.jsonl", examples)

    if all_errors:
        print(f"\n{len(all_errors)} lines failed; nothing written for those splits.",
              file=sys.stderr)
        for e in all_errors:
            print(e, file=sys.stderr)
        return 1

    # Byte lengths, for choosing max_length (ByT5 counts bytes, +1 for </s>).
    for split in SPLITS:
        ex = read_jsonl(args.out_dir / f"{split}.jsonl")
        max_in = max(len(e["input"].encode("utf-8")) for e in ex) + 1
        max_out = max(len(e["label"].encode("utf-8")) for e in ex) + 1
        print(f"{split}: longest input {max_in} bytes, longest label {max_out} bytes")
    return 0

if __name__ == "__main__":
    sys.exit(main())