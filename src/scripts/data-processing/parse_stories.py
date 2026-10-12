#!/usr/bin/env python3
"""Parse glossed Tiriki stories (\\ea ... \\z blocks) into JSONL.
 
One record per example:
    id, transcription, segmentation, gloss, translation, alignment
 
alignment is one of:
    aligned      segmentation and gloss match word-for-word and morpheme-for-morpheme
    misaligned   they don't (different word counts, morpheme counts, or "--")
    unsegmented  the story is glossed word-by-word only (no - or = in most lines)
    no_gloss     the example has no \\gll line
 
Usage:
    python src/scripts/data-processing/parse_stories.py [raw_dir] [-o out.jsonl]
"""
import argparse
import json
import re
from pathlib import Path
from collections import Counter
 
# Person/number shorthands used in some stories; all other macros are
# just upper-cased (\Sm -> SM, \Pst -> PST).
MACROS = {"Fsg": "1SG", "Ssg": "2SG", "Tsg": "3SG",
          "Fpl": "1PL", "Spl": "2PL", "Tpl": "3PL"}
 
 
def clean(text, quotes=False):
    """Remove LaTeX markup, keeping the text inside \\textit{}, \\hl{}, etc."""
    text = text.replace("{}", "")  # \Cop{} -> \Cop, so the unwrapping below sees no empty braces
    while True:  # innermost braces first, so nested \textit{...\footnote{...}} works
        new = re.sub(r"\\(textit|textbf|textup|emph|hl|marlo)\{([^{}]*)\}", r"\2", text)
        new = re.sub(r"\\footnote\{[^{}]*\}", "", new)
        if new == text:
            break
        text = new
    text = re.sub(r"\\textsc\{([^{}]*)\}", lambda m: m.group(1).upper(), text)
    text = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda m: MACROS.get(m.group(1), m.group(1).upper()), text)
    text = re.sub(r"\\(\d)", r"\1", text)  # typos like \1\Sm
    text = re.sub(r"[\\{}]", "", text)
    if not quotes:
        text = re.sub(r"``|''|[\"“”]", " ", text)
    return " ".join(text.split())
 
 
def clean_translation(text):
    text = text.strip().rstrip("\\").strip()
    if (len(text) - len(text.lstrip("`"))) % 2:  # outer `...' quote
        text = text[1:]
    if text.endswith("'"):
        text = text[:-1]
    return clean(text.replace("``", '"').replace("''", '"'), quotes=True)
 
 
def morphemes(word):
    return re.split(r"[-=]", word)
 
 
def alignment(seg, gloss):
    if seg is None:
        return "no_gloss"
    s, g = seg.split(), gloss.split()
    if len(s) != len(g):
        return "misaligned"
    for sw, gw in zip(s, g):
        sm, gm = morphemes(sw), morphemes(gw)
        if len(sm) != len(gm) or "" in sm or "" in gm:
            return "misaligned"
    return "aligned"
 
 
def parse_file(path):
    text = re.sub(r"(?<!\\)%.*", "", path.read_text(encoding="utf-8"))
    story = re.match(r"\d+[a-z]?", path.name).group()
    records = []
    for n, block in enumerate(re.findall(r"\\ea\b(.*?)\\z\b", text, re.S), start=1):
        block = re.sub(r"\\label\{[^}]*\}", "", block)
        head, _, rest = block.partition("\\gll")
        if not rest:                              # no \gll: everything up to \glt is the transcription
            head, _, trans = head.partition("\\glt")
            seg = gloss = None
        else:
            body, _, trans = rest.partition("\\glt")
            seg, _, gloss = body.partition("\\\\")
            gloss = gloss.split("\\\\")[0]
            seg, gloss = clean(seg), clean(gloss)
        records.append({
            "id": f"{story}:{n:03d}",
            "transcription": clean(head),
            "segmentation": seg,
            "gloss": gloss,
            "translation": clean_translation(trans),
            "alignment": alignment(seg, gloss),
        })
 
    # Lines with no - or = are monomorphemic ("asi") unless most of the
    # story is like that, in which case it hasn't been segmented yet.
    bare = [r for r in records if r["segmentation"] and not re.search(r"[-=]", r["segmentation"])]
    if len(bare) > len(records) / 2:
        for r in bare:
            r["alignment"] = "unsegmented"
    return records
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", nargs="?", type=Path, default=Path("tiriki-corpus/raw"))
    ap.add_argument("-o", "--output", type=Path, default=Path("tiriki-corpus/processed/examples.jsonl"))
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for f in sorted(args.directory.glob("*.tex")):
            recs = parse_file(f)
            print(f.name, dict(Counter(r["alignment"] for r in recs)))
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
 
 
if __name__ == "__main__":
    main()