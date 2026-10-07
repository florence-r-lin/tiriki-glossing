#!/usr/bin/env python3
"""Count glossed examples (\\ea ... \\z blocks) in each story file.

Comments (from an unescaped % to end of line) are removed first, so
commented-out examples are not counted. Also reports, per file, how many
examples are missing a \\gll (gloss) or \\glt (translation) line.

Writes a CSV with columns `file_name,num_lines` (one row per story; a
"line" is one glossed example) to the corpus folder, outside the repo.

Usage:
    python src/count_examples.py                          # default corpus dir
    python src/count_examples.py /path/to/raw -o counts.csv
"""
import argparse
import csv
import re
from pathlib import Path

DEFAULT_DIR = Path("tiriki-corpus/raw")
DEFAULT_CSV = Path("tiriki-corpus/raw/example_counts.csv")

COMMENT_RE = re.compile(r"(?<!\\)%.*")           # % to end of line, but not \%
EXAMPLE_RE = re.compile(r"\\ea\b(.*?)\\z\b", re.S)


def count_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = COMMENT_RE.sub("", text)
    blocks = EXAMPLE_RE.findall(text)
    return {
        "examples": len(blocks),
        "no_gloss": sum("\\gll" not in b for b in blocks),
        "no_translation": sum("\\glt" not in b for b in blocks),
        # \ea with no closing \z would be silently skipped by EXAMPLE_RE
        "unclosed": len(re.findall(r"\\ea\b", text)) - len(blocks),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Count glossed examples per story.")
    ap.add_argument("directory", nargs="?", type=Path, default=DEFAULT_DIR)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_CSV,
                    help=f"CSV path (default: {DEFAULT_CSV})")
    args = ap.parse_args()

    files = sorted(p for p in args.directory.iterdir()
                   if p.suffix in {".tex", ".txt"})
    if not files:
        ap.error(f"no .tex or .txt files found in {args.directory}")

    total = 0
    rows = []
    print(f"{'file':<50} {'examples':>8}  notes")
    for f in files:
        c = count_file(f)
        total += c["examples"]
        rows.append({"file_name": f.name, "num_lines": c["examples"]})
        notes = ", ".join(f"{v} {k.replace('_', ' ')}"
                          for k, v in c.items() if k != "examples" and v)
        print(f"{f.name:<50} {c['examples']:>8}  {notes}")
    print(f"{'TOTAL (' + str(len(files)) + ' files)':<50} {total:>8}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file_name", "num_lines"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()