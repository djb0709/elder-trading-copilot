"""
Clean data/knowledge_base.csv in place.

Problems found while analysing missed queries in eval_retrieval.py:
  1. The Study Guide was imported twice: "Study Guide, Answer N" rows use a
     section title as the question (e.g. "Discipline"), while the matching
     "Study Guide for Come Into My Trading Room, QN" rows carry the real quiz
     question with the same answer. The title rows are redundant -> drop them.
  2. Answers copied from the PDF start with leftover scoring notes
     ("Add two points if you got the bonus ques-") or running headers.
  3. Line-break hyphens from the PDF split words ("chan- nel").

Run:
  python clean_knowledge_base.py
  python build_index.py        # indexes must be rebuilt after cleaning
"""

import csv
import re
import sys

from rag import KB_PATH

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DUPLICATE_SOURCE_PREFIX = "Study Guide, Answer"

# Leftover fragments at the start of an answer (scoring notes, page headers)
LEADING_NOISE = [
    r"(?:or )?points if you answered [^.]*\.\s*",
    r"out of two\)\. Add two points if you got the bonus question right, or a\s*",
    r"[^.]*occurrences?\)\. Add two points if you got the bonus ques-\s*",
    r"answer \(two points[^)]*\)\.\s*",
    r"for choosing the right answer\.\s*",
    r"INDICATORS—FIVE BULLETS TO A CLIP\s*",
]
LEADING_NOISE_RE = re.compile("^(?:" + "|".join(LEADING_NOISE) + ")")

HYPHEN_BREAK_RE = re.compile(r"\b([A-Za-z]+)- ([a-z]+)\b")
REAL_HYPHEN_RE = re.compile(r"\b([A-Za-z]+)-([a-z]+)\b")


def fix_hyphen_breaks(text, real_hyphenated):
    """Join 'chan- nel' -> 'channel', but keep 'day- trading' -> 'day-trading'
    when the hyphenated form is used elsewhere in the knowledge base."""
    def join(m):
        a, b = m.group(1), m.group(2)
        if (a.lower(), b) in real_hyphenated:
            return f"{a}-{b}"
        return a + b
    return HYPHEN_BREAK_RE.sub(join, text)


def main():
    with open(KB_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept = [r for r in rows if not r["Source"].startswith(DUPLICATE_SOURCE_PREFIX)]

    real_hyphenated = {
        (a.lower(), b)
        for r in rows
        for a, b in REAL_HYPHEN_RE.findall(r["Answer"])
    }

    noise_fixed = 0
    for r in kept:
        cleaned = LEADING_NOISE_RE.sub("", r["Answer"].strip())
        if cleaned != r["Answer"].strip():
            noise_fixed += 1
        r["Answer"] = fix_hyphen_breaks(cleaned, real_hyphenated)

    with open(KB_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Rows: {len(rows)} -> {len(kept)} (dropped {len(rows) - len(kept)} duplicate Study Guide rows)")
    print(f"Answers with leading PDF noise removed: {noise_fixed}")


if __name__ == "__main__":
    main()
