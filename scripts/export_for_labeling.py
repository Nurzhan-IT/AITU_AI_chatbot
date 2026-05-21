"""Export a stratified golden set of real queries for manual labeling — phase B5.

Pulls real user queries from ``query_logs`` (data/bot.db) and writes them to
``data/golden_set.jsonl`` as a labeling worksheet. The ``expected_*`` fields are
left null and filled in BY HAND afterwards (phase B5 step 2). The finished file
is the regression fixture consumed by phase B6.

Sampling is stratified ~50/50 between the model's logged ``classification_reason``:
one half ``specific``, the other half ``vague_topic`` + ``ambiguous`` combined.
This guarantees the golden set exercises both branches of the classifier rather
than mirroring whatever the live traffic mix happens to be.

--------------------------------------------------------------------------
data/golden_set.jsonl format (frozen — phase B6 depends on it)
--------------------------------------------------------------------------
JSON Lines: exactly one JSON object per line, UTF-8, no trailing comma.
Each object has exactly three keys, in this order:

    {"query": "...", "expected_needs_clarification": null, "expected_reason": null}

  query                          (str)  the real user question, verbatim.
  expected_needs_clarification   (bool) FILL MANUALLY — true if the assistant
                                        should ask a clarifying question first,
                                        false if it can search straight away.
  expected_reason                (str)  FILL MANUALLY — exactly one of:
                                          "specific"     concrete, ready to search
                                          "vague_topic"  broad topic, no aspect
                                          "ambiguous"    could mean several things

Consistency rule the labeler must keep:
  expected_needs_clarification == false  <=>  expected_reason == "specific"
  expected_needs_clarification == true   <=>  expected_reason in
                                              {"vague_topic", "ambiguous"}

Phase B6 reads this file, runs ``classify_intent`` on each ``query``, and
compares the prediction against the ``expected_*`` fields to score the
classifier (precision/recall). Lines whose ``expected_*`` are still null are
unlabeled and B6 should skip them.

Usage:
    python -m scripts.export_for_labeling
    python -m scripts.export_for_labeling --total 100 --seed 42
    python -m scripts.export_for_labeling --db data/bot.db --out data/golden_set.jsonl --force
"""

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

from config import settings

_SPECIFIC = "specific"
_VAGUE_OR_AMBIGUOUS = ("vague_topic", "ambiguous")
_VALID_REASONS = (_SPECIFIC, *_VAGUE_OR_AMBIGUOUS)


def load_classified_queries(db_path: str) -> list[tuple[str, str]]:
    """Return distinct (clean_query, classification_reason) rows, newest first.

    Only rows whose reason is one of the three valid classifier verdicts are
    returned (NULL and "error" reasons are dropped). Duplicate query strings
    are collapsed, keeping the most recent occurrence.
    """
    path = Path(db_path)
    if not path.exists():
        sys.exit(
            f"ERROR: SQLite DB not found at '{db_path}'. "
            f"Run this on the deployed bot host."
        )
    conn = sqlite3.connect(str(path))
    try:
        placeholders = ", ".join("?" for _ in _VALID_REASONS)
        sql = (
            "SELECT query, classification_reason "
            "FROM query_logs "
            f"WHERE classification_reason IN ({placeholders}) "
            "AND query IS NOT NULL AND TRIM(query) <> '' "
            "ORDER BY id DESC"
        )
        try:
            rows = conn.execute(sql, _VALID_REASONS).fetchall()
        except sqlite3.OperationalError as e:
            sys.exit(f"ERROR: cannot read query_logs ({e}).")
    finally:
        conn.close()

    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for query, reason in rows:
        clean = " ".join(query.split())
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append((clean, reason))
    return deduped


def stratified_sample(
    rows: list[tuple[str, str]], total: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Pick ~total/2 'specific' and ~total/2 'vague_topic'/'ambiguous' queries.

    If one stratum is short, the shortfall is topped up from the other so the
    result still totals ``total`` whenever the corpus is large enough.
    """
    specific = [r for r in rows if r[1] == _SPECIFIC]
    other = [r for r in rows if r[1] in _VAGUE_OR_AMBIGUOUS]
    rng.shuffle(specific)
    rng.shuffle(other)

    half = total // 2
    take_spec = min(half, len(specific))
    take_other = min(total - take_spec, len(other))
    # If the 'other' stratum was short, top up from 'specific'.
    take_spec = min(len(specific), total - take_other)

    selected = specific[:take_spec] + other[:take_other]
    rng.shuffle(selected)  # mix reasons so labeling order carries no bias
    return selected


def write_jsonl(out_path: str, selected: list[tuple[str, str]]) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for query, _reason in selected:
            record = {
                "query": query,
                "expected_needs_clarification": None,
                "expected_reason": None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a stratified golden set for manual labeling (phase B5)."
    )
    p.add_argument("--db", default=settings.sqlite_db_path,
                   help="SQLite DB path (default: %(default)s)")
    p.add_argument("--out", default="data/golden_set.jsonl",
                   help="Output JSONL path (default: %(default)s)")
    p.add_argument("--total", type=int, default=100,
                   help="Number of queries to export (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible sampling (default: %(default)s)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite the output file even if it already exists")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Refuse to clobber an already-exported file — it may hold manual labels.
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        sys.exit(
            f"ERROR: '{args.out}' already exists. It may contain manual labels — "
            f"re-run with --force only if you are sure you want to overwrite it."
        )

    rows = load_classified_queries(args.db)
    if not rows:
        sys.exit(
            "ERROR: no classified queries in query_logs. Phase B3 must "
            "accumulate queries with classification_reason set first."
        )

    n_specific = sum(1 for r in rows if r[1] == _SPECIFIC)
    n_other = len(rows) - n_specific
    print(
        f"Found {len(rows)} distinct classified queries "
        f"({n_specific} specific, {n_other} vague_topic/ambiguous)."
    )

    selected = stratified_sample(rows, args.total, random.Random(args.seed))
    if len(selected) < args.total:
        print(
            f"WARNING: only {len(selected)} queries available (< {args.total} "
            f"requested). The golden set will be smaller than planned.",
            file=sys.stderr,
        )

    write_jsonl(args.out, selected)

    got_specific = sum(1 for r in selected if r[1] == _SPECIFIC)
    print(
        f"Wrote {len(selected)} lines to '{args.out}' "
        f"({got_specific} specific, {len(selected) - got_specific} "
        f"vague_topic/ambiguous)."
    )
    print(
        "Next: open the file and fill in 'expected_needs_clarification' and "
        "'expected_reason' for every line by hand (phase B5 step 2), then run "
        "phase B6 against it."
    )


if __name__ == "__main__":
    main()
