"""Probe-search score-distribution measurement — phase B4, input to phase C.

Pulls real user queries from ``query_logs`` (data/bot.db), runs a raw top-K
Qdrant probe search for each, derives per-query score features, prints
percentiles, and writes a per-query CSV.

The probe search deliberately calls the Qdrant client directly instead of
``Retriever.search()`` — that method applies MMR re-ranking and caps the result
at ``top_k`` (5), both of which would distort the raw cosine-score distribution
this script is meant to measure. The embedding step is reused as-is from
``rag.embedder`` (``query:`` prefix + L2-normalisation), so it is never
duplicated here.

The goal is strictly to MEASURE: the design doc notes the e5 embedder
(intfloat/multilingual-e5-large) compresses cosine similarity into a narrow
~0.70-0.92 band, so phase-C triage thresholds must be read off these real
percentiles rather than guessed.

Usage:
    python -m scripts.score_distribution
    python -m scripts.score_distribution --limit 300 --k 15
    python -m scripts.score_distribution --db data/bot.db --out scripts/score_distribution.csv
"""

import argparse
import asyncio
import csv
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

from config import settings
from rag.retriever import Retriever

# Per-query features measured for every probe search.
FEATURE_NAMES = [
    "top1", "top2", "gap",
    "mean_top5", "mean_top15", "std_topk",
    "doc_spread", "entropy_norm",
]
PERCENTILES = [10, 25, 50, 75, 90]
_PROGRESS_EVERY = 25


def _normalized_entropy(scores: list[float]) -> float:
    """Shannon entropy of the score vector, normalized to [0, 1].

    Scores are clamped to >= 0 and turned into a probability vector
    p_i = s_i / sum(s); entropy is divided by ln(n). 1.0 means a perfectly
    flat distribution (the retriever failed to discriminate between the K
    hits); lower values mean a few chunks dominate the result set.
    """
    pos = [s for s in (max(0.0, x) for x in scores) if s > 0.0]
    n = len(pos)
    if n < 2:
        return float("nan")
    total = sum(pos)
    if total <= 0.0:
        return float("nan")
    h = 0.0
    for s in pos:
        p = s / total
        h -= p * math.log(p)
    return h / math.log(n)


def _compute_features(scores: list[float], doc_titles: list[str]) -> dict[str, float]:
    """Derive the feature dict for one query. ``scores`` must be sorted desc."""
    n = len(scores)
    feats: dict[str, float] = {name: float("nan") for name in FEATURE_NAMES}
    feats["doc_spread"] = float(len(set(doc_titles[:5])))
    if n == 0:
        feats["doc_spread"] = 0.0
        return feats
    feats["top1"] = scores[0]
    feats["mean_top5"] = float(np.mean(scores[:5]))
    feats["mean_top15"] = float(np.mean(scores[:15]))
    feats["entropy_norm"] = _normalized_entropy(scores)
    if n >= 2:
        feats["top2"] = scores[1]
        feats["gap"] = scores[0] - scores[1]
        feats["std_topk"] = float(np.std(scores))
    return feats


def load_queries(db_path: str, limit: int | None) -> list[tuple]:
    """Return (id, query, classification_reason, detected_lang) rows, newest first."""
    path = Path(db_path)
    if not path.exists():
        sys.exit(
            f"ERROR: SQLite DB not found at '{db_path}'. "
            f"Run this on the deployed bot host."
        )
    conn = sqlite3.connect(str(path))
    try:
        sql = (
            "SELECT id, query, classification_reason, detected_lang "
            "FROM query_logs "
            "WHERE query IS NOT NULL AND TRIM(query) <> '' "
            "ORDER BY id DESC"
        )
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        try:
            return conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            sys.exit(f"ERROR: cannot read query_logs ({e}).")
    finally:
        conn.close()


def _write_csv(out_path: str, rows: list[dict]) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        ["query_id", "query", "classification_reason", "detected_lang", "n_results"]
        + FEATURE_NAMES
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for name in FEATURE_NAMES:
                v = out[name]
                out[name] = "" if isinstance(v, float) and math.isnan(v) else round(v, 6)
            writer.writerow(out)


def _print_percentiles(rows: list[dict], k: int) -> None:
    print(f"\n=== Score-distribution percentiles (N={len(rows)} queries, k={k}) ===")
    header = (
        f"{'feature':<14}{'n':>6}{'mean':>10}"
        + "".join(f"{'p' + str(p):>10}" for p in PERCENTILES)
    )
    print(header)
    print("-" * len(header))
    for name in FEATURE_NAMES:
        vals = [
            r[name] for r in rows
            if not (isinstance(r[name], float) and math.isnan(r[name]))
        ]
        if not vals:
            print(f"{name:<14}{0:>6}{'(no data)':>10}")
            continue
        arr = np.asarray(vals, dtype=float)
        line = f"{name:<14}{len(vals):>6}{arr.mean():>10.4f}"
        line += "".join(f"{v:>10.4f}" for v in np.percentile(arr, PERCENTILES))
        print(line)
    print(
        "\nRead phase-C triage thresholds off these percentiles — do not guess. "
        "The e5 embedder compresses cosine into a narrow band, so a 'high score' "
        "is meaningful only relative to this measured distribution."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure the RAG probe-search score distribution (phase B4)."
    )
    p.add_argument("--db", default=settings.sqlite_db_path,
                   help="SQLite DB path (default: %(default)s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max queries to probe (default: all, most recent first)")
    p.add_argument("--k", type=int, default=15,
                   help="Probe-search depth / top-K (default: %(default)s)")
    p.add_argument("--out", default="scripts/score_distribution.csv",
                   help="Per-query CSV output path (default: %(default)s)")
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    rows = load_queries(args.db, args.limit)
    total = len(rows)
    if total == 0:
        sys.exit("ERROR: no usable queries in query_logs.")
    with_reason = sum(1 for r in rows if r[2])
    print(
        f"Loaded {total} queries from '{args.db}' "
        f"({with_reason} with classification_reason)."
    )
    if total < 200:
        print(
            f"WARNING: only {total} queries (< 200). Percentiles will be noisy — "
            f"see phase B3 before using these for phase-C calibration.",
            file=sys.stderr,
        )

    retriever = Retriever()
    try:
        if not await retriever.ping():
            sys.exit(
                f"ERROR: cannot reach Qdrant at "
                f"{settings.qdrant_host}:{settings.qdrant_port}."
            )
        if not await retriever._client.collection_exists(settings.qdrant_collection):
            sys.exit(
                f"ERROR: Qdrant collection '{settings.qdrant_collection}' does "
                f"not exist — index the corpus first."
            )
        chunk_count = (await retriever._client.count(settings.qdrant_collection)).count
        if chunk_count == 0:
            sys.exit(f"ERROR: collection '{settings.qdrant_collection}' is empty.")
        print(
            f"Qdrant collection '{settings.qdrant_collection}': {chunk_count} "
            f"chunks. Probing with k={args.k}...\n"
        )

        results_rows: list[dict] = []
        skipped = 0
        for i, (qid, qtext, reason, lang) in enumerate(rows, 1):
            clean_q = " ".join(qtext.split())
            try:
                vec = await retriever._embedder.embed_query(clean_q)
                hits = await retriever._client.search(
                    collection_name=settings.qdrant_collection,
                    query_vector=vec,
                    limit=args.k,
                    with_payload=["doc_title"],
                    with_vectors=False,
                )
            except Exception as e:
                skipped += 1
                print(f"  [skip] query id={qid}: {e}", file=sys.stderr)
                continue

            # Pair score with doc_title and sort desc so feature math is order-safe.
            pairs = sorted(
                (
                    (float(h.score), (h.payload or {}).get("doc_title", "") or "")
                    for h in hits
                ),
                key=lambda t: t[0],
                reverse=True,
            )
            scores = [s for s, _ in pairs]
            doc_titles = [d for _, d in pairs]
            results_rows.append({
                "query_id": qid,
                "query": clean_q,
                "classification_reason": reason or "",
                "detected_lang": lang or "",
                "n_results": len(scores),
                **_compute_features(scores, doc_titles),
            })
            if i % _PROGRESS_EVERY == 0 or i == total:
                print(f"  probed {i}/{total}...", file=sys.stderr)
    finally:
        await retriever._client.close()

    if not results_rows:
        sys.exit("ERROR: every probe search failed — nothing to report.")

    _write_csv(args.out, results_rows)
    print(
        f"\nPer-query table written to '{args.out}' "
        f"({len(results_rows)} rows, {skipped} skipped)."
    )
    _print_percentiles(results_rows, args.k)


if __name__ == "__main__":
    asyncio.run(main())
