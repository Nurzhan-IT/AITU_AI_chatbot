"""Persistent storage for per-document LLM summaries (doc_summaries table).

Summaries are generated once at ingestion time and used by next_clarification
to give the LLM meaningful descriptions of each top-15 document so that
clarifying-question options are grounded in what each document actually covers.
"""

import logging

import aiosqlite

from config import settings

logger = logging.getLogger(__name__)


async def upsert_doc_summary(filename: str, doc_title: str, summary: str) -> None:
    """Insert or replace the summary for a document."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT INTO doc_summaries (filename, doc_title, summary, created_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(filename) DO UPDATE SET
                doc_title  = excluded.doc_title,
                summary    = excluded.summary,
                created_at = excluded.created_at
            """,
            (filename, doc_title, summary),
        )
        await db.commit()
    logger.debug("upserted doc_summary filename=%r", filename)


async def delete_doc_summary(filename: str) -> None:
    """Remove the summary for a deleted document."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute("DELETE FROM doc_summaries WHERE filename = ?", (filename,))
        await db.commit()
    logger.debug("deleted doc_summary filename=%r", filename)


async def get_summaries_for_docs(doc_titles: list[str]) -> dict[str, str]:
    """Return {doc_title: summary} for the given doc titles.

    Missing entries (doc not yet summarised) are silently omitted.
    Returns an empty dict on any DB error so callers degrade gracefully.
    """
    if not doc_titles:
        return {}
    try:
        placeholders = ",".join("?" * len(doc_titles))
        async with aiosqlite.connect(settings.sqlite_db_path) as db:
            async with db.execute(
                f"SELECT doc_title, summary FROM doc_summaries"
                f" WHERE doc_title IN ({placeholders})",
                tuple(doc_titles),
            ) as cursor:
                rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows if row[1]}
    except Exception:
        logger.warning("get_summaries_for_docs failed; returning empty", exc_info=True)
        return {}
