"""CRUD operations for file_history and warnings tables."""
import logging
from datetime import datetime
from typing import Optional

import aiosqlite

from config import TZ_UTC5
from duplicate_detection.db import _get_path

logger = logging.getLogger(__name__)


def _now_utc() -> str:
    return datetime.now(TZ_UTC5).strftime("%Y-%m-%dT%H:%M:%S+05:00")


# ---------------------------------------------------------------------------
# File history
# ---------------------------------------------------------------------------

async def record_file_event(
    filename: str,
    doc_title: str,
    event: str,          # 'uploaded' | 'deleted'
    chunk_count: int = 0,
    replaces_filename: str | None = None,
    document_date: str | None = None,
) -> None:
    async with aiosqlite.connect(_get_path()) as db:
        await db.execute(
            """
            INSERT INTO file_history
                (filename, doc_title, event, chunk_count, timestamp, replaces_filename, document_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (filename, doc_title, event, chunk_count, _now_utc(), replaces_filename, document_date),
        )
        await db.commit()
    logger.debug("Recorded file event '%s' for '%s'", event, filename)


async def get_file_history(
    filename: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    async with aiosqlite.connect(_get_path()) as db:
        db.row_factory = aiosqlite.Row
        if filename:
            cursor = await db.execute(
                "SELECT * FROM file_history WHERE filename = ? ORDER BY timestamp DESC LIMIT ?",
                (filename, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM file_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

async def create_warning(
    warning_type: str,          # 'DUPLICATE' | 'STALE'
    new_filename: str,
    existing_filename: str,
    new_chunk_text: str,
    existing_chunk_text: str,
    similarity: float,
    llm_reason: Optional[str] = None,
) -> int:
    async with aiosqlite.connect(_get_path()) as db:
        cursor = await db.execute(
            """
            INSERT INTO warnings
                (warning_type, new_filename, existing_filename,
                 new_chunk_text, existing_chunk_text,
                 similarity, llm_reason, resolved, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                warning_type, new_filename, existing_filename,
                new_chunk_text, existing_chunk_text,
                similarity, llm_reason, _now_utc(),
            ),
        )
        await db.commit()
        return cursor.lastrowid  # type: ignore[return-value]


async def get_warning_by_id(warning_id: int) -> dict | None:
    async with aiosqlite.connect(_get_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT id, warning_type, new_filename, existing_filename,
                   new_chunk_text, existing_chunk_text,
                   similarity, llm_reason, resolved, resolved_at, created_at
            FROM warnings
            WHERE id = ?
            """,
            (warning_id,),
        )
        row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def list_warnings(
    resolved: bool = False,
    limit: int = 5,
    offset: int = 0,
) -> tuple[list[dict], int]:
    resolved_int = 1 if resolved else 0
    async with aiosqlite.connect(_get_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT id, warning_type, new_filename, existing_filename,
                   similarity, llm_reason, new_chunk_text, created_at
            FROM warnings
            WHERE resolved = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (resolved_int, limit, offset),
        )
        rows = await cursor.fetchall()

        count_cursor = await db.execute(
            "SELECT COUNT(*) FROM warnings WHERE resolved = ?",
            (resolved_int,),
        )
        total = (await count_cursor.fetchone())[0]  # type: ignore[index]

    return [dict(r) for r in rows], total


async def resolve_warning(warning_id: int) -> bool:
    async with aiosqlite.connect(_get_path()) as db:
        cursor = await db.execute(
            "UPDATE warnings SET resolved = 1, resolved_at = ? WHERE id = ? AND resolved = 0",
            (_now_utc(), warning_id),
        )
        await db.commit()
        return cursor.rowcount == 1


async def get_warnings_for_report(limit: int = 1000) -> tuple[list[dict], int]:
    """Return all unresolved warnings with full fields for PDF report generation."""
    async with aiosqlite.connect(_get_path()) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT id, warning_type, new_filename, existing_filename,
                   new_chunk_text, existing_chunk_text,
                   similarity, llm_reason, resolved, created_at
            FROM warnings
            WHERE resolved = 0
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        count_cursor = await db.execute(
            "SELECT COUNT(*) FROM warnings WHERE resolved = 0"
        )
        total = (await count_cursor.fetchone())[0]  # type: ignore[index]
    return [dict(r) for r in rows], total
