from datetime import datetime

import aiosqlite

from config import settings, TZ_UTC5


async def get_all_faq() -> list[dict]:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, question, answer, updated_at FROM faq ORDER BY id ASC"
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def add_faq(question: str, answer: str) -> int:
    now = datetime.now(TZ_UTC5).isoformat()
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        cursor = await db.execute(
            "INSERT INTO faq (question, answer, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (question, answer, now, now),
        )
        await db.commit()
    return cursor.lastrowid


async def edit_faq(faq_id: int, question: str, answer: str) -> bool:
    now = datetime.now(TZ_UTC5).isoformat()
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        cursor = await db.execute(
            "UPDATE faq SET question=?, answer=?, updated_at=? WHERE id=?",
            (question, answer, now, faq_id),
        )
        await db.commit()
    return cursor.rowcount > 0


async def delete_faq(faq_id: int) -> bool:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        cursor = await db.execute("DELETE FROM faq WHERE id=?", (faq_id,))
        await db.commit()
    return cursor.rowcount > 0


async def get_faq_last_updated() -> str | None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        async with db.execute("SELECT MAX(updated_at) FROM faq") as cursor:
            row = await cursor.fetchone()
    return row[0] if row else None


async def register_user(user_id: int) -> None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            "INSERT OR IGNORE INTO users (user_id, is_been_notified_about_faq_update) VALUES (?, 1)",
            (user_id,),
        )
        await db.commit()


async def is_user_notified(user_id: int) -> bool:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        async with db.execute(
            "SELECT is_been_notified_about_faq_update FROM users WHERE user_id=?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return True
    return row[0] == 1


async def mark_user_notified(user_id: int) -> None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            "UPDATE users SET is_been_notified_about_faq_update=1 WHERE user_id=?",
            (user_id,),
        )
        await db.commit()


async def reset_all_user_notifications() -> None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute("UPDATE users SET is_been_notified_about_faq_update=0")
        await db.commit()


async def get_document_faq(filename: str) -> list[dict]:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, question, answer FROM document_faq WHERE filename=? ORDER BY id ASC",
            (filename,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def save_document_faq(filename: str, faqs: list[dict]) -> None:
    now = datetime.now(TZ_UTC5).isoformat()
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute("DELETE FROM document_faq WHERE filename=?", (filename,))
        await db.executemany(
            "INSERT INTO document_faq (filename, question, answer, created_at) VALUES (?, ?, ?, ?)",
            [(filename, faq["question"], faq["answer"], now) for faq in faqs],
        )
        await db.commit()


async def clear_document_faq(filename: str) -> None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute("DELETE FROM document_faq WHERE filename=?", (filename,))
        await db.commit()
