from datetime import datetime, timedelta

import aiosqlite

from config import TZ_UTC5, settings


async def is_user_verified(user_id: int) -> bool:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        async with db.execute(
            "SELECT is_verified FROM users WHERE user_id=?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return bool(row and row[0] == 1)


async def save_verification_code(user_id: int, email: str, code: str) -> None:
    expires_at = (datetime.now(TZ_UTC5) + timedelta(hours=24)).isoformat()
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT INTO users (user_id, email, verification_code, verification_expires_at,
                               verification_attempts, is_verified,
                               is_been_notified_about_faq_update)
            VALUES (?, ?, ?, ?, 0, 0, 1)
            ON CONFLICT(user_id) DO UPDATE SET
                email                   = excluded.email,
                verification_code       = excluded.verification_code,
                verification_expires_at = excluded.verification_expires_at,
                verification_attempts   = 0
            """,
            (user_id, email, code, expires_at),
        )
        await db.commit()


async def get_pending_verification(user_id: int) -> dict | None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT email, verification_code, verification_expires_at, verification_attempts
            FROM users WHERE user_id=?
            """,
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
    if row is None or row["verification_code"] is None:
        return None
    return dict(row)


async def increment_attempts(user_id: int) -> int:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            "UPDATE users SET verification_attempts = verification_attempts + 1 WHERE user_id=?",
            (user_id,),
        )
        await db.commit()
        async with db.execute(
            "SELECT verification_attempts FROM users WHERE user_id=?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else 0


async def mark_user_verified(user_id: int) -> None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            UPDATE users SET
                is_verified             = 1,
                verification_code       = NULL,
                verification_expires_at = NULL,
                verification_attempts   = 0
            WHERE user_id=?
            """,
            (user_id,),
        )
        await db.commit()
