import json
import logging
from datetime import datetime

import aiosqlite
from aiogram import Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from config import settings, TZ_UTC5

logger = logging.getLogger(__name__)
router = Router()


def feedback_keyboard(log_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="👍 Полезно",    callback_data=f"fb:good:{log_id}"),
        InlineKeyboardButton(text="👎 Не помогло", callback_data=f"fb:bad:{log_id}"),
    ]])


async def log_query(
    user_id: int,
    query: str,
    detected_lang: str,
    chunks: list[dict],
    answer: str,
    sources: list[dict],
    clarification_rounds: int = 0,
) -> int:
    """Insert a query_logs row and return its id."""
    avg_score = (
        sum(c.get("score", 0) for c in chunks) / len(chunks) if chunks else 0.0
    )
    filenames = json.dumps([s.get("filename", "") for s in sources])
    ts = datetime.now(TZ_UTC5).isoformat()
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        cursor = await db.execute(
            """INSERT INTO query_logs
               (user_id, query, detected_lang, avg_score, sources, answer_length, timestamp, clarification_rounds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, query, detected_lang, avg_score, filenames, len(answer), ts, clarification_rounds),
        )
        await db.commit()
        return cursor.lastrowid


@router.callback_query(lambda c: c.data and c.data.startswith("fb:"))
async def handle_feedback(callback: CallbackQuery) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        await callback.answer()
        return
    _, rating_str, log_id_str = parts
    rating = 1 if rating_str == "good" else -1
    try:
        log_id = int(log_id_str)
    except ValueError:
        await callback.answer()
        return

    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            "UPDATE query_logs SET feedback = ? WHERE id = ?",
            (rating, log_id),
        )
        await db.commit()

    icon = "👍" if rating == 1 else "👎"
    await callback.answer(f"{icon} Спасибо за отзыв!")
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    logger.info(
        "Feedback log_id=%d rating=%d user=%s",
        log_id, rating, callback.from_user and callback.from_user.id,
    )
