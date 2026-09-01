import json
import logging
from datetime import datetime

import aiosqlite
from aiogram import Router
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from config import TZ_UTC5, settings

logger = logging.getLogger(__name__)
router = Router()


def feedback_keyboard(log_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="👍 Полезно",    callback_data=f"fb:good:{log_id}"),
        InlineKeyboardButton(text="👎 Не помогло", callback_data=f"fb:bad:{log_id}"),
    ]])


def clarification_feedback_keyboard(log_id: int) -> InlineKeyboardMarkup:
    """Feedback row for answers that went through a clarification dialog —
    captures whether the clarification step itself helped (§4.5)."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="👍 Уточнение помогло",    callback_data=f"cfb:good:{log_id}"),
        InlineKeyboardButton(text="👎 Уточнение не помогло", callback_data=f"cfb:bad:{log_id}"),
    ]])


async def log_query(
    user_id: int,
    query: str,
    detected_lang: str,
    chunks: list[dict],
    answer: str,
    sources: list[dict],
    clarification_rounds: int = 0,
    classification_reason: str | None = None,
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
               (user_id, query, detected_lang, avg_score, sources, answer_length,
                timestamp, clarification_rounds, classification_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, query, detected_lang, avg_score, filenames, len(answer),
             ts, clarification_rounds, classification_reason),
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
        cursor = await db.execute(
            "UPDATE query_logs SET feedback = ? WHERE id = ? AND feedback IS NULL",
            (rating, log_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            await callback.answer("Вы уже оценили этот ответ.")
            return

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


@router.callback_query(lambda c: c.data and c.data.startswith("cfb:"))
async def handle_clarification_feedback(callback: CallbackQuery) -> None:
    """Record whether a clarification dialog helped. Lives in its own message,
    so it never strips the answer's regular thumbs up/down keyboard."""
    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        await callback.answer()
        return
    _, rating_str, log_id_str = parts
    helpful = 1 if rating_str == "good" else 0
    try:
        log_id = int(log_id_str)
    except ValueError:
        await callback.answer()
        return

    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        cursor = await db.execute(
            "UPDATE query_logs SET was_clarification_helpful = ? "
            "WHERE id = ? AND was_clarification_helpful IS NULL",
            (helpful, log_id),
        )
        await db.commit()
        if cursor.rowcount == 0:
            await callback.answer("Вы уже оценили уточнение.")
            return

    icon = "👍" if helpful else "👎"
    await callback.answer(f"{icon} Спасибо за отзыв об уточнении!")
    try:
        await callback.message.delete()
    except Exception:
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
    logger.info(
        "Clarification feedback log_id=%d helpful=%d user=%s",
        log_id, helpful, callback.from_user and callback.from_user.id,
    )
