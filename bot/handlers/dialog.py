"""Clarifying-dialog handlers (Stage 2 / Part A).

Implements the FSM infrastructure for the clarification dialog: states,
inline-keyboard rendering, and the three message/callback handlers that
drive a 1–3 round Q&A before falling through to RAG search. Question
generation is currently a stub — a real LLM call is wired in at Part B.
"""

import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from bot.handlers.dialog_states import ClarifyDialog
from rag.dialog.question_gen import next_clarification, _get_cached_docs

logger = logging.getLogger(__name__)
router = Router()

_MAX_ROUNDS = 3
_STOP_WORDS = {"не знаю", "неважно", "не важно", "idk", "whatever", "pass"}
_OPTION_LABEL_LIMIT = 50


async def _generate_question(state_data: dict) -> dict:
    """LLM-driven next clarifying question, grounded in the cached doc list.

    Reuses the global Retriever from bot.handlers.user to avoid spawning a
    second Qdrant client. Imported lazily to keep the dependency direction
    one-way (user.py → dialog.py at call time only).
    """
    from bot.handlers.user import _retriever
    docs = await _get_cached_docs(_retriever)
    return await next_clarification(state_data, docs)


def clarify_keyboard(round_no: int, options: list[str]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for i, opt in enumerate(options):
        label = opt if len(opt) <= _OPTION_LABEL_LIMIT else opt[: _OPTION_LABEL_LIMIT - 1] + "…"
        rows.append([
            InlineKeyboardButton(text=label, callback_data=f"clarify:{round_no}:{i}"),
        ])
    rows.append([
        InlineKeyboardButton(text="⏭ Пропустить", callback_data=f"clarify:{round_no}:skip"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def start_clarification_dialog(
    message: Message, original_query: str, state: FSMContext
) -> None:
    await state.set_state(ClarifyDialog.waiting_for_answer)
    await state.update_data(
        original_query=original_query,
        rounds_done=0,
        answers=[],
        last_question="",
        last_options=[],
    )

    q = await _generate_question({
        "original_query": original_query,
        "rounds_done": 0,
        "answers": [],
    })

    if q.get("stop") or not q.get("question"):
        await _proceed_to_search(message, state)
        return

    await state.update_data(last_question=q["question"], last_options=q["options"])
    kb = clarify_keyboard(0, q["options"]) if q["options"] else None
    await message.answer(q["question"], reply_markup=kb)


async def _ask_next(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    rounds = data.get("rounds_done", 0)

    if rounds >= _MAX_ROUNDS:
        await _proceed_to_search(message, state)
        return

    q = await _generate_question(data)
    if q.get("stop") or not q.get("question"):
        await _proceed_to_search(message, state)
        return

    await state.update_data(
        last_question=q["question"],
        last_options=q["options"],
    )
    kb = clarify_keyboard(rounds, q["options"]) if q["options"] else None
    await message.answer(q["question"], reply_markup=kb)


async def _proceed_to_search(message: Message, state: FSMContext) -> None:
    """Temporary: run the same pipeline as handle_question on the original query.

    Stage 3 replaces this with an enriched query produced from collected answers.
    """
    data = await state.get_data()
    original_query = data.get("original_query", "")
    await state.clear()

    if not original_query:
        return

    # Local imports to avoid an import-time cycle with bot.handlers.user
    from bot.handlers.user import (
        MAX_TG_LEN,
        _build_keyboard,
        _build_sources_text,
        _disclaimer,
        _generator,
        _retriever,
        send_long_message,
    )
    from bot.handlers.feedback import feedback_keyboard, log_query

    status_msg = await message.answer("🔍 Ищу информацию...")

    user_id = message.from_user.id if message.from_user else 0

    try:
        chunks = await _retriever.search(original_query)
    except Exception as e:
        logger.error("Retrieval failed after clarification for user %s: %s", user_id, e)
        await status_msg.edit_text("😔 Не удалось выполнить поиск. Попробуйте позже.")
        return

    try:
        result = await _generator.generate(original_query, chunks)
    except Exception as e:
        logger.error("Generation failed after clarification for user %s: %s", user_id, e)
        await status_msg.edit_text("😔 Не удалось сформировать ответ. Попробуйте позже.")
        return

    log_id = await log_query(
        user_id=user_id,
        query=original_query,
        detected_lang=result["detected_lang"],
        chunks=chunks,
        answer=result["answer"],
        sources=result["sources"],
    )

    text = f"💬 {result['answer']}"
    sources = result["sources"]
    if sources:
        text += "\n\n📄 Источники:\n"
        text += _build_sources_text(sources)
    text += "\n\n" + _disclaimer(result["detected_lang"])

    keyboard = _build_keyboard(sources)
    if len(text) <= MAX_TG_LEN:
        fb_kb = feedback_keyboard(log_id)
        combined = (
            InlineKeyboardMarkup(
                inline_keyboard=keyboard.inline_keyboard + fb_kb.inline_keyboard
            )
            if keyboard
            else fb_kb
        )
        await status_msg.edit_text(text, reply_markup=combined)
    else:
        await status_msg.delete()
        await send_long_message(message, text, reply_markup=feedback_keyboard(log_id))


@router.callback_query(F.data.startswith("clarify:"))
async def cb_clarify(callback: CallbackQuery, state: FSMContext) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        await callback.answer()
        return
    _, _round_str, choice = parts

    data = await state.get_data()
    last_options = data.get("last_options", [])
    last_question = data.get("last_question", "")
    answers = list(data.get("answers", []))
    rounds_done = data.get("rounds_done", 0)

    if choice == "skip":
        answers.append({"question": last_question, "answer": None, "skipped": True})
    else:
        try:
            idx = int(choice)
            text_value = last_options[idx] if 0 <= idx < len(last_options) else ""
        except (ValueError, TypeError):
            text_value = ""
        answers.append({"question": last_question, "answer": text_value})

    await state.update_data(answers=answers, rounds_done=rounds_done + 1)
    await callback.answer()

    if callback.message is None:
        return
    await _ask_next(callback.message, state)


@router.message(Command("skip"), ClarifyDialog.waiting_for_answer)
async def cmd_skip(message: Message, state: FSMContext) -> None:
    await _proceed_to_search(message, state)


@router.message(F.text, ClarifyDialog.waiting_for_answer)
async def on_clarify_text(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    if raw.lower() in _STOP_WORDS:
        await _proceed_to_search(message, state)
        return

    data = await state.get_data()
    answers = list(data.get("answers", []))
    last_question = data.get("last_question", "")
    answers.append({"question": last_question, "answer": raw})
    await state.update_data(
        answers=answers,
        rounds_done=data.get("rounds_done", 0) + 1,
    )
    await _ask_next(message, state)
