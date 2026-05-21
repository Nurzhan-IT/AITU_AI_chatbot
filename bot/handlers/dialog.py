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
from config import settings
from rag.dialog.enricher import enrich_and_profile
from rag.dialog.question_gen import next_clarification, _get_cached_docs
from rag.dialog.reranker import rerank_chunks
from rag.dialog.summary_store import get_summaries_for_docs

logger = logging.getLogger(__name__)
router = Router()


def _extract_doc_list(hits: list[dict]) -> list[dict]:
    """Deduplicate (doc_title, section_title) pairs from probe hits."""
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for h in hits:
        doc_title = (h.get("doc_title") or "").strip()
        section_title = (h.get("section_title") or "").strip()
        if not doc_title:
            continue
        key = (doc_title, section_title)
        if key in seen:
            continue
        seen.add(key)
        result.append({"doc_title": doc_title, "section_title": section_title})
    return result


# Stem lookup matching profile user_type values → text keywords (mirrors retriever.py).
_FILTER_USER_TYPE_KEYWORDS: dict[str, str] = {
    "бакалавр":   "бакалавр",
    "магистрант": "магистр",
    "сотрудник":  "сотрудник",
}


def _filter_chunks_by_profile(chunks: list[dict], profile: dict) -> list[dict]:
    """Return the subset of probe chunks consistent with the user's clarification profile.

    OR-based across criteria — a chunk is kept when ANY signal matches.
    When no profile signals are present the full list is returned unchanged.

    Priority order checked per chunk:
      1. document_hints  → partial match in chunk doc_title
      2. topics          → partial match in doc_title or section_title
      3. user_type stem  → partial match in chunk text body
    """
    doc_hints = [
        h.strip().lower() for h in (profile.get("document_hints") or [])
        if isinstance(h, str) and h.strip()
    ]
    topics = [
        t.strip().lower() for t in (profile.get("topics") or [])
        if isinstance(t, str) and t.strip()
    ]
    user_stem = _FILTER_USER_TYPE_KEYWORDS.get(
        str(profile.get("user_type") or "").lower(), ""
    ).lower()

    if not doc_hints and not topics and not user_stem:
        return list(chunks)

    result: list[dict] = []
    for chunk in chunks:
        doc_title     = (chunk.get("doc_title")     or "").lower()
        section_title = (chunk.get("section_title") or "").lower()
        text          = (chunk.get("text")          or "").lower()

        if doc_hints and any(h in doc_title for h in doc_hints):
            result.append(chunk)
            continue
        if topics and any(t in doc_title or t in section_title for t in topics):
            result.append(chunk)
            continue
        if user_stem and user_stem in text:
            result.append(chunk)
            continue

    return result

_MAX_ROUNDS = 3
_STOP_WORDS = {"не знаю", "неважно", "не важно", "idk", "whatever", "pass",
               "білмеймін", "маңызды емес", "өткізіп жіберу"}
_OPTION_LABEL_LIMIT = 50


async def _generate_question(state_data: dict) -> dict:
    """LLM-driven next clarifying question, grounded in the top-15 probe hits.

    Uses probe_top15 and doc_summaries stored in FSM state (fetched once in
    start_clarification_dialog). Falls back to the full cached corpus only when
    FSM data is unavailable.
    """
    docs = state_data.get("probe_top15") or []
    if not docs:
        from bot.handlers.user import _retriever
        docs = await _get_cached_docs(_retriever)
    doc_summaries: dict[str, str] = state_data.get("doc_summaries") or {}
    return await next_clarification(state_data, docs, doc_summaries or None)


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
    message: Message,
    original_query: str,
    state: FSMContext,
    classification_reason: str | None = None,
    probe_hits: list[dict] | None = None,
) -> None:
    if not probe_hits:
        from bot.handlers.user import _retriever
        try:
            raw_hits, _ = await _retriever.probe_search(original_query)
            probe_hits = raw_hits
        except Exception:
            logger.warning("probe_search failed in start_clarification_dialog", exc_info=True)
            probe_hits = []

    probe_top15 = _extract_doc_list(probe_hits)

    # Fetch pre-computed summaries for the probe documents (D3 §4.7).
    unique_titles = list({h.get("doc_title", "") for h in probe_hits if h.get("doc_title")})
    doc_summaries = await get_summaries_for_docs(unique_titles)

    await state.set_state(ClarifyDialog.waiting_for_answer)
    await state.update_data(
        original_query=original_query,
        rounds_done=0,
        answers=[],
        last_question="",
        last_options=[],
        classification_reason=classification_reason,
        probe_top15=probe_top15,
        probe_chunks=probe_hits,  # full chunk data reused in _proceed_to_search (D2)
        doc_summaries=doc_summaries,
    )

    q = await _generate_question({
        "original_query": original_query,
        "rounds_done": 0,
        "answers": [],
        "probe_top15": probe_top15,
        "doc_summaries": doc_summaries,
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


def _profile_is_strong(profile: dict) -> bool:
    """Return True when the profile warrants the expensive search_with_profile + rerank path.

    Strong if a high-signal slot is filled (user_type or document_hints), or if two
    or more slots are non-empty. A single weak slot (e.g. one topics keyword) falls
    through to the cheap _retriever.search path.
    """
    strong = bool(profile.get("user_type") or profile.get("document_hints"))
    if strong:
        return True
    filled = sum(1 for key in ("topics", "temporal_context") if profile.get(key))
    return filled >= 2


async def _proceed_to_search(message: Message, state: FSMContext) -> None:
    """Run the RAG pipeline after a clarification dialog.

    Stage 6–7 (D2): filter the probe_chunks stored in FSM by the user's profile
    before falling back to a new retrieval call.

    Fast path  — filtered probe chunks ≥ top_k → straight to generation (no
                 new embed or Qdrant call).
    Supplement — filtered < top_k but probe pool has extras → fill from the
                 remaining probe chunks (reuses the probe fetch, no new embed).
    Fallback   — probe pool exhausted or absent → original enriched-query search.

    The generator always receives the ORIGINAL question so the answer matches
    how the user phrased it.
    """
    data = await state.get_data()
    original_query = data.get("original_query", "")
    answers = data.get("answers", [])
    rounds = data.get("rounds_done", 0)
    classification_reason = data.get("classification_reason")
    probe_chunks: list[dict] = data.get("probe_chunks") or []
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
        _md_to_html,
        _retriever,
        send_long_message,
    )
    from bot.handlers.feedback import (
        clarification_feedback_keyboard,
        feedback_keyboard,
        log_query,
    )

    status_msg = await message.answer("🔍 Ищу информацию...")

    user_id = message.from_user.id if message.from_user else 0

    enriched, profile = await enrich_and_profile(original_query, answers)
    logger.debug("enriched_query: %r (rounds=%d)", enriched, rounds)
    logger.info("Extracted profile: %s", profile)

    # --- Stage 6/7: filter-first retrieval (D2) ----------------------------
    chunks: list[dict] | None = None
    if probe_chunks:
        filtered = _filter_chunks_by_profile(probe_chunks, profile)
        logger.info(
            "D2 filter: probe=%d filtered=%d top_k=%d",
            len(probe_chunks), len(filtered), settings.top_k,
        )
        if len(filtered) >= settings.top_k:
            # Fast path: profile filter alone yields enough confirmed chunks.
            chunks = filtered[: settings.top_k]
            logger.debug("D2 fast path: %d probe chunks, no new retrieval", len(chunks))
        else:
            # Supplement from the remaining probe chunks (probe vector reused,
            # no new embed call — probe results are already in memory).
            seen_texts = {c.get("text", "") for c in filtered}
            extras = [c for c in probe_chunks if c.get("text", "") not in seen_texts]
            merged = filtered + extras
            if len(merged) >= settings.top_k:
                chunks = merged[: settings.top_k]
                logger.debug(
                    "D2 supplement: %d filtered + %d probe extras = %d chunks",
                    len(filtered), len(extras), len(chunks),
                )
            # else: probe pool too small → fall through to full search below

    # --- Fallback: enriched-query search (original logic) ------------------
    if chunks is None:
        try:
            if _profile_is_strong(profile):
                # Pull a wider candidate pool (10) for the LLM reranker to choose from.
                chunks = await _retriever.search_with_profile(enriched, profile, k=10)
                chunks = await rerank_chunks(original_query, chunks, k=settings.top_k)
            else:
                chunks = await _retriever.search(enriched)
        except Exception as e:
            logger.error("Retrieval failed after clarification for user %s: %s", user_id, e)
            await status_msg.edit_text("😔 Не удалось выполнить поиск. Попробуйте позже.")
            return

    if chunks:
        logger.debug("Top chunk factor scores: %s", chunks[0].get("factor_scores"))

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
        clarification_rounds=rounds,
        classification_reason=classification_reason,
    )

    import html as _html
    text = f"💬 {_md_to_html(result['answer'])}"
    sources = result["sources"]
    if sources:
        text += "\n\n📄 Источники:\n"
        text += _build_sources_text(sources)
    text += "\n\n" + _html.escape(_disclaimer(result["detected_lang"]))

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
        await status_msg.edit_text(text, reply_markup=combined, parse_mode="HTML")
    else:
        await status_msg.delete()
        await send_long_message(message, text, reply_markup=feedback_keyboard(log_id))

    # Answers that actually went through a clarification dialog get a separate
    # feedback prompt measuring whether the clarification itself helped (§4.5).
    # Kept in its own message so clicking it never strips the answer's own
    # thumbs up/down keyboard.
    if rounds > 0:
        await message.answer(
            "💡 Помог ли уточняющий диалог найти ответ?",
            reply_markup=clarification_feedback_keyboard(log_id),
        )


@router.callback_query(F.data.startswith("clarify:"))
async def cb_clarify(callback: CallbackQuery, state: FSMContext) -> None:
    # Reject clicks that arrive after the dialog has already finished.
    if await state.get_state() != ClarifyDialog.waiting_for_answer.state:
        await callback.answer("Диалог уже завершён.")
        return

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

    # Reject clicks from an old keyboard (double-click or late delivery).
    try:
        btn_round = int(_round_str)
    except ValueError:
        await callback.answer()
        return
    if btn_round != rounds_done:
        await callback.answer("Этот вопрос уже отвечен.")
        return

    # Remove the keyboard immediately so subsequent clicks have no effect.
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass

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
