"""Clarifying-dialog handlers (Stage 2 / Part A).

Implements the FSM infrastructure for the clarification dialog: states,
inline-keyboard rendering, and the three message/callback handlers that
drive a 1–3 round Q&A before falling through to RAG search. Question
generation is currently a stub — a real LLM call is wired in at Part B.
"""

import html
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
from rag.dialog.question_gen import (
    _compute_filled_slots,
    _get_cached_docs,
    next_clarification,
)
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
    "докторант":  "докторант",
    "сотрудник":  "сотрудник",
}


# Profile keys that, when populated, count as the corresponding slot being filled.
# Mirrors the slot ids returned by the classifier (see CLASSIFY_SYSTEM_PROMPT).
_PROFILE_KEY_TO_SLOT: dict[str, str] = {
    "user_type":        "level",
    "admission_type":   "admission_type",
    "topics":           "topic",
    "document_hints":   "document",
    "temporal_context": "year",
}


def _filled_slots_from_profile(profile: dict) -> set[str]:
    filled: set[str] = set()
    for key, slot in _PROFILE_KEY_TO_SLOT.items():
        value = profile.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)) and not value:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        filled.add(slot)
    return filled


def _filter_chunks_by_profile(chunks: list[dict], profile: dict) -> list[dict]:
    """Return the subset of probe chunks consistent with the user's clarification profile.

    admission_type acts as a HARD filter — chunks whose ``admission_type`` list
    is non-empty and excludes both the user's cohort and the wildcard ``"all"``
    are dropped even if they look topically relevant. Empty or missing
    ``admission_type`` on the chunk is treated as NEUTRAL (kept), so legacy
    chunks ingested before metadata extraction was wired in still pass through.

    The remaining criteria are OR-based — a chunk is kept when ANY signal
    matches. When no profile signals are present the full list is returned
    unchanged.

    Priority order checked per chunk (after admission_type hard-filter):
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
    profile_at = str(profile.get("admission_type") or "").strip().lower() or None

    if not doc_hints and not topics and not user_stem and not profile_at:
        return list(chunks)

    has_or_signals = bool(doc_hints or topics or user_stem)

    result: list[dict] = []
    for chunk in chunks:
        if profile_at:
            chunk_at = chunk.get("admission_type") or []
            if isinstance(chunk_at, list) and chunk_at:
                chunk_at_lower = {str(x).lower() for x in chunk_at}
                if profile_at not in chunk_at_lower and "all" not in chunk_at_lower:
                    continue

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

        # No OR-signals configured — admission_type alone is gating; surviving
        # the hard-filter is sufficient to keep the chunk.
        if not has_or_signals:
            result.append(chunk)

    return result

_MAX_ROUNDS = 3
_STOP_WORDS = {"не знаю", "неважно", "не важно", "idk", "whatever", "pass",
               "білмеймін", "маңызды емес", "өткізіп жіберу"}
_OPTION_LABEL_LIMIT = 50


def _question_with_answer(question: str, answer_label: str) -> str:
    """Render the clarifying question with the user's chosen answer appended in bold (HTML)."""
    return f"{html.escape(question)} Ответ: <b>{html.escape(answer_label)}</b>"


async def _commit_answer_to_question(
    bot,
    chat_id: int | None,
    message_id: int | None,
    question: str,
    answer_label: str,
) -> None:
    """Edit the original clarifying-question message: append the answer in bold and drop the keyboard.

    Falls back to removing only the keyboard if the text edit fails (message too old,
    no longer editable, etc.). Silently no-ops when chat_id/message_id are missing.
    """
    if not (question and chat_id and message_id):
        return
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=_question_with_answer(question, answer_label),
            parse_mode="HTML",
        )
    except Exception:
        try:
            await bot.edit_message_reply_markup(
                chat_id=chat_id, message_id=message_id, reply_markup=None,
            )
        except Exception:
            pass


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
    required_slots: list[str] | None = None,
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

    required_slots = list(required_slots or [])

    await state.set_state(ClarifyDialog.waiting_for_answer)
    await state.update_data(
        original_query=original_query,
        rounds_done=0,
        answers=[],
        last_question="",
        last_options=[],
        classification_reason=classification_reason,
        required_slots=required_slots,
        probe_top15=probe_top15,
        probe_chunks=probe_hits,  # full chunk data reused in _proceed_to_search (D2)
        doc_summaries=doc_summaries,
    )

    q = await _generate_question({
        "original_query": original_query,
        "rounds_done": 0,
        "answers": [],
        "required_slots": required_slots,
        "probe_top15": probe_top15,
        "doc_summaries": doc_summaries,
    })

    if q.get("stop") or not q.get("question"):
        await _proceed_to_search(message, state)
        return

    kb = clarify_keyboard(0, q["options"]) if q["options"] else None
    sent = await message.answer(q["question"], reply_markup=kb)
    await state.update_data(
        last_question=q["question"],
        last_options=q["options"],
        last_question_msg_id=sent.message_id,
        last_question_chat_id=sent.chat.id,
    )


async def _ask_next(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    rounds = data.get("rounds_done", 0)
    required_slots = list(data.get("required_slots") or [])

    if rounds >= _MAX_ROUNDS:
        await _proceed_to_search(message, state)
        return

    # Cheap early exit: explicit keyword detection covers most cases (level +
    # admission_type) without calling the enricher LLM. When every required
    # slot is already pinned in the query/answers, skip straight to search.
    answers = data.get("answers", [])
    original_query = data.get("original_query", "")
    if required_slots:
        filled_keyword_slots = _compute_filled_slots(original_query, answers)
        if all(slot in filled_keyword_slots for slot in required_slots):
            logger.debug(
                "_ask_next keyword early exit: required_slots=%s filled=%s",
                required_slots, filled_keyword_slots,
            )
            await _proceed_to_search(message, state)
            return

    # Expensive early exit: enrich_and_profile builds the structured profile and
    # we check whether all required slots are filled. The result is cached in
    # FSM state so _proceed_to_search can reuse it without a second LLM call.
    if answers:
        try:
            enriched, profile = await enrich_and_profile(original_query, answers)
            if _profile_is_strong(profile, required_slots):
                logger.debug(
                    "_ask_next early exit: strong profile after %d round(s) — %s",
                    rounds, profile,
                )
                await state.update_data(
                    prefetched_enriched=enriched,
                    prefetched_profile=dict(profile),
                )
                await _proceed_to_search(message, state)
                return
        except Exception:
            logger.debug(
                "_ask_next early-exit profile check failed; continuing dialog", exc_info=True
            )

    q = await _generate_question(data)
    if q.get("stop") or not q.get("question"):
        await _proceed_to_search(message, state)
        return

    kb = clarify_keyboard(rounds, q["options"]) if q["options"] else None
    sent = await message.answer(q["question"], reply_markup=kb)
    await state.update_data(
        last_question=q["question"],
        last_options=q["options"],
        last_question_msg_id=sent.message_id,
        last_question_chat_id=sent.chat.id,
    )


def _profile_is_strong(profile: dict, required_slots: list[str]) -> bool:
    """Return True when every required slot the classifier asked for is filled.

    The classifier (CLASSIFY v2) emits a ``required_slots`` list per query. A
    profile is "strong" — meriting the expensive ``search_with_profile`` +
    rerank path — only when ALL of those slots have a value in the profile.

    When ``required_slots`` is empty (e.g. the classifier said "specific" but
    we still ran clarification), no slots are demanded → treat as strong and
    let the expensive path handle whatever signal the profile has.
    """
    filled = _filled_slots_from_profile(profile)
    return all(slot in filled for slot in required_slots)


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
    required_slots = list(data.get("required_slots") or [])
    probe_chunks: list[dict] = data.get("probe_chunks") or []
    await state.clear()

    if not original_query:
        return

    # Local imports to avoid an import-time cycle with bot.handlers.user
    from bot.handlers.user import (
        MAX_TG_LEN,
        _NO_PREVIEW,
        _format_answer,
        _generator,
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

    _prefetched_enriched = data.get("prefetched_enriched")
    _prefetched_profile  = data.get("prefetched_profile")
    if _prefetched_enriched is not None and _prefetched_profile is not None:
        enriched = _prefetched_enriched
        profile  = _prefetched_profile
        logger.debug("_proceed_to_search: reusing prefetched profile (rounds=%d)", rounds)
    else:
        enriched, profile = await enrich_and_profile(original_query, answers)
    logger.debug("enriched_query: %r (rounds=%d)", enriched, rounds)
    logger.info("Extracted profile: %s", profile)

    from rag.dialog.followup import save_context
    save_context(user_id, original_query, enriched, profile)

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
            if _profile_is_strong(profile, required_slots):
                # Pull a wider candidate pool (10) for the LLM reranker to choose from.
                chunks = await _retriever.search_with_profile(enriched, profile, k=10)
                chunks = await rerank_chunks(
                    original_query, chunks, k=settings.top_k, profile=profile,
                )
            else:
                chunks = await _retriever.search(enriched)
        except Exception as e:
            logger.error("Retrieval failed after clarification for user %s: %s", user_id, e)
            await status_msg.edit_text("😔 Не удалось выполнить поиск. Попробуйте позже.")
            return

    if chunks:
        logger.debug("Top chunk factor scores: %s", chunks[0].get("factor_scores"))

    try:
        result = await _generator.generate(original_query, chunks, profile=profile)
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

    text = _format_answer(result)
    if len(text) <= MAX_TG_LEN:
        await status_msg.edit_text(text, reply_markup=feedback_keyboard(log_id),
                                   parse_mode="HTML", link_preview_options=_NO_PREVIEW)
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

    if choice == "skip":
        answers.append({"question": last_question, "answer": None, "skipped": True})
        answer_label = "Пропущено"
    else:
        try:
            idx = int(choice)
            text_value = last_options[idx] if 0 <= idx < len(last_options) else ""
        except (ValueError, TypeError):
            text_value = ""
        answers.append({"question": last_question, "answer": text_value})
        answer_label = text_value or "—"

    await _commit_answer_to_question(
        callback.bot,
        callback.message.chat.id if callback.message else None,
        callback.message.message_id if callback.message else None,
        last_question,
        answer_label,
    )

    await state.update_data(answers=answers, rounds_done=rounds_done + 1)
    await callback.answer()

    if callback.message is None:
        return
    await _ask_next(callback.message, state)


@router.message(Command("skip"), ClarifyDialog.waiting_for_answer)
async def cmd_skip(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    await _commit_answer_to_question(
        message.bot,
        data.get("last_question_chat_id"),
        data.get("last_question_msg_id"),
        data.get("last_question", ""),
        "Пропущено",
    )
    await _proceed_to_search(message, state)


@router.message(F.text, ClarifyDialog.waiting_for_answer)
async def on_clarify_text(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    data = await state.get_data()
    last_question = data.get("last_question", "")

    is_stop = raw.lower() in _STOP_WORDS
    if is_stop or not raw:
        answer_label = "Пропущено"
    else:
        answer_label = raw

    await _commit_answer_to_question(
        message.bot,
        data.get("last_question_chat_id"),
        data.get("last_question_msg_id"),
        last_question,
        answer_label,
    )

    if is_stop or not raw:
        await _proceed_to_search(message, state)
        return

    answers = list(data.get("answers", []))
    answers.append({"question": last_question, "answer": raw})
    await state.update_data(
        answers=answers,
        rounds_done=data.get("rounds_done", 0) + 1,
    )
    await _ask_next(message, state)
