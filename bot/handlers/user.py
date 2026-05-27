import asyncio
import html as _html
import logging
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import time
from urllib.parse import quote, urlsplit, urlunsplit

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import LinkPreviewOptions, Message

from bot.faq_repository import (
    get_all_faq,
    get_faq_last_updated,
    is_user_notified,
    mark_user_notified,
    register_user,
)
from bot.handlers.feedback import feedback_keyboard, log_query
from bot.keyboards.admin import admin_keyboard
from bot.keyboards.user import user_keyboard
from config import settings
from rag.dialog.classifier import classify_intent
from rag.dialog.triage import triage
from rag.generator import Generator
from rag.retriever import Retriever

logger = logging.getLogger(__name__)
router = Router()

_retriever = Retriever()
_generator = Generator()

_RATE_LIMIT = 10          # max requests per minute per user
_RATE_WINDOW = 60.0       # seconds
_user_timestamps: dict[int, list[float]] = defaultdict(list)


def _is_rate_limited(user_id: int) -> bool:
    now = time()
    _user_timestamps[user_id] = [
        t for t in _user_timestamps[user_id] if now - t < _RATE_WINDOW
    ]
    if len(_user_timestamps[user_id]) >= _RATE_LIMIT:
        return True
    _user_timestamps[user_id].append(now)
    return False


# ---------------------------------------------------------------------------
# Shadow-mode triage (phase C, §3.2)
# ---------------------------------------------------------------------------
# The retrieval-grounded triage cascade runs OFF the hot path: a fire-and-forget
# task logs its verdict next to the legacy classify_intent result for offline
# comparison. It never affects the answer the user receives — wiring the verdict
# into handler behavior is a later phase, gated by settings.triage_enabled.
_shadow_triage_tasks: set[asyncio.Task] = set()


def _fmt_feat(value) -> str:
    """Render a possibly-missing / NaN feature value for a log line."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.3f}"
    return str(value)


def _spawn_shadow_triage(question: str, legacy: dict) -> None:
    """Schedule a background triage run; keep a task ref so it is not GC'd."""
    task = asyncio.create_task(_run_shadow_triage(question, dict(legacy)))
    _shadow_triage_tasks.add(task)
    task.add_done_callback(_shadow_triage_tasks.discard)


async def _run_shadow_triage(question: str, legacy: dict) -> None:
    try:
        result = await triage(question, _retriever)
    except Exception:
        logger.warning("TRIAGE_SHADOW cascade crashed for q=%.80r", question, exc_info=True)
        return
    feats = result["features"]
    agree = bool(result["needs_clarification"]) == bool(legacy.get("needs_clarification"))
    logger.info(
        "TRIAGE_SHADOW q=%.80r agree=%s | legacy needs=%s reason=%s | "
        "triage needs=%s reason=%s by=%s rule=%s conf=%.2f | probe n=%s "
        "top1=%s gap_ratio=%s entropy=%s doc_spread=%s vec=%s",
        question, agree,
        legacy.get("needs_clarification"), legacy.get("reason"),
        result["needs_clarification"], result["reason"], result["decided_by"],
        result["rule"] or "-", result["confidence"],
        _fmt_feat(feats.get("n")), _fmt_feat(feats.get("top1")),
        _fmt_feat(feats.get("gap_ratio")), _fmt_feat(feats.get("entropy")),
        _fmt_feat(feats.get("doc_spread")),
        "yes" if result["query_vector"] else "no",
    )


_START_TEXT = (
    "👋 Привет! Я университетский консультант-ассистент.\n\n"
    "Я отвечаю на вопросы на основе официальных документов университета "
    "(уставы, правила, приказы и др.).\n\n"
    "Примеры вопросов:\n"
    "• Каковы условия перевода на другую специальность?\n"
    "• Как оформить академический отпуск?\n"
    "• Какие требования к итоговой аттестации?\n\n"
    "Просто напишите свой вопрос — на русском, английском или казахском 🇰🇿"
)

_HELP_TEXT = (
    "ℹ️ Как пользоваться ботом\n\n"
    "1. Напишите вопрос обычным текстом.\n"
    "2. Бот найдёт релевантные фрагменты в документах и сформирует ответ.\n"
    "3. В ответе будут указаны источники со ссылками на PDF.\n\n"
    "Поддерживаемые языки: RU / EN / KZ\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/help — эта справка"
)

MAX_TG_LEN = 4000
_DIVIDER = "─" * 22


async def _check_and_send_faq_notification(message: Message) -> None:
    if not message.from_user:
        return
    user_id = message.from_user.id
    notified = await is_user_notified(user_id)
    if not notified:
        await message.answer(
            "📢 FAQ был обновлён! Нажмите кнопку «FAQ» в меню, "
            "чтобы ознакомиться с актуальной версией."
        )
        await mark_user_notified(user_id)


def _md_to_html(text: str) -> str:
    text = _html.escape(text)
    # Markdown headings → bold
    text = re.sub(r'^#{1,3}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    # Italic *text*
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    # Inline code `text`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Citation markers [N] → monospace
    text = re.sub(r'\[(\d+)\]', r'<code>[\1]</code>', text)
    # Horizontal rule --- or ___ on its own line → Unicode divider
    text = re.sub(r'^[-_]{3,}\s*$', _DIVIDER, text, flags=re.MULTILINE)
    return text


_NO_PREVIEW = LinkPreviewOptions(is_disabled=True)


async def send_long_message(message, text: str, reply_markup=None):
    if len(text) <= MAX_TG_LEN:
        await message.answer(text, reply_markup=reply_markup, parse_mode="HTML",
                             link_preview_options=_NO_PREVIEW)
        return
    parts = []
    while len(text) > MAX_TG_LEN:
        split_at = text.rfind("\n\n", 0, MAX_TG_LEN)
        if split_at == -1:
            split_at = MAX_TG_LEN
        parts.append(text[:split_at])
        text = text[split_at:].lstrip()
    parts.append(text)
    for i, part in enumerate(parts):
        await message.answer(part, parse_mode="HTML",
                             link_preview_options=_NO_PREVIEW,
                             reply_markup=reply_markup if i == len(parts) - 1 else None)


# ---------------------------------------------------------------------------
# /start  /help
# ---------------------------------------------------------------------------

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    if message.from_user and message.from_user.id == settings.admin_telegram_id:
        await message.answer(_START_TEXT, reply_markup=admin_keyboard())
    else:
        await message.answer(_START_TEXT, reply_markup=user_keyboard())
        await register_user(message.from_user.id)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(_HELP_TEXT)


# ---------------------------------------------------------------------------
# FAQ button
# ---------------------------------------------------------------------------

@router.message(F.text == "📖 FAQ")
async def handle_faq_button(message: Message) -> None:
    await _check_and_send_faq_notification(message)
    entries = await get_all_faq()
    if not entries:
        await message.answer("📭 FAQ пока пуст.")
        return
    last_updated = await get_faq_last_updated()
    try:
        formatted_date = datetime.fromisoformat(last_updated).strftime("%d.%m.%Y %H:%M")
    except (TypeError, ValueError):
        formatted_date = "—"
    await message.answer(
        f"📖 <b>FAQ</b>\n🕐 Последнее обновление: {formatted_date}",
        parse_mode="HTML",
    )
    for entry in entries:
        await message.answer(
            f"❓ <b>{entry['question']}</b>\n\n💬 {entry['answer']}",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# .docx / .doc → extract text → RAG pipeline
# ---------------------------------------------------------------------------

def _extract_docx(path: Path) -> str:
    from docx import Document  # noqa: PLC0415
    doc = Document(str(path))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def _extract_doc_antiword(path: Path) -> str:
    result = subprocess.run(
        ["antiword", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "antiword failed")
    return result.stdout


@router.message(F.document)
async def handle_document(message: Message) -> None:
    doc = message.document
    if not doc or not doc.file_name:
        return

    name_lower = doc.file_name.lower()
    if not (name_lower.endswith(".docx") or name_lower.endswith(".doc")):
        return  # ignore other file types silently

    await _check_and_send_faq_notification(message)

    user_id = message.from_user.id if message.from_user else 0
    if _is_rate_limited(user_id):
        await message.answer("⏳ Слишком много запросов. Пожалуйста, подождите немного.")
        return

    status_msg = await message.answer("📥 Читаю файл...")

    suffix = ".docx" if name_lower.endswith(".docx") else ".doc"
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    tmp_path = Path(tmp_name)
    try:
        import os; os.close(tmp_fd)

        try:
            await message.bot.download(doc, destination=tmp_path)
        except Exception as e:
            logger.error("Failed to download document from user %s: %s", user_id, e)
            await status_msg.edit_text("❌ Не удалось скачать файл.")
            return

        loop = asyncio.get_running_loop()
        if name_lower.endswith(".docx"):
            try:
                question = await loop.run_in_executor(None, _extract_docx, tmp_path)
            except Exception as e:
                logger.error("Failed to extract .docx from user %s: %s", user_id, e)
                await status_msg.edit_text("❌ Не удалось прочитать файл .docx.")
                return
        else:
            if not shutil.which("antiword"):
                await status_msg.edit_text(
                    "❌ Формат .doc не поддерживается.\n"
                    "Пожалуйста, конвертируйте файл в .docx и попробуйте снова."
                )
                return
            try:
                question = await loop.run_in_executor(None, _extract_doc_antiword, tmp_path)
            except Exception as e:
                logger.error("Failed to extract .doc from user %s: %s", user_id, e)
                await status_msg.edit_text("❌ Не удалось прочитать файл .doc.")
                return
    finally:
        tmp_path.unlink(missing_ok=True)

    question = question.strip()
    if not question:
        await status_msg.edit_text("❌ Файл не содержит текста.")
        return

    await status_msg.edit_text("✅ Файл прочитан. Ищу информацию...")

    try:
        chunks = await _retriever.search(question)
    except Exception as e:
        logger.error("Retrieval failed for user %s: %s", user_id, e)
        await status_msg.edit_text("😔 Не удалось выполнить поиск. Попробуйте позже.")
        return

    try:
        result = await _generator.generate(question, chunks)
    except Exception as e:
        logger.error("Generation failed for user %s: %s", user_id, e)
        await status_msg.edit_text("😔 Не удалось сформировать ответ. Попробуйте позже.")
        return

    log_id = await log_query(
        user_id=user_id,
        query=question,
        detected_lang=result["detected_lang"],
        chunks=chunks,
        answer=result["answer"],
        sources=result["sources"],
    )

    text = _format_answer(result)

    if len(text) <= MAX_TG_LEN:
        await status_msg.edit_text(text, reply_markup=feedback_keyboard(log_id),
                                   parse_mode="HTML", link_preview_options=_NO_PREVIEW)
    else:
        await status_msg.delete()
        await send_long_message(message, text, reply_markup=feedback_keyboard(log_id))


# ---------------------------------------------------------------------------
# Text → RAG pipeline
# ---------------------------------------------------------------------------

@router.message(F.text)
async def handle_question(message: Message, state: FSMContext) -> None:
    await _check_and_send_faq_notification(message)

    question = (message.text or "").strip()
    if not question:
        return

    user_id = message.from_user.id if message.from_user else 0
    if _is_rate_limited(user_id):
        await message.answer(
            "⏳ Слишком много запросов. Пожалуйста, подождите немного."
        )
        return

    # Follow-up detection: if a fresh completed-turn context exists for this user,
    # ask the LLM whether the new message continues that topic.  When it does, skip
    # the clarification dialog entirely and go straight to search with the merged
    # query + inherited profile.
    from rag.dialog.followup import check_followup, save_context
    is_followup, merged_query, saved_profile = await check_followup(user_id, question)

    classification_reason: str
    if is_followup:
        classification_reason = "followup"
        logger.info(
            "follow-up path: user_id=%d q=%.60r merged=%.60r",
            user_id, question, merged_query,
        )
    else:
        intent = await classify_intent(question)
        logger.info("Intent classification: q='%.60s' result=%s", question, intent)

        # Shadow-mode triage: logs a comparison verdict off the hot path, never
        # changes the response. See _run_shadow_triage and rag/dialog/triage.py.
        _spawn_shadow_triage(question, intent)
        classification_reason = intent["reason"]

        if intent["needs_clarification"]:
            from bot.handlers.dialog import start_clarification_dialog
            await start_clarification_dialog(
                message, question, state,
                classification_reason=intent["reason"],
                required_slots=list(intent.get("required_slots") or []),
            )
            return

    status_msg = await message.answer("🔍 Ищу информацию...")

    try:
        if is_followup:
            from rag.dialog.reranker import rerank_chunks
            profile_strong = bool(
                saved_profile.get("user_type") or saved_profile.get("document_hints")
            ) or sum(
                1 for k in ("topics", "temporal_context") if saved_profile.get(k)
            ) >= 2
            if profile_strong:
                chunks = await _retriever.search_with_profile(merged_query, saved_profile, k=10)
                chunks = await rerank_chunks(
                    question, chunks, k=settings.top_k, profile=saved_profile,
                )
            else:
                chunks = await _retriever.search(merged_query)
        else:
            chunks = await _retriever.search(question)
    except Exception as e:
        logger.error("Retrieval failed for user %s: %s", user_id, e)
        await status_msg.edit_text("😔 Не удалось выполнить поиск. Попробуйте позже.")
        return

    try:
        result = await _generator.generate(
            question, chunks,
            profile=saved_profile if is_followup else None,
        )
    except Exception as e:
        logger.error("Generation failed for user %s: %s", user_id, e)
        await status_msg.edit_text("😔 Не удалось сформировать ответ. Попробуйте позже.")
        return

    log_id = await log_query(
        user_id=user_id,
        query=question,
        detected_lang=result["detected_lang"],
        chunks=chunks,
        answer=result["answer"],
        sources=result["sources"],
        clarification_rounds=0,
        classification_reason=classification_reason,
    )

    # Save turn context so subsequent follow-ups can reuse it.
    from rag.dialog.enricher import UserProfile as _UserProfile
    _profile_to_save = saved_profile if is_followup else _UserProfile(
        topics=[], user_type=None, admission_type=None,
        document_hints=[], temporal_context=None,
    )
    save_context(user_id, question, merged_query if is_followup else question, _profile_to_save)

    text = _format_answer(result)

    if len(text) <= MAX_TG_LEN:
        await status_msg.edit_text(text, reply_markup=feedback_keyboard(log_id),
                                   parse_mode="HTML", link_preview_options=_NO_PREVIEW)
    else:
        await status_msg.delete()
        await send_long_message(message, text, reply_markup=feedback_keyboard(log_id))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_url(url: str) -> str:
    parts = urlsplit(url)
    encoded_path = quote(parts.path, safe="/")
    return urlunsplit((parts.scheme, parts.netloc, encoded_path, parts.query, parts.fragment))


def _format_uploaded_at(raw: str) -> str:
    """Parse ISO timestamp and return DD.MM.YYYY, or empty string if absent/invalid."""
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%d.%m.%Y")
    except ValueError:
        return ""


def _build_sources_text(sources: list[dict]) -> str:
    lines = []
    for src in sources:
        doc_title = _html.escape(src.get("doc_title", ""))
        pages = src.get("pages", [])
        pages_str = ", ".join(str(p) for p in pages)
        date_str = _format_uploaded_at(src.get("uploaded_at", ""))
        url = src.get("url", "")

        first_page = pages[0] if pages else None
        if url:
            encoded = _encode_url(url)
            href = f"{encoded}#page={first_page}" if first_page else encoded
            title_part = f'<a href="{href}">{doc_title}</a>'
        else:
            title_part = f"<b>{doc_title}</b>"

        line = f"• {title_part}"
        if pages_str:
            line += f" — стр. {pages_str}"
        if date_str:
            line += f" <i>({date_str})</i>"
        lines.append(line)
    return "\n".join(lines)


def _format_answer(result: dict) -> str:
    text = f"💬 {_md_to_html(result['answer'])}"

    sources = result["sources"]
    if sources:
        text += f"\n\n{_DIVIDER}\n📄 <b>Источники</b>\n\n"
        text += _build_sources_text(sources)

    text += f"\n\n{_DIVIDER}\n<i>{_html.escape(_disclaimer(result['detected_lang']))}</i>"
    return text


_DISCLAIMER = {
    "Russian": "⚠️ ИИ может допускать ошибки. Проверяйте важную информацию.",
    "Kazakh": "⚠️ ЖИ қателесуі мүмкін. Маңызды ақпаратты тексеріңіз.",
    "English": "⚠️ AI may make mistakes. Please verify important information.",
}


def _disclaimer(detected_lang: str) -> str:
    return _DISCLAIMER.get(detected_lang, _DISCLAIMER["English"])


