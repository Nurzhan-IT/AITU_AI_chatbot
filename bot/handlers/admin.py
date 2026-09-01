import asyncio
import hashlib
import html as _html
import logging
import math
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
)

from bot.faq_repository import clear_document_faq
from config import TZ_UTC5, settings
from duplicate_detection import detector, repository
from ingestion.convert import convert_to_pdf
from ingestion.ingest import ingest_pdf
from rag.dialog.summary_store import delete_doc_summary
from rag.retriever import Retriever

logger = logging.getLogger(__name__)
router = Router()
_NO_PREVIEW = LinkPreviewOptions(is_disabled=True)

_PDFS_DIR = Path("pdfs")
_WARNINGS_PAGE_SIZE = 5
_DOCS_PAGE_SIZE = 5


def _filename_key(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:8]


async def _filename_from_key(key: str) -> str | None:
    retriever = Retriever()
    try:
        docs = await retriever.get_all_documents()
    except Exception:
        return None
    for doc in docs:
        if _filename_key(doc["filename"]) == key:
            return doc["filename"]
    return None


def _find_similar_filename(new_name: str, existing_names: list[str], threshold: float = 0.6) -> str | None:
    """Return the most similar existing filename if similarity >= threshold, else None."""
    best_ratio = 0.0
    best_match = None
    for name in existing_names:
        ratio = SequenceMatcher(None, new_name.lower(), name.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = name
    if best_ratio >= threshold and best_match != new_name:
        return best_match
    return None


# ---------------------------------------------------------------------------
# Admin guard — applied to the entire router
# ---------------------------------------------------------------------------

def _is_admin(message: Message) -> bool:
    return message.from_user is not None and settings.is_admin(message.from_user.id)


async def _admin_filter(message: Message) -> bool:
    return _is_admin(message)


async def _admin_callback_filter(callback: CallbackQuery) -> bool:
    if callback.from_user is not None and settings.is_admin(callback.from_user.id):
        return True
    logger.warning("Unauthorized admin callback attempt from user_id=%s", callback.from_user and callback.from_user.id)
    return False


router.message.filter(_admin_filter)
router.callback_query.filter(_admin_callback_filter)


# ---------------------------------------------------------------------------
# /upload
# ---------------------------------------------------------------------------

@router.message(Command("upload"))
async def cmd_upload(message: Message, bot: Bot) -> None:
    # The PDF must be sent as a document in the same message
    if not message.document:
        await message.answer(
            "📎 Отправьте PDF-файл вместе с командой /upload как документ."
        )
        return

    doc = message.document
    if not doc.file_name or not doc.file_name.lower().endswith((".pdf", ".docx", ".doc")):
        await message.answer("❌ Принимаются только PDF, DOCX и DOC файлы.")
        return

    _PDFS_DIR.mkdir(exist_ok=True)
    dest = _PDFS_DIR / doc.file_name

    status_msg = await message.answer("⏳ Скачиваю файл...")

    try:
        await bot.download(doc, destination=dest)
        logger.info("Downloaded '%s' (%d bytes)", doc.file_name, doc.file_size or 0)
    except Exception as e:
        logger.error("Failed to download '%s': %s", doc.file_name, e)
        await status_msg.edit_text(f"❌ Ошибка при скачивании файла: {e}")
        return

    needs_conversion = doc.file_name.lower().endswith((".docx", ".doc"))
    await status_msg.edit_text(
        "⏳ Конвертирую в PDF..." if needs_conversion else "⏳ Начинаю индексацию в фоне..."
    )

    title = dest.stem

    async def _progress(text: str) -> None:
        try:
            await status_msg.edit_text(text)
        except Exception:
            pass

    async def _run_ingest() -> None:
        nonlocal dest, title
        filename = doc.file_name

        if needs_conversion:
            try:
                pdf_path = await convert_to_pdf(dest, _PDFS_DIR)
            except Exception as e:
                logger.error("Failed to convert '%s' to PDF: %s", filename, e)
                await _progress(f"❌ Ошибка конвертации в PDF: {e}")
                return

            try:
                dest.unlink()
            except Exception as e:
                logger.warning("Could not remove source file '%s': %s", dest, e)

            dest = pdf_path
            filename = pdf_path.name
            title = pdf_path.stem
            await _progress("⏳ Начинаю индексацию в фоне...")

        try:
            n_chunks = await ingest_pdf(dest, title=title, progress_cb=_progress)
        except Exception as e:
            logger.error("Failed to ingest '%s': %s", filename, e)
            await _progress(f"❌ Ошибка при индексации: {e}")
            return

        await repository.record_file_event(filename, title, "uploaded", n_chunks)
        await clear_document_faq(filename)

        await _progress(
            f"✅ <b>{filename}</b> проиндексирован.\n"
            f"Загружено чанков: <b>{n_chunks}</b>\n"
            f"⏳ Анализирую на дубликаты...",
        )

        # Run duplicate/stale detection (non-critical — upload already succeeded)
        try:
            found = await detector.analyze_new_document(
                filepath=dest,
                filename=filename,
                doc_title=title,
                bot=bot,
                admin_ids=settings.admin_telegram_ids,
            )
            suffix = (
                f"⚠️ Найдено предупреждений: <b>{len(found)}</b>. Используйте /warnings"
                if found
                else "✅ Дубликаты и устаревшие данные не обнаружены."
            )
        except Exception as e:
            logger.error("Detection pipeline failed: %s", e)
            found = []
            suffix = "⚠️ Анализ дубликатов не выполнен."

        await status_msg.edit_text(
            f"✅ <b>{filename}</b> проиндексирован.\n"
            f"Загружено чанков: <b>{n_chunks}</b>\n{suffix}",
            parse_mode="HTML",
        )

    asyncio.create_task(_run_ingest())


# ---------------------------------------------------------------------------
# Content builders (shared by commands and callbacks)
# ---------------------------------------------------------------------------

async def _get_list_content(page: int = 1) -> tuple[str, InlineKeyboardMarkup | None]:
    retriever = Retriever()
    try:
        docs = await retriever.get_all_documents()
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        return f"❌ Ошибка при получении списка: {e}", None

    if not docs:
        return "📭 В базе нет ни одного документа.", None

    total = len(docs)
    total_pages = math.ceil(total / _DOCS_PAGE_SIZE)
    page = max(1, min(page, total_pages))
    offset = (page - 1) * _DOCS_PAGE_SIZE
    page_docs = docs[offset: offset + _DOCS_PAGE_SIZE]
    start = offset + 1
    end = min(offset + _DOCS_PAGE_SIZE, total)

    lines = [f"📚 <b>Документы в базе ({start}–{end} из {total}):</b>\n"]
    keyboard_rows = []
    for i, doc in enumerate(page_docs, start):
        doc_title = _html.escape(doc.get("doc_title", ""))
        url = doc.get("url", "")
        if url:
            parts = urlsplit(url)
            encoded_url = urlunsplit(
                (parts.scheme, parts.netloc, quote(parts.path, safe="/"), parts.query, parts.fragment)
            )
            title_part = f'<a href="{encoded_url}">{doc_title}</a>'
        else:
            title_part = f"<b>{doc_title}</b>"
        lines.append(f"{i}. {title_part}\n")
        key = _filename_key(doc["filename"])
        keyboard_rows.append([
            InlineKeyboardButton(text=f"🗑️ Удалить #{i}", callback_data=f"delete_ask:{key}"),
            InlineKeyboardButton(text=f"📜 История #{i}", callback_data=f"history:{key}"),
        ])

    nav_row: list[InlineKeyboardButton] = []
    if page > 1:
        nav_row.append(InlineKeyboardButton(text="◀ Назад", callback_data=f"docs_page:{page - 1}"))
    if total_pages > 1:
        nav_row.append(InlineKeyboardButton(text=f"Стр. {page}/{total_pages}", callback_data="noop"))
    if page < total_pages:
        nav_row.append(InlineKeyboardButton(text="Вперёд ▶", callback_data=f"docs_page:{page + 1}"))
    if nav_row:
        keyboard_rows.append(nav_row)

    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=keyboard_rows)


async def _get_warnings_content(page: int) -> tuple[str, InlineKeyboardMarkup]:
    page = max(1, page)
    offset = (page - 1) * _WARNINGS_PAGE_SIZE

    rows, total = await repository.list_warnings(
        resolved=False, limit=_WARNINGS_PAGE_SIZE, offset=offset
    )

    if total == 0:
        return "✅ Нет активных предупреждений.", InlineKeyboardMarkup(inline_keyboard=[])

    total_pages = math.ceil(total / _WARNINGS_PAGE_SIZE)
    start = offset + 1
    end = min(offset + _WARNINGS_PAGE_SIZE, total)

    lines = [f"📋 <b>Предупреждения ({start}–{end} из {total})</b>\n"]
    keyboard_rows = []

    base = settings.pdf_base_url.rstrip("/")
    for row in rows:
        wtype = row["warning_type"]
        badge = "🔁 DUPLICATE" if wtype == "DUPLICATE" else "📅 STALE"
        sim = f"{row['similarity']:.0%}"
        date = row["created_at"][:10]

        def _doc_link(filename: str, doc_title: str | None) -> str:
            label = _html.escape(doc_title or filename)
            parts = urlsplit(f"{base}/{filename}")
            encoded = urlunsplit((parts.scheme, parts.netloc, quote(parts.path, safe="/"), parts.query, parts.fragment))
            return f'<a href="{encoded}">{label}</a>'

        new_link = _doc_link(row["new_filename"], row.get("new_doc_title"))
        ex_link = _doc_link(row["existing_filename"], row.get("existing_doc_title"))

        lines.append(
            f"[<b>#{row['id']}</b>] {badge}\n"
            f"  Новый: {new_link}\n"
            f"  Существующий: {ex_link}\n"
            f"  Сходство: {sim}  |  {date}"
        )
        if wtype == "STALE" and row.get("llm_reason"):
            lines.append(f"  Причина: {row['llm_reason']}")
        lines.append("")
        keyboard_rows.append([
            InlineKeyboardButton(text=f"✅ Закрыть #{row['id']}", callback_data=f"resolve:{row['id']}"),
            InlineKeyboardButton(text=f"📄 Репорт #{row['id']}", callback_data=f"report:{row['id']}"),
        ])

    # Navigation row
    nav_row: list[InlineKeyboardButton] = []
    if page > 1:
        nav_row.append(InlineKeyboardButton(text="◀ Назад", callback_data=f"warnings_page:{page - 1}"))
    nav_row.append(InlineKeyboardButton(text=f"Страница {page}/{total_pages}", callback_data="noop"))
    if page < total_pages:
        nav_row.append(InlineKeyboardButton(text="Вперёд ▶", callback_data=f"warnings_page:{page + 1}"))
    keyboard_rows.append(nav_row)

    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=keyboard_rows)


def _current_warnings_page(callback: CallbackQuery) -> int:
    """Extract current page number from the noop 'Страница X/Y' button."""
    msg = callback.message
    if msg and hasattr(msg, "reply_markup") and msg.reply_markup:
        for row in msg.reply_markup.inline_keyboard:
            for btn in row:
                if btn.callback_data == "noop" and "Страница" in (btn.text or ""):
                    try:
                        return int(btn.text.split()[1].split("/")[0])
                    except (IndexError, ValueError):
                        pass
    return 1


# ---------------------------------------------------------------------------
# Handler helpers (called by both commands and text buttons)
# ---------------------------------------------------------------------------

async def _list_handler(message: Message) -> None:
    text, markup = await _get_list_content(page=1)
    await message.answer(text, parse_mode="HTML", reply_markup=markup,
                         link_preview_options=_NO_PREVIEW)


async def _warnings_handler(message: Message, page: int = 1) -> None:
    text, markup = await _get_warnings_content(page)
    await message.answer(text, parse_mode="HTML", reply_markup=markup,
                         link_preview_options=_NO_PREVIEW)


async def _health_handler(message: Message) -> None:
    from rag.embedder import Embedder as Emb
    from rag.generator import Generator as Gen

    retriever = Retriever()
    generator = Gen()
    embedder = Emb()

    qdrant_ok  = await retriever.ping()
    llm_ok     = await generator.ping()
    embed_ok   = await embedder.ping()

    lines = [
        "🩺 <b>Health Check</b>",
        f"Qdrant:    {'✅' if qdrant_ok  else '❌'}",
        f"LLM (Groq/OpenRouter): {'✅' if llm_ok else '❌'}",
        f"Embedder:  {'✅' if embed_ok  else '❌'}",
    ]
    await message.answer("\n".join(lines), parse_mode="HTML")


async def _stats_handler(message: Message) -> None:
    import aiosqlite

    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row

        row = await (await db.execute(
            """SELECT
                 COUNT(*)                                  AS total,
                 COUNT(CASE WHEN feedback = 1  THEN 1 END) AS thumbs_up,
                 COUNT(CASE WHEN feedback = -1 THEN 1 END) AS thumbs_down,
                 ROUND(AVG(avg_score), 3)                  AS avg_sim,
                 COUNT(CASE WHEN detected_lang='Russian' THEN 1 END) AS lang_ru,
                 COUNT(CASE WHEN detected_lang='English' THEN 1 END) AS lang_en,
                 COUNT(CASE WHEN detected_lang='Kazakh'  THEN 1 END) AS lang_kk
               FROM query_logs
               WHERE timestamp >= datetime('now', '-7 days')"""
        )).fetchone()

        top_rows = await (await db.execute(
            """SELECT query, COUNT(*) AS cnt
               FROM query_logs
               WHERE timestamp >= datetime('now', '-7 days')
               GROUP BY lower(trim(query))
               ORDER BY cnt DESC
               LIMIT 5"""
        )).fetchall()

    lines = [
        "📊 <b>Статистика за 7 дней</b>",
        f"Запросов: <b>{row['total']}</b>",
        f"👍 {row['thumbs_up']}  👎 {row['thumbs_down']}",
        f"Средняя близость: {row['avg_sim'] or '—'}",
        f"RU: {row['lang_ru']}  EN: {row['lang_en']}  KZ: {row['lang_kk']}",
        "",
        "<b>Топ-5 вопросов:</b>",
    ]
    for r in top_rows:
        q = r["query"][:80].replace("<", "&lt;")
        lines.append(f"• ({r['cnt']}×) {q}")

    await message.answer("\n".join(lines), parse_mode="HTML")


async def _report_handler(message: Message) -> None:
    status = await message.answer("⏳ Генерирую PDF-отчёт...")
    try:
        from duplicate_detection.report_generator import generate_report
        pdf_bytes = await generate_report()
        ts = datetime.now(TZ_UTC5).strftime("%Y%m%d_%H%M")
        filename = f"warnings_report_{ts}.pdf"
        await message.answer_document(
            BufferedInputFile(pdf_bytes, filename=filename),
            caption=(
                "📊 <b>Отчёт о предупреждениях</b>\n"
                "Содержит все нерешённые дубликаты и устаревшие данные."
            ),
            parse_mode="HTML",
        )
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        await message.answer(f"❌ Ошибка генерации отчёта: {exc}")
    finally:
        await status.delete()


# ---------------------------------------------------------------------------
# /list
# ---------------------------------------------------------------------------

@router.message(Command("list"))
async def cmd_list(message: Message) -> None:
    await _list_handler(message)


# ---------------------------------------------------------------------------
# /delete <filename>
# ---------------------------------------------------------------------------

@router.message(Command("delete"))
async def cmd_delete(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer(
            "✏️ Использование: <code>/delete filename.pdf</code>",
            parse_mode="HTML",
        )
        return

    filename = parts[1].strip()

    retriever = Retriever()
    try:
        deleted = await retriever.delete_document(filename)
    except Exception as e:
        logger.error("Failed to delete '%s': %s", filename, e)
        await message.answer(f"❌ Ошибка при удалении: {e}")
        return

    if deleted == 0:
        await message.answer(
            f"⚠️ Документ <code>{filename}</code> не найден в базе.",
            parse_mode="HTML",
        )
        return

    local_file = _PDFS_DIR / filename
    if local_file.exists():
        try:
            local_file.unlink()
            logger.info("Removed local file '%s'", local_file)
        except Exception as e:
            logger.warning("Could not remove local file '%s': %s", local_file, e)

    await repository.record_file_event(filename, filename, "deleted", deleted)
    await clear_document_faq(filename)
    await delete_doc_summary(filename)

    await message.answer(
        f"🗑️ Документ <code>{filename}</code> удалён.\n"
        f"Удалено чанков: <b>{deleted}</b>",
        parse_mode="HTML",
    )


# ---------------------------------------------------------------------------
# /warnings [page]
# ---------------------------------------------------------------------------

@router.message(Command("warnings"))
async def cmd_warnings(message: Message) -> None:
    parts = (message.text or "").split()
    page = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
    await _warnings_handler(message, page)


# ---------------------------------------------------------------------------
# /resolve <id>
# ---------------------------------------------------------------------------

@router.message(Command("resolve"))
async def cmd_resolve(message: Message) -> None:
    parts = (message.text or "").split()
    if len(parts) < 2 or not parts[1].isdigit():
        await message.answer(
            "✏️ Использование: <code>/resolve 42</code>",
            parse_mode="HTML",
        )
        return

    warning_id = int(parts[1])
    success = await repository.resolve_warning(warning_id)

    if success:
        await message.answer(
            f"✅ Предупреждение <code>#{warning_id}</code> закрыто.",
            parse_mode="HTML",
        )
    else:
        await message.answer(
            f"⚠️ Предупреждение <code>#{warning_id}</code> не найдено или уже закрыто.",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# /history [filename]
# ---------------------------------------------------------------------------

@router.message(Command("history"))
async def cmd_history(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)
    filename = parts[1].strip() if len(parts) > 1 else None

    rows = await repository.get_file_history(filename=filename, limit=10)

    if not rows:
        subject = f"<code>{filename}</code>" if filename else "документов"
        await message.answer(
            f"📭 История изменений {subject} пуста.",
            parse_mode="HTML",
        )
        return

    title = f"📜 <b>История: {filename}</b>" if filename else "📜 <b>Последние 10 событий</b>"
    lines = [title, ""]

    event_icon = {"uploaded": "⬆️", "deleted": "🗑️"}
    for row in rows:
        icon = event_icon.get(row["event"], "•")
        date = row["timestamp"][:16].replace("T", " ")
        lines.append(
            f"{icon} <b>{row['event'].upper()}</b>  {date}\n"
            f"   <code>{row['filename']}</code>"
            + (f"  ({row['chunk_count']} чанков)" if row["chunk_count"] else "")
        )

    await message.answer("\n".join(lines), parse_mode="HTML")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@router.message(Command("health"))
async def cmd_health(message: Message) -> None:
    await _health_handler(message)


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------

@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    await _stats_handler(message)


# ---------------------------------------------------------------------------
# /report
# ---------------------------------------------------------------------------

@router.message(Command("report"))
async def cmd_report(message: Message) -> None:
    parts = (message.text or "").split()
    if len(parts) > 1 and parts[1].isdigit():
        warning_id = int(parts[1])
        status = await message.answer(f"⏳ Генерирую PDF-отчёт для #{warning_id}...")
        try:
            from duplicate_detection.report_generator import generate_warning_report
            pdf_bytes = await generate_warning_report(warning_id)
            ts = datetime.now(TZ_UTC5).strftime("%Y%m%d_%H%M")
            await message.answer_document(
                BufferedInputFile(pdf_bytes, f"warning_{warning_id}_{ts}.pdf"),
                caption=f"📊 Подробный отчёт по предупреждению #{warning_id}",
            )
        except ValueError as exc:
            await message.answer(f"⚠️ {exc}")
        except Exception as exc:
            logger.error("Per-warning report failed for #%d: %s", warning_id, exc)
            await message.answer(f"❌ Ошибка генерации отчёта: {exc}")
        finally:
            await status.delete()
    else:
        await _report_handler(message)


# ---------------------------------------------------------------------------
# Reply keyboard text-button handlers
# ---------------------------------------------------------------------------

@router.message(F.text == "📋 Документы")
async def btn_documents(message: Message) -> None:
    await _list_handler(message)


@router.message(F.text == "⚠️ Предупреждения")
async def btn_warnings(message: Message) -> None:
    await _warnings_handler(message, page=1)


@router.message(F.text == "🩺 Состояние")
async def btn_health(message: Message) -> None:
    await _health_handler(message)


@router.message(F.text == "📊 Статистика")
async def btn_stats(message: Message) -> None:
    await _stats_handler(message)


@router.message(F.text == "📄 Отчёт")
async def btn_report(message: Message) -> None:
    await _report_handler(message)


@router.message(F.text == "📤 Загрузить")
async def btn_upload(message: Message) -> None:
    await message.answer("📎 Отправьте PDF-файл вместе с командой /upload как документ.")


# ---------------------------------------------------------------------------
# Inline callbacks — delete (two-step confirmation)
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("delete_ask:"))
async def cb_delete_ask(callback: CallbackQuery) -> None:
    key = callback.data.split(":", maxsplit=1)[1]
    filename = await _filename_from_key(key)
    if not filename:
        await callback.answer("❌ Документ не найден.", show_alert=True)
        return
    markup = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Подтвердить удаление", callback_data=f"delete_confirm:{key}"),
        InlineKeyboardButton(text="❌ Отмена",               callback_data=f"delete_cancel:{key}"),
    ]])
    await callback.message.edit_text(
        f"🗑️ Удалить <code>{filename}</code>? Это действие необратимо.",
        parse_mode="HTML",
        reply_markup=markup,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("delete_confirm:"))
async def cb_delete_confirm(callback: CallbackQuery) -> None:
    key = callback.data.split(":", maxsplit=1)[1]
    filename = await _filename_from_key(key)
    if not filename:
        await callback.answer("❌ Документ не найден.", show_alert=True)
        return

    retriever = Retriever()
    try:
        deleted = await retriever.delete_document(filename)
    except Exception as e:
        logger.error("Failed to delete '%s' via callback: %s", filename, e)
        await callback.answer(f"❌ Ошибка при удалении: {e}", show_alert=True)
        return

    if deleted == 0:
        await callback.message.edit_text(
            f"⚠️ Документ <code>{filename}</code> не найден в базе.",
            parse_mode="HTML",
        )
    else:
        local_file = _PDFS_DIR / filename
        if local_file.exists():
            try:
                local_file.unlink()
                logger.info("Removed local file '%s'", local_file)
            except Exception as e:
                logger.warning("Could not remove local file '%s': %s", local_file, e)

        await repository.record_file_event(filename, filename, "deleted", deleted)
        await clear_document_faq(filename)
        await delete_doc_summary(filename)
        await callback.message.edit_text(
            f"✅ Документ <code>{filename}</code> удалён.\n"
            f"Удалено чанков: <b>{deleted}</b>",
            parse_mode="HTML",
        )

    await callback.answer()


@router.callback_query(F.data.startswith("delete_cancel:"))
async def cb_delete_cancel(callback: CallbackQuery) -> None:
    await callback.message.edit_text("❌ Удаление отменено.")
    await callback.answer()


# ---------------------------------------------------------------------------
# Inline callbacks — history + back to list
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("history:"))
async def cb_history(callback: CallbackQuery) -> None:
    key = callback.data.split(":", maxsplit=1)[1]
    filename = await _filename_from_key(key)
    if not filename:
        await callback.answer("❌ Документ не найден.", show_alert=True)
        return

    rows = await repository.get_file_history(filename=filename, limit=10)

    if not rows:
        await callback.message.edit_text(
            f"📭 История изменений <code>{filename}</code> пуста.",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="🔙 Назад к списку", callback_data="back_to_list"),
            ]]),
        )
        await callback.answer()
        return

    title = f"📜 <b>История: {filename}</b>"
    lines = [title, ""]

    event_icon = {"uploaded": "⬆️", "deleted": "🗑️"}
    for row in rows:
        icon = event_icon.get(row["event"], "•")
        date = row["timestamp"][:16].replace("T", " ")
        lines.append(
            f"{icon} <b>{row['event'].upper()}</b>  {date}\n"
            f"   <code>{row['filename']}</code>"
            + (f"  ({row['chunk_count']} чанков)" if row["chunk_count"] else "")
        )

    markup = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🔙 Назад к списку", callback_data="back_to_list"),
    ]])
    await callback.message.edit_text(
        "\n".join(lines), parse_mode="HTML", reply_markup=markup
    )
    await callback.answer()


@router.callback_query(F.data == "back_to_list")
async def cb_back_to_list(callback: CallbackQuery) -> None:
    text, markup = await _get_list_content(page=1)
    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=markup,
                                     link_preview_options=_NO_PREVIEW)
    await callback.answer()


@router.callback_query(F.data.startswith("docs_page:"))
async def cb_docs_page(callback: CallbackQuery) -> None:
    page = int(callback.data.split(":")[1])
    text, markup = await _get_list_content(page=page)
    try:
        await callback.message.edit_text(text, parse_mode="HTML", reply_markup=markup,
                                         link_preview_options=_NO_PREVIEW)
    except Exception:
        pass  # "message is not modified" — ignore
    await callback.answer()


# ---------------------------------------------------------------------------
# Inline callbacks — warnings navigation & resolve
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("warnings_page:"))
async def cb_warnings_page(callback: CallbackQuery) -> None:
    page = int(callback.data.split(":")[1])
    text, markup = await _get_warnings_content(page)
    try:
        await callback.message.edit_text(
            text, parse_mode="HTML", reply_markup=markup,
            link_preview_options=_NO_PREVIEW,
        )
    except Exception:
        pass  # "message is not modified" — ignore
    await callback.answer()


@router.callback_query(F.data == "noop")
async def cb_noop(callback: CallbackQuery) -> None:
    await callback.answer()


@router.callback_query(F.data.startswith("report:"))
async def cb_warning_report(callback: CallbackQuery, bot: Bot) -> None:
    warning_id = int(callback.data.split(":")[1])
    await callback.answer()
    status = await callback.message.answer(f"⏳ Генерирую PDF-отчёт для #{warning_id}...")
    try:
        from duplicate_detection.report_generator import generate_warning_report
        pdf_bytes = await generate_warning_report(warning_id)
        ts = datetime.now(TZ_UTC5).strftime("%Y%m%d_%H%M")
        await bot.send_document(
            callback.message.chat.id,
            BufferedInputFile(pdf_bytes, f"warning_{warning_id}_{ts}.pdf"),
            caption=f"📊 Подробный отчёт по предупреждению #{warning_id}",
        )
    except ValueError as exc:
        await callback.message.answer(f"⚠️ {exc}")
    except Exception as exc:
        logger.error("Per-warning report failed for #%d: %s", warning_id, exc)
        await callback.message.answer(f"❌ Ошибка генерации отчёта: {exc}")
    finally:
        await status.delete()


@router.callback_query(F.data.startswith("resolve:"))
async def cb_resolve(callback: CallbackQuery) -> None:
    warning_id = int(callback.data.split(":")[1])
    success = await repository.resolve_warning(warning_id)

    if success:
        await callback.answer(f"✅ Предупреждение #{warning_id} закрыто.")
    else:
        await callback.answer("⚠️ Не найдено или уже закрыто.", show_alert=True)

    page = _current_warnings_page(callback)
    text, markup = await _get_warnings_content(page)
    try:
        await callback.message.edit_text(
            text, parse_mode="HTML", reply_markup=markup,
            link_preview_options=_NO_PREVIEW,
        )
    except Exception:
        pass  # "message is not modified" — ignore


