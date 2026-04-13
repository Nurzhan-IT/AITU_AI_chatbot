import asyncio
import logging
import math
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message

from config import settings, TZ_UTC5
from ingestion.ingest import ingest_pdf
from rag.retriever import Retriever
from duplicate_detection import repository, detector

logger = logging.getLogger(__name__)
router = Router()

_PDFS_DIR = Path("pdfs")
_WARNINGS_PAGE_SIZE = 5
_pending_replace: dict[int, tuple[str, str]] = {}


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
    return message.from_user is not None and message.from_user.id == settings.admin_telegram_id


async def _admin_filter(message: Message) -> bool:
    if _is_admin(message):
        return True
    await message.answer("⛔ Доступ запрещён.")
    logger.warning("Unauthorized admin access attempt from user_id=%s", message.from_user and message.from_user.id)
    return False


router.message.filter(_admin_filter)


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
    if not doc.file_name or not doc.file_name.lower().endswith(".pdf"):
        await message.answer("❌ Принимаются только PDF-файлы.")
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

    await status_msg.edit_text("⏳ Начинаю индексацию в фоне...")

    title = dest.stem

    async def _progress(text: str) -> None:
        try:
            await status_msg.edit_text(text)
        except Exception:
            pass

    async def _run_ingest() -> None:
        try:
            n_chunks = await ingest_pdf(dest, title=title, progress_cb=_progress)
        except Exception as e:
            logger.error("Failed to ingest '%s': %s", doc.file_name, e)
            await _progress(f"❌ Ошибка при индексации: {e}")
            return

        await repository.record_file_event(doc.file_name, title, "uploaded", n_chunks)

        await _progress(
            f"✅ <b>{doc.file_name}</b> проиндексирован.\n"
            f"Загружено чанков: <b>{n_chunks}</b>\n"
            f"⏳ Анализирую на дубликаты...",
        )

        # Run duplicate/stale detection (non-critical — upload already succeeded)
        try:
            found = await detector.analyze_new_document(
                filepath=dest,
                filename=doc.file_name,
                doc_title=title,
                bot=bot,
                admin_id=settings.admin_telegram_id,
            )
            suffix = (
                f"⚠️ Найдено предупреждений: <b>{len(found)}</b>. Используйте /warnings"
                if found
                else "✅ Дубликаты и устаревшие данные не обнаружены."
            )
        except Exception as e:
            logger.error("Detection pipeline failed: %s", e)
            suffix = "⚠️ Анализ дубликатов не выполнен."

        await status_msg.edit_text(
            f"✅ <b>{doc.file_name}</b> проиндексирован.\n"
            f"Загружено чанков: <b>{n_chunks}</b>\n{suffix}",
            parse_mode="HTML",
        )

    asyncio.create_task(_run_ingest())


# ---------------------------------------------------------------------------
# /confirm_replace  /skip_replace
# ---------------------------------------------------------------------------

@router.message(Command("confirm_replace"))
async def cmd_confirm_replace(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    pending = _pending_replace.pop(user_id, None)
    if not pending:
        await message.answer("Нет ожидающего подтверждения замены.")
        return
    new_name, old_name = pending
    await repository.record_file_event(new_name, new_name, "uploaded", 0, replaces_filename=old_name)
    await message.answer(
        f"✅ Записано: <code>{new_name}</code> заменяет <code>{old_name}</code>.",
        parse_mode="HTML",
    )


@router.message(Command("skip_replace"))
async def cmd_skip_replace(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    _pending_replace.pop(user_id, None)
    await message.answer("ℹ️ Связь с предыдущей версией не записана.")


# ---------------------------------------------------------------------------
# /list
# ---------------------------------------------------------------------------

@router.message(Command("list"))
async def cmd_list(message: Message) -> None:
    retriever = Retriever()
    try:
        docs = await retriever.get_all_documents()
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        await message.answer(f"❌ Ошибка при получении списка: {e}")
        return

    if not docs:
        await message.answer("📭 В базе нет ни одного документа.")
        return

    lines = ["📚 <b>Документы в базе:</b>\n"]
    for i, doc in enumerate(docs, 1):
        lines.append(
            f"{i}. <b>{doc['doc_title']}</b>\n"
            f"   📄 <code>{doc['filename']}</code>"
        )

    await message.answer("\n".join(lines), parse_mode="HTML")


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

    # Remove the local file if it exists
    local_file = _PDFS_DIR / filename
    if local_file.exists():
        try:
            local_file.unlink()
            logger.info("Removed local file '%s'", local_file)
        except Exception as e:
            logger.warning("Could not remove local file '%s': %s", local_file, e)

    # Record deletion in file history
    await repository.record_file_event(filename, filename, "deleted", deleted)

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
    page = max(1, page)
    offset = (page - 1) * _WARNINGS_PAGE_SIZE

    rows, total = await repository.list_warnings(
        resolved=False, limit=_WARNINGS_PAGE_SIZE, offset=offset
    )

    if total == 0:
        await message.answer("✅ Нет активных предупреждений.")
        return

    total_pages = math.ceil(total / _WARNINGS_PAGE_SIZE)
    start = offset + 1
    end = min(offset + _WARNINGS_PAGE_SIZE, total)

    lines = [f"📋 <b>Предупреждения ({start}–{end} из {total})</b>\n"]

    for row in rows:
        wtype = row["warning_type"]
        badge = "🔁 DUPLICATE" if wtype == "DUPLICATE" else "📅 STALE"
        sim = f"{row['similarity']:.0%}"
        date = row["created_at"][:10]  # YYYY-MM-DD
        lines.append(
            f"[<b>#{row['id']}</b>] {badge}\n"
            f"  Новый: <code>{row['new_filename']}</code>\n"
            f"  Существующий: <code>{row['existing_filename']}</code>\n"
            f"  Сходство: {sim}  |  {date}"
        )
        if wtype == "STALE" and row.get("llm_reason"):
            lines.append(f"  Причина: {row['llm_reason']}")
        lines.append("")

    lines.append(f"Страница {page}/{total_pages}")
    if page < total_pages:
        lines.append(f"Следующая: /warnings {page + 1}")
    lines.append("\nЗакрыть предупреждение: /resolve &lt;id&gt;")

    await message.answer("\n".join(lines), parse_mode="HTML")


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
        date = row["timestamp"][:16].replace("T", " ")  # YYYY-MM-DD HH:MM
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
    from rag.generator import Generator as Gen
    from rag.embedder import Embedder as Emb

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


# ---------------------------------------------------------------------------
# /stats
# ---------------------------------------------------------------------------

@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    """Show basic query analytics for the last 7 days."""
    import aiosqlite
    from config import settings

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


# ---------------------------------------------------------------------------
# /report
# ---------------------------------------------------------------------------

@router.message(Command("report"))
async def cmd_report(message: Message) -> None:
    """Generate and send a PDF report of all unresolved warnings."""
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
