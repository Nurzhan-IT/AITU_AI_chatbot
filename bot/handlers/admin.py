import logging
from pathlib import Path

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.types import Message

from config import settings
from ingestion.ingest import ingest_pdf
from rag.retriever import Retriever

logger = logging.getLogger(__name__)
router = Router()

_PDFS_DIR = Path("pdfs")


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

    await status_msg.edit_text("⏳ Индексирую...")

    try:
        title = dest.stem
        n_chunks = await ingest_pdf(dest, title=title)
    except Exception as e:
        logger.error("Failed to ingest '%s': %s", doc.file_name, e)
        await status_msg.edit_text(f"❌ Ошибка при индексации: {e}")
        return

    await status_msg.edit_text(
        f"✅ <b>{doc.file_name}</b> проиндексирован.\n"
        f"Загружено чанков: <b>{n_chunks}</b>",
        parse_mode="HTML",
    )


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

    await message.answer(
        f"🗑️ Документ <code>{filename}</code> удалён.\n"
        f"Удалено чанков: <b>{deleted}</b>",
        parse_mode="HTML",
    )
