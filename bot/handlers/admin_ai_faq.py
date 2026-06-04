import hashlib
import logging

from aiogram import F, Router
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from bot.faq_repository import get_document_faq, save_document_faq
from config import settings
from rag.generator import generate_document_faq
from rag.retriever import Retriever

logger = logging.getLogger(__name__)
router = Router()


def _is_admin(message: Message) -> bool:
    return message.from_user is not None and settings.is_admin(message.from_user.id)


async def _admin_filter(message: Message) -> bool:
    return _is_admin(message)


async def _admin_callback_filter(callback: CallbackQuery) -> bool:
    if callback.from_user is not None and settings.is_admin(callback.from_user.id):
        return True
    logger.warning("Unauthorized AI FAQ callback from user_id=%s", callback.from_user and callback.from_user.id)
    return False


router.message.filter(_admin_filter)
router.callback_query.filter(_admin_callback_filter)


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


@router.message(F.text == "🤖 AI FAQ")
async def btn_ai_faq(message: Message) -> None:
    retriever = Retriever()
    try:
        docs = await retriever.get_all_documents()
    except Exception as e:
        await message.answer(f"❌ Ошибка при получении списка документов: {e}")
        return

    if not docs:
        await message.answer("📭 В базе нет ни одного документа.")
        return

    lines = ["📚 <b>Выберите документ для генерации FAQ:</b>\n"]
    keyboard_rows = []
    for i, doc in enumerate(docs, 1):
        lines.append(
            f"{i}. <b>{doc['doc_title']}</b>\n"
            f"   📄 <code>{doc['filename']}</code>"
        )
        key = _filename_key(doc["filename"])
        keyboard_rows.append([
            InlineKeyboardButton(
                text=f"🤖 FAQ для #{i}: {doc['doc_title'][:30]}",
                callback_data=f"ai_faq_doc:{key}",
            )
        ])

    await message.answer(
        "\n".join(lines),
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard_rows),
    )


@router.callback_query(F.data.startswith("ai_faq_doc:"))
async def cb_ai_faq_doc(callback: CallbackQuery) -> None:
    key = callback.data.split(":", maxsplit=1)[1]
    filename = await _filename_from_key(key)

    if not filename:
        await callback.answer("❌ Документ не найден.", show_alert=True)
        return

    await callback.answer()

    existing = await get_document_faq(filename)
    if existing:
        await _send_faq_message(callback.message, filename, existing, from_cache=True)
        return

    status = await callback.message.answer(
        f"⏳ Генерирую 10 FAQ для <b>{filename}</b>...", parse_mode="HTML"
    )
    try:
        retriever = Retriever()
        chunks = await retriever.get_document_chunks(filename)

        if not chunks:
            await status.edit_text(
                f"❌ Не найдено данных для <b>{filename}</b>.", parse_mode="HTML"
            )
            return

        faqs = await generate_document_faq(chunks)

        if not faqs:
            await status.edit_text("❌ Не удалось сгенерировать FAQ. Попробуйте позже.")
            return

        await save_document_faq(filename, faqs)
        await status.delete()
        await _send_faq_message(callback.message, filename, faqs, from_cache=False)

    except Exception as e:
        logger.error("AI FAQ generation failed for '%s': %s", filename, e)
        await status.edit_text(f"❌ Ошибка генерации FAQ: {e}")


async def _send_faq_message(
    message: Message, filename: str, faqs: list[dict], from_cache: bool
) -> None:
    source = "из базы данных" if from_cache else "только что сгенерировано"
    header = f"🤖 <b>AI FAQ: {filename}</b> <i>({source})</i>\n\n"

    entries: list[str] = []
    for i, faq in enumerate(faqs, 1):
        q = faq["question"].replace("<", "&lt;").replace(">", "&gt;")
        a = faq["answer"].replace("<", "&lt;").replace(">", "&gt;")
        entries.append(f"<b>{i}. {q}</b>\n{a}")

    # Telegram message limit is 4096 chars; split if needed
    current = header
    for entry in entries:
        block = entry + "\n\n"
        if len(current) + len(block) > 4000:
            await message.answer(current.rstrip(), parse_mode="HTML")
            current = block
        else:
            current += block

    if current.strip():
        await message.answer(current.rstrip(), parse_mode="HTML")
