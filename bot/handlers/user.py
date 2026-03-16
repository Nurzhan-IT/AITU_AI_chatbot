import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message

from rag.generator import Generator
from rag.retriever import Retriever

logger = logging.getLogger(__name__)
router = Router()

_retriever = Retriever()
_generator = Generator()

_START_TEXT = (
    "👋 <b>Привет! Я университетский консультант-ассистент.</b>\n\n"
    "Я отвечаю на вопросы на основе официальных документов университета "
    "(уставы, правила, приказы и др.).\n\n"
    "<b>Примеры вопросов:</b>\n"
    "• Каковы условия перевода на другую специальность?\n"
    "• Как оформить академический отпуск?\n"
    "• Какие требования к итоговой аттестации?\n\n"
    "Просто напишите свой вопрос — на русском, английском или казахском 🇰🇿"
)

_HELP_TEXT = (
    "ℹ️ <b>Как пользоваться ботом</b>\n\n"
    "1. Напишите вопрос обычным текстом.\n"
    "2. Бот найдёт релевантные фрагменты в документах и сформирует ответ.\n"
    "3. В ответе будут указаны источники со ссылками на PDF.\n\n"
    "<b>Поддерживаемые языки:</b> RU / EN / KZ\n\n"
    "<b>Команды:</b>\n"
    "/start — приветствие\n"
    "/help — эта справка"
)


# ---------------------------------------------------------------------------
# /start  /help
# ---------------------------------------------------------------------------

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(_START_TEXT, parse_mode="HTML")


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(_HELP_TEXT, parse_mode="HTML")


# ---------------------------------------------------------------------------
# Text → RAG pipeline
# ---------------------------------------------------------------------------

@router.message(F.text)
async def handle_question(message: Message) -> None:
    question = (message.text or "").strip()
    if not question:
        return

    status_msg = await message.answer("🔍 Ищу информацию...")

    try:
        chunks = await _retriever.search(question)
    except Exception as e:
        logger.error("Retrieval failed for user %s: %s", message.from_user and message.from_user.id, e)
        await status_msg.edit_text("😔 Не удалось выполнить поиск. Попробуйте позже.")
        return

    try:
        result = await _generator.generate(question, chunks)
    except Exception as e:
        logger.error("Generation failed for user %s: %s", message.from_user and message.from_user.id, e)
        await status_msg.edit_text("😔 Не удалось сформировать ответ. Попробуйте позже.")
        return

    text = f"💬 {result['answer']}"

    sources_block = _build_sources(result["sources"])
    if sources_block:
        text += f"\n\n{sources_block}"

    await status_msg.edit_text(text, parse_mode="HTML")


# ---------------------------------------------------------------------------
# Sources formatter
# ---------------------------------------------------------------------------

def _build_sources(sources: list[dict]) -> str:
    """Format generator sources [{doc_title, url, pages}, ...] into HTML block."""
    if not sources:
        return ""

    lines = ["📄 <b>Источники:</b>"]
    for src in sources:
        doc_title = src.get("doc_title", "")
        url = src.get("url", "")
        pages = src.get("pages", [])

        pages_str = ", ".join(str(p) for p in pages)
        label = f"{doc_title} — стр. {pages_str}" if pages_str else doc_title
        if url:
            lines.append(f'• {label}  <a href="{url}">📎 открыть</a>')
        else:
            lines.append(f"• {label}")

    return "\n".join(lines)
