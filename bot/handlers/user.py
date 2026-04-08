import logging
from urllib.parse import quote, urlsplit, urlunsplit

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from rag.generator import Generator, detect_language
from rag.retriever import Retriever

logger = logging.getLogger(__name__)
router = Router()

_retriever = Retriever()
_generator = Generator()

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


async def send_long_message(message, text: str):
    if len(text) <= MAX_TG_LEN:
        await message.answer(text)
        return
    parts = []
    while len(text) > MAX_TG_LEN:
        split_at = text.rfind("\n\n", 0, MAX_TG_LEN)
        if split_at == -1:
            split_at = MAX_TG_LEN
        parts.append(text[:split_at])
        text = text[split_at:].lstrip()
    parts.append(text)
    for part in parts:
        await message.answer(part)


# ---------------------------------------------------------------------------
# /start  /help
# ---------------------------------------------------------------------------

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(_START_TEXT)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(_HELP_TEXT)


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

    sources = result["sources"]
    if sources:
        text += "\n\n📄 Источники:\n"
        text += _build_sources_text(sources)

    keyboard = _build_keyboard(sources)
    if len(text) <= MAX_TG_LEN:
        await status_msg.edit_text(text, reply_markup=keyboard)
    else:
        await status_msg.delete()
        await send_long_message(message, text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_url(url: str) -> str:
    parts = urlsplit(url)
    encoded_path = quote(parts.path, safe="/")
    return urlunsplit((parts.scheme, parts.netloc, encoded_path, parts.query, parts.fragment))


def _build_sources_text(sources: list[dict]) -> str:
    lines = []
    for src in sources:
        doc_title = src.get("doc_title", "")
        pages = src.get("pages", [])
        pages_str = ", ".join(str(p) for p in pages)
        lines.append(f"• {doc_title} — стр. {pages_str}" if pages_str else f"• {doc_title}")
    return "\n".join(lines)


def _build_keyboard(sources: list[dict]) -> InlineKeyboardMarkup | None:
    buttons = []
    for src in sources:
        url = src.get("url", "")
        if not url:
            continue
        doc_title = src.get("doc_title", "Документ")
        label = f"📎 {doc_title}"
        if len(label) > 64:
            label = label[:61] + "..."
        buttons.append([InlineKeyboardButton(text=label, url=_encode_url(url))])

    if not buttons:
        return None
    return InlineKeyboardMarkup(inline_keyboard=buttons)
