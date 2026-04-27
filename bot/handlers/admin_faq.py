import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from bot.faq_repository import (
    add_faq,
    delete_faq,
    edit_faq,
    get_all_faq,
    reset_all_user_notifications,
)
from bot.keyboards.admin import admin_faq_keyboard, admin_keyboard
from config import settings

logger = logging.getLogger(__name__)
router = Router()


# ---------------------------------------------------------------------------
# Admin guard — identical to admin.py
# ---------------------------------------------------------------------------

def _is_admin(message: Message) -> bool:
    return message.from_user is not None and message.from_user.id == settings.admin_telegram_id


async def _admin_filter(message: Message) -> bool:
    return _is_admin(message)


async def _admin_callback_filter(callback: CallbackQuery) -> bool:
    if callback.from_user is not None and callback.from_user.id == settings.admin_telegram_id:
        return True
    await callback.answer("⛔ Доступ запрещён.", show_alert=True)
    logger.warning("Unauthorized admin callback attempt from user_id=%s", callback.from_user and callback.from_user.id)
    return False


router.message.filter(_admin_filter)
router.callback_query.filter(_admin_callback_filter)


# ---------------------------------------------------------------------------
# FSM States
# ---------------------------------------------------------------------------

class FAQForm(StatesGroup):
    add_question  = State()
    add_answer    = State()
    edit_question = State()
    edit_answer   = State()


# ---------------------------------------------------------------------------
# Helper: build FAQ list text + inline keyboard
# ---------------------------------------------------------------------------

async def _get_faq_list_content() -> tuple[str, InlineKeyboardMarkup | None]:
    entries = await get_all_faq()
    if not entries:
        return "📭 FAQ пуст.", None

    n = len(entries)
    lines = [f"📋 <b>Список FAQ ({n} записей)</b>\n"]
    rows = []
    for i, entry in enumerate(entries, 1):
        lines.append(f"{i}. ❓ {entry['question']}\n    💬 {entry['answer']}\n")
        rows.append([
            InlineKeyboardButton(
                text=f"🗑️ Удалить #{entry['id']}",
                callback_data=f"faq_del_ask:{entry['id']}",
            ),
            InlineKeyboardButton(
                text=f"✏️ Редактировать #{entry['id']}",
                callback_data=f"faq_edit:{entry['id']}",
            ),
        ])

    return "".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


# ---------------------------------------------------------------------------
# "📖 FAQ" — open FAQ sub-menu
# ---------------------------------------------------------------------------

@router.message(F.text == "📖 FAQ")
async def btn_faq_menu(message: Message) -> None:
    await message.answer("📖 Меню FAQ:", reply_markup=admin_faq_keyboard())


# ---------------------------------------------------------------------------
# "◀ Назад" — return to main admin menu
# ---------------------------------------------------------------------------

@router.message(F.text == "◀ Назад")
async def btn_back(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("🏠 Главное меню:", reply_markup=admin_keyboard())


# ---------------------------------------------------------------------------
# "➕ Добавить FAQ" / /faq_add — start add FSM
# ---------------------------------------------------------------------------

@router.message(F.text == "➕ Добавить FAQ")
@router.message(Command("faq_add"))
async def btn_add_faq(message: Message, state: FSMContext) -> None:
    await state.set_state(FAQForm.add_question)
    await message.answer("✏️ Введите вопрос для нового FAQ:\n\n(отправьте /cancel для отмены)")


# FSM Step 1: receive question
@router.message(FAQForm.add_question)
async def fsm_add_question(message: Message, state: FSMContext) -> None:
    if (message.text or "").strip() == "/cancel":
        await state.clear()
        await message.answer("❌ Добавление отменено.")
        return

    await state.update_data(question=message.text)
    await state.set_state(FAQForm.add_answer)
    await message.answer("✏️ Теперь введите ответ:\n\n(отправьте /cancel для отмены)")


# FSM Step 2: receive answer
@router.message(FAQForm.add_answer)
async def fsm_add_answer(message: Message, state: FSMContext) -> None:
    if (message.text or "").strip() == "/cancel":
        await state.clear()
        await message.answer("❌ Добавление отменено.")
        return

    data = await state.get_data()
    question = data["question"]
    answer = message.text

    new_id = await add_faq(question, answer)
    await reset_all_user_notifications()
    await state.clear()

    await message.answer(f"✅ FAQ добавлен (#{new_id}).")

    text, markup = await _get_faq_list_content()
    await message.answer(text, parse_mode="HTML", reply_markup=markup)


# ---------------------------------------------------------------------------
# "📋 Список FAQ" / /faq_list — show list
# ---------------------------------------------------------------------------

@router.message(F.text == "📋 Список FAQ")
@router.message(Command("faq_list"))
async def btn_list_faq(message: Message) -> None:
    text, markup = await _get_faq_list_content()
    await message.answer(text, parse_mode="HTML", reply_markup=markup)


# ---------------------------------------------------------------------------
# Callbacks: Delete (two-step confirmation)
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("faq_del_ask:"))
async def cb_faq_del_ask(callback: CallbackQuery) -> None:
    faq_id = int(callback.data.split(":", maxsplit=1)[1])
    markup = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Подтвердить", callback_data=f"faq_del_confirm:{faq_id}"),
        InlineKeyboardButton(text="❌ Отмена",      callback_data=f"faq_del_cancel:{faq_id}"),
    ]])
    await callback.message.edit_text(
        f"🗑️ Удалить FAQ #{faq_id}? Это действие необратимо.",
        reply_markup=markup,
    )
    await callback.answer()


@router.callback_query(F.data.startswith("faq_del_confirm:"))
async def cb_faq_del_confirm(callback: CallbackQuery) -> None:
    faq_id = int(callback.data.split(":", maxsplit=1)[1])
    deleted = await delete_faq(faq_id)

    if deleted:
        await callback.message.edit_text(f"✅ FAQ #{faq_id} удалён.")
        text, markup = await _get_faq_list_content()
        await callback.message.answer(text, parse_mode="HTML", reply_markup=markup)
    else:
        await callback.message.edit_text(f"⚠️ FAQ #{faq_id} не найден.")

    await callback.answer()


@router.callback_query(F.data.startswith("faq_del_cancel:"))
async def cb_faq_del_cancel(callback: CallbackQuery) -> None:
    await callback.message.edit_text("❌ Удаление отменено.")
    text, markup = await _get_faq_list_content()
    await callback.message.answer(text, parse_mode="HTML", reply_markup=markup)
    await callback.answer()


# ---------------------------------------------------------------------------
# Callback: Edit — start FSM
# ---------------------------------------------------------------------------

@router.callback_query(F.data.startswith("faq_edit:"))
async def cb_faq_edit(callback: CallbackQuery, state: FSMContext) -> None:
    faq_id = int(callback.data.split(":", maxsplit=1)[1])
    entries = await get_all_faq()
    entry = next((e for e in entries if e["id"] == faq_id), None)

    if entry is None:
        await callback.answer(f"⚠️ FAQ #{faq_id} не найден.", show_alert=True)
        return

    await callback.answer()
    await callback.message.answer(
        f"📄 <b>Текущий FAQ #{faq_id}:</b>\n\n❓ {entry['question']}\n\n💬 {entry['answer']}",
        parse_mode="HTML",
    )
    await callback.message.answer(
        "✏️ Введите новый вопрос:\n\n(отправьте /cancel для отмены)"
    )
    await state.set_state(FAQForm.edit_question)
    await state.update_data(faq_id=faq_id)


# FSM Step 1: receive new question
@router.message(FAQForm.edit_question)
async def fsm_edit_question(message: Message, state: FSMContext) -> None:
    if (message.text or "").strip() == "/cancel":
        await state.clear()
        await message.answer("❌ Редактирование отменено.")
        return

    await state.update_data(new_question=message.text)
    await state.set_state(FAQForm.edit_answer)
    await message.answer("✏️ Введите новый ответ:\n\n(отправьте /cancel для отмены)")


# FSM Step 2: receive new answer
@router.message(FAQForm.edit_answer)
async def fsm_edit_answer(message: Message, state: FSMContext) -> None:
    if (message.text or "").strip() == "/cancel":
        await state.clear()
        await message.answer("❌ Редактирование отменено.")
        return

    data = await state.get_data()
    faq_id = data["faq_id"]
    new_question = data["new_question"]
    answer = message.text

    success = await edit_faq(faq_id, new_question, answer)
    if success:
        await reset_all_user_notifications()
    await state.clear()

    await message.answer(f"✅ FAQ #{faq_id} обновлён.")

    text, markup = await _get_faq_list_content()
    await message.answer(text, parse_mode="HTML", reply_markup=markup)
