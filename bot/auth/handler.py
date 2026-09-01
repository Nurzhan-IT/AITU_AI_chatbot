import logging
import secrets
from datetime import datetime

from aiogram import F, Router
from aiogram.filters import BaseFilter, Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from bot.auth.email_service import send_verification_code
from bot.auth.repository import (
    get_pending_verification,
    increment_attempts,
    is_user_verified,
    mark_user_verified,
    save_verification_code,
)
from bot.auth.states import EmailVerification
from bot.keyboards.user import user_keyboard
from config import TZ_UTC5, settings

logger = logging.getLogger(__name__)
router = Router()

_ALLOWED_DOMAIN = "astanait.edu.kz"
_MAX_ATTEMPTS = 3

_ASK_EMAIL_TEXT = (
    "👋 Добро пожаловать в AITU Chatbot!\n\n"
    "Бот доступен только для студентов и сотрудников университета.\n"
    "Введите вашу университетскую почту:\n\n"
    "<code>username@astanait.edu.kz</code>"
)

_WELCOME_TEXT = (
    "✅ Почта подтверждена! Добро пожаловать в <b>AITU Assistant</b>.\n\n"
    "Я отвечаю на вопросы на основе официальных документов университета. "
    "Задайте любой вопрос — на русском, английском или казахском 🇰🇿"
)


def _resend_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Отправить код заново", callback_data="auth:resend")]
        ]
    )


def _generate_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


class IsNotVerified(BaseFilter):
    async def __call__(self, message: Message) -> bool:
        if not settings.email_verification_enabled:
            return False
        if message.from_user is None:
            return False
        if settings.is_admin(message.from_user.id):
            return False
        return not await is_user_verified(message.from_user.id)


# --- Catch-all for non-verified users outside any FSM state ---

@router.message(IsNotVerified(), StateFilter(None))
async def start_verification(message: Message, state: FSMContext) -> None:
    await state.set_state(EmailVerification.waiting_for_email)
    await message.answer(_ASK_EMAIL_TEXT)


# --- Email input ---

@router.message(EmailVerification.waiting_for_email)
async def handle_email(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip().lower()
    if not raw.endswith(f"@{_ALLOWED_DOMAIN}"):
        await message.answer(
            f"❌ Только почта <b>@{_ALLOWED_DOMAIN}</b> принимается.\n\n"
            "Введите вашу университетскую почту:"
        )
        return

    code = _generate_code()
    await save_verification_code(message.from_user.id, raw, code)

    sent = await send_verification_code(raw, code)
    if not sent:
        await message.answer(
            "⚠️ Не удалось отправить письмо. Попробуйте позже или обратитесь к администратору."
        )
        await state.clear()
        return

    await state.set_state(EmailVerification.waiting_for_code)
    await state.update_data(email=raw)
    await message.answer(
        f"✉️ Код подтверждения отправлен на <code>{raw}</code>\n\n"
        "Введите 6-значный код (действителен 24 часа).",
        reply_markup=_resend_keyboard(),
    )


# --- Code input ---

@router.message(EmailVerification.waiting_for_code)
async def handle_code(message: Message, state: FSMContext) -> None:
    entered = (message.text or "").strip()
    user_id = message.from_user.id

    pending = await get_pending_verification(user_id)
    if pending is None:
        await state.set_state(EmailVerification.waiting_for_email)
        await message.answer("Сессия устарела. " + _ASK_EMAIL_TEXT)
        return

    # Expiry check
    expires_at = datetime.fromisoformat(pending["verification_expires_at"])
    if datetime.now(TZ_UTC5) > expires_at:
        await state.set_state(EmailVerification.waiting_for_email)
        await message.answer(
            "⏰ Код истёк. Пожалуйста, введите почту заново."
        )
        return

    if entered == pending["verification_code"]:
        await mark_user_verified(user_id)
        await state.clear()
        await message.answer(_WELCOME_TEXT, reply_markup=user_keyboard())
        return

    attempts = await increment_attempts(user_id)
    remaining = _MAX_ATTEMPTS - attempts
    if remaining <= 0:
        await state.set_state(EmailVerification.waiting_for_email)
        await message.answer(
            "❌ Превышено число попыток. Введите почту заново."
        )
        return

    await message.answer(
        f"❌ Неверный код. Осталось попыток: <b>{remaining}</b>",
        reply_markup=_resend_keyboard(),
    )


# --- Resend callback ---

@router.callback_query(F.data == "auth:resend", EmailVerification.waiting_for_code)
async def resend_code(callback: CallbackQuery, state: FSMContext) -> None:
    user_id = callback.from_user.id
    data = await state.get_data()
    email = data.get("email")

    if not email:
        await callback.message.answer(_ASK_EMAIL_TEXT)
        await state.set_state(EmailVerification.waiting_for_email)
        await callback.answer()
        return

    code = _generate_code()
    await save_verification_code(user_id, email, code)
    sent = await send_verification_code(email, code)

    if sent:
        await callback.message.answer(
            f"✉️ Новый код отправлен на <code>{email}</code>",
            reply_markup=_resend_keyboard(),
        )
    else:
        await callback.message.answer(
            "⚠️ Не удалось отправить письмо. Попробуйте позже."
        )
    await callback.answer()


# --- Cancel ---

@router.message(
    Command("cancel"),
    StateFilter(EmailVerification.waiting_for_email, EmailVerification.waiting_for_code),
)
async def cancel_verification(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Верификация отменена. Отправьте любое сообщение, чтобы начать снова.")
