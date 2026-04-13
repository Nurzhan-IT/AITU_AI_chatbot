from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def admin_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📋 Документы"),  KeyboardButton(text="⚠️ Предупреждения")],
            [KeyboardButton(text="🩺 Состояние"),  KeyboardButton(text="📊 Статистика"),  KeyboardButton(text="📄 Отчёт")],
            [KeyboardButton(text="📤 Загрузить")],
        ],
        resize_keyboard=True,
    )
