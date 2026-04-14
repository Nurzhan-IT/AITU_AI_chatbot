from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def user_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📖 FAQ")]],
        resize_keyboard=True,
    )
