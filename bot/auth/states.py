from aiogram.fsm.state import State, StatesGroup


class EmailVerification(StatesGroup):
    waiting_for_email = State()
    waiting_for_code = State()
