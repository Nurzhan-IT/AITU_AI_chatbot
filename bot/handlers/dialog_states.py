from aiogram.fsm.state import State, StatesGroup


class ClarifyDialog(StatesGroup):
    waiting_for_answer = State()
