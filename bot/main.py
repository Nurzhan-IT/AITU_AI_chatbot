import asyncio
import logging
import time
from datetime import datetime

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from config import settings, TZ_UTC5
from bot.auth.handler import router as auth_router
from bot.handlers import admin, admin_faq, admin_ai_faq, dialog, user
from bot.handlers.feedback import router as feedback_router
from rag.retriever import Retriever

logger = logging.getLogger(__name__)


async def on_startup(bot: Bot) -> None:
    # Ensure Qdrant collection exists before the first request
    retriever = Retriever()
    await retriever._ensure_collection()

    from duplicate_detection.db import init_db
    await init_db(settings.sqlite_db_path, settings.duplicate_detection_log)

    me = await bot.get_me()
    logger.info("Bot started: @%s (id=%d)", me.username, me.id)
    logger.info("Admin Telegram ID: %d", settings.admin_telegram_id)


async def on_shutdown(bot: Bot) -> None:
    logger.info("Bot shutting down, closing bot session...")
    await bot.session.close()


async def main() -> None:
    logging.Formatter.converter = lambda *args: datetime.now(TZ_UTC5).timetuple()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    bot = Bot(
        token=settings.telegram_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher(storage=MemoryStorage())

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    # Auth router first — intercepts unverified users before any other handler
    dp.include_router(auth_router)
    # Admin router must be registered before user — its filter rejects non-admins
    dp.include_router(admin.router)
    dp.include_router(admin_faq.router)
    dp.include_router(admin_ai_faq.router)
    dp.include_router(feedback_router)
    # Dialog router before user router — its state-scoped F.text handler must
    # intercept clarification answers before the generic text handler.
    dp.include_router(dialog.router)
    dp.include_router(user.router)

    logger.info("Starting polling...")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
