import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from config import settings
from bot.handlers import admin, user
from rag.retriever import Retriever

logger = logging.getLogger(__name__)


async def on_startup(bot: Bot) -> None:
    # Ensure Qdrant collection exists before the first request
    retriever = Retriever()
    await retriever._ensure_collection()

    me = await bot.get_me()
    logger.info("Bot started: @%s (id=%d)", me.username, me.id)
    logger.info("Admin Telegram ID: %d", settings.admin_telegram_id)


async def on_shutdown(bot: Bot) -> None:
    logger.info("Bot shutting down, closing bot session...")
    await bot.session.close()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    bot = Bot(
        token=settings.telegram_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher()

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    # Admin router must be registered first — its filter rejects non-admins
    # before user router's catch-all F.text handler sees the message
    dp.include_router(admin.router)
    dp.include_router(user.router)

    logger.info("Starting polling...")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    asyncio.run(main())
