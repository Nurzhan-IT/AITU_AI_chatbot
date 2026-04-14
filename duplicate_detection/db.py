import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

import aiosqlite

from config import TZ_UTC5

logger = logging.getLogger(__name__)

_DB_PATH: str = "data/bot.db"

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _setup_file_logger(log_path: str) -> None:
    """Attach a rotating file handler to the duplicate_detection logger.

    Called once from init_db(). Safe to call multiple times — skips setup
    if a FileHandler pointing to the same path is already attached.
    """
    dd_logger = logging.getLogger("duplicate_detection")

    # Avoid adding duplicate handlers on reload / hot-restart
    for h in dd_logger.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            return

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    fmt.converter = lambda *args: datetime.now(TZ_UTC5).timetuple()
    handler.setFormatter(fmt)

    dd_logger.addHandler(handler)
    dd_logger.setLevel(logging.DEBUG)


def _get_path() -> str:
    return _DB_PATH


async def init_db(db_path: str = "data/bot.db", log_path: str = "duplicate_detection.log") -> None:
    global _DB_PATH
    _DB_PATH = db_path

    _setup_file_logger(log_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS file_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT    NOT NULL,
                doc_title   TEXT    NOT NULL,
                event       TEXT    NOT NULL CHECK(event IN ('uploaded', 'deleted')),
                chunk_count INTEGER NOT NULL DEFAULT 0,
                timestamp   TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS warnings (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                warning_type        TEXT    NOT NULL CHECK(warning_type IN ('DUPLICATE', 'STALE')),
                new_filename        TEXT    NOT NULL,
                existing_filename   TEXT    NOT NULL,
                new_chunk_text      TEXT    NOT NULL,
                existing_chunk_text TEXT    NOT NULL,
                similarity          REAL    NOT NULL,
                llm_reason          TEXT,
                resolved            INTEGER NOT NULL DEFAULT 0,
                resolved_at         TEXT,
                created_at          TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_warnings_resolved  ON warnings(resolved);
            CREATE INDEX IF NOT EXISTS idx_warnings_new_file  ON warnings(new_filename);
            CREATE INDEX IF NOT EXISTS idx_file_history_fname ON file_history(filename);
        """)

        # Add new columns if migrating from older schema (idempotent)
        for col, definition in [
            ("replaces_filename", "TEXT"),
            ("document_date",     "TEXT"),
        ]:
            try:
                await db.execute(
                    f"ALTER TABLE file_history ADD COLUMN {col} {definition}"
                )
            except Exception:
                pass  # Column already exists
        await db.commit()

        await db.executescript("""

            CREATE TABLE IF NOT EXISTS query_logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER,
                query         TEXT    NOT NULL,
                detected_lang TEXT,
                avg_score     REAL,
                sources       TEXT,
                answer_length INTEGER,
                timestamp     TEXT    NOT NULL,
                feedback      INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_query_logs_user     ON query_logs(user_id);
            CREATE INDEX IF NOT EXISTS idx_query_logs_ts       ON query_logs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_query_logs_feedback ON query_logs(feedback);

            CREATE TABLE IF NOT EXISTS faq (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                question   TEXT    NOT NULL,
                answer     TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                updated_at TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                user_id                            INTEGER PRIMARY KEY,
                is_been_notified_about_faq_update  INTEGER NOT NULL DEFAULT 1
            );
        """)
        await db.commit()

    logger.info("SQLite database initialised at '%s'", db_path)
