"""Shared pytest fixtures for the test suite.

`asyncio_mode = auto` is set in pytest.ini, so async fixtures and async
test functions are picked up by pytest-asyncio without extra markers.
"""
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

import config
import duplicate_detection.db as dd_db


@pytest.fixture
async def tmp_sqlite_db(tmp_path, monkeypatch):
    """Create a fresh SQLite DB with the full app schema and point every
    settings/module reader at it.

    duplicate_detection/repository.py reads the DB path via
    duplicate_detection.db._get_path() (a module global set by init_db),
    while bot/faq_repository.py and bot/auth/repository.py read
    config.settings.sqlite_db_path directly on every call. init_db() is the
    only place that creates the full schema (users, faq, warnings,
    file_history, query_logs, document_faq, doc_summaries), so both readers
    must be pointed at the same freshly-initialised file.
    """
    db_path = str(tmp_path / "test.db")
    log_path = str(tmp_path / "dd.log")

    await dd_db.init_db(db_path, log_path)
    monkeypatch.setattr(config.settings, "sqlite_db_path", db_path)

    return db_path


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeChatCompletion:
    choices: list[_FakeChoice] = field(default_factory=list)


@pytest.fixture
def fake_llm_response():
    """Factory: fake_llm_response("some text") -> object shaped like a
    groq/openai chat.completions.create() response, i.e.
    response.choices[0].message.content == "some text".
    """
    def _make(content: str) -> _FakeChatCompletion:
        return _FakeChatCompletion(choices=[_FakeChoice(message=_FakeMessage(content=content))])

    return _make


@pytest.fixture
def fake_async_client():
    """Minimal async mock standing in for groq.AsyncGroq / openai.AsyncOpenAI.

    Configure per-test:
        fake_async_client.chat.completions.create.return_value = fake_llm_response("...")
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
    """
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client
