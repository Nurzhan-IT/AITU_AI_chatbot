from unittest.mock import AsyncMock, MagicMock

from duplicate_detection import notifier


def _dup(wid=1, existing="existing.pdf", sim=0.95, text="the new duplicated text"):
    return {
        "id": wid,
        "warning_type": "DUPLICATE",
        "existing_filename": existing,
        "similarity": sim,
        "new_chunk_text": text,
    }


def _stale(wid=2, existing="existing.pdf", sim=0.8, reason="supersedes the old policy"):
    return {
        "id": wid,
        "warning_type": "STALE",
        "existing_filename": existing,
        "similarity": sim,
        "new_chunk_text": "irrelevant for stale rendering",
        "llm_reason": reason,
    }


class TestSendUploadWarnings:
    async def test_empty_warnings_does_not_send(self):
        bot = MagicMock()
        bot.send_message = AsyncMock()

        await notifier.send_upload_warnings(bot, [1, 2], "f.pdf", [])

        bot.send_message.assert_not_called()

    async def test_duplicate_and_stale_sections_formatted(self):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        warnings = [_dup(), _stale()]

        await notifier.send_upload_warnings(bot, [1], "new.pdf", warnings)

        bot.send_message.assert_awaited_once()
        args, kwargs = bot.send_message.call_args
        text = args[1]
        assert kwargs.get("parse_mode") == "HTML"

        assert "🔁 ДУБЛИКАТЫ (1)" in text
        assert "📅 УСТАРЕВШИЕ (1)" in text
        assert "existing.pdf" in text
        assert "95%" in text
        assert "80%" in text
        assert "supersedes the old policy" in text
        assert "/warnings" in text
        assert "/resolve" in text

    async def test_only_duplicates_omits_stale_section(self):
        bot = MagicMock()
        bot.send_message = AsyncMock()

        await notifier.send_upload_warnings(bot, [1], "new.pdf", [_dup()])

        text = bot.send_message.call_args.args[1]
        assert "ДУБЛИКАТЫ" in text
        assert "УСТАРЕВШИЕ" not in text

    async def test_only_stale_omits_duplicate_section(self):
        bot = MagicMock()
        bot.send_message = AsyncMock()

        await notifier.send_upload_warnings(bot, [1], "new.pdf", [_stale()])

        text = bot.send_message.call_args.args[1]
        assert "УСТАРЕВШИЕ" in text
        assert "ДУБЛИКАТЫ" not in text

    async def test_multiple_admins_all_receive_message(self):
        bot = MagicMock()
        bot.send_message = AsyncMock()

        await notifier.send_upload_warnings(bot, [1, 2, 3], "new.pdf", [_dup()])

        assert bot.send_message.await_count == 3
        recipients = [call.args[0] for call in bot.send_message.call_args_list]
        assert recipients == [1, 2, 3]

    async def test_one_admin_failure_does_not_block_others(self):
        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=[RuntimeError("blocked bot"), None, None])

        await notifier.send_upload_warnings(bot, [1, 2, 3], "new.pdf", [_dup()])

        assert bot.send_message.await_count == 3
        recipients = [call.args[0] for call in bot.send_message.call_args_list]
        assert recipients == [1, 2, 3]


class TestSnippet:
    def test_collapses_whitespace(self):
        assert notifier._snippet("hello   \n\n  world") == "hello world"

    def test_short_text_unchanged(self):
        assert notifier._snippet("short text") == "short text"

    def test_long_text_truncated_with_ellipsis(self):
        text = "a" * 100
        result = notifier._snippet(text)
        assert result == "a" * 80 + "…"
