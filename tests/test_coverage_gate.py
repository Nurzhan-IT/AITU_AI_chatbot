import asyncio

import pytest

from rag.coverage_gate import (
    _build_user_message,
    _extract_json_object,
    _format_profile_slot,
    check_coverage,
)

# ---------------------------------------------------------------------------
# _format_profile_slot()
# ---------------------------------------------------------------------------

class TestFormatProfileSlot:
    def test_none_is_unknown(self):
        assert _format_profile_slot(None) == "UNKNOWN"

    def test_empty_string_is_unknown(self):
        assert _format_profile_slot("   ") == "UNKNOWN"

    def test_plain_value_is_stripped(self):
        assert _format_profile_slot(" бакалавр ") == "бакалавр"

    def test_empty_list_is_unknown(self):
        assert _format_profile_slot([]) == "UNKNOWN"

    def test_list_of_blanks_is_unknown(self):
        assert _format_profile_slot([None, "  ", ""]) == "UNKNOWN"

    def test_list_joins_non_empty_items(self):
        assert _format_profile_slot(["дедлайны", " стипендия ", None]) == "дедлайны, стипендия"


# ---------------------------------------------------------------------------
# _build_user_message()
# ---------------------------------------------------------------------------

class TestBuildUserMessage:
    def test_empty_profile_and_fragments(self):
        msg = _build_user_message("Вопрос?", {}, [])
        assert "QUESTION:\nВопрос?" in msg
        assert "user_type: UNKNOWN" in msg
        assert "FRAGMENTS:\n(none)" in msg

    def test_fragments_are_numbered_with_titles_and_excerpt(self):
        fragments = [
            {"doc_title": "Doc A", "section_title": "Sec 1", "text": "hello world"},
            {"doc_title": "Doc B", "section_title": "", "text": "second"},
        ]
        msg = _build_user_message("Q", {"user_type": "бакалавр"}, fragments)
        assert 'user_type: бакалавр' in msg
        assert '[1] doc_title="Doc A", section_title="Sec 1"' in msg
        assert "hello world" in msg
        assert '[2] doc_title="Doc B", section_title=""' in msg

    def test_fragment_text_truncated_to_excerpt_len(self):
        long_text = "x" * 1000
        fragments = [{"doc_title": "D", "section_title": "S", "text": long_text}]
        msg = _build_user_message("Q", {}, fragments)
        assert "x" * 500 in msg
        assert "x" * 501 not in msg


# ---------------------------------------------------------------------------
# _extract_json_object()
# ---------------------------------------------------------------------------

class TestExtractJsonObject:
    def test_plain_json(self):
        data = _extract_json_object('{"verdict": "sufficient", "missing": []}')
        assert data == {"verdict": "sufficient", "missing": []}

    def test_json_fenced_with_language_tag(self):
        content = '```json\n{"verdict": "partial", "missing": ["x"]}\n```'
        data = _extract_json_object(content)
        assert data == {"verdict": "partial", "missing": ["x"]}

    def test_json_fenced_without_language_tag(self):
        content = '```\n{"verdict": "insufficient"}\n```'
        data = _extract_json_object(content)
        assert data == {"verdict": "insufficient"}

    def test_no_json_object_raises(self):
        with pytest.raises(ValueError):
            _extract_json_object("no json here at all")

    def test_extra_text_around_json_is_ignored(self):
        content = 'Sure, here is the result:\n{"verdict": "sufficient"}\nThanks.'
        data = _extract_json_object(content)
        assert data == {"verdict": "sufficient"}


# ---------------------------------------------------------------------------
# check_coverage()
# ---------------------------------------------------------------------------

def _patch_client(monkeypatch, fake_async_client):
    monkeypatch.setattr(
        "rag.coverage_gate._make_llm_client", lambda: fake_async_client
    )


class TestCheckCoverage:
    async def test_valid_sufficient_verdict(self, monkeypatch, fake_async_client, fake_llm_response):
        content = '{"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await check_coverage("Q", {}, [])
        assert result == {"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}

    async def test_valid_partial_verdict_with_missing_and_wrong_cohort(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        content = (
            '{"verdict": "partial", "missing": ["deadline"], '
            '"wrong_cohort_indices": [2, 3]}'
        )
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await check_coverage("Q", {}, [{"text": "a"}, {"text": "b"}, {"text": "c"}])
        assert result == {
            "verdict": "partial",
            "missing": ["deadline"],
            "wrong_cohort_indices": [2, 3],
        }

    async def test_valid_insufficient_verdict(self, monkeypatch, fake_async_client, fake_llm_response):
        content = '{"verdict": "insufficient", "missing": ["topic"], "wrong_cohort_indices": []}'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await check_coverage("Q", {}, [])
        assert result["verdict"] == "insufficient"
        assert result["missing"] == ["topic"]

    async def test_response_wrapped_in_json_fence(self, monkeypatch, fake_async_client, fake_llm_response):
        content = '```json\n{"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}\n```'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await check_coverage("Q", {}, [])
        assert result["verdict"] == "sufficient"

    async def test_invalid_verdict_fails_open(self, monkeypatch, fake_async_client, fake_llm_response):
        content = '{"verdict": "maybe", "missing": [], "wrong_cohort_indices": []}'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await check_coverage("Q", {}, [])
        assert result == {"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}

    async def test_missing_json_fails_open(self, monkeypatch, fake_async_client, fake_llm_response):
        fake_async_client.chat.completions.create.return_value = fake_llm_response("not json at all")
        _patch_client(monkeypatch, fake_async_client)

        result = await check_coverage("Q", {}, [])
        assert result == {"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}

    async def test_timeout_fails_open(self, monkeypatch, fake_async_client):
        async def _raise_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        fake_async_client.chat.completions.create.side_effect = _raise_timeout
        _patch_client(monkeypatch, fake_async_client)

        result = await check_coverage("Q", {}, [])
        assert result == {"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}

    async def test_client_exception_fails_open(self, monkeypatch, fake_async_client):
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_client(monkeypatch, fake_async_client)

        result = await check_coverage("Q", {}, [])
        assert result == {"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}

    async def test_make_client_raising_fails_open(self, monkeypatch):
        def _raise():
            raise RuntimeError("no client")

        monkeypatch.setattr("rag.coverage_gate._make_llm_client", _raise)

        result = await check_coverage("Q", {}, [])
        assert result == {"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}

    async def test_non_int_wrong_cohort_indices_are_skipped(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        content = '{"verdict": "partial", "missing": [], "wrong_cohort_indices": ["a", 1, null]}'
        fake_async_client.chat.completions.create.return_value = fake_llm_response(content)
        _patch_client(monkeypatch, fake_async_client)

        result = await check_coverage("Q", {}, [])
        assert result["wrong_cohort_indices"] == [1]
