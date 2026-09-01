import json

import pytest

from rag.dialog import enricher
from rag.dialog.enricher import (
    _coerce_optional_str,
    _coerce_str_list,
    _empty_profile,
    _strip_wrapping_quotes,
    enrich_and_profile,
    enrich_query,
)


def _patch_client(monkeypatch, fake_async_client):
    monkeypatch.setattr(enricher, "_make_llm_client", lambda: fake_async_client)


# ---------------------------------------------------------------------------
# _empty_profile()
# ---------------------------------------------------------------------------

class TestEmptyProfile:
    def test_shape(self):
        assert _empty_profile() == {
            "topics": [],
            "user_type": None,
            "admission_type": None,
            "document_hints": [],
            "temporal_context": None,
        }


# ---------------------------------------------------------------------------
# _coerce_str_list()
# ---------------------------------------------------------------------------

class TestCoerceStrList:
    def test_non_list_returns_empty(self):
        assert _coerce_str_list("x", 5) == []
        assert _coerce_str_list(None, 5) == []

    def test_filters_none_and_blank(self):
        assert _coerce_str_list(["a", None, "  ", "b"], 5) == ["a", "b"]

    def test_stops_at_limit(self):
        assert _coerce_str_list(["a", "b", "c"], 2) == ["a", "b"]

    def test_non_string_items_stringified(self):
        assert _coerce_str_list([1, 2.5], 5) == ["1", "2.5"]


# ---------------------------------------------------------------------------
# _coerce_optional_str()
# ---------------------------------------------------------------------------

class TestCoerceOptionalStr:
    def test_none_returns_none(self):
        assert _coerce_optional_str(None) is None

    def test_blank_returns_none(self):
        assert _coerce_optional_str("   ") is None

    def test_strips_whitespace(self):
        assert _coerce_optional_str("  hello  ") == "hello"

    def test_non_string_stringified(self):
        assert _coerce_optional_str(42) == "42"


# ---------------------------------------------------------------------------
# _strip_wrapping_quotes()
# ---------------------------------------------------------------------------

class TestStripWrappingQuotes:
    def test_double_quotes_stripped(self):
        assert _strip_wrapping_quotes('"hello"') == "hello"

    def test_single_quotes_stripped(self):
        assert _strip_wrapping_quotes("'hello'") == "hello"

    def test_mismatched_guillemets_not_stripped(self):
        # first char '«' != last char '»' so this does not match the check
        assert _strip_wrapping_quotes("«hello»") == "«hello»"

    def test_unwrapped_text_unchanged(self):
        assert _strip_wrapping_quotes("hello") == "hello"

    def test_quote_in_middle_not_stripped(self):
        assert _strip_wrapping_quotes('he said "hi" ok') == 'he said "hi" ok'

    def test_single_char_text_unchanged(self):
        assert _strip_wrapping_quotes('"') == '"'

    def test_only_outer_quotes_stripped_once(self):
        assert _strip_wrapping_quotes("\"quoted 'inner' text\"") == "quoted 'inner' text"


# ---------------------------------------------------------------------------
# enrich_query()
# ---------------------------------------------------------------------------

class TestEnrichQuery:
    async def test_no_answers_returns_original_without_llm_call(
        self, monkeypatch, fake_async_client
    ):
        _patch_client(monkeypatch, fake_async_client)

        result = await enrich_query("original query", [])

        assert result == "original query"
        fake_async_client.chat.completions.create.assert_not_called()

    async def test_all_blank_answers_returns_original(self, monkeypatch, fake_async_client):
        _patch_client(monkeypatch, fake_async_client)
        answers = [{"question": "Q1", "answer": None}, {"question": "Q2", "answer": "  "}]

        result = await enrich_query("original query", answers)

        assert result == "original query"
        fake_async_client.chat.completions.create.assert_not_called()

    async def test_happy_path_strips_wrapping_quotes(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '"обучение на бакалавриате инженерные специальности"'
        )
        _patch_client(monkeypatch, fake_async_client)
        answers = [{"question": "уровень?", "answer": "бакалавриат"}]

        result = await enrich_query("расскажи про обучение", answers)

        assert result == "обучение на бакалавриате инженерные специальности"

    async def test_empty_llm_response_falls_back_to_original(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response("   ")
        _patch_client(monkeypatch, fake_async_client)
        answers = [{"question": "q", "answer": "a"}]

        result = await enrich_query("original query", answers)

        assert result == "original query"

    async def test_llm_exception_falls_back_to_original(self, monkeypatch, fake_async_client):
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_client(monkeypatch, fake_async_client)
        answers = [{"question": "q", "answer": "a"}]

        result = await enrich_query("original query", answers)

        assert result == "original query"


# ---------------------------------------------------------------------------
# enrich_and_profile()
# ---------------------------------------------------------------------------

class TestEnrichAndProfile:
    async def test_no_filtered_answers_returns_original_and_empty_profile(
        self, monkeypatch, fake_async_client
    ):
        _patch_client(monkeypatch, fake_async_client)

        enriched, profile = await enrich_and_profile("original", [{"question": "q", "answer": ""}])

        assert enriched == "original"
        assert profile == _empty_profile()
        fake_async_client.chat.completions.create.assert_not_called()

    async def test_happy_path_parses_enriched_and_profile(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        payload = {
            "enriched": "обучение на бакалавриате",
            "profile": {
                "topics": ["поступление", "бакалавриат"],
                "user_type": "Бакалавр",
                "admission_type": "Обычный",
                "document_hints": ["диплом"],
                "temporal_context": "2024",
            },
        }
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            json.dumps(payload, ensure_ascii=False)
        )
        _patch_client(monkeypatch, fake_async_client)
        answers = [{"question": "уровень?", "answer": "бакалавриат"}]

        enriched, profile = await enrich_and_profile("расскажи", answers)

        assert enriched == "обучение на бакалавриате"
        assert profile["topics"] == ["поступление", "бакалавриат"]
        assert profile["user_type"] == "бакалавр"
        assert profile["admission_type"] == "обычный"
        assert profile["document_hints"] == ["диплом"]
        assert profile["temporal_context"] == "2024"

    async def test_invalid_user_type_and_admission_type_become_none(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        payload = {
            "enriched": "query",
            "profile": {"user_type": "инопланетянин", "admission_type": "летний"},
        }
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            json.dumps(payload, ensure_ascii=False)
        )
        _patch_client(monkeypatch, fake_async_client)
        answers = [{"question": "q", "answer": "a"}]

        _, profile = await enrich_and_profile("original", answers)

        assert profile["user_type"] is None
        assert profile["admission_type"] is None

    async def test_missing_enriched_falls_back_to_original(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            json.dumps({"profile": {}})
        )
        _patch_client(monkeypatch, fake_async_client)
        answers = [{"question": "q", "answer": "a"}]

        enriched, _ = await enrich_and_profile("original query", answers)

        assert enriched == "original query"

    async def test_no_json_in_response_falls_back(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response("garbage")
        _patch_client(monkeypatch, fake_async_client)
        answers = [{"question": "q", "answer": "a"}]

        enriched, profile = await enrich_and_profile("original query", answers)

        assert enriched == "original query"
        assert profile == _empty_profile()

    async def test_llm_exception_falls_back(self, monkeypatch, fake_async_client):
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_client(monkeypatch, fake_async_client)
        answers = [{"question": "q", "answer": "a"}]

        enriched, profile = await enrich_and_profile("original query", answers)

        assert enriched == "original query"
        assert profile == _empty_profile()
