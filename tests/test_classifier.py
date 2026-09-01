from dataclasses import dataclass
from typing import Optional

import pytest

from rag.dialog import classifier
from rag.dialog.classifier import (
    _cache_get,
    _cache_key,
    _cache_put,
    _coerce_required_slots,
    _describe_choice,
    classify_intent,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    classifier._cache.clear()
    yield
    classifier._cache.clear()


def _patch_client(monkeypatch, fake_async_client):
    monkeypatch.setattr(classifier, "_make_llm_client", lambda: fake_async_client)


# ---------------------------------------------------------------------------
# _cache_key()
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic(self):
        assert _cache_key("Hello world") == _cache_key("Hello world")

    def test_case_and_whitespace_insensitive(self):
        assert _cache_key("Hello   World") == _cache_key("hello world")

    def test_different_questions_differ(self):
        assert _cache_key("hello") != _cache_key("goodbye")


# ---------------------------------------------------------------------------
# _cache_get() / _cache_put()
# ---------------------------------------------------------------------------

class TestCacheGetPut:
    def test_round_trip(self):
        result = classifier.ClassificationResult(
            needs_clarification=False, reason="specific", required_slots=[], confidence=0.9
        )
        _cache_put("k1", result)
        assert _cache_get("k1") == result

    def test_missing_key_returns_none(self):
        assert _cache_get("missing") is None

    def test_expired_entry_returns_none_and_is_evicted(self, monkeypatch):
        result = classifier.ClassificationResult(
            needs_clarification=False, reason="specific", required_slots=[], confidence=0.9
        )
        fake_now = [1000.0]
        monkeypatch.setattr(classifier.time, "time", lambda: fake_now[0])
        _cache_put("k1", result)

        fake_now[0] = 1000.0 + classifier._CACHE_TTL + 1
        assert _cache_get("k1") is None
        assert "k1" not in classifier._cache

    def test_evicts_oldest_when_over_capacity(self, monkeypatch):
        monkeypatch.setattr(classifier, "_CACHE_MAX", 2)
        result = classifier.ClassificationResult(
            needs_clarification=False, reason="specific", required_slots=[], confidence=0.9
        )
        _cache_put("k1", result)
        _cache_put("k2", result)
        _cache_put("k3", result)
        assert len(classifier._cache) == 2
        assert "k1" not in classifier._cache


# ---------------------------------------------------------------------------
# _coerce_required_slots()
# ---------------------------------------------------------------------------

class TestCoerceRequiredSlots:
    def test_non_list_returns_empty(self):
        assert _coerce_required_slots("level") == []
        assert _coerce_required_slots(None) == []

    def test_filters_invalid_slots(self):
        assert _coerce_required_slots(["level", "bogus", "topic"]) == ["level", "topic"]

    def test_filters_non_string_items(self):
        assert _coerce_required_slots(["level", 5, None]) == ["level"]

    def test_dedupes_preserving_order(self):
        assert _coerce_required_slots(["topic", "level", "topic"]) == ["topic", "level"]

    def test_strips_whitespace(self):
        assert _coerce_required_slots([" level "]) == ["level"]


# ---------------------------------------------------------------------------
# _describe_choice()
# ---------------------------------------------------------------------------

@dataclass
class _Msg:
    content: str
    extra: Optional[dict] = None

    def model_dump(self):
        d = {"content": self.content}
        if self.extra:
            d.update(self.extra)
        return d


@dataclass
class _Choice:
    finish_reason: Optional[str] = None
    message: Optional[_Msg] = None


class TestDescribeChoice:
    def test_no_message(self):
        choice = _Choice(finish_reason="stop", message=None)
        assert _describe_choice(choice) == "finish_reason='stop'"

    def test_includes_finish_reason_and_extra_fields(self):
        choice = _Choice(finish_reason="length", message=_Msg(content="", extra={"role": "assistant"}))
        desc = _describe_choice(choice)
        assert "finish_reason='length'" in desc
        assert "role" in desc

    def test_no_diagnostic_fields_when_message_empty_and_no_extra(self):
        choice = _Choice(finish_reason=None, message=_Msg(content=""))
        assert _describe_choice(choice) == "no diagnostic fields"

    def test_no_message_and_no_finish_reason(self):
        choice = _Choice(finish_reason=None, message=None)
        assert _describe_choice(choice) == "no message in choice"


# ---------------------------------------------------------------------------
# classify_intent()
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    async def test_happy_path_needs_clarification(self, monkeypatch, fake_async_client, fake_llm_response):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '{"needs_clarification": true, "reason": "vague_topic", '
            '"required_slots": ["level", "topic"], "confidence": 0.8}'
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await classify_intent("расскажи про поступление")

        assert result["needs_clarification"] is True
        assert result["reason"] == "vague_topic"
        assert result["required_slots"] == ["level", "topic"]
        assert result["confidence"] == pytest.approx(0.8)

    async def test_happy_path_specific(self, monkeypatch, fake_async_client, fake_llm_response):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '{"needs_clarification": false, "reason": "specific", '
            '"required_slots": [], "confidence": 0.95}'
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await classify_intent("какой срок подачи документов на бакалавриат 2024")

        assert result["needs_clarification"] is False
        assert result["reason"] == "specific"
        assert result["required_slots"] == []

    async def test_required_slots_cleared_when_not_needed(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '{"needs_clarification": false, "reason": "specific", '
            '"required_slots": ["level"], "confidence": 0.9}'
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await classify_intent("сколько стоит обучение на программе X")

        assert result["required_slots"] == []

    async def test_invalid_reason_defaults_based_on_needs(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '{"needs_clarification": true, "reason": "not_a_real_reason", '
            '"required_slots": [], "confidence": 0.6}'
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await classify_intent("бла бла бла вопрос один")

        assert result["reason"] == "vague_topic"

    async def test_invalid_confidence_defaults_to_half(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '{"needs_clarification": false, "reason": "specific", '
            '"required_slots": [], "confidence": "not-a-number"}'
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await classify_intent("вопрос про что-то конкретное здесь")

        assert result["confidence"] == pytest.approx(0.5)

    async def test_out_of_range_confidence_defaults_to_half(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '{"needs_clarification": false, "reason": "specific", '
            '"required_slots": [], "confidence": 5.0}'
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await classify_intent("еще один отдельный вопрос текста")

        assert result["confidence"] == pytest.approx(0.5)

    async def test_missing_needs_clarification_raises_and_falls_back(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '{"reason": "specific", "required_slots": [], "confidence": 0.5}'
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await classify_intent("нормальный многословный вопрос про учебу")

        # Falls back to Stage 1 triage heuristics: multi-token, not all stopwords.
        assert result["needs_clarification"] is False
        assert result["reason"] == "specific"

    async def test_no_json_in_response_falls_back(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response("not json at all")
        _patch_client(monkeypatch, fake_async_client)

        result = await classify_intent("что")

        assert result["needs_clarification"] is True
        assert result["reason"] == "ambiguous"
        assert result["confidence"] == pytest.approx(0.5)

    async def test_llm_exception_falls_back_to_stage1_short_query(
        self, monkeypatch, fake_async_client
    ):
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_client(monkeypatch, fake_async_client)

        result = await classify_intent("что")

        assert result == classifier.ClassificationResult(
            needs_clarification=True, reason="ambiguous", required_slots=[], confidence=0.5
        )

    async def test_llm_exception_falls_back_to_stage1_normal_query(
        self, monkeypatch, fake_async_client
    ):
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_client(monkeypatch, fake_async_client)

        result = await classify_intent("как поступить в магистратуру на инженерную специальность")

        assert result == classifier.ClassificationResult(
            needs_clarification=False, reason="specific", required_slots=[], confidence=0.5
        )

    async def test_cache_hit_avoids_second_llm_call(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '{"needs_clarification": false, "reason": "specific", '
            '"required_slots": [], "confidence": 0.9}'
        )
        _patch_client(monkeypatch, fake_async_client)

        first = await classify_intent("уникальный вопрос про кэширование ответа")
        second = await classify_intent("уникальный вопрос про кэширование ответа")

        assert first == second
        fake_async_client.chat.completions.create.assert_called_once()
