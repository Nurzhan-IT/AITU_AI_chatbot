import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.dialog import question_gen
from rag.dialog.question_gen import (
    _compute_filled_slots,
    _detect_slots_in_text,
    _get_cached_docs,
    _normalize_options,
    invalidate_docs_cache,
    next_clarification,
)


@pytest.fixture(autouse=True)
def _reset_docs_cache():
    question_gen._docs_cache["items"] = None
    question_gen._docs_cache["expires_at"] = 0.0
    yield
    question_gen._docs_cache["items"] = None
    question_gen._docs_cache["expires_at"] = 0.0


def _patch_client(monkeypatch, fake_async_client):
    monkeypatch.setattr(question_gen, "_make_llm_client", lambda: fake_async_client)


# ---------------------------------------------------------------------------
# _detect_slots_in_text()
# ---------------------------------------------------------------------------

class TestDetectSlotsInText:
    def test_detects_bachelor_level(self):
        assert _detect_slots_in_text("Я поступаю на бакалавриат") == {"level": "бакалавр"}

    def test_detects_master_level_english(self):
        assert _detect_slots_in_text("I want a master program") == {"level": "магистрант"}

    def test_detects_phd(self):
        assert _detect_slots_in_text("докторантура интересует") == {"level": "докторант"}

    def test_no_match_returns_empty(self):
        assert _detect_slots_in_text("что-то совсем другое") == {}

    def test_regular_admission_ru(self):
        assert _detect_slots_in_text("обычный приём документов") == {"admission_type": "обычный"}

    def test_winter_admission_ru(self):
        assert _detect_slots_in_text("зимний приём заявок") == {"admission_type": "зимний"}

    def test_regular_admission_en(self):
        assert _detect_slots_in_text("this is regular admission") == {"admission_type": "обычный"}

    def test_winter_admission_en(self):
        assert _detect_slots_in_text("winter admission period") == {"admission_type": "зимний"}

    def test_level_and_admission_type_both_detected(self):
        result = _detect_slots_in_text("бакалавриат, зимний приём")
        assert result == {"level": "бакалавр", "admission_type": "зимний"}

    def test_empty_text_returns_empty(self):
        assert _detect_slots_in_text("") == {}

    def test_first_matching_keyword_wins(self):
        # "бакалавриат" appears before the generic "бакалавр" in the table but
        # both map to the same value; ensure no crash / consistent value.
        assert _detect_slots_in_text("бакалавр")["level"] == "бакалавр"


# ---------------------------------------------------------------------------
# _compute_filled_slots()
# ---------------------------------------------------------------------------

class TestComputeFilledSlots:
    def test_slot_from_original_query(self):
        result = _compute_filled_slots("хочу поступить на бакалавриат", [])
        assert result == {"level": "бакалавр"}

    def test_slot_from_answers(self):
        answers = [{"question": "уровень?", "answer": "магистратура"}]
        result = _compute_filled_slots("общий вопрос", answers)
        assert result == {"level": "магистрант"}

    def test_original_query_takes_precedence_over_answers(self):
        answers = [{"question": "уровень?", "answer": "докторантура"}]
        result = _compute_filled_slots("бакалавриат интересует", answers)
        assert result["level"] == "бакалавр"

    def test_none_and_blank_answers_skipped(self):
        answers = [{"question": "q", "answer": None}, {"question": "q2", "answer": "   "}]
        result = _compute_filled_slots("общий вопрос без слотов", answers)
        assert result == {}

    def test_answers_none_defaults_to_empty(self):
        assert _compute_filled_slots("общий вопрос", None) == {}


# ---------------------------------------------------------------------------
# _normalize_options()
# ---------------------------------------------------------------------------

class TestNormalizeOptions:
    def test_non_list_returns_empty(self):
        assert _normalize_options("not a list") == []

    def test_filters_blank_labels(self):
        assert _normalize_options(["a", "  ", "b"]) == ["a", "b"]

    def test_caps_at_four_options(self):
        assert _normalize_options(["a", "b", "c", "d", "e"]) == ["a", "b", "c", "d"]

    def test_truncates_long_label_with_ellipsis(self):
        long_label = "x" * 60
        result = _normalize_options([long_label])
        assert len(result[0]) == 50
        assert result[0].endswith("…")

    def test_short_label_unchanged(self):
        assert _normalize_options(["short"]) == ["short"]


# ---------------------------------------------------------------------------
# invalidate_docs_cache() / _get_cached_docs()
# ---------------------------------------------------------------------------

def _fake_retriever(records, next_offset=None):
    client = MagicMock()
    client.scroll = AsyncMock(return_value=(records, next_offset))
    return SimpleNamespace(_client=client)


def _rec(doc_title, section_title=""):
    return SimpleNamespace(payload={"doc_title": doc_title, "section_title": section_title})


class TestGetCachedDocs:
    async def test_fetches_and_dedupes(self):
        retriever = _fake_retriever([_rec("Doc A", "S1"), _rec("Doc A", "S1"), _rec("Doc B")])

        items = await _get_cached_docs(retriever)

        assert items == [
            {"doc_title": "Doc A", "section_title": "S1"},
            {"doc_title": "Doc B", "section_title": ""},
        ]
        retriever._client.scroll.assert_awaited_once()

    async def test_second_call_uses_cache_without_refetching(self):
        retriever = _fake_retriever([_rec("Doc A")])

        await _get_cached_docs(retriever)
        await _get_cached_docs(retriever)

        retriever._client.scroll.assert_awaited_once()

    async def test_invalidate_forces_refetch(self):
        retriever = _fake_retriever([_rec("Doc A")])

        await _get_cached_docs(retriever)
        invalidate_docs_cache()
        await _get_cached_docs(retriever)

        assert retriever._client.scroll.await_count == 2

    async def test_missing_doc_title_skipped(self):
        retriever = _fake_retriever([_rec(""), _rec("Doc A")])

        items = await _get_cached_docs(retriever)

        assert items == [{"doc_title": "Doc A", "section_title": ""}]

    async def test_scroll_exception_returns_empty_and_is_cached(self):
        client = MagicMock()
        client.scroll = AsyncMock(side_effect=RuntimeError("boom"))
        retriever = SimpleNamespace(_client=client)

        items = await _get_cached_docs(retriever)
        assert items == []

        # Even the empty result is cached for the TTL window.
        items_again = await _get_cached_docs(retriever)
        assert items_again == []
        client.scroll.assert_awaited_once()


# ---------------------------------------------------------------------------
# next_clarification()
# ---------------------------------------------------------------------------

class TestNextClarification:
    async def test_all_required_slots_filled_short_circuits_without_llm(
        self, monkeypatch, fake_async_client
    ):
        _patch_client(monkeypatch, fake_async_client)
        state_data = {
            "original_query": "поступление на бакалавриат",
            "answers": [],
            "required_slots": ["level"],
        }

        result = await next_clarification(state_data, [])

        assert result["stop"] is True
        assert result["slot"] == ""
        fake_async_client.chat.completions.create.assert_not_called()

    async def test_no_required_slots_short_circuits(self, monkeypatch, fake_async_client):
        _patch_client(monkeypatch, fake_async_client)
        state_data = {"original_query": "что угодно", "answers": [], "required_slots": []}

        result = await next_clarification(state_data, [])

        assert result["stop"] is True
        fake_async_client.chat.completions.create.assert_not_called()

    async def test_happy_path_returns_question_and_options(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            json.dumps({
                "slot": "level",
                "question": "На каком уровне обучения?",
                "options": ["Бакалавриат", "Магистратура"],
                "stop": False,
            })
        )
        _patch_client(monkeypatch, fake_async_client)
        state_data = {
            "original_query": "хочу поступить",
            "answers": [],
            "required_slots": ["level"],
        }

        result = await next_clarification(state_data, [])

        assert result["slot"] == "level"
        assert result["question"] == "На каком уровне обучения?"
        assert result["options"] == ["Бакалавриат", "Магистратура"]
        assert result["stop"] is False

    async def test_invalid_slot_is_cleared(self, monkeypatch, fake_async_client, fake_llm_response):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            json.dumps({"slot": "bogus_slot", "question": "Q?", "options": [], "stop": False})
        )
        _patch_client(monkeypatch, fake_async_client)
        state_data = {
            "original_query": "хочу поступить",
            "answers": [],
            "required_slots": ["level"],
        }

        result = await next_clarification(state_data, [])

        assert result["slot"] == ""

    async def test_llm_exception_returns_stop_true(self, monkeypatch, fake_async_client):
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_client(monkeypatch, fake_async_client)
        state_data = {
            "original_query": "хочу поступить",
            "answers": [],
            "required_slots": ["level"],
        }

        result = await next_clarification(state_data, [])

        assert result == {"slot": "", "question": "", "options": [], "stop": True}

    async def test_no_json_in_response_returns_stop_true(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response("garbage")
        _patch_client(monkeypatch, fake_async_client)
        state_data = {
            "original_query": "хочу поступить",
            "answers": [],
            "required_slots": ["level"],
        }

        result = await next_clarification(state_data, [])

        assert result["stop"] is True
