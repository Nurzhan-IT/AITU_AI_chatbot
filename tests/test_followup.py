import json

import pytest

from rag.dialog import followup
from rag.dialog.enricher import _empty_profile
from rag.dialog.followup import _apply_patch, _get_fresh, check_followup, save_context


@pytest.fixture(autouse=True)
def _clear_store():
    followup._store.clear()
    yield
    followup._store.clear()


def _patch_client(monkeypatch, fake_async_client):
    monkeypatch.setattr("rag.generator._make_llm_client", lambda: fake_async_client)


# ---------------------------------------------------------------------------
# save_context() / _get_fresh()
# ---------------------------------------------------------------------------

class TestSaveAndGetFresh:
    def test_save_then_fresh_round_trip(self):
        profile = {"topics": ["a"], "user_type": "бакалавр", "admission_type": None,
                   "document_hints": [], "temporal_context": None}
        save_context(1, "original", "enriched", profile)

        entry = _get_fresh(1)

        assert entry is not None
        assert entry.original_query == "original"
        assert entry.enriched_query == "enriched"
        assert entry.profile == profile

    def test_missing_user_returns_none(self):
        assert _get_fresh(999) is None

    def test_stored_profile_is_a_copy(self):
        profile = {"topics": ["a"], "user_type": None, "admission_type": None,
                   "document_hints": [], "temporal_context": None}
        save_context(1, "o", "e", profile)
        profile["topics"].append("mutated")  # mutates the shared list object

        entry = _get_fresh(1)
        # dict(profile) is a shallow copy, so the top-level dict differs but
        # nested lists are shared -- document the actual (shallow) contract.
        assert entry.profile is not profile

    def test_expired_entry_returns_none_and_is_evicted(self, monkeypatch):
        save_context(1, "o", "e", _empty_profile())
        followup._store[1].saved_at = 1000.0
        monkeypatch.setattr(followup, "time", lambda: 1000.0 + followup._FOLLOWUP_TTL + 1)

        assert _get_fresh(1) is None
        assert 1 not in followup._store

    def test_entry_within_ttl_is_returned(self, monkeypatch):
        save_context(1, "o", "e", _empty_profile())
        followup._store[1].saved_at = 1000.0
        monkeypatch.setattr(followup, "time", lambda: 1000.0 + followup._FOLLOWUP_TTL - 1)

        assert _get_fresh(1) is not None


# ---------------------------------------------------------------------------
# _apply_patch()
# ---------------------------------------------------------------------------

def _base_profile(**overrides):
    base = {
        "topics": ["поступление"],
        "user_type": "бакалавр",
        "admission_type": "обычный",
        "document_hints": ["диплом"],
        "temporal_context": "2024",
    }
    base.update(overrides)
    return base


class TestApplyPatch:
    def test_empty_patch_keeps_base_values(self):
        base = _base_profile()
        result = _apply_patch(base, {})
        assert result == base

    def test_valid_user_type_overrides(self):
        base = _base_profile()
        result = _apply_patch(base, {"user_type": "Магистрант"})
        assert result["user_type"] == "магистрант"

    def test_invalid_user_type_falls_back_to_base(self):
        base = _base_profile()
        result = _apply_patch(base, {"user_type": "инопланетянин"})
        assert result["user_type"] == "бакалавр"

    def test_valid_admission_type_overrides(self):
        base = _base_profile()
        result = _apply_patch(base, {"admission_type": "Зимний"})
        assert result["admission_type"] == "зимний"

    def test_invalid_admission_type_falls_back_to_base(self):
        base = _base_profile()
        result = _apply_patch(base, {"admission_type": "летний"})
        assert result["admission_type"] == "обычный"

    def test_topics_are_appended_and_deduped(self):
        base = _base_profile(topics=["a", "b"])
        result = _apply_patch(base, {"topics": ["b", "c"]})
        assert result["topics"] == ["a", "b", "c"]

    def test_topics_capped_at_ten(self):
        base = _base_profile(topics=[f"t{i}" for i in range(9)])
        result = _apply_patch(base, {"topics": ["new1", "new2"]})
        assert len(result["topics"]) == 10

    def test_topics_ignored_when_not_a_list(self):
        base = _base_profile(topics=["a"])
        result = _apply_patch(base, {"topics": "not-a-list"})
        assert result["topics"] == ["a"]

    def test_document_hints_replaced_when_list(self):
        base = _base_profile(document_hints=["old.pdf"])
        result = _apply_patch(base, {"document_hints": ["new.pdf"]})
        assert result["document_hints"] == ["new.pdf"]

    def test_document_hints_ignored_when_not_a_list(self):
        base = _base_profile(document_hints=["old.pdf"])
        result = _apply_patch(base, {"document_hints": "not-a-list"})
        assert result["document_hints"] == ["old.pdf"]

    def test_temporal_context_overridden_when_present(self):
        base = _base_profile(temporal_context="2023")
        result = _apply_patch(base, {"temporal_context": "2024 winter intake"})
        assert result["temporal_context"] == "2024 winter intake"

    def test_temporal_context_blank_falls_back_to_base(self):
        base = _base_profile(temporal_context="2023")
        result = _apply_patch(base, {"temporal_context": "   "})
        assert result["temporal_context"] == "2023"


# ---------------------------------------------------------------------------
# check_followup()
# ---------------------------------------------------------------------------

class TestCheckFollowup:
    async def test_no_saved_context_returns_false_without_llm_call(
        self, monkeypatch, fake_async_client
    ):
        _patch_client(monkeypatch, fake_async_client)

        is_followup, query, profile = await check_followup(1, "new message")

        assert is_followup is False
        assert query == "new message"
        assert profile == _empty_profile()
        fake_async_client.chat.completions.create.assert_not_called()

    async def test_followup_detected_merges_query_and_profile(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        save_context(1, "поступление на бакалавриат", "поступление бакалавриат 2024",
                     _base_profile())
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            json.dumps({
                "is_followup": True,
                "merged_query": "поступление бакалавриат 2024 документы",
                "profile_patch": {"topics": ["документы"]},
            })
        )
        _patch_client(monkeypatch, fake_async_client)

        is_followup, query, profile = await check_followup(1, "а какие документы нужны?")

        assert is_followup is True
        assert query == "поступление бакалавриат 2024 документы"
        assert "документы" in profile["topics"]

    async def test_not_followup_returns_new_message_and_empty_profile(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        save_context(1, "original", "enriched", _base_profile())
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            json.dumps({"is_followup": False})
        )
        _patch_client(monkeypatch, fake_async_client)

        is_followup, query, profile = await check_followup(1, "unrelated new question")

        assert is_followup is False
        assert query == "unrelated new question"
        assert profile == _empty_profile()

    async def test_missing_merged_query_falls_back_to_stored_enriched(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        save_context(1, "original", "stored enriched query", _base_profile())
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            json.dumps({"is_followup": True})
        )
        _patch_client(monkeypatch, fake_async_client)

        is_followup, query, _ = await check_followup(1, "follow up msg")

        assert is_followup is True
        assert query == "stored enriched query"

    async def test_llm_exception_falls_back(self, monkeypatch, fake_async_client):
        save_context(1, "original", "enriched", _base_profile())
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_client(monkeypatch, fake_async_client)

        is_followup, query, profile = await check_followup(1, "new message")

        assert is_followup is False
        assert query == "new message"
        assert profile == _empty_profile()

    async def test_no_json_in_response_falls_back(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        save_context(1, "original", "enriched", _base_profile())
        fake_async_client.chat.completions.create.return_value = fake_llm_response("garbage")
        _patch_client(monkeypatch, fake_async_client)

        is_followup, query, profile = await check_followup(1, "new message")

        assert is_followup is False
        assert query == "new message"
        assert profile == _empty_profile()
