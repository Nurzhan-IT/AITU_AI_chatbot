import json

import pytest

from rag.dialog import reranker
from rag.dialog.reranker import (
    _build_user_content,
    _coerce_str_list,
    _parse_order,
    _profile_payload,
    rerank_chunks,
)


def _patch_client(monkeypatch, fake_async_client):
    monkeypatch.setattr(reranker, "_make_llm_client", lambda: fake_async_client)


# ---------------------------------------------------------------------------
# _profile_payload()
# ---------------------------------------------------------------------------

class TestProfilePayload:
    def test_none_profile_returns_defaults(self):
        assert _profile_payload(None) == {
            "user_type": None,
            "admission_type": None,
            "document_hints": [],
            "temporal_context": None,
        }

    def test_empty_dict_returns_defaults(self):
        assert _profile_payload({}) == {
            "user_type": None,
            "admission_type": None,
            "document_hints": [],
            "temporal_context": None,
        }

    def test_populated_profile_is_passed_through(self):
        profile = {
            "user_type": "бакалавр",
            "admission_type": "обычный",
            "document_hints": ["диплом"],
            "temporal_context": "2024",
        }
        assert _profile_payload(profile) == profile

    def test_missing_document_hints_defaults_to_empty_list(self):
        result = _profile_payload({"user_type": "магистрант"})
        assert result["document_hints"] == []


# ---------------------------------------------------------------------------
# _coerce_str_list()
# ---------------------------------------------------------------------------

class TestCoerceStrList:
    def test_list_of_strings(self):
        assert _coerce_str_list(["a", "b"]) == ["a", "b"]

    def test_filters_none_and_blank(self):
        assert _coerce_str_list(["a", None, "  ", "b"]) == ["a", "b"]

    def test_none_returns_empty(self):
        assert _coerce_str_list(None) == []

    def test_scalar_wrapped_in_list(self):
        assert _coerce_str_list("solo") == ["solo"]

    def test_blank_scalar_returns_empty(self):
        assert _coerce_str_list("   ") == []

    def test_non_string_items_stringified(self):
        assert _coerce_str_list([1, 2]) == ["1", "2"]


# ---------------------------------------------------------------------------
# _build_user_content()
# ---------------------------------------------------------------------------

class TestBuildUserContent:
    def test_includes_question_and_profile(self):
        content = _build_user_content("Какой срок?", [], {"user_type": "бакалавр"})
        assert "QUESTION: Какой срок?" in content
        assert "PROFILE:" in content
        assert '"user_type": "бакалавр"' in content

    def test_chunk_header_includes_index_title_and_section(self):
        chunks = [{"doc_title": "Правила", "section_title": "Приём", "text": "текст"}]
        content = _build_user_content("Q", chunks, None)
        assert "[0] Правила — Приём" in content

    def test_untitled_chunk_uses_placeholder(self):
        chunks = [{"text": "текст"}]
        content = _build_user_content("Q", chunks, None)
        assert "[0] (untitled)" in content

    def test_excerpt_truncated_to_300_chars(self):
        chunks = [{"doc_title": "Doc", "text": "a" * 500}]
        content = _build_user_content("Q", chunks, None)
        assert "a" * 300 in content
        assert "a" * 301 not in content

    def test_applies_to_and_admission_type_serialized(self):
        chunks = [{"doc_title": "Doc", "text": "t", "applies_to": ["бакалавр"], "admission_type": "зимний"}]
        content = _build_user_content("Q", chunks, None)
        assert 'applies_to=["бакалавр"]' in content
        assert 'admission_type=["зимний"]' in content


# ---------------------------------------------------------------------------
# _parse_order()
# ---------------------------------------------------------------------------

class TestParseOrder:
    def test_valid_permutation(self):
        assert _parse_order([2, 0, 1], 3) == [2, 0, 1]

    def test_not_a_list_raises(self):
        with pytest.raises(ValueError):
            _parse_order("0,1,2", 3)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            _parse_order([0, 1], 3)

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError):
            _parse_order([0, 1, 5], 3)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError):
            _parse_order([-1, 1, 2], 3)

    def test_duplicate_index_raises(self):
        with pytest.raises(ValueError):
            _parse_order([0, 0, 1], 3)


# ---------------------------------------------------------------------------
# rerank_chunks()
# ---------------------------------------------------------------------------

def _chunks(n):
    return [{"doc_title": f"Doc {i}", "text": f"text {i}"} for i in range(n)]


class TestRerankChunks:
    async def test_fewer_chunks_than_k_returns_unchanged_without_llm_call(
        self, monkeypatch, fake_async_client
    ):
        _patch_client(monkeypatch, fake_async_client)
        chunks = _chunks(3)

        result = await rerank_chunks("Q", chunks, k=5)

        assert result == chunks
        fake_async_client.chat.completions.create.assert_not_called()

    async def test_valid_order_reorders_and_attaches_reasons(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        chunks = _chunks(4)
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            json.dumps({"order": [3, 1, 0, 2], "reasons": ["best", "second"]})
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await rerank_chunks("Q", chunks, k=2)

        assert [c["doc_title"] for c in result] == ["Doc 3", "Doc 1"]
        assert result[0]["rerank_reason"] == "best"
        assert result[1]["rerank_reason"] == "second"

    async def test_invalid_json_falls_back_to_original_top_k(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        chunks = _chunks(4)
        fake_async_client.chat.completions.create.return_value = fake_llm_response("not json")
        _patch_client(monkeypatch, fake_async_client)

        result = await rerank_chunks("Q", chunks, k=2)

        assert result == chunks[:2]

    async def test_out_of_range_order_falls_back_to_original_top_k(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        chunks = _chunks(4)
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            json.dumps({"order": [0, 1, 2, 99], "reasons": []})
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await rerank_chunks("Q", chunks, k=2)

        assert result == chunks[:2]

    async def test_llm_exception_falls_back_to_original_top_k(
        self, monkeypatch, fake_async_client
    ):
        chunks = _chunks(4)
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_client(monkeypatch, fake_async_client)

        result = await rerank_chunks("Q", chunks, k=2)

        assert result == chunks[:2]
