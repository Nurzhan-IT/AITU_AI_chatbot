from unittest.mock import AsyncMock, MagicMock

import pytest

from config import settings
from rag.generator import (
    _PARTIAL_COHORT_NOTE,
    Generator,
    _build_context,
    _deduplicate_sources,
    _make_llm_client,
    _strip_citation_artifacts,
    detect_language,
    supports_json_object,
    supports_json_schema,
    verify_chunk_answers_query,
)

# ---------------------------------------------------------------------------
# detect_language()
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_russian(self, monkeypatch):
        monkeypatch.setattr("rag.generator.detect", lambda text: "ru")
        assert detect_language("Привет") == "Russian"

    def test_english(self, monkeypatch):
        monkeypatch.setattr("rag.generator.detect", lambda text: "en")
        assert detect_language("Hello") == "English"

    def test_kazakh(self, monkeypatch):
        monkeypatch.setattr("rag.generator.detect", lambda text: "kk")
        assert detect_language("Сәлем") == "Kazakh"

    def test_unknown_code_defaults_to_russian(self, monkeypatch):
        monkeypatch.setattr("rag.generator.detect", lambda text: "fr")
        assert detect_language("Bonjour") == "Russian"

    def test_exception_defaults_to_russian(self, monkeypatch):
        def _raise(text):
            raise ValueError("no features in text")

        monkeypatch.setattr("rag.generator.detect", _raise)
        assert detect_language("") == "Russian"


# ---------------------------------------------------------------------------
# supports_json_schema() / supports_json_object()
# ---------------------------------------------------------------------------

class TestSupportsJsonSchema:
    def test_groq_allowlisted_model_true(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "llm_model", "moonshotai/kimi-k2-instruct-0905")
        assert supports_json_schema() is True

    def test_groq_non_allowlisted_model_false(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "llm_model", "llama-3.3-70b-versatile")
        assert supports_json_schema() is False

    def test_openrouter_normal_model_true(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "openrouter")
        monkeypatch.setattr(settings, "llm_model", "anthropic/claude-3.5-sonnet")
        assert supports_json_schema() is True

    def test_openrouter_reasoning_model_false(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "openrouter")
        monkeypatch.setattr(settings, "llm_model", "openai/o1-mini")
        assert supports_json_schema() is False

    def test_openrouter_gpt_oss_prefix_false(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "openrouter")
        monkeypatch.setattr(settings, "llm_model", "openai/gpt-oss-120b")
        assert supports_json_schema() is False

    def test_other_provider_false(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "something_else")
        assert supports_json_schema() is False


class TestSupportsJsonObject:
    def test_groq_true(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        assert supports_json_object() is True

    def test_openrouter_true(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "openrouter")
        assert supports_json_object() is True

    def test_other_provider_false(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "something_else")
        assert supports_json_object() is False


# ---------------------------------------------------------------------------
# _make_llm_client()
# ---------------------------------------------------------------------------

class TestMakeLlmClient:
    def test_groq_provider_uses_async_groq_with_api_key(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "gk-test-key")

        fake_groq_cls = MagicMock()
        monkeypatch.setattr("groq.AsyncGroq", fake_groq_cls)

        _make_llm_client()

        fake_groq_cls.assert_called_once_with(api_key="gk-test-key")

    def test_non_groq_provider_uses_async_openai_with_openrouter_base_url(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "openrouter")
        monkeypatch.setattr(settings, "openrouter_api_key", "ork-test-key")
        monkeypatch.setattr(settings, "openrouter_base_url", "https://example.com/v1")

        fake_openai_cls = MagicMock()
        monkeypatch.setattr("openai.AsyncOpenAI", fake_openai_cls)

        _make_llm_client()

        fake_openai_cls.assert_called_once_with(
            api_key="ork-test-key", base_url="https://example.com/v1"
        )


# ---------------------------------------------------------------------------
# _build_context()
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_includes_header_with_page_range_section_and_paragraph(self):
        chunks = [
            {
                "doc_title": "Doc A",
                "page": 3,
                "page_end": 5,
                "section_title": "Раздел 1",
                "paragraph_range": "п.1-2",
                "text": "some text",
            }
        ]
        context = _build_context(chunks)
        assert "[1] Doc A" in context
        assert "Раздел: «Раздел 1»" in context
        assert "п.1-2" in context
        assert "стр. 3–5" in context
        assert "some text" in context

    def test_single_page_header_when_page_equals_page_end(self):
        chunks = [{"doc_title": "Doc A", "page": 4, "page_end": 4, "text": "t"}]
        context = _build_context(chunks)
        assert "стр. 4" in context
        assert "стр. 4–4" not in context

    def test_truncates_chunks_exceeding_token_budget(self):
        chunks = [
            {"doc_title": f"Doc {i}", "page": i, "text": "слово " * 200}
            for i in range(1, 6)
        ]
        # A tiny budget should only fit the first chunk (or none), never overflow.
        context = _build_context(chunks, max_tokens=50)
        assert "[1] Doc 1" in context or context == ""
        assert "[5] Doc 5" not in context

    def test_empty_chunks_yields_empty_context(self):
        assert _build_context([]) == ""


# ---------------------------------------------------------------------------
# _deduplicate_sources()
# ---------------------------------------------------------------------------

class TestDeduplicateSources:
    def test_groups_by_filename_and_merges_pages(self):
        chunks = [
            {"filename": "a.pdf", "doc_title": "Doc A", "url": "u", "uploaded_at": "t", "page": 3},
            {"filename": "a.pdf", "doc_title": "Doc A", "url": "u", "uploaded_at": "t", "page": 1},
            {"filename": "a.pdf", "doc_title": "Doc A", "url": "u", "uploaded_at": "t", "page": 3},
        ]
        sources = _deduplicate_sources(chunks)
        assert len(sources) == 1
        assert sources[0]["pages"] == [1, 3]

    def test_page_end_added_to_pages_when_different_from_page(self):
        chunks = [{"filename": "a.pdf", "page": 2, "page_end": 4}]
        sources = _deduplicate_sources(chunks)
        assert sources[0]["pages"] == [2, 4]

    def test_chunk_without_filename_is_skipped(self):
        chunks = [{"doc_title": "No filename", "page": 1}]
        sources = _deduplicate_sources(chunks)
        assert sources == []

    def test_multiple_filenames_produce_separate_sources(self):
        chunks = [
            {"filename": "a.pdf", "page": 1},
            {"filename": "b.pdf", "page": 2},
        ]
        sources = _deduplicate_sources(chunks)
        filenames = {s["filename"] for s in sources}
        assert filenames == {"a.pdf", "b.pdf"}


# ---------------------------------------------------------------------------
# _strip_citation_artifacts()
# ---------------------------------------------------------------------------

class TestStripCitationArtifacts:
    def test_removes_fullwidth_lenticular_brackets(self):
        assert _strip_citation_artifacts("текст 【1】 продолжение") == "текст  продолжение"

    def test_removes_tortoise_shell_brackets(self):
        assert _strip_citation_artifacts("текст 〔2〕 продолжение") == "текст  продолжение"

    def test_removes_angle_brackets(self):
        assert _strip_citation_artifacts("текст ⟨3⟩ продолжение") == "текст  продолжение"

    def test_removes_dagger_and_reference_mark(self):
        assert _strip_citation_artifacts("текст† и ※текст") == "текст и текст"

    def test_keeps_plain_ascii_brackets(self):
        assert _strip_citation_artifacts("Ответ [1] и [2].") == "Ответ [1] и [2]."


# ---------------------------------------------------------------------------
# verify_chunk_answers_query()
# ---------------------------------------------------------------------------

def _patch_client(monkeypatch, fake_async_client):
    monkeypatch.setattr("rag.generator._make_llm_client", lambda: fake_async_client)


class TestVerifyChunkAnswersQuery:
    async def test_empty_chunk_text_returns_false_without_calling_llm(
        self, monkeypatch, fake_async_client
    ):
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_chunk_answers_query("Q?", {"text": "   "})

        assert result is False
        fake_async_client.chat.completions.create.assert_not_called()

    async def test_valid_true_response(self, monkeypatch, fake_async_client, fake_llm_response):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '{"answers_question": true}'
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_chunk_answers_query("Q?", {"text": "some fragment"})

        assert result is True

    async def test_valid_false_response(self, monkeypatch, fake_async_client, fake_llm_response):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            '{"answers_question": false}'
        )
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_chunk_answers_query("Q?", {"text": "some fragment"})

        assert result is False

    async def test_missing_key_returns_false(self, monkeypatch, fake_async_client, fake_llm_response):
        fake_async_client.chat.completions.create.return_value = fake_llm_response("{}")
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_chunk_answers_query("Q?", {"text": "some fragment"})

        assert result is False

    async def test_exception_returns_false(self, monkeypatch, fake_async_client):
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_chunk_answers_query("Q?", {"text": "some fragment"})

        assert result is False

    async def test_no_json_object_in_response_returns_false(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response("garbage")
        _patch_client(monkeypatch, fake_async_client)

        result = await verify_chunk_answers_query("Q?", {"text": "some fragment"})

        assert result is False


# ---------------------------------------------------------------------------
# Generator.generate() -- fully fake LLM client, no real network calls.
#
# check_coverage and verify_citations are imported *locally* inside
# generate(), so patching must target their source modules
# (rag.coverage_gate / rag.citation_verifier) -- patching rag.generator's
# module namespace would be invisible to that local `from ... import ...`.
# ---------------------------------------------------------------------------

@pytest.fixture
def generator(monkeypatch):
    """A Generator whose constructor doesn't hit the network -- groq.AsyncGroq
    is mocked out so _make_llm_client() succeeds without real credentials.
    Each test then overwrites the instance's `._client` with a fully fake client."""
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr("groq.AsyncGroq", MagicMock())
    return Generator()


class TestGenerateNoContext:
    async def test_all_chunks_below_min_score_skips_coverage_and_citations(
        self, monkeypatch, generator, fake_async_client, fake_llm_response
    ):
        generator._client = fake_async_client
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            "no context answer"
        )
        check_coverage_mock = AsyncMock()
        verify_citations_mock = AsyncMock()
        monkeypatch.setattr("rag.coverage_gate.check_coverage", check_coverage_mock)
        monkeypatch.setattr("rag.citation_verifier.verify_citations", verify_citations_mock)

        chunks = [{"text": "irrelevant", "score": 0.1, "filename": "a.pdf"}]
        result = await generator.generate("Вопрос?", chunks)

        assert result["answer"] == "no context answer"
        assert result["sources"] == []
        check_coverage_mock.assert_not_called()
        verify_citations_mock.assert_not_called()


class TestGenerateCoverageInsufficient:
    @pytest.mark.parametrize(
        "lang_code,expected_prefix",
        [("ru", "Уточните"), ("en", "Please clarify"), ("kk", "Нақтылаңыз")],
    )
    async def test_insufficient_coverage_returns_clarify_text_in_detected_language(
        self, monkeypatch, generator, fake_async_client, lang_code, expected_prefix
    ):
        monkeypatch.setattr("rag.generator.detect", lambda text: lang_code)
        generator._client = fake_async_client

        async def fake_check_coverage(question, profile, chunks):
            return {"verdict": "insufficient", "missing": ["дата"], "wrong_cohort_indices": []}

        monkeypatch.setattr("rag.coverage_gate.check_coverage", fake_check_coverage)

        chunks = [{"text": "some relevant text", "score": 0.9, "filename": "a.pdf"}]
        result = await generator.generate("Когда дедлайн?", chunks)

        assert result["sources"] == []
        assert result["answer"].startswith(expected_prefix)
        fake_async_client.chat.completions.create.assert_not_called()


class TestGeneratePartialCoverage:
    async def test_all_chunks_wrong_cohort_returns_clarify_text(
        self, monkeypatch, generator, fake_async_client
    ):
        generator._client = fake_async_client

        async def fake_check_coverage(question, profile, chunks):
            return {"verdict": "partial", "missing": [], "wrong_cohort_indices": [1]}

        monkeypatch.setattr("rag.coverage_gate.check_coverage", fake_check_coverage)

        chunks = [{"text": "text1", "score": 0.9, "filename": "a.pdf"}]
        result = await generator.generate("Question?", chunks)

        assert result["sources"] == []
        fake_async_client.chat.completions.create.assert_not_called()

    async def test_some_chunks_remaining_adds_partial_cohort_note_to_system_prompt(
        self, monkeypatch, generator, fake_async_client, fake_llm_response
    ):
        generator._client = fake_async_client
        fake_async_client.chat.completions.create.return_value = fake_llm_response("Ответ [1]")

        async def fake_check_coverage(question, profile, chunks):
            return {"verdict": "partial", "missing": [], "wrong_cohort_indices": [1]}

        async def fake_verify_citations(answer, fragments):
            return {"verdict": "clean", "invalid_citations": []}

        monkeypatch.setattr("rag.coverage_gate.check_coverage", fake_check_coverage)
        monkeypatch.setattr("rag.citation_verifier.verify_citations", fake_verify_citations)

        chunks = [
            {"text": "wrong cohort text", "score": 0.9, "filename": "a.pdf", "page": 1},
            {"text": "correct cohort text", "score": 0.9, "filename": "b.pdf", "page": 2},
        ]
        result = await generator.generate("Question?", chunks)

        call_kwargs = fake_async_client.chat.completions.create.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        assert _PARTIAL_COHORT_NOTE in system_content
        assert result["answer"] == "Ответ [1]"


class TestGenerateNoAnswerMarker:
    async def test_no_answer_marker_stripped_and_sources_emptied(
        self, monkeypatch, generator, fake_async_client, fake_llm_response
    ):
        generator._client = fake_async_client
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            "[NO_ANSWER] Не могу ответить на этот вопрос."
        )

        async def fake_check_coverage(question, profile, chunks):
            return {"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}

        verify_citations_mock = AsyncMock()
        monkeypatch.setattr("rag.coverage_gate.check_coverage", fake_check_coverage)
        monkeypatch.setattr("rag.citation_verifier.verify_citations", verify_citations_mock)

        chunks = [{"text": "text", "score": 0.9, "filename": "a.pdf"}]
        result = await generator.generate("Question?", chunks)

        assert result["answer"] == "Не могу ответить на этот вопрос."
        assert result["sources"] == []
        verify_citations_mock.assert_not_called()


class TestGenerateCitationVerification:
    async def test_unsupported_verdict_returns_localized_message_and_empty_sources(
        self, monkeypatch, generator, fake_async_client, fake_llm_response
    ):
        monkeypatch.setattr("rag.generator.detect", lambda text: "ru")
        generator._client = fake_async_client
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            "Ответ [1] и [2]"
        )

        async def fake_check_coverage(question, profile, chunks):
            return {"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}

        async def fake_verify_citations(answer, fragments):
            return {
                "verdict": "unsupported",
                "invalid_citations": [{"index": 2, "claim": "x", "reason": "not found"}],
            }

        monkeypatch.setattr("rag.coverage_gate.check_coverage", fake_check_coverage)
        monkeypatch.setattr("rag.citation_verifier.verify_citations", fake_verify_citations)

        chunks = [{"text": "text", "score": 0.9, "filename": "a.pdf"}]
        result = await generator.generate("Вопрос?", chunks)

        assert result["sources"] == []
        assert "Не удалось найти источники" in result["answer"]

    async def test_minor_verdict_returns_answer_unchanged(
        self, monkeypatch, generator, fake_async_client, fake_llm_response
    ):
        generator._client = fake_async_client
        fake_async_client.chat.completions.create.return_value = fake_llm_response("Ответ [1]")

        async def fake_check_coverage(question, profile, chunks):
            return {"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}

        async def fake_verify_citations(answer, fragments):
            return {
                "verdict": "minor",
                "invalid_citations": [{"index": 1, "claim": "x", "reason": "vague"}],
            }

        monkeypatch.setattr("rag.coverage_gate.check_coverage", fake_check_coverage)
        monkeypatch.setattr("rag.citation_verifier.verify_citations", fake_verify_citations)

        chunks = [{"text": "text", "score": 0.9, "filename": "a.pdf", "page": 1}]
        result = await generator.generate("Question?", chunks)

        assert result["answer"] == "Ответ [1]"
        assert len(result["sources"]) == 1


class TestGenerateHappyPath:
    async def test_chunks_with_different_filenames_produce_deduplicated_sources(
        self, monkeypatch, generator, fake_async_client, fake_llm_response
    ):
        generator._client = fake_async_client
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            "Финальный ответ [1][2]"
        )

        async def fake_check_coverage(question, profile, chunks):
            return {"verdict": "sufficient", "missing": [], "wrong_cohort_indices": []}

        async def fake_verify_citations(answer, fragments):
            return {"verdict": "clean", "invalid_citations": []}

        monkeypatch.setattr("rag.coverage_gate.check_coverage", fake_check_coverage)
        monkeypatch.setattr("rag.citation_verifier.verify_citations", fake_verify_citations)

        chunks = [
            {"text": "t1", "score": 0.9, "filename": "a.pdf", "doc_title": "Doc A", "url": "u1", "page": 1},
            {"text": "t2", "score": 0.9, "filename": "a.pdf", "doc_title": "Doc A", "url": "u1", "page": 3},
            {"text": "t3", "score": 0.9, "filename": "b.pdf", "doc_title": "Doc B", "url": "u2", "page": 5},
        ]
        result = await generator.generate("Question?", chunks)

        assert result["answer"] == "Финальный ответ [1][2]"
        sources_by_filename = {s["filename"]: s for s in result["sources"]}
        assert set(sources_by_filename) == {"a.pdf", "b.pdf"}
        assert sources_by_filename["a.pdf"]["pages"] == [1, 3]
        assert sources_by_filename["b.pdf"]["pages"] == [5]
