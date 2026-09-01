from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from config import settings
from duplicate_detection import detector
from duplicate_detection import notifier as dd_notifier
from duplicate_detection import repository as dd_repo

# ---------------------------------------------------------------------------
# _snip
# ---------------------------------------------------------------------------

class TestSnip:
    def test_short_text_unchanged(self):
        assert detector._snip("hello world") == "hello world"

    def test_long_text_truncated_with_ellipsis(self):
        text = "a" * 200
        assert detector._snip(text, n=10) == "a" * 10 + "…"

    def test_collapses_whitespace(self):
        text = "hello    \n\n   world"
        assert detector._snip(text) == "hello world"

    def test_exact_length_no_ellipsis(self):
        text = "a" * 10
        assert detector._snip(text, n=10) == text

    def test_one_over_length_gets_ellipsis(self):
        text = "a" * 11
        assert detector._snip(text, n=10) == "a" * 10 + "…"


# ---------------------------------------------------------------------------
# _deduplicate
# ---------------------------------------------------------------------------

def _w(wtype="DUPLICATE", new="a.pdf", existing="b.pdf", sim=0.9, **extra):
    return {
        "warning_type": wtype,
        "new_filename": new,
        "existing_filename": existing,
        "similarity": sim,
        **extra,
    }


class TestDeduplicate:
    def test_empty_list(self):
        assert detector._deduplicate([]) == []

    def test_single_warning_kept(self):
        w = _w()
        assert detector._deduplicate([w]) == [w]

    def test_higher_similarity_replaces_lower_for_same_key(self):
        low = _w(sim=0.8, marker="low")
        high = _w(sim=0.95, marker="high")
        result = detector._deduplicate([low, high])
        assert result == [high]

    def test_equal_similarity_keeps_first_seen(self):
        first = _w(sim=0.9, marker="first")
        second = _w(sim=0.9, marker="second")
        result = detector._deduplicate([first, second])
        assert result == [first]

    def test_lower_similarity_after_higher_does_not_replace(self):
        high = _w(sim=0.95, marker="high")
        low = _w(sim=0.8, marker="low")
        result = detector._deduplicate([high, low])
        assert result == [high]

    def test_different_keys_all_kept(self):
        w1 = _w(wtype="DUPLICATE", new="a.pdf", existing="b.pdf")
        w2 = _w(wtype="STALE", new="a.pdf", existing="b.pdf")
        w3 = _w(wtype="DUPLICATE", new="a.pdf", existing="c.pdf")
        result = detector._deduplicate([w1, w2, w3])
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _classify_with_llm
# ---------------------------------------------------------------------------

def _patch_groq(monkeypatch, fake_async_client):
    monkeypatch.setattr(
        "duplicate_detection.detector.AsyncGroq", lambda api_key=None: fake_async_client
    )


class TestClassifyWithLlm:
    async def test_stale_response_parsed(self, monkeypatch, fake_async_client, fake_llm_response):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            "STALE: new document supersedes the old policy"
        )
        _patch_groq(monkeypatch, fake_async_client)

        verdict, reason = await detector._classify_with_llm(
            "new chunk", "existing chunk", "New Title", "Existing Title", 0.8, 0
        )

        assert verdict == "STALE"
        assert reason == "new document supersedes the old policy"

    async def test_similar_only_response(self, monkeypatch, fake_async_client, fake_llm_response):
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            "SIMILAR_ONLY: unrelated topics"
        )
        _patch_groq(monkeypatch, fake_async_client)

        verdict, reason = await detector._classify_with_llm(
            "new chunk", "existing chunk", "New Title", "Existing Title", 0.8, 0
        )

        assert verdict == "SIMILAR_ONLY"
        assert reason == ""

    async def test_unexpected_text_defaults_to_similar_only(
        self, monkeypatch, fake_async_client, fake_llm_response
    ):
        fake_async_client.chat.completions.create.return_value = fake_llm_response("garbage output")
        _patch_groq(monkeypatch, fake_async_client)

        verdict, reason = await detector._classify_with_llm(
            "new chunk", "existing chunk", "New Title", "Existing Title", 0.8, 0
        )

        assert verdict == "SIMILAR_ONLY"
        assert reason == ""

    async def test_client_exception_fails_safe_to_similar_only(self, monkeypatch, fake_async_client):
        fake_async_client.chat.completions.create.side_effect = RuntimeError("boom")
        _patch_groq(monkeypatch, fake_async_client)

        verdict, reason = await detector._classify_with_llm(
            "new chunk", "existing chunk", "New Title", "Existing Title", 0.8, 0
        )

        assert verdict == "SIMILAR_ONLY"
        assert reason == ""


# ---------------------------------------------------------------------------
# analyze_new_document / _run_analysis — full pipeline, everything mocked
# ---------------------------------------------------------------------------

def _make_chunk(text, section_title="Sec", paragraph_range="1", page=1, page_end=1):
    return {
        "text": text,
        "section_title": section_title,
        "paragraph_range": paragraph_range,
        "page": page,
        "page_end": page_end,
    }


def _hit(score, filename="existing.pdf", text="existing chunk text", doc_title="Existing Doc"):
    return SimpleNamespace(
        score=score,
        payload={
            "filename": filename,
            "text": text,
            "doc_title": doc_title,
            "section_title": "Sec",
            "paragraph_range": "1",
            "page": 1,
        },
    )


class _FakeEmbedder:
    """Assigns passage i the vector [float(i)] so the fake Qdrant client can
    deterministically map a search call back to the chunk that triggered it,
    regardless of asyncio.gather() scheduling order."""

    def __init__(self):
        self.embed_calls: list[list[str]] = []
        self.closed = False

    async def embed_passages(self, texts):
        self.embed_calls.append(list(texts))
        return [[float(i)] for i in range(len(texts))]

    async def aclose(self):
        self.closed = True


def _make_fake_qdrant_cls(results_by_index: dict, search_calls: list):
    class FakeQdrantClient:
        def __init__(self, host=None, port=None):
            pass

        async def search(
            self, *, collection_name, query_vector, query_filter, limit, with_payload, score_threshold
        ):
            search_calls.append(
                {
                    "collection_name": collection_name,
                    "query_filter": query_filter,
                    "score_threshold": score_threshold,
                    "query_vector": query_vector,
                }
            )
            idx = int(query_vector[0])
            return results_by_index.get(idx, [])

        async def close(self):
            pass

    return FakeQdrantClient


def _setup_pipeline(monkeypatch, chunks, results_by_index=None):
    monkeypatch.setattr("duplicate_detection.detector.parse_pdf", lambda filepath: "full text")
    monkeypatch.setattr("duplicate_detection.detector.build_fitz_page_map", lambda filepath: {})
    monkeypatch.setattr(
        "duplicate_detection.detector.chunk_by_sections", lambda *a, **kw: chunks
    )

    fake_embedder = _FakeEmbedder()
    monkeypatch.setattr("duplicate_detection.detector.Embedder", lambda: fake_embedder)

    search_calls: list[dict] = []
    fake_qdrant_cls = _make_fake_qdrant_cls(results_by_index or {}, search_calls)
    monkeypatch.setattr("duplicate_detection.detector.AsyncQdrantClient", fake_qdrant_cls)

    notify_mock = AsyncMock()
    monkeypatch.setattr(dd_notifier, "send_upload_warnings", notify_mock)

    return fake_embedder, search_calls, notify_mock


class TestRunAnalysisPipeline:
    async def test_no_chunks_returns_empty_and_skips_rest_of_pipeline(
        self, tmp_sqlite_db, monkeypatch
    ):
        fake_embedder, search_calls, notify_mock = _setup_pipeline(monkeypatch, chunks=[])
        bot = MagicMock()

        result = await detector.analyze_new_document(Path("f.pdf"), "f.pdf", "Title", bot, [1])

        assert result == []
        assert fake_embedder.embed_calls == []
        assert search_calls == []
        notify_mock.assert_not_called()

    async def test_only_header_only_chunks_yields_empty_result(self, tmp_sqlite_db, monkeypatch):
        chunks = [_make_chunk("short header")]  # well below _MIN_ANALYSIS_CHARS (150)
        fake_embedder, search_calls, notify_mock = _setup_pipeline(monkeypatch, chunks=chunks)
        bot = MagicMock()

        result = await detector.analyze_new_document(Path("f.pdf"), "f.pdf", "Title", bot, [1])

        assert result == []
        assert fake_embedder.embed_calls == []
        notify_mock.assert_not_called()

    async def test_high_score_is_duplicate_without_llm_call(self, tmp_sqlite_db, monkeypatch):
        long_text = "x" * 200
        chunks = [_make_chunk(long_text)]
        hit = _hit(score=0.95, filename="existing.pdf")
        fake_embedder, search_calls, notify_mock = _setup_pipeline(
            monkeypatch, chunks=chunks, results_by_index={0: [hit]}
        )

        def _groq_should_not_be_called(*args, **kwargs):
            raise AssertionError("AsyncGroq should not be instantiated for a DUPLICATE-range score")

        monkeypatch.setattr("duplicate_detection.detector.AsyncGroq", _groq_should_not_be_called)
        bot = MagicMock()

        result = await detector.analyze_new_document(Path("new.pdf"), "new.pdf", "New Title", bot, [111])

        assert len(result) == 1
        assert result[0]["warning_type"] == "DUPLICATE"
        assert result[0]["existing_filename"] == "existing.pdf"
        assert result[0]["similarity"] == 0.95

        saved = await dd_repo.get_warning_by_id(result[0]["id"])
        assert saved is not None
        assert saved["warning_type"] == "DUPLICATE"
        assert saved["new_filename"] == "new.pdf"

        notify_mock.assert_awaited_once()
        call_args = notify_mock.call_args.args
        assert call_args[0] is bot
        assert call_args[1] == [111]
        assert call_args[2] == "new.pdf"
        assert call_args[3] == result

    async def test_ambiguous_score_routes_to_llm_and_saves_stale(
        self, tmp_sqlite_db, monkeypatch, fake_async_client, fake_llm_response
    ):
        long_text = "y" * 200
        chunks = [_make_chunk(long_text)]
        hit = _hit(score=0.80, filename="existing.pdf")
        fake_embedder, search_calls, notify_mock = _setup_pipeline(
            monkeypatch, chunks=chunks, results_by_index={0: [hit]}
        )
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            "STALE: replaces the old policy document"
        )
        _patch_groq(monkeypatch, fake_async_client)
        bot = MagicMock()

        result = await detector.analyze_new_document(Path("new.pdf"), "new.pdf", "New Title", bot, [1])

        assert len(result) == 1
        assert result[0]["warning_type"] == "STALE"
        assert result[0]["llm_reason"] == "replaces the old policy document"

        saved = await dd_repo.get_warning_by_id(result[0]["id"])
        assert saved["warning_type"] == "STALE"
        assert saved["llm_reason"] == "replaces the old policy document"

    async def test_ambiguous_score_similar_only_produces_no_warning(
        self, tmp_sqlite_db, monkeypatch, fake_async_client, fake_llm_response
    ):
        long_text = "y" * 200
        chunks = [_make_chunk(long_text)]
        hit = _hit(score=0.80, filename="existing.pdf")
        fake_embedder, search_calls, notify_mock = _setup_pipeline(
            monkeypatch, chunks=chunks, results_by_index={0: [hit]}
        )
        fake_async_client.chat.completions.create.return_value = fake_llm_response(
            "SIMILAR_ONLY: independent topics"
        )
        _patch_groq(monkeypatch, fake_async_client)
        bot = MagicMock()

        result = await detector.analyze_new_document(Path("new.pdf"), "new.pdf", "New Title", bot, [1])

        assert result == []
        notify_mock.assert_not_called()

    async def test_below_threshold_ignored_and_exclude_self_filter_applied(
        self, tmp_sqlite_db, monkeypatch
    ):
        long_text = "z" * 200
        chunks = [_make_chunk(long_text)]
        # No results for any vector -> simulates Qdrant's own score_threshold
        # filtering everything out.
        fake_embedder, search_calls, notify_mock = _setup_pipeline(
            monkeypatch, chunks=chunks, results_by_index={}
        )
        bot = MagicMock()

        result = await detector.analyze_new_document(Path("new.pdf"), "new.pdf", "New Title", bot, [1])

        assert result == []
        notify_mock.assert_not_called()

        assert len(search_calls) == 1
        assert search_calls[0]["score_threshold"] == settings.stale_threshold_low
        query_filter = search_calls[0]["query_filter"]
        assert query_filter.must_not[0].key == "filename"
        assert query_filter.must_not[0].match.value == "new.pdf"

    async def test_pipeline_exception_is_caught_and_returns_empty_list(
        self, tmp_sqlite_db, monkeypatch
    ):
        def _boom(filepath):
            raise RuntimeError("parse failed")

        monkeypatch.setattr("duplicate_detection.detector.parse_pdf", _boom)
        bot = MagicMock()

        result = await detector.analyze_new_document(Path("bad.pdf"), "bad.pdf", "Title", bot, [1])

        assert result == []
