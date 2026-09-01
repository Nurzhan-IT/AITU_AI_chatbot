import math
from datetime import datetime, timezone

import pytest
from qdrant_client.models import Fusion, FusionQuery

from config import settings
from rag.bm25 import BM25Stats
from rag.retriever import Retriever, _clamp01, _compute_factors, _cosine, mmr

# ---------------------------------------------------------------------------
# _clamp01()
# ---------------------------------------------------------------------------

class TestClamp01:
    def test_below_zero_clamped_to_zero(self):
        assert _clamp01(-0.5) == 0.0

    def test_above_one_clamped_to_one(self):
        assert _clamp01(1.5) == 1.0

    def test_nan_clamped_to_zero(self):
        assert _clamp01(float("nan")) == 0.0

    def test_normal_range_passthrough(self):
        assert _clamp01(0.3) == 0.3


# ---------------------------------------------------------------------------
# _cosine()
# ---------------------------------------------------------------------------

class TestCosine:
    def test_orthogonal_vectors_yield_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_identical_vectors_yield_one(self):
        assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_zero_vector_yields_zero_without_division_error(self):
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert _cosine([1.0, 1.0], [0.0, 0.0]) == 0.0
        assert _cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# _compute_factors()
# ---------------------------------------------------------------------------

def _make_chunk(**overrides):
    chunk = {
        "text": "some default chunk text",
        "section_title": "",
        "doc_title": "",
        "uploaded_at": "",
        "score": 0.5,
    }
    chunk.update(overrides)
    return chunk


class TestComputeFactorsUserTypeMatch:
    def test_none_keyword_yields_neutral(self):
        chunk = _make_chunk(text="бакалавриат правила")
        factors = _compute_factors(chunk, None, [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["user_type_match"] == 0.5

    def test_keyword_present_in_text(self):
        chunk = _make_chunk(text="это правила для магистр программы")
        factors = _compute_factors(chunk, "магистр", [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["user_type_match"] == 1.0

    def test_keyword_present_in_section_title(self):
        chunk = _make_chunk(text="общие правила", section_title="магистратура")
        factors = _compute_factors(chunk, "магистр", [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["user_type_match"] == 1.0

    def test_keyword_absent_yields_zero(self):
        chunk = _make_chunk(text="общие правила", section_title="прочее")
        factors = _compute_factors(chunk, "докторант", [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["user_type_match"] == 0.0


class TestComputeFactorsDocMatch:
    def test_empty_hints_yields_neutral(self):
        chunk = _make_chunk(doc_title="Положение о стипендиях")
        factors = _compute_factors(chunk, None, [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["doc_match"] == 0.5

    def test_high_similarity_hint(self):
        chunk = _make_chunk(doc_title="положение о стипендиях")
        factors = _compute_factors(
            chunk, None, ["положение о стипендиях"], [0.0], {}, datetime.now(timezone.utc)
        )
        assert factors["doc_match"] == pytest.approx(1.0)

    def test_low_similarity_hint(self):
        chunk = _make_chunk(doc_title="положение о стипендиях")
        factors = _compute_factors(
            chunk, None, ["zzzzzzzzzzzz"], [0.0], {}, datetime.now(timezone.utc)
        )
        assert factors["doc_match"] < 0.3


class TestComputeFactorsPositionBonus:
    def test_empty_section_title_yields_neutral(self):
        chunk = _make_chunk(section_title="", text="whatever text")
        factors = _compute_factors(chunk, None, [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["position_bonus"] == 0.5

    def test_section_word_found_in_prefix_yields_full_bonus(self):
        chunk = _make_chunk(
            section_title="Стипендии",
            text="Раздел про стипендии и условия их получения " + "x" * 100,
        )
        factors = _compute_factors(chunk, None, [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["position_bonus"] == 1.0

    def test_section_word_not_found_in_prefix_yields_neutral(self):
        chunk = _make_chunk(section_title="Стипендии", text="совершенно не связанный текст")
        factors = _compute_factors(chunk, None, [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["position_bonus"] == 0.5

    def test_short_words_below_min_length_are_filtered_out(self):
        # Both words in section_title are shorter than _POSITION_MIN_WORD (3),
        # so section_words ends up empty even though "в п" literally appears
        # in the text -- must fall back to neutral, not match.
        chunk = _make_chunk(section_title="в п", text="в п " + "x" * 100)
        factors = _compute_factors(chunk, None, [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["position_bonus"] == 0.5

    def test_match_beyond_prefix_window_is_not_found(self):
        padding = "x" * 1600
        chunk = _make_chunk(section_title="Стипендии", text=padding + " стипендии")
        factors = _compute_factors(chunk, None, [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["position_bonus"] == 0.5


class TestComputeFactorsRecency:
    def test_empty_uploaded_at_yields_neutral(self):
        chunk = _make_chunk(uploaded_at="")
        factors = _compute_factors(chunk, None, [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["recency"] == 0.5

    def test_invalid_timestamp_string_yields_neutral(self):
        chunk = _make_chunk(uploaded_at="not-a-real-date")
        factors = _compute_factors(chunk, None, [], [0.0], {}, datetime.now(timezone.utc))
        assert factors["recency"] == 0.5

    def test_zero_age_yields_recency_one(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        chunk = _make_chunk(uploaded_at="2026-01-01T00:00:00+00:00")
        factors = _compute_factors(chunk, None, [], [0.0], {}, now)
        assert factors["recency"] == pytest.approx(1.0)

    def test_exponential_decay_at_one_half_decay_period(self):
        now = datetime(2027, 1, 1, tzinfo=timezone.utc)
        chunk = _make_chunk(uploaded_at="2026-01-01T00:00:00+00:00")
        factors = _compute_factors(chunk, None, [], [0.0], {}, now)
        age_days = (now - datetime(2026, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0
        expected = math.exp(-age_days / 365.0)
        assert factors["recency"] == pytest.approx(expected, rel=1e-6)


class TestComputeFactorsSectionSim:
    def test_empty_section_title_yields_neutral(self):
        chunk = _make_chunk(section_title="")
        factors = _compute_factors(chunk, None, [], [1.0, 0.0], {}, datetime.now(timezone.utc))
        assert factors["section_sim"] == 0.5

    def test_vector_in_cache_uses_cosine(self):
        chunk = _make_chunk(section_title="Стипендии")
        cache = {"Стипендии": [1.0, 0.0]}
        factors = _compute_factors(chunk, None, [], [1.0, 0.0], cache, datetime.now(timezone.utc))
        assert factors["section_sim"] == pytest.approx(1.0)

    def test_vector_missing_from_cache_yields_neutral(self):
        chunk = _make_chunk(section_title="Стипендии")
        factors = _compute_factors(chunk, None, [], [1.0, 0.0], {}, datetime.now(timezone.utc))
        assert factors["section_sim"] == 0.5


# ---------------------------------------------------------------------------
# mmr()
# ---------------------------------------------------------------------------

class TestMmr:
    def test_empty_candidates_returns_empty_list(self):
        assert mmr([1.0, 0.0], [], k=5) == []

    def test_k_larger_than_pool_returns_whole_pool(self):
        candidates = [
            {"id": "A", "_vector": [1.0, 0.0]},
            {"id": "B", "_vector": [0.0, 1.0]},
        ]
        result = mmr([1.0, 0.0], candidates, k=10)
        assert len(result) == 2

    def test_vector_key_is_stripped_from_results(self):
        candidates = [
            {"id": "A", "_vector": [1.0, 0.0]},
            {"id": "B", "_vector": [0.0, 1.0]},
        ]
        result = mmr([1.0, 0.0], candidates, k=2)
        for item in result:
            assert "_vector" not in item

    def test_first_selection_is_argmax_of_relevance_not_first_in_list(self):
        # B has the highest dot product with q, but is not first in the list --
        # the first pick must still be B, not A.
        candidates = [
            {"id": "A", "_vector": [0.2, 0.9]},   # dot = 0.2
            {"id": "B", "_vector": [0.9, 0.1]},   # dot = 0.9
            {"id": "C", "_vector": [0.5, 0.5]},   # dot = 0.5
        ]
        result = mmr([1.0, 0.0], candidates, k=1)
        assert result[0]["id"] == "B"


# ---------------------------------------------------------------------------
# Retriever integration tests -- fully fake Qdrant client + embedder,
# no real network calls.
# ---------------------------------------------------------------------------

class FakePoint:
    """Stands in for a qdrant_client ScoredPoint."""

    def __init__(self, id="pt", payload=None, score=0.0, vector=None):
        self.id = id
        self.payload = payload or {}
        self.score = score
        self.vector = vector


class FakeQueryPointsResult:
    """Stands in for the response of AsyncQdrantClient.query_points()."""

    def __init__(self, points):
        self.points = points


class FakeScrollRecord:
    """Stands in for a qdrant_client Record returned by scroll()."""

    def __init__(self, id="rec", payload=None):
        self.id = id
        self.payload = payload or {}


class FakeCountResult:
    def __init__(self, count):
        self.count = count


class FakeQdrantClient:
    """In-memory fake for AsyncQdrantClient -- implements only what the
    Retriever methods under test actually call."""

    def __init__(self):
        self.search_result: list = []
        self.query_points_result = FakeQueryPointsResult([])
        self.scroll_pages: list[tuple[list, object]] = []
        self.count_result = FakeCountResult(0)

        self.search_calls: list[dict] = []
        self.query_points_calls: list[dict] = []
        self.scroll_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_result

    async def query_points(self, **kwargs):
        self.query_points_calls.append(kwargs)
        return self.query_points_result

    async def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        return self.scroll_pages.pop(0)

    async def count(self, **kwargs):
        return self.count_result

    async def delete(self, **kwargs):
        self.delete_calls.append(kwargs)


class FakeEmbedder:
    """Deterministic fixed-vector fake -- no real embedding calls."""

    def __init__(self, vector=None):
        self._vector = vector if vector is not None else [1.0, 0.0]

    async def embed_query(self, text):
        return self._vector

    async def embed_passages(self, texts):
        return [self._vector for _ in texts]


def _payload(**overrides):
    payload = {
        "filename": "doc.pdf",
        "doc_title": "Doc",
        "text": "",
        "uploaded_at": "",
        "page": 1,
        "section_title": "",
        "paragraph_range": "",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def retriever():
    """A Retriever with its real (network-free) constructor; _client and
    _embedder get swapped for fakes by each test."""
    return Retriever()


# ---------------------------------------------------------------------------
# Retriever.search()
# ---------------------------------------------------------------------------

class TestRetrieverSearchVectorExtraction:
    async def test_legacy_unnamed_vector_extracted_correctly(self, retriever):
        fake_client = FakeQdrantClient()
        # B comes first in the list but is orthogonal to the query vector;
        # A is aligned with it -- MMR must pick A first, proving the dense
        # vector was correctly read straight off point.vector (a plain list).
        fake_client.search_result = [
            FakePoint(payload=_payload(filename="b.pdf"), score=0.7, vector=[0.0, 1.0]),
            FakePoint(payload=_payload(filename="a.pdf"), score=0.9, vector=[1.0, 0.0]),
        ]
        retriever._client = fake_client
        retriever._embedder = FakeEmbedder([1.0, 0.0])
        retriever._collection_checked = True
        retriever._uses_named_vectors = False
        retriever._hybrid_active = False

        hits = await retriever.search("query")

        assert hits[0]["filename"] == "a.pdf"
        assert "_vector" not in hits[0]

    async def test_named_dense_vector_extracted_correctly(self, retriever):
        fake_client = FakeQdrantClient()
        fake_client.search_result = [
            FakePoint(payload=_payload(filename="b.pdf"), score=0.7, vector={"dense": [0.0, 1.0]}),
            FakePoint(payload=_payload(filename="a.pdf"), score=0.9, vector={"dense": [1.0, 0.0]}),
        ]
        retriever._client = fake_client
        retriever._embedder = FakeEmbedder([1.0, 0.0])
        retriever._collection_checked = True
        retriever._uses_named_vectors = True
        retriever._hybrid_active = False

        hits = await retriever.search("query")

        assert hits[0]["filename"] == "a.pdf"
        assert "_vector" not in hits[0]


class TestRetrieverSearchHybrid:
    async def test_nonempty_bm25_tokens_uses_query_points_rrf_fusion(self, retriever):
        fake_client = FakeQdrantClient()
        fake_client.query_points_result = FakeQueryPointsResult([
            FakePoint(payload=_payload(filename="a.pdf"), score=0.9, vector={"dense": [1.0, 0.0]}),
        ])
        retriever._client = fake_client
        retriever._embedder = FakeEmbedder([1.0, 0.0])
        retriever._collection_checked = True
        retriever._uses_named_vectors = True
        retriever._hybrid_active = True
        stats = BM25Stats()
        stats.total_chunks = 10
        retriever._bm25_stats = stats

        hits = await retriever.search("университет стипендия")

        assert len(fake_client.query_points_calls) == 1
        assert fake_client.search_calls == []
        call_kwargs = fake_client.query_points_calls[0]
        assert isinstance(call_kwargs["query"], FusionQuery)
        assert call_kwargs["query"].fusion == Fusion.RRF
        assert hits[0]["filename"] == "a.pdf"

    async def test_all_stopword_tokens_falls_back_to_dense_search(self, retriever):
        fake_client = FakeQdrantClient()
        fake_client.search_result = [
            FakePoint(payload=_payload(filename="a.pdf"), score=0.9, vector={"dense": [1.0, 0.0]}),
        ]
        retriever._client = fake_client
        retriever._embedder = FakeEmbedder([1.0, 0.0])
        retriever._collection_checked = True
        retriever._uses_named_vectors = True
        retriever._hybrid_active = True
        stats = BM25Stats()
        stats.total_chunks = 10
        retriever._bm25_stats = stats

        # "и", "в", "на" are all shorter than the BM25 min token length (3),
        # so tokenize() yields an empty token list regardless of stopwords.
        hits = await retriever.search("и в на")

        assert fake_client.query_points_calls == []
        assert len(fake_client.search_calls) == 1
        assert hits[0]["filename"] == "a.pdf"


# ---------------------------------------------------------------------------
# Retriever.probe_search()
# ---------------------------------------------------------------------------

class TestRetrieverProbeSearch:
    async def test_no_mmr_no_topk_cap_no_vector_key(self, retriever, monkeypatch):
        monkeypatch.setattr(settings, "top_k", 3)  # smaller than probe result count
        fake_client = FakeQdrantClient()
        points = [
            FakePoint(payload=_payload(filename=f"doc{i}.pdf"), score=1.0 - i * 0.01)
            for i in range(7)
        ]
        fake_client.search_result = points
        retriever._client = fake_client
        retriever._embedder = FakeEmbedder([1.0, 0.0])
        retriever._collection_checked = True
        retriever._uses_named_vectors = False
        retriever._hybrid_active = False

        hits, vector = await retriever.probe_search("query", k=7)

        assert len(hits) == 7  # not capped to top_k, unlike search()
        assert [h["filename"] for h in hits] == [f"doc{i}.pdf" for i in range(7)]
        for h in hits:
            assert "_vector" not in h  # probe_search never adds it -- no mmr() call
        assert vector == [1.0, 0.0]
        assert fake_client.search_calls[0]["with_vectors"] is False

    async def test_default_k_uses_triage_probe_k_setting(self, retriever, monkeypatch):
        monkeypatch.setattr(settings, "triage_probe_k", 3)
        fake_client = FakeQdrantClient()
        fake_client.search_result = []
        retriever._client = fake_client
        retriever._embedder = FakeEmbedder([1.0, 0.0])
        retriever._collection_checked = True
        retriever._uses_named_vectors = False
        retriever._hybrid_active = False

        await retriever.probe_search("query")

        assert fake_client.search_calls[0]["limit"] == 3


# ---------------------------------------------------------------------------
# Retriever.search_with_profile()
# ---------------------------------------------------------------------------

class TestRetrieverSearchWithProfile:
    async def test_with_profile_sorts_by_final_score_descending(self, retriever):
        fake_client = FakeQdrantClient()
        fake_client.search_result = [
            FakePoint(
                payload=_payload(
                    filename="low.pdf", doc_title="Другое", text="общий текст без ключевых слов",
                ),
                score=0.5,
                vector={"dense": [1.0, 0.0]},
            ),
            FakePoint(
                payload=_payload(
                    filename="high.pdf",
                    doc_title="Положение о стипендиях",
                    text="это правила для магистрантов",
                ),
                score=0.9,
                vector={"dense": [1.0, 0.0]},
            ),
        ]
        retriever._client = fake_client
        retriever._embedder = FakeEmbedder([1.0, 0.0])
        retriever._collection_checked = True
        retriever._uses_named_vectors = True
        retriever._hybrid_active = False

        profile = {"user_type": "магистрант", "document_hints": ["положение о стипендиях"]}
        results = await retriever.search_with_profile("query", profile, k=2)

        assert [r["filename"] for r in results] == ["high.pdf", "low.pdf"]
        assert results[0]["final_score"] > results[1]["final_score"]
        for r in results:
            assert "_vector" not in r
            assert "factor_scores" in r

    async def test_without_profile_factors_default_to_neutral(self, retriever):
        fake_client = FakeQdrantClient()
        fake_client.search_result = [
            FakePoint(payload=_payload(filename="only.pdf"), score=0.7, vector={"dense": [1.0, 0.0]}),
        ]
        retriever._client = fake_client
        retriever._embedder = FakeEmbedder([1.0, 0.0])
        retriever._collection_checked = True
        retriever._uses_named_vectors = True
        retriever._hybrid_active = False

        results = await retriever.search_with_profile("query", {}, k=5)

        factors = results[0]["factor_scores"]
        assert factors["user_type_match"] == 0.5
        assert factors["doc_match"] == 0.5
        assert factors["position_bonus"] == 0.5
        assert factors["recency"] == 0.5
        assert factors["section_sim"] == 0.5
        assert factors["semantic_sim"] == pytest.approx(0.7)
        expected_final = 0.35 * 0.7 + 0.20 * 0.5 + 0.15 * 0.5 + 0.10 * 0.5 + 0.10 * 0.5 + 0.10 * 0.5
        assert results[0]["final_score"] == pytest.approx(expected_final)


# ---------------------------------------------------------------------------
# Retriever.get_all_documents()
# ---------------------------------------------------------------------------

class TestRetrieverGetAllDocuments:
    async def test_paginates_scroll_and_dedupes_by_filename(self, retriever):
        fake_client = FakeQdrantClient()
        page1 = [
            FakeScrollRecord(payload={"doc_title": "Doc A", "filename": "a.pdf"}),
            FakeScrollRecord(payload={"doc_title": "Doc B", "filename": "b.pdf"}),
        ]
        page2 = [
            FakeScrollRecord(payload={"doc_title": "Doc B", "filename": "b.pdf"}),  # duplicate
            FakeScrollRecord(payload={"doc_title": "Doc C", "filename": "c.pdf"}),
        ]
        fake_client.scroll_pages = [
            (page1, "offset-1"),
            (page2, None),
        ]
        retriever._client = fake_client
        retriever._collection_checked = True

        docs = await retriever.get_all_documents()

        assert len(fake_client.scroll_calls) == 2
        assert [d["filename"] for d in docs] == ["a.pdf", "b.pdf", "c.pdf"]


# ---------------------------------------------------------------------------
# Retriever.delete_document()
# ---------------------------------------------------------------------------

class TestRetrieverDeleteDocument:
    async def test_zero_count_skips_delete_call(self, retriever):
        fake_client = FakeQdrantClient()
        fake_client.count_result = FakeCountResult(0)
        retriever._client = fake_client
        retriever._collection_checked = True

        deleted = await retriever.delete_document("missing.pdf")

        assert deleted == 0
        assert fake_client.delete_calls == []

    async def test_nonzero_count_calls_delete_and_returns_count(self, retriever):
        fake_client = FakeQdrantClient()
        fake_client.count_result = FakeCountResult(3)
        retriever._client = fake_client
        retriever._collection_checked = True

        deleted = await retriever.delete_document("present.pdf")

        assert deleted == 3
        assert len(fake_client.delete_calls) == 1
