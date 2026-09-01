import math
from datetime import datetime, timezone

import pytest

from rag.retriever import _clamp01, _cosine, _compute_factors, mmr


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
