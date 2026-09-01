import json
import logging
import math

import pytest

from rag.bm25 import BM25Stats, bm25_doc_vector, bm25_query_vector, term_hash, tokenize

# ---------------------------------------------------------------------------
# tokenize()
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercases_and_keeps_cyrillic_latin_digits(self):
        assert tokenize("Привет мир 123 test") == ["привет", "мир", "123", "test"]

    def test_filters_tokens_shorter_than_min_length(self):
        # "vps" (3) kept, "ok" (2) and "я" (1) dropped by length alone
        assert tokenize("vps ok я") == ["vps"]

    def test_filters_russian_stopwords(self):
        # "это" (3 chars) is long enough to survive length filtering but is
        # a stopword; "тест" and "бота" are not stopwords and survive.
        assert tokenize("это тест бота") == ["тест", "бота"]

    def test_filters_english_stopwords(self):
        # "the" is 3 chars (passes length filter) but is a stopword.
        assert tokenize("the cat sat") == ["cat", "sat"]

    def test_filters_kazakh_stopwords(self):
        # "үшін" and "туралы" are stopwords; "қала" is not.
        assert tokenize("үшін туралы қала") == ["қала"]

    def test_all_stopwords_yields_empty_list(self):
        assert tokenize("и в на the a an") == []

    def test_kazakh_specific_letters_are_kept(self):
        assert tokenize("әдіс ғылым қала") == ["әдіс", "ғылым", "қала"]

    def test_empty_string(self):
        assert tokenize("") == []


# ---------------------------------------------------------------------------
# term_hash()
# ---------------------------------------------------------------------------

class TestTermHash:
    def test_deterministic_across_calls(self):
        assert term_hash("cat") == term_hash("cat")

    def test_different_terms_differ(self):
        assert term_hash("cat") != term_hash("dog")

    def test_is_unsigned_32_bit(self):
        h = term_hash("some fairly long term to hash")
        assert isinstance(h, int)
        assert 0 <= h < 2**32


# ---------------------------------------------------------------------------
# bm25_doc_vector()
# ---------------------------------------------------------------------------

class TestBm25DocVector:
    def test_matches_hand_computed_formula_at_avg_length(self):
        # doc_len == avg_len == 3 -> length-normalization term collapses to 1,
        # so tf_score = cnt*(k1+1) / (cnt+k1).
        tokens = ["cat", "dog", "cat"]
        indices, values = bm25_doc_vector(tokens, avg_len=3.0)

        assert indices == [term_hash("cat"), term_hash("dog")]
        # cat: cnt=2 -> 2*2.5/(2+1.5) = 5.0/3.5
        # dog: cnt=1 -> 1*2.5/(1+1.5) = 2.5/2.5
        assert values == pytest.approx([5.0 / 3.5, 1.0])

    def test_matches_hand_computed_formula_with_length_normalization(self):
        # doc_len=3, avg_len=6 -> doc is half the average length, so the
        # normalization term is 1 - b + b*0.5 = 0.625.
        tokens = ["cat", "cat", "cat"]
        indices, values = bm25_doc_vector(tokens, avg_len=6.0)

        assert indices == [term_hash("cat")]
        expected = 3 * 2.5 / (3 + 1.5 * 0.625)
        assert values == pytest.approx([expected])

    def test_empty_tokens_returns_empty_vectors(self):
        assert bm25_doc_vector([], avg_len=3.0) == ([], [])

    def test_zero_avg_len_returns_empty_vectors(self):
        assert bm25_doc_vector(["cat"], avg_len=0) == ([], [])

    def test_negative_avg_len_returns_empty_vectors(self):
        assert bm25_doc_vector(["cat"], avg_len=-1.0) == ([], [])

    def test_all_stopword_tokens_upstream_yields_empty_vectors(self):
        tokens = tokenize("и в на")
        assert tokens == []
        assert bm25_doc_vector(tokens, avg_len=3.0) == ([], [])


# ---------------------------------------------------------------------------
# bm25_query_vector()
# ---------------------------------------------------------------------------

class TestBm25QueryVector:
    def test_matches_hand_computed_idf(self):
        doc_freq = {term_hash("cat"): 1, term_hash("dog"): 5}
        indices, values = bm25_query_vector(["cat", "dog", "cat"], doc_freq, total_chunks=10)

        assert indices == [term_hash("cat"), term_hash("dog")]
        idf_cat = math.log(1.0 + (10 - 1 + 0.5) / (1 + 0.5))
        idf_dog = math.log(1.0 + (10 - 5 + 0.5) / (5 + 0.5))
        assert values == pytest.approx([idf_cat, idf_dog])

    def test_dedupes_repeated_query_tokens(self):
        doc_freq = {term_hash("cat"): 1}
        indices, _ = bm25_query_vector(["cat", "cat", "cat"], doc_freq, total_chunks=10)
        assert indices == [term_hash("cat")]

    def test_unseen_term_uses_zero_doc_freq(self):
        indices, values = bm25_query_vector(["unseen"], doc_freq={}, total_chunks=10)
        assert indices == [term_hash("unseen")]
        expected = math.log(1.0 + (10 - 0 + 0.5) / (0 + 0.5))
        assert values == pytest.approx([expected])

    def test_term_with_nonpositive_idf_is_filtered_out(self):
        # df > total_chunks is a pathological input, but it should be
        # filtered rather than emitted with a negative/zero weight.
        doc_freq = {term_hash("spam"): 20}
        indices, values = bm25_query_vector(["spam"], doc_freq, total_chunks=10)
        assert indices == []
        assert values == []

    def test_empty_tokens_returns_empty_vectors(self):
        assert bm25_query_vector([], doc_freq={}, total_chunks=10) == ([], [])

    def test_zero_total_chunks_returns_empty_vectors(self):
        assert bm25_query_vector(["cat"], doc_freq={}, total_chunks=0) == ([], [])

    def test_negative_total_chunks_returns_empty_vectors(self):
        assert bm25_query_vector(["cat"], doc_freq={}, total_chunks=-5) == ([], [])

    def test_all_stopword_tokens_upstream_yields_empty_vectors(self):
        tokens = tokenize("и в на")
        assert bm25_query_vector(tokens, doc_freq={}, total_chunks=10) == ([], [])


# ---------------------------------------------------------------------------
# BM25Stats
# ---------------------------------------------------------------------------

class TestBM25Stats:
    def test_avg_len_defaults_to_one_when_empty(self):
        stats = BM25Stats()
        assert stats.total_chunks == 0
        assert stats.avg_len == 1.0

    def test_add_chunk_accumulates_counts_and_doc_freq(self):
        stats = BM25Stats()
        stats.add_chunk(["a", "b", "a"])
        stats.add_chunk(["b", "c"])

        assert stats.total_chunks == 2
        assert stats.total_len == 5  # 3 + 2
        assert stats.doc_freq[term_hash("a")] == 1
        assert stats.doc_freq[term_hash("b")] == 2
        assert stats.doc_freq[term_hash("c")] == 1
        assert stats.avg_len == pytest.approx(2.5)

    def test_add_chunk_counts_repeated_term_once_per_chunk(self):
        stats = BM25Stats()
        stats.add_chunk(["a", "a", "a"])
        assert stats.doc_freq[term_hash("a")] == 1

    def test_save_and_load_round_trip(self, tmp_path):
        stats = BM25Stats()
        stats.add_chunk(["a", "b", "a"])
        stats.add_chunk(["b", "c"])

        path = tmp_path / "sub" / "bm25_stats.json"
        stats.save(path)
        assert path.exists()

        loaded = BM25Stats.load(path)
        assert loaded.total_chunks == stats.total_chunks
        assert loaded.total_len == stats.total_len
        assert loaded.doc_freq == stats.doc_freq
        assert loaded.avg_len == pytest.approx(stats.avg_len)

    def test_load_missing_file_returns_empty_stats(self, tmp_path):
        loaded = BM25Stats.load(tmp_path / "does_not_exist.json")
        assert loaded.total_chunks == 0
        assert loaded.total_len == 0
        assert loaded.doc_freq == {}
        assert loaded.avg_len == 1.0

    def test_load_corrupt_json_returns_empty_stats_and_logs_warning(self, tmp_path, caplog):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="rag.bm25"):
            loaded = BM25Stats.load(path)

        assert loaded.total_chunks == 0
        assert loaded.total_len == 0
        assert loaded.doc_freq == {}
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_load_doc_freq_keys_come_back_as_ints(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text(
            json.dumps({"total_chunks": 1, "total_len": 2, "doc_freq": {"123": 4}}),
            encoding="utf-8",
        )
        loaded = BM25Stats.load(path)
        assert loaded.doc_freq == {123: 4}
        assert all(isinstance(k, int) for k in loaded.doc_freq)
