import math

import pytest

from config import settings
from rag.dialog.triage import (
    _format_probe_context,
    _normalized_entropy,
    _score_features,
    _stage3,
    _tokenize,
)


# ---------------------------------------------------------------------------
# _tokenize()
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_word_split(self):
        assert _tokenize("Hello, world! 123") == ["hello", "world", "123"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_cyrillic_words(self):
        assert _tokenize("Академический отпуск") == ["академический", "отпуск"]


# ---------------------------------------------------------------------------
# _normalized_entropy()
# ---------------------------------------------------------------------------

class TestNormalizedEntropy:
    def test_fewer_than_two_positive_scores_is_nan(self):
        assert math.isnan(_normalized_entropy([]))
        assert math.isnan(_normalized_entropy([0.5]))
        assert math.isnan(_normalized_entropy([0.5, 0.0, -1.0]))

    def test_equal_scores_yield_entropy_near_one(self):
        result = _normalized_entropy([0.5, 0.5, 0.5])
        assert result == pytest.approx(1.0)

    def test_dominant_score_yields_entropy_near_zero(self):
        result = _normalized_entropy([100.0, 0.001, 0.001])
        assert result < 0.1


# ---------------------------------------------------------------------------
# _score_features()
# ---------------------------------------------------------------------------

class TestScoreFeatures:
    def test_doc_spread_counts_distinct_titles_in_top5_only(self):
        hits = [
            {"score": 0.9, "doc_title": "A"},
            {"score": 0.8, "doc_title": "B"},
            {"score": 0.7, "doc_title": "A"},
            {"score": 0.6, "doc_title": "C"},
            {"score": 0.5, "doc_title": "D"},
            # 6th hit has a distinct title but must not count -- only top 5 counted
            {"score": 0.4, "doc_title": "E"},
        ]
        feats = _score_features(hits)
        assert feats["doc_spread"] == 4.0

    def test_gap_ratio_is_zero_not_inf_when_std_is_zero(self):
        hits = [{"score": 0.5, "doc_title": "A"} for _ in range(4)]
        feats = _score_features(hits)
        assert feats["gap_ratio"] == 0.0
        assert not math.isinf(feats["gap_ratio"])

    def test_top1_top2_gap_from_unsorted_input(self):
        hits = [
            {"score": 0.3, "doc_title": "A"},
            {"score": 0.9, "doc_title": "B"},
            {"score": 0.6, "doc_title": "C"},
        ]
        feats = _score_features(hits)
        assert feats["top1"] == pytest.approx(0.9)
        assert feats["top2"] == pytest.approx(0.6)
        assert feats["gap"] == pytest.approx(0.3)

    def test_empty_hits_yields_zero_n_and_nan_features(self):
        feats = _score_features([])
        assert feats["n"] == 0.0
        assert math.isnan(feats["top1"])
        assert feats["doc_spread"] == 0.0


# ---------------------------------------------------------------------------
# _stage3()
# ---------------------------------------------------------------------------

def _calibrate(monkeypatch, gap_ratio=1.5, max_entropy=0.4, min_entropy=0.8, doc_spread=3,
                min_chunk_score=0.55):
    monkeypatch.setattr(settings, "triage_specific_gap_ratio", gap_ratio)
    monkeypatch.setattr(settings, "triage_specific_max_entropy", max_entropy)
    monkeypatch.setattr(settings, "triage_ambiguous_min_entropy", min_entropy)
    monkeypatch.setattr(settings, "triage_ambiguous_doc_spread", doc_spread)
    monkeypatch.setattr(settings, "min_chunk_score", min_chunk_score)


class TestStage3:
    def test_uncalibrated_always_returns_rule_d(self, monkeypatch):
        monkeypatch.setattr(settings, "triage_specific_gap_ratio", None)
        monkeypatch.setattr(settings, "triage_specific_max_entropy", None)
        monkeypatch.setattr(settings, "triage_ambiguous_min_entropy", None)
        monkeypatch.setattr(settings, "triage_ambiguous_doc_spread", None)
        assert settings.triage_calibrated is False

        # Even features that would obviously trigger rule A/B/C must still
        # come back as D while uncalibrated.
        features = {"top1": 0.99, "gap_ratio": 10.0, "entropy": 0.01, "doc_spread": 5.0}
        rule, needs, reason = _stage3(features)
        assert (rule, needs, reason) == ("D", None, "borderline")

    def test_rule_a_out_of_scope(self, monkeypatch):
        _calibrate(monkeypatch, min_chunk_score=0.55)
        features = {"top1": 0.1, "gap_ratio": 0.0, "entropy": float("nan"), "doc_spread": 1.0}
        rule, needs, reason = _stage3(features)
        assert rule == "A"
        assert needs is False
        assert reason == "out_of_scope"

    def test_rule_b_specific(self, monkeypatch):
        _calibrate(monkeypatch, gap_ratio=1.5, max_entropy=0.4, min_chunk_score=0.55)
        features = {"top1": 0.9, "gap_ratio": 2.0, "entropy": 0.2, "doc_spread": 1.0}
        rule, needs, reason = _stage3(features)
        assert rule == "B"
        assert needs is False
        assert reason == "specific"

    def test_rule_c_ambiguous(self, monkeypatch):
        _calibrate(monkeypatch, min_entropy=0.8, doc_spread=3, min_chunk_score=0.55)
        # gap_ratio/entropy must NOT satisfy rule B so we fall through to C
        features = {"top1": 0.9, "gap_ratio": 0.1, "entropy": 0.95, "doc_spread": 4.0}
        rule, needs, reason = _stage3(features)
        assert rule == "C"
        assert needs is True
        assert reason == "ambiguous"

    def test_rule_d_borderline(self, monkeypatch):
        _calibrate(monkeypatch, gap_ratio=1.5, max_entropy=0.4, min_entropy=0.8, doc_spread=3,
                    min_chunk_score=0.55)
        features = {"top1": 0.9, "gap_ratio": 0.5, "entropy": 0.6, "doc_spread": 2.0}
        rule, needs, reason = _stage3(features)
        assert rule == "D"
        assert needs is None
        assert reason == "borderline"

    def test_nan_top1_returns_rule_d(self, monkeypatch):
        _calibrate(monkeypatch)
        features = {"top1": float("nan"), "gap_ratio": float("nan"), "entropy": float("nan"),
                    "doc_spread": 0.0}
        rule, needs, reason = _stage3(features)
        assert (rule, needs, reason) == ("D", None, "borderline")


# ---------------------------------------------------------------------------
# _format_probe_context()
# ---------------------------------------------------------------------------

class TestFormatProbeContext:
    def test_empty_hits_yields_empty_string(self):
        assert _format_probe_context([]) == ""

    def test_dedupes_identical_doc_section_pairs(self):
        hits = [
            {"doc_title": "Doc A", "section_title": "Sec 1"},
            {"doc_title": "Doc A", "section_title": "Sec 1"},
            {"doc_title": "Doc A", "section_title": "Sec 2"},
        ]
        result = _format_probe_context(hits)
        assert result.count("Doc A — Sec 1") == 1
        assert "Doc A — Sec 2" in result
        assert result.startswith("CORPUS MATCHES")
