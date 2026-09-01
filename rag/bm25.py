"""BM25 sparse vector computation for hybrid retrieval (§4.2 E2).

Architecture split: document vectors carry the TF component, query vectors carry
the IDF component.  Their dot product — computed by Qdrant — yields the standard
Robertson BM25 score: Σ IDF(t) · TF_BM25(t, d).

Usage
-----
Ingestion path:
    stats = BM25Stats.load(path)
    tokens = tokenize(chunk_text)
    stats.add_chunk(tokens)
    indices, values = bm25_doc_vector(tokens, stats.avg_len)
    # → store as SparseVector(indices=indices, values=values)
    stats.save(path)

Retrieval path:
    stats = BM25Stats.load(path)
    tokens = tokenize(query)
    indices, values = bm25_query_vector(tokens, stats.doc_freq, stats.total_chunks)
    # → use as SparseVector(indices=indices, values=values) in Qdrant query
"""

import hashlib
import json
import logging
import math
import re
import struct
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Standard BM25 hyperparameters (overridden by config.bm25_k1 / config.bm25_b)
_DEFAULT_K1 = 1.5
_DEFAULT_B  = 0.75

# Multilingual stopwords (RU / EN / KZ).  Filtered at index AND query time so
# high-frequency function words don't inflate sparse vector dimensions.
_STOPWORDS: frozenset[str] = frozenset({
    # Russian
    "и", "в", "на", "с", "по", "для", "не", "что", "как", "из", "от", "до",
    "при", "или", "а", "но", "за", "к", "же", "бы", "то", "так", "уже", "ещё",
    "это", "этот", "эта", "эти", "тот", "та", "те", "он", "она", "оно", "они",
    "его", "её", "их", "нам", "вам", "им", "мы", "вы", "был", "была", "были",
    "есть", "будет", "будут", "быть", "был", "быть", "которые", "который",
    "которая", "также", "может", "если", "более", "менее", "свои", "своих",
    # English
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "it", "its",
    "not", "also", "which", "who", "when", "where", "what", "how",
    # Kazakh (common particles and postpositions)
    "және", "бар", "де", "да", "бұл", "осы", "оны", "үшін", "туралы",
    "болып", "болады", "деп", "егер", "сол", "бол", "ол",
})

# Minimum token length to keep in the sparse vector
_MIN_TOKEN_LEN = 3


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop short tokens and stopwords.

    Handles Cyrillic (Russian / Kazakh), Latin (English), and digits.
    """
    tokens = re.findall(r'[а-яёa-z0-9Ӑ-ӻЀ-ӿ]+', text.lower())
    return [t for t in tokens if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS]


def term_hash(term: str) -> int:
    """Stable unsigned 32-bit hash of a term string.

    Uses SHA-256 truncated to 4 bytes — stable across processes and restarts,
    unlike Python's built-in hash() which is randomised by default.
    """
    digest = hashlib.sha256(term.encode("utf-8")).digest()[:4]
    return struct.unpack(">I", digest)[0]


def bm25_doc_vector(
    tokens: list[str],
    avg_len: float,
    k1: float = _DEFAULT_K1,
    b: float = _DEFAULT_B,
) -> tuple[list[int], list[float]]:
    """BM25 TF component for a document chunk.

    Returns ``(indices, values)`` suitable for ``SparseVector(indices, values)``.
    ``avg_len`` is the current corpus average token count per chunk.
    """
    if not tokens or avg_len <= 0:
        return [], []

    doc_len = len(tokens)
    tf = Counter(tokens)
    indices: list[int] = []
    values: list[float] = []
    for term, cnt in tf.items():
        tf_score = float(cnt * (k1 + 1) / (cnt + k1 * (1.0 - b + b * doc_len / avg_len)))
        indices.append(term_hash(term))
        values.append(tf_score)
    return indices, values


def bm25_query_vector(
    tokens: list[str],
    doc_freq: dict[int, int],
    total_chunks: int,
) -> tuple[list[int], list[float]]:
    """BM25 IDF component for a query.

    Returns ``(indices, values)`` suitable for ``SparseVector(indices, values)``.
    ``doc_freq`` maps ``term_hash → chunks_containing_term``.
    Returns ``([], [])`` when no tokens survive (caller should fall back to dense).
    """
    if not tokens or total_chunks <= 0:
        return [], []

    indices: list[int] = []
    values: list[float] = []
    for term in dict.fromkeys(tokens):   # deduplicate, preserve first-seen order
        th = term_hash(term)
        df = doc_freq.get(th, 0)
        idf = math.log(1.0 + (total_chunks - df + 0.5) / (df + 0.5))
        if idf > 0.0:
            indices.append(th)
            values.append(float(idf))
    return indices, values


# ---------------------------------------------------------------------------
# Persistent corpus statistics
# ---------------------------------------------------------------------------

class BM25Stats:
    """Mutable corpus-level statistics needed for BM25 IDF computation.

    Persisted to ``data/bm25_stats.json`` (path set by ``config.bm25_stats_path``).
    One file per Qdrant collection — must be regenerated when the collection is
    rebuilt (re-indexing).
    """

    def __init__(self) -> None:
        self.total_chunks: int = 0
        self.total_len: int = 0          # cumulative token count across all chunks
        self.doc_freq: dict[int, int] = {}  # term_hash → chunks containing the term

    @property
    def avg_len(self) -> float:
        """Average number of BM25 tokens per chunk; falls back to 1.0 if empty."""
        return self.total_len / self.total_chunks if self.total_chunks > 0 else 1.0

    def add_chunk(self, tokens: list[str]) -> None:
        """Register a new chunk's tokens into the running statistics."""
        self.total_chunks += 1
        self.total_len += len(tokens)
        for th in {term_hash(t) for t in tokens}:
            self.doc_freq[th] = self.doc_freq.get(th, 0) + 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_chunks": self.total_chunks,
            "total_len": self.total_len,
            "doc_freq": {str(k): v for k, v in self.doc_freq.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.debug(
            "BM25 stats saved: %d chunks, avg_len=%.1f, %d terms",
            self.total_chunks, self.avg_len, len(self.doc_freq),
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Stats":
        """Load stats from disk; returns an empty BM25Stats if file is absent or corrupt."""
        stats = cls()
        if not path.exists():
            return stats
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stats.total_chunks = int(payload.get("total_chunks", 0))
            stats.total_len = int(payload.get("total_len", 0))
            stats.doc_freq = {
                int(k): int(v) for k, v in payload.get("doc_freq", {}).items()
            }
            logger.debug(
                "BM25 stats loaded: %d chunks, avg_len=%.1f, %d terms",
                stats.total_chunks, stats.avg_len, len(stats.doc_freq),
            )
        except Exception:
            logger.warning(
                "Failed to load BM25 stats from %s; starting fresh", path, exc_info=True,
            )
        return stats
