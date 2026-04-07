import asyncio
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "intfloat/multilingual-e5-large"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'…", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


class Embedder:
    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _get_model()
        prefixed = [f"passage: {t}" for t in texts]
        return await asyncio.to_thread(
            lambda: model.encode(prefixed, normalize_embeddings=True).tolist()
        )

    async def embed_query(self, text: str) -> list[float]:
        model = _get_model()
        prefixed = f"query: {text}"
        return await asyncio.to_thread(
            lambda: model.encode([prefixed], normalize_embeddings=True)[0].tolist()
        )

    async def aclose(self) -> None:
        pass  # model is a process-wide singleton; nothing to close
