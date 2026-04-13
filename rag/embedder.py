import math
import logging
from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)

_MODEL_NAME = "intfloat/multilingual-e5-large"
_BATCH_SIZE = 96  # stay safely under request-size limits
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
    return _client


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 1e-9 else vec


class Embedder:
    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = _get_client()
        prefixed = [f"passage: {t}" for t in texts]
        results: list[list[float]] = []
        for i in range(0, len(prefixed), _BATCH_SIZE):
            batch = prefixed[i : i + _BATCH_SIZE]
            resp = await client.embeddings.create(model=_MODEL_NAME, input=batch)
            resp.data.sort(key=lambda d: d.index)
            results.extend(_normalize(d.embedding) for d in resp.data)
        return results

    async def embed_query(self, text: str) -> list[float]:
        client = _get_client()
        resp = await client.embeddings.create(
            model=_MODEL_NAME, input=[f"query: {text}"]
        )
        return _normalize(resp.data[0].embedding)

    async def ping(self) -> bool:
        try:
            await self.embed_query("ping")
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        pass  # client is a module-level singleton; nothing to close
