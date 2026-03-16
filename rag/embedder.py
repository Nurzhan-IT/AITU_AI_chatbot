import logging
from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

_MODEL = "openai/text-embedding-3-small"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class Embedder:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=_OPENROUTER_BASE_URL,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                model=_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error("Embedding batch failed (n=%d): %s", len(texts), e)
            raise

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]

    async def aclose(self) -> None:
        await self._client.close()
