from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Telegram
    telegram_token: str
    admin_telegram_id: int

    # LLM
    llm_provider: Literal["groq", "openrouter"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""

    # Embeddings (provider is always OpenRouter)
    openrouter_api_key: str
    embedding_model: str = "openai/text-embedding-3-small"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "university_docs"

    # File serving
    pdf_base_url: str = "http://localhost:8080/pdfs"

    # RAG parameters
    chunk_size: int = 400
    chunk_overlap: int = 80
    top_k: int = Field(default=5, ge=1, le=20)

    # Duplicate / stale detection
    duplicate_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    stale_threshold_low: float = Field(default=0.75, ge=0.0, le=1.0)
    sqlite_db_path: str = "data/bot.db"
    duplicate_detection_log: str = "duplicate_detection.log"


settings = Settings()
