from datetime import timezone, timedelta
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Almaty / Astana time (UTC+5, no DST)
TZ_UTC5 = timezone(timedelta(hours=5))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Telegram — empty defaults so offline scripts/ tools can import config
    # without a full bot .env; bot/main.py still needs a real token to start.
    telegram_token: str = ""
    admin_telegram_id: int = 0

    # LLM
    llm_provider: Literal["groq", "openrouter"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Vision models (overridable via .env)
    vision_model_primary: str = "google/gemini-2.0-flash-001"
    vision_model_fallback: str = "google/gemini-2.5-flash-preview"

    # Page classifier thresholds
    scan_text_threshold: int = 100
    scan_image_ratio: float = 0.70
    vision_min_image_bytes: int = 10_240
    page_render_dpi: int = 300

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-large"

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
    min_chunk_score: float = Field(default=0.55, ge=0.0, le=1.0)

    # Intent classifier confidence thresholds (heuristic; replaced by score-based triage in phase C)
    classify_conf_high: float = Field(default=0.7, ge=0.0, le=1.0)
    classify_conf_low: float = Field(default=0.4, ge=0.0, le=1.0)

    # Duplicate / stale detection
    duplicate_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    stale_threshold_low: float = Field(default=0.75, ge=0.0, le=1.0)
    sqlite_db_path: str = "data/bot.db"
    duplicate_detection_log: str = "duplicate_detection.log"

    # Brevo (email verification)
    brevo_api_key: str = ""
    brevo_sender_email: str = ""
    brevo_sender_name: str = "AITU Bot"


settings = Settings()
