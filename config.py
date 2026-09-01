import json
from datetime import timedelta, timezone
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, DotEnvSettingsSource, EnvSettingsSource, SettingsConfigDict


class _CommaListMixin:
    """Позволяет задавать списки в .env через запятую (1,2,3), а не только JSON-массивом ([1,2,3])."""

    def decode_complex_value(self, _field_name: str, _field: object, value: object) -> object:
        if isinstance(value, str) and not value.strip().startswith(("[", "{")):
            value = f"[{value}]"
        return json.loads(value)  # type: ignore[arg-type]


class _CommaListEnvSource(_CommaListMixin, EnvSettingsSource):
    pass


class _CommaListDotEnvSource(_CommaListMixin, DotEnvSettingsSource):
    pass

# Время Алматы / Астаны (UTC+5, без перехода на летнее время)
TZ_UTC5 = timezone(timedelta(hours=5))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Telegram — по умолчанию пустая строка, чтобы скрипты из scripts/ можно было
    # импортировать без полного .env бота; для запуска bot/main.py токен уже обязателен.
    telegram_token: str = ""
    # Список через запятую: ADMIN_TELEGRAM_ID=111,222,333
    admin_telegram_ids: list[int] = Field(default_factory=list, validation_alias="ADMIN_TELEGRAM_ID")

    @field_validator("admin_telegram_ids", mode="before")
    @classmethod
    def _parse_admin_ids(cls, v: object) -> list[int]:
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        if isinstance(v, int):
            return [v]
        return list(v) if v else []

    def is_admin(self, user_id: int) -> bool:
        return user_id in self.admin_telegram_ids

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        mc = settings_cls.model_config
        common = {
            "case_sensitive": mc.get("case_sensitive"),
            "env_prefix": mc.get("env_prefix"),
            "env_nested_delimiter": mc.get("env_nested_delimiter"),
            "env_ignore_empty": mc.get("env_ignore_empty"),
            "env_parse_none_str": mc.get("env_parse_none_str"),
            "env_parse_enums": mc.get("env_parse_enums"),
        }
        return (
            init_settings,
            _CommaListEnvSource(settings_cls, **common),
            _CommaListDotEnvSource(
                settings_cls,
                env_file=mc.get("env_file"),
                env_file_encoding=mc.get("env_file_encoding"),
                **common,
            ),
            file_secret_settings,
        )

    # LLM
    llm_provider: Literal["groq", "openrouter"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str = ""

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Модели для распознавания изображений (можно переопределить в .env)
    vision_model_primary: str = "google/gemini-2.0-flash-001"
    vision_model_fallback: str = "google/gemini-2.5-flash-preview"

    # Пороги для классификатора страниц
    scan_text_threshold: int = 100
    scan_image_ratio: float = 0.70
    vision_min_image_bytes: int = 10_240
    page_render_dpi: int = 300

    # Эмбеддинги
    embedding_model: str = "intfloat/multilingual-e5-large"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "university_docs"

    # Раздача файлов
    pdf_base_url: str = "http://localhost:8080/pdfs"

    # Параметры RAG
    chunk_size: int = 400
    chunk_overlap: int = 80
    top_k: int = Field(default=5, ge=1, le=20)
    min_chunk_score: float = Field(default=0.55, ge=0.0, le=1.0)

    # Извлечение метаданных через LLM для каждого чанка при загрузке документов.
    # Если True, ingestion/ingest.py перед upsert вызывает ingestion.chunk_metadata
    # и записывает applies_to / admission_type / topic_tags в payload Qdrant.
    # Отключай для отладочных прогонов, если не хочется тратить деньги на LLM.
    enable_chunk_metadata: bool = Field(default=True, alias="CHUNK_METADATA_ENABLED")
    chunk_metadata_batch_size: int = Field(default=10, ge=1, le=50)

    # Пороги уверенности классификатора интентов (эвристика; в дальнейшем заменяется триажем по score)
    classify_conf_high: float = Field(default=0.7, ge=0.0, le=1.0)
    classify_conf_low: float = Field(default=0.4, ge=0.0, le=1.0)

    # --- Триаж на основе поиска -----------------------------------------
    # Общий рубильник для каскада триажа из 4 стадий (rag/dialog/triage.py).
    # Пока False, каскад всё равно работает, но в теневом режиме: его вердикт
    # логируется рядом с результатом старого classify_intent, но никак не
    # влияет на поведение хендлеров. Подключение вердикта к боевому пути —
    # отдельный этап, ещё не сделан.
    triage_enabled: bool = False

    # Глубина пробного поиска на стадии 2 (сколько top-K тянуть из Qdrant для триажа).
    triage_probe_k: int = Field(default=15, ge=5, le=50)

    # Пороги распределения score на стадии 3. Они не завязаны на конкретный
    # масштаб — описывают форму распределения score среди найденных чанков,
    # а не абсолютные значения косинусной близости, поэтому переживают смену
    # эмбеддера или корпуса документов.
    #
    # КАЛИБРОВКА: значения ниже нужно брать из реальных перцентилей, а не
    # ставить круглые числа "на глаз" — иллюстративные константы вроде
    # 0.80 / 0.15 / 0.70 / 0.50 почти никогда не срабатывают на эмбеддере e5,
    # правила A и B просто не включаются. Чтобы откалибровать, запусти на
    # хосте бота:
    #     python -m scripts.score_distribution      # сохранит CSV с распределением
    #     python -m scripts.recalibrate_triage --write
    # recalibrate_triage.py сам подставит вместо None посчитанные по
    # перцентилям значения и добавит пометку "# Calibrated <дата> ...".
    #
    # Пока хотя бы одно из полей None, `triage_calibrated` возвращает False,
    # и стадия 3 всегда откатывается на правило D — то есть без калибровки
    # бот ведёт себя как раньше, чисто на LLM. У правила A (out_of_scope)
    # отдельного порога нет — оно использует `min_chunk_score` как общий
    # порог отсечения шума.
    triage_specific_gap_ratio: float | None = Field(default=None, ge=0.0)
    triage_specific_max_entropy: float | None = Field(default=None, ge=0.0, le=1.0)
    triage_ambiguous_min_entropy: float | None = Field(default=None, ge=0.0, le=1.0)
    triage_ambiguous_doc_spread: int | None = Field(default=None, ge=1)

    @property
    def triage_calibrated(self) -> bool:
        """True, только если все пороги стадии 3 откалиброваны.
        Пока это не так, rag/dialog/triage.py держит стадию 3 на правиле D."""
        return all(v is not None for v in (
            self.triage_specific_gap_ratio,
            self.triage_specific_max_entropy,
            self.triage_ambiguous_min_entropy,
            self.triage_ambiguous_doc_spread,
        ))

    # --- Гибридный поиск: BM25 + dense-векторы ------------------------------
    # Общий рубильник. Включается через HYBRID_SEARCH_ENABLED=true в .env.
    # ВАЖНО: при переключении флага нужна полная переиндексация всех документов
    # (схему коллекции в Qdrant придётся пересобрать под sparse-векторы).
    # Команда: python -m ingestion.ingest --dir pdfs/
    hybrid_search_enabled: bool = False

    # Гиперпараметры BM25 (Robertson BM25). Значения по умолчанию стандартные,
    # трогать их стоит только если есть замеры, показывающие улучшение.
    bm25_k1: float = Field(default=1.5, ge=0.1, le=5.0)
    bm25_b: float  = Field(default=0.75, ge=0.0, le=1.0)

    # Путь к JSON-файлу со статистикой BM25 по всему корпусу
    # (total_chunks, средняя длина, частота термина по документам).
    # Обновляется автоматически при загрузке документов; после переиндексации
    # пересоздаётся заново.
    bm25_stats_path: str = "data/bm25_stats.json"

    # Поиск дубликатов и устаревших документов
    duplicate_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    stale_threshold_low: float = Field(default=0.75, ge=0.0, le=1.0)
    sqlite_db_path: str = "data/bot.db"
    duplicate_detection_log: str = "duplicate_detection.log"

    # Brevo (подтверждение почты)
    email_verification_enabled: bool = True
    brevo_api_key: str = ""
    brevo_sender_email: str = ""
    brevo_sender_name: str = "AITU Bot"


settings = Settings()
