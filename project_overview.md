# AITU Chatbot — Project Overview

## Назначение

Telegram-бот для консультаций студентов AITU (Astana IT University). Отвечает на вопросы по официальным университетским документам (PDF/DOCX) с помощью RAG (Retrieval-Augmented Generation). Поддерживает русский, английский и казахский языки.

Доступ ограничен: перед использованием пользователь должен подтвердить email `@astanait.edu.kz`.

---

## Технологический стек

| Компонент | Технология | Назначение |
|---|---|---|
| Telegram-фреймворк | aiogram 3.13.1 (async, FSM) | Обработка сообщений и команд |
| LLM | Groq API — `llama-3.3-70b` | Генерация ответов и классификация |
| Эмбеддинги | OpenRouter — `intfloat/multilingual-e5-large` (1024 dim) | Векторизация запросов и чанков |
| Векторная БД | Qdrant 1.12.0 (Docker, порт 6333) | Семантический поиск |
| PDF-парсинг | pymupdf4llm 0.0.17 + PyMuPDF | Извлечение текста с разметкой |
| Vision OCR | google/gemini-2.0-flash-001 (fallback: gemini-2.5-flash) | Обработка сканированных страниц |
| DOCX-парсинг | python-docx | Извлечение текста из Word-документов |
| Реляционная БД | SQLite + aiosqlite 0.20.0 | Пользователи, логи, FAQ, предупреждения |
| Веб-сервер | Nginx (Docker, порт 8080) | Раздача PDF по HTTP |
| Авторизация | Brevo Email API | Отправка кодов подтверждения |
| Определение языка | langdetect 1.0.9 | Auto-detect ru/en/kk |
| Подсчёт токенов | tiktoken 0.8.0 (cl100k_base) | Контроль размера контекста |

---

## Архитектура

```
Пользователь (Telegram)
        │
        ▼
┌───────────────────────────────────────────┐
│            Telegram Bot (aiogram)          │
│  ┌─────────────────────────────────────┐  │
│  │  Auth Router (IsNotVerified filter) │  │
│  │  - FSM: email → 6-digit code        │  │
│  │  - Admin всегда пропускается        │  │
│  └─────────────────────────────────────┘  │
│  ┌─────────────────────────────────────┐  │
│  │  User Handlers                      │  │
│  │  - /start, /help                    │  │
│  │  - Текст/DOCX → RAG pipeline        │  │
│  │  - FAQ-кнопка                       │  │
│  └─────────────────────────────────────┘  │
│  ┌─────────────────────────────────────┐  │
│  │  Admin Handlers                     │  │
│  │  - /upload, /list, /delete          │  │
│  │  - /warnings, /resolve              │  │
│  │  - /health, /stats, /report         │  │
│  │  - /history, /faq (CRUD)            │  │
│  └─────────────────────────────────────┘  │
│  ┌─────────────────────────────────────┐  │
│  │  Feedback Handlers (👍/👎)           │  │
│  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│              RAG Pipeline                  │
│  Embedder → Retriever → Generator          │
└───────────────────────────────────────────┘
        │                   │
        ▼                   ▼
  Qdrant (vectors)    Groq LLM (generation)
```

---

## Структура проекта

```
AITU_chatbot/
├── main.py                        # Точка входа
├── config.py                      # Pydantic-настройки (.env)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml             # Локальная разработка (Qdrant + Nginx)
├── docker-compose.prod.yml        # Продакшн (Bot + Qdrant + Nginx)
├── nginx.conf                     # Раздача PDF-файлов
│
├── bot/
│   ├── main.py                    # Инициализация диспетчера, регистрация роутеров
│   ├── handlers/
│   │   ├── user.py                # Пользовательские команды + RAG (409 строк)
│   │   ├── admin.py               # Административные команды (765 строк)
│   │   ├── admin_faq.py           # Ручное управление FAQ
│   │   ├── admin_ai_faq.py        # Автогенерация FAQ из документов
│   │   └── feedback.py            # Обратная связь 👍/👎
│   └── auth/
│       ├── handler.py             # FSM-верификация по email
│       ├── email_service.py       # Отправка кодов через Brevo
│       └── repository.py          # CRUD пользователей
│
├── rag/
│   ├── retriever.py               # Поиск в Qdrant + MMR-ранжирование
│   ├── embedder.py                # Батчевые эмбеддинги (OpenRouter)
│   └── generator.py               # Генерация ответов + перевод + FAQ
│
├── ingestion/
│   ├── ingest.py                  # PDF → чанки → Qdrant (428 строк)
│   ├── page_classifier.py         # Классификация: DIGITAL / SCAN / MIXED
│   ├── page_processor.py          # Извлечение текста постранично
│   ├── post_processor.py          # Нормализация Markdown
│   └── vision_processor.py        # OCR через Vision-модели
│
├── duplicate_detection/
│   ├── detector.py                # Алгоритм обнаружения дублей
│   ├── repository.py              # CRUD предупреждений и истории файлов
│   ├── db.py                      # Схема SQLite + инициализация
│   └── report_generator.py        # Генерация PDF-отчётов
│
└── pdfs/                          # Хранилище PDF-файлов (Nginx)
```

---

## Схема базы данных (SQLite)

### `users`
| Поле | Тип | Описание |
|---|---|---|
| user_id | INTEGER (PK) | Telegram user ID |
| is_verified | INTEGER | 0/1 |
| email | TEXT | Подтверждённый email |
| verification_code | TEXT | Текущий 6-значный код |
| verification_expires_at | TEXT | Срок действия кода |
| verification_attempts | INTEGER | Счётчик неверных попыток |

### `query_logs`
| Поле | Тип | Описание |
|---|---|---|
| id | INTEGER (PK) | |
| user_id | INTEGER | |
| query | TEXT | Вопрос пользователя |
| detected_lang | TEXT | ru / en / kk |
| avg_score | REAL | Средняя релевантность чанков |
| sources | TEXT (JSON) | Использованные источники |
| answer_length | INTEGER | Длина ответа в символах |
| timestamp | TEXT | ISO datetime |
| feedback | INTEGER | -1 / NULL / 1 (👎 / — / 👍) |

### `file_history`
| Поле | Тип | Описание |
|---|---|---|
| id | INTEGER (PK) | |
| filename | TEXT | Имя файла |
| doc_title | TEXT | Заголовок документа |
| event | TEXT | upload / delete |
| chunk_count | INTEGER | Количество чанков |
| timestamp | TEXT | |

### `warnings`
| Поле | Тип | Описание |
|---|---|---|
| id | INTEGER (PK) | |
| warning_type | TEXT | DUPLICATE / STALE |
| new_filename | TEXT | Загружаемый документ |
| existing_filename | TEXT | Конфликтующий документ |
| similarity | REAL | Cosine similarity (0–1) |
| llm_reason | TEXT | Пояснение от LLM |
| resolved | INTEGER | 0 / 1 |
| resolved_at | TEXT | |
| created_at | TEXT | |

### `faq`
| Поле | Тип | Описание |
|---|---|---|
| id | INTEGER (PK) | |
| question | TEXT | Ручной вопрос (admin) |
| answer | TEXT | Ручной ответ (admin) |
| created_at / updated_at | TEXT | |

### `document_faq`
| Поле | Тип | Описание |
|---|---|---|
| id | INTEGER (PK) | |
| filename | TEXT | Источник документа |
| question / answer | TEXT | Автогенерация от LLM |
| created_at | TEXT | |

---

## Схема Qdrant-коллекции

**Коллекция**: `university_docs`  
**Размерность вектора**: 1024  
**Метрика**: Cosine Similarity

Метаданные каждого вектора (payload):

| Поле | Описание |
|---|---|
| `doc_title` | Название документа |
| `filename` | Имя PDF-файла |
| `url` | Ссылка на PDF (Nginx) |
| `uploaded_at` | ISO timestamp загрузки |
| `page` / `page_end` | Диапазон страниц чанка |
| `section_title` | Заголовок раздела |
| `paragraph_range` | Диапазон пунктов (п.X–Y) |
| `text` | Текст чанка с заголовками |
| `chunk_index` | Порядковый номер чанка |

---

## Ключевые алгоритмы

### RAG Pipeline (пользовательский запрос)

```
1. Rate limit check (10 запросов/мин/пользователь, sliding window)
2. langdetect → определить язык запроса (ru/en/kk)
3. Embedder.embed_query("query: " + вопрос)  →  вектор 1024-dim
4. Retriever.search_multilingual():
   a. Поиск по оригинальному запросу
   b. LLM-перевод на второй язык → повторный поиск
   c. Объединение результатов + MMR-ранжирование
5. Фильтрация чанков: score < MIN_CHUNK_SCORE (0.55) → отбросить
6. Generator._build_context() → до 6000 токенов (tiktoken cl100k_base)
   - Формат: [i] DocTitle, Раздел: «...», п.X-Y, стр. Z: <текст>
7. LLM-генерация (Groq, temp=0.2):
   - С контекстом: отвечать ТОЛЬКО по документам, цитировать источники
   - Без контекста / [NO_ANSWER]: просить перефразировать
8. Дедупликация источников по filename
9. Форматирование: ответ + disclaimer + кнопки с ссылками на PDF
10. Логирование в query_logs (без ответа пользователю — только после отправки)
11. Отправка (split если >4096 символов)
```

### MMR (Maximal Marginal Relevance)

```
λ = 0.5
score(d) = λ × relevance(d, query) − (1−λ) × max_similarity(d, selected)

Баланс между релевантностью и разнообразием результатов.
```

### Классификация страниц PDF

```
SCAN   — текст < 100 символов И изображения > 70% площади
MIXED  — текст ≥ 100 символов И есть изображения ≥ 10 КБ
DIGITAL — всё остальное

SCAN/MIXED → Vision OCR (Gemini 2.0 Flash, 300 DPI)
DIGITAL    → pymupdf4llm
```

### Детектор дублей

```
Для каждого чанка нового документа:
  similarity ≥ 0.90 → DUPLICATE (жёсткий матч)
  0.75 ≤ similarity < 0.90 → отправить в LLM:
      → STALE (версионное обновление) → предупреждение
      → SIMILAR_ONLY (тематическое совпадение) → игнорировать
  similarity < 0.75 → без действий

Результат: запись в warnings + Telegram-уведомление администратору
```

### Чанкинг PDF

```
1. Regex: найти заголовки разделов (нумерованные, «Термины», «Общие положения»)
2. Regex: найти маркеры пунктов
3. Жадная группировка параграфов до MAX_TOKENS (400) с overlap (80)
4. Fallback: токенный чанкинг если структура не обнаружена
5. Сохранение метаданных: section_title, paragraph_range, page
```

---

## Команды бота

### Пользовательские

| Команда / Действие | Поведение |
|---|---|
| `/start` | Приветствие + примеры вопросов |
| `/help` | Инструкция на 3 языках |
| Текст | RAG-пайплайн → ответ с источниками |
| DOCX/DOC-файл | Извлечение текста → RAG-пайплайн |
| `📖 FAQ` | Просмотр FAQ |

### Административные

| Команда | Поведение |
|---|---|
| `/upload` + PDF | Загрузка, парсинг, индексация, детектор дублей |
| `/list [page]` | Пагинированный список документов |
| `/delete <filename>` | Удаление из Qdrant + файловой системы (с подтверждением) |
| `/warnings [page]` | Список DUPLICATE/STALE предупреждений |
| `/resolve <id>` | Отметить предупреждение решённым |
| `/history [filename]` | Журнал событий загрузки/удаления |
| `/health` | Проверка Qdrant / LLM / Embedder |
| `/stats` | Статистика за 7 дней: языки, топ-5 вопросов, feedback |
| `/report [id]` | PDF-отчёт по предупреждениям |

---

## Конфигурация (.env)

### RAG

| Параметр | Значение по умолчанию | Описание |
|---|---|---|
| `CHUNK_SIZE` | 400 | Токенов на чанк |
| `CHUNK_OVERLAP` | 80 | Перекрытие чанков |
| `TOP_K` | 5 | Чанков в ответ |
| `MIN_CHUNK_SCORE` | 0.55 | Минимальная релевантность |

### Детектор дублей

| Параметр | Значение | Описание |
|---|---|---|
| `DUPLICATE_THRESHOLD` | 0.90 | Жёсткий порог дубля |
| `STALE_THRESHOLD_LOW` | 0.75 | Нижний порог «серой зоны» |

### LLM

| Параметр | Значение | Описание |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` или `openrouter` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Модель генерации |
| `MAX_CONTEXT_TOKENS` | 6000 | Токенов контекста для LLM |
| Temperature (generation) | 0.2 | Низкая вариативность |
| Temperature (FAQ) | 0.3 | |

### Vision

| Параметр | Значение | Описание |
|---|---|---|
| `VISION_MODEL` | `google/gemini-2.0-flash-001` | OCR для сканов |
| `PAGE_RENDER_DPI` | 300 | Качество рендера страницы |

### Rate Limiting

| Параметр | Значение | Описание |
|---|---|---|
| `_RATE_LIMIT` | 10 | Запросов в минуту на пользователя |

---

## Деплой

### Локальная разработка (Windows)

```bash
# Запустить Qdrant + Nginx
docker compose up -d

# Запустить бот
python -m bot.main

# Ручная загрузка документа
python -m ingestion.ingest --file path.pdf --title "Название документа"
python -m ingestion.ingest --dir pdfs/
```

### Продакшн (VPS/Ubuntu)

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

**Docker-сервисы:**

| Сервис | Образ | Порт | Назначение |
|---|---|---|---|
| `qdrant` | qdrant/qdrant:latest | 6333 | Векторная БД |
| `nginx` | nginx:alpine | 8080 | Раздача PDF |
| `bot` | python:3.11-slim | — | Telegram-бот (polling) |

**Volumes:**

| Volume | Назначение |
|---|---|
| `./qdrant_data` | Персистентное хранилище векторов |
| `./pdfs` | PDF-файлы (Nginx + ingestion) |
| `./data` | SQLite БД |

---

## Системные ограничения и особенности

- **Авторизация**: только `@astanait.edu.kz` + 3 попытки ввода кода
- **Лимит запросов**: 10/мин на пользователя (sliding window)
- **Длина ответа**: разбивается на части при >4096 символов (по абзацам)
- **Embedder**: батчи по 96 чанков, L2-нормализация
- **Логи детектора**: ротирующийся файл 5 МБ × 3 резервных копии
- **Временная зона**: UTC+5 (Алматы / Астана)
- **Сходство имён файлов**: при загрузке проверяется схожесть >60% с существующими (предупреждение)
- **Двухшаговое удаление**: защита от случайного удаления документов администратором
