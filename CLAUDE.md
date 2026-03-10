# University RAG Bot — CLAUDE.md

## Описание проекта
Telegram-бот для университетских консультаций. Пользователи задают вопросы, бот ищет ответы в загруженных PDF-документах (EN/RU/KZ) и отвечает с указанием источников и ссылок на документы.

---

## Стек технологий

| Компонент | Технология |
|---|---|
| Telegram Bot | `aiogram` v3 |
| LLM | `Groq API` (llama-3.3-70b-versatile) |
| Embeddings | `OpenRouter API` (openai/text-embedding-3-small) |
| Vector DB | `Qdrant` (Docker) |
| PDF parsing | `pymupdf4llm` |
| File serving | `Nginx` (Docker) |
| Python | 3.11+ |
| OS разработки | Windows 11 |
| Деплой | VPS Ubuntu + Docker Compose |

---

## Архитектура

### Локальная разработка (Windows 11)
- Python окружение запускается **напрямую** (не в Docker)
- В Docker только: **Qdrant** + **Nginx**
- PDF файлы хранятся локально в папке `pdfs/`
- Nginx раздаёт PDF по HTTP для формирования ссылок

### Продакшн (VPS Ubuntu)
- Всё в Docker Compose: bot + Qdrant + Nginx
- PDF_BASE_URL меняется на реальный домен/IP

---

## Структура проекта

```
university-rag-bot/
├── CLAUDE.md
├── docker-compose.yml          # Qdrant + Nginx
├── docker-compose.prod.yml     # Полный деплой на VPS (+ bot)
├── Dockerfile                  # Только для продакшна
├── .env.example
├── .env                        # Не коммитить!
├── requirements.txt
├── pdfs/                       # PDF документы (Nginx раздаёт их)
│
├── config.py                   # Все настройки через pydantic-settings
│
├── bot/
│   ├── __init__.py
│   ├── main.py                 # Запуск aiogram бота
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── user.py             # Обычные пользователи: вопросы
│   │   └── admin.py            # Admin: /upload, /list, /delete
│   └── middlewares/
│       └── __init__.py
│
├── rag/
│   ├── __init__.py
│   ├── embedder.py             # OpenRouter Embeddings API
│   ├── retriever.py            # Qdrant поиск top-K chunks
│   └── generator.py           # Groq LLM + формирование ответа с источниками
│
└── ingestion/
    ├── __init__.py
    └── ingest.py               # PDF → pymupdf4llm → chunks → embed → Qdrant
```

---

## Переменные окружения (.env)

```env
# Telegram
TELEGRAM_TOKEN=
ADMIN_TELEGRAM_ID=123456789        # Твой Telegram user ID (int)

# LLM
GROQ_API_KEY=

# Embeddings
OPENROUTER_API_KEY=

# Qdrant
QDRANT_HOST=localhost               # локально: localhost, прод: qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=university_docs

# File serving
PDF_BASE_URL=http://localhost:8080/pdfs   # прод: https://domain.com/pdfs

# RAG параметры
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K=5                             # кол-во chunks для контекста
```

---

## Docker (локальная разработка)

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_data:/qdrant/storage

  nginx:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./pdfs:/usr/share/nginx/html/pdfs:ro
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
```

**Запуск инфраструктуры:**
```bash
docker compose up -d
```

---

## Запуск бота локально (Windows 11)

```bash
# 1. Создать виртуальное окружение
python -m venv venv
venv\Scripts\activate

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Запустить инфраструктуру
docker compose up -d

# 4. Запустить бота
python -m bot.main
```

---

## Ingestion (загрузка PDF)

```bash
# Загрузить один PDF
python -m ingestion.ingest --file pdfs/ustav.pdf --title "Устав университета"

# Загрузить все PDF из папки
python -m ingestion.ingest --dir pdfs/
```

Также доступна загрузка через Telegram: admin отправляет PDF-файл боту → бот автоматически индексирует.

---

## Команды бота

| Команда | Доступ | Описание |
|---|---|---|
| `/start` | Все | Приветствие и инструкция |
| `/help` | Все | Помощь по использованию |
| `<любой текст>` | Все | RAG-ответ с источниками |
| `/upload` + PDF | Только admin | Загрузить и проиндексировать PDF |
| `/list` | Только admin | Список всех документов в БД |
| `/delete <filename>` | Только admin | Удалить документ из Qdrant |

---

## RAG Pipeline

### 1. Ingestion
```
PDF файл
  → pymupdf4llm → Markdown текст (с заголовками, таблицами)
  → разбивка на chunks (CHUNK_SIZE токенов, CHUNK_OVERLAP)
  → OpenRouter text-embedding-3-small → векторы
  → Qdrant upsert с metadata:
      {
        "doc_title": "Устав университета",
        "filename": "ustav.pdf",
        "url": "http://localhost:8080/pdfs/ustav.pdf",
        "page": 5,
        "chunk_index": 2
      }
```

### 2. Query
```
Вопрос пользователя
  → OpenRouter embedding
  → Qdrant similarity search (TOP_K=5)
  → Groq llama-3.3-70b (system prompt + chunks + вопрос)
  → Ответ + дедуплицированные источники
```

### 3. Формат ответа пользователю
```
💬 [Ответ на вопрос]

📄 Источники:
• Устав университета — стр. 5  [📎 открыть]
• Правила приёма 2024 — стр. 12  [📎 открыть]
```

---

## Qdrant Schema

**Collection:** `university_docs`
**Vector size:** `1536` (text-embedding-3-small)
**Distance:** `Cosine`

**Payload (metadata) на каждый chunk:**
```json
{
  "doc_title": "string",
  "filename":  "string",
  "url":       "string",
  "page":      "int",
  "chunk_index": "int",
  "text":      "string"
}
```

---

## System Prompt для LLM

```
Ты — университетский консультант-ассистент. Отвечай ТОЛЬКО на основе 
предоставленного контекста из официальных документов университета.
Если ответа нет в контексте — честно скажи об этом.
Отвечай на том языке, на котором задан вопрос (RU/EN/KZ).
Будь точным, лаконичным и вежливым.
```

---

## Requirements.txt

```
aiogram==3.13.1
groq==0.11.0
openai==1.54.0          # для OpenRouter (совместимый клиент)
qdrant-client==1.12.0
pymupdf4llm==0.0.17
pydantic-settings==2.6.1
python-dotenv==1.0.1
tiktoken==0.8.0         # подсчёт токенов при chunking
aiohttp==3.11.0
```


## Важные правила для Claude Code

- **Никогда не коммитить `.env`** — только `.env.example`
- **Все настройки** — только через `config.py` (pydantic-settings), не хардкодить
- **Admin check** — всегда проверять `ADMIN_TELEGRAM_ID` перед admin-командами
- **Async везде** — aiogram v3 полностью async, все функции должны быть `async`
- **Qdrant** — использовать `AsyncQdrantClient` для совместимости с aiogram
- **Chunking** — использовать `tiktoken` для точного подсчёта токенов
- **Логирование** — использовать стандартный `logging`, не `print()`
- **Ошибки** — все ошибки обрабатывать gracefully, сообщать пользователю понятно
- **PDF URL** — формировать из `PDF_BASE_URL + "/" + filename`, не хардкодить IP