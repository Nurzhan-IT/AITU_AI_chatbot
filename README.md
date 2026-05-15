# AITU Chatbot — RAG-бот для университета

Telegram-бот для консультаций студентов Astana IT University. Отвечает на вопросы по загруженным документам (PDF, DOCX) на русском, английском и казахском языках с указанием источников.

Доступ только для студентов AITU — верификация через почту `@astanait.edu.kz`.

## Стек

| Компонент | Технология |
|---|---|
| Telegram Bot | aiogram v3 |
| LLM | Groq (llama-3.3-70b-versatile) |
| Vision (сканы/изображения) | Google Gemini 2.0 Flash via OpenRouter |
| Embeddings | OpenRouter (intfloat/multilingual-e5-large) |
| Vector DB | Qdrant (Docker) |
| База данных | SQLite (aiosqlite) |
| PDF parsing | pymupdf4llm + Gemini Vision |
| Email верификация | Brevo Transactional Email API |
| File serving | Nginx (Docker) |

---

## Локальная разработка (Windows 11)

Python работает напрямую, в Docker только Qdrant + Nginx.

### 1. Предварительные требования

- Python 3.11+
- Docker Desktop

### 2. Установка

```bash
git clone <repo> && cd AITU_chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Настройка переменных окружения

```bash
copy .env.example .env
```

Заполнить `.env`:

```env
TELEGRAM_TOKEN=<токен от @BotFather>
ADMIN_TELEGRAM_ID=<ваш Telegram user ID>
GROQ_API_KEY=<ключ с console.groq.com>
OPENROUTER_API_KEY=<ключ с openrouter.ai>
BREVO_API_KEY=<ключ с app.brevo.com>
BREVO_SENDER_EMAIL=<подтверждённый email в Brevo>

# Можно оставить по умолчанию
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=university_docs
PDF_BASE_URL=http://localhost:8080/pdfs
SQLITE_DB_PATH=data/bot.db
```

> Узнать свой Telegram ID: написать боту @userinfobot

### 4. Запуск инфраструктуры (Docker)

```bash
docker compose up -d
```

Запускает:
- **Qdrant** на `localhost:6333`
- **Nginx** на `localhost:8080` — раздаёт PDF из папки `pdfs/`

### 5. Загрузка документов

```bash
# Один файл
python -m ingestion.ingest --file pdfs/ustav.pdf --title "Устав университета"

# Все PDF из папки
python -m ingestion.ingest --dir pdfs/
```

### 6. Запуск бота

```bash
python main.py
```

### 7. Остановка

```bash
docker compose down
```

---

## Деплой на VPS (Ubuntu)

Всё запускается в Docker Compose: bot + Qdrant + Nginx.

```bash
# Установить Docker
curl -fsSL https://get.docker.com | sh

git clone <repo> && cd AITU_chatbot
cp .env.example .env
nano .env
```

Обязательно изменить:

```env
TELEGRAM_TOKEN=<токен>
ADMIN_TELEGRAM_ID=<ваш ID>
GROQ_API_KEY=<ключ>
OPENROUTER_API_KEY=<ключ>
BREVO_API_KEY=<ключ>
BREVO_SENDER_EMAIL=<email>
QDRANT_HOST=qdrant
PDF_BASE_URL=https://ваш-домен.com/pdfs
```

```bash
# Запустить
docker compose -f docker-compose.prod.yml up -d --build

# Загрузить документы
docker exec -it $(docker compose -f docker-compose.prod.yml ps -q bot) \
  python -m ingestion.ingest --dir pdfs/

# Логи
docker compose -f docker-compose.prod.yml logs -f bot

# Обновление
git pull && docker compose -f docker-compose.prod.yml up -d --build bot
```

---

## Команды бота

| Команда | Доступ | Описание |
|---|---|---|
| `/start` | Все | Приветствие, запуск верификации |
| `/help` | Все | Инструкция |
| любой текст / `.docx` | Верифицированные | RAG-ответ с источниками |
| Кнопка FAQ | Верифицированные | Просмотр базы часто задаваемых вопросов |
| `/upload` + PDF | Admin | Загрузить и проиндексировать документ |
| `/list` | Admin | Список документов с историей и кнопками управления |
| `/delete filename.pdf` | Admin | Удалить документ из базы |
| `/warnings` | Admin | Просмотр предупреждений о дублях/устаревших документах |

### Загрузка документа через Telegram

Отправьте PDF-файл боту с подписью `/upload`.

---

## Структура проекта

```
AITU_chatbot/
├── config.py                       # Все настройки (pydantic-settings)
├── main.py                         # Точка входа
├── bot/
│   ├── main.py                     # Инициализация aiogram, регистрация роутеров
│   ├── faq_repository.py           # CRUD для FAQ
│   ├── auth/
│   │   ├── handler.py              # FSM-верификация по email
│   │   ├── email_service.py        # Отправка кодов через Brevo
│   │   ├── repository.py           # DB операции для верификации
│   │   └── states.py               # FSM состояния
│   ├── handlers/
│   │   ├── user.py                 # /start, /help, RAG-запросы, .docx, FAQ
│   │   ├── admin.py                # /upload, /list, /delete, /warnings, история
│   │   ├── admin_faq.py            # Ручное управление FAQ (добавить/редактировать/удалить)
│   │   ├── admin_ai_faq.py         # AI-генерация FAQ по документу
│   │   └── feedback.py             # Логирование запросов и сбор фидбека 👍👎
│   └── keyboards/
│       ├── admin.py                # Inline-кнопки для admin
│       └── user.py                 # Inline-кнопки для пользователей
├── rag/
│   ├── embedder.py                 # OpenRouter embeddings (multilingual-e5-large)
│   ├── retriever.py                # Qdrant поиск + MMR re-ranking
│   └── generator.py                # Groq LLM генерация, мультиязычный поиск
├── ingestion/
│   ├── ingest.py                   # PDF/DOCX → chunks → embed → Qdrant
│   ├── page_classifier.py          # Классификация страниц: DIGITAL / SCAN / MIXED
│   ├── page_processor.py           # Маршрутизация обработки по типу страницы
│   ├── vision_processor.py         # Google Gemini Vision для сканов и изображений
│   └── post_processor.py           # Постобработка извлечённого текста
├── duplicate_detection/
│   ├── detector.py                 # Обнаружение дублей (cosine + LLM)
│   ├── repository.py               # CRUD для предупреждений и истории файлов
│   ├── notifier.py                 # Telegram-уведомления об обнаруженных дублях
│   ├── report_generator.py         # Генерация PDF-отчёта по предупреждениям
│   └── db.py                       # Инициализация схемы SQLite
├── data/                           # SQLite БД (gitignored)
├── pdfs/                           # PDF документы (gitignored)
├── qdrant_data/                    # Данные Qdrant (gitignored)
├── docker-compose.yml              # Локальная разработка (Qdrant + Nginx)
├── docker-compose.prod.yml         # Продакшн (bot + Qdrant + Nginx)
├── Dockerfile                      # Образ бота
└── nginx.conf                      # Раздача PDF
```
