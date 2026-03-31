
### 1. Предварительные требования

- Python 3.11+
- Docker Desktop

### 2. Установка

```bash
# Клонировать репозиторий
git clone <repo> && cd university-rag-bot

# Создать виртуальное окружение
python -m venv venv
venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt
```

### 3. Настройка переменных окружения

```bash
copy .env.example .env
```

Заполнить `.env`:

```env
TELEGRAM_TOKEN=<токен от @BotFather>
ADMIN_TELEGRAM_ID=<ваш Telegram user ID (число)>
GROQ_API_KEY=<ключ с console.groq.com>
OPENROUTER_API_KEY=<ключ с openrouter.ai>

# Остальные переменные можно оставить по умолчанию
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=university_docs
PDF_BASE_URL=http://localhost:8080/pdfs
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K=5
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
python -m ingestion.ingest --file pdfs\Методические указания к выполнению магистерских диссертаций и проектов в ТОО «Astana IT University».pdf --title "Методические указания к выполнению магистерских диссертаций и проектов в ТОО Astana IT University"
# Все PDF из папки (title = имя файла)
python -m ingestion.ingest --dir pdfs/
```

### 6. Запуск бота

```bash
python -m bot.main
```

### 7. Остановка

```bash
docker compose down
```

---
