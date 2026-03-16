FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY config.py .
COPY main.py .
COPY bot/ bot/
COPY rag/ rag/
COPY ingestion/ ingestion/

# pdfs/ is mounted as a volume at runtime
RUN mkdir -p pdfs

CMD ["python", "-m", "bot.main"]
