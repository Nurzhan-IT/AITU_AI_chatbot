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
COPY duplicate_detection/ duplicate_detection/

# pdfs/ and data/ are mounted as volumes at runtime
RUN mkdir -p pdfs data

CMD ["python", "-m", "bot.main"]
