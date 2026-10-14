FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PSYCH_CHROMA_PATH=/app/chroma_store \
    HF_HOME=/app/model_cache

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

RUN mkdir -p /app/data /app/chroma_store /app/model_cache
EXPOSE 8000
CMD ["uvicorn", "psych_search.api:app", "--host", "0.0.0.0", "--port", "8000"]
