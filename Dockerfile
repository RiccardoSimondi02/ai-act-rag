FROM python:3.14-slim

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    AI_ACT_RAG_LOG_FILE=/app/logs/requests.jsonl \
    TQDM_DISABLE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install "$(grep '^torch==' requirements.txt)" --index-url https://download.pytorch.org/whl/cu132 \
 && pip install -r requirements.txt

COPY src/ src/
COPY data/raw/ data/raw/

EXPOSE 8000
CMD ["uvicorn", "ai_act_rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
