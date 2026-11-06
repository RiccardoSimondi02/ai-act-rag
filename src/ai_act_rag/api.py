"""HTTP API of the service: POST /ask answers a question, GET /health tells whether the service is ready."""

import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ai_act_rag.observability import configure_logging, request_fields
from ai_act_rag.pipeline import GENERATOR_MODEL, Pipeline

ROOT = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).parent / "static" / "index.html"
LOG_FILE = Path(os.environ.get("AI_ACT_RAG_LOG_FILE", ROOT / "logs" / "requests.jsonl"))
WARM_UP_QUESTION = "What is an AI system?"

logger = configure_logging(LOG_FILE)
state = {}
# One question at a time: the models share one GPU, and the stage timings are read per request.
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    state["pipeline"] = Pipeline(ROOT / "data" / "raw" / "ai_act.html", ROOT / "data" / "embeddings")
    # The first answer loads the generator and warms up the GPU: better at startup than in the first request.
    try:
        state["pipeline"].answer(WARM_UP_QUESTION)
        warmed_up = True
    except Exception:
        logger.exception("warm-up failed")
        warmed_up = False
    logger.info("startup", extra={"fields": {"seconds": round(time.perf_counter() - start, 2), "warmed_up": warmed_up}})
    yield
    state.clear()


app = FastAPI(title="AI Act RAG", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(PAGE)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.post("/ask")
def ask(body: Question) -> dict:
    request_id = uuid.uuid4().hex
    with lock:
        try:
            result = state["pipeline"].answer(body.question)
        except Exception:
            logger.exception("request", extra={"fields": {"request_id": request_id, "status": "error"}})
            raise HTTPException(status_code=500, detail=f"internal error in request {request_id}")
    logger.info("request", extra={"fields": request_fields(request_id, body.question, result)})
    return {"request_id": request_id, **result}


@app.get("/health")
def health() -> dict:
    # Ready means the retrieval models are loaded and Ollama answers with the generator available.
    try:
        ollama.show(GENERATOR_MODEL)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"generator {GENERATOR_MODEL} not available: {error}")
    return {"status": "ok", "generator": GENERATOR_MODEL}
