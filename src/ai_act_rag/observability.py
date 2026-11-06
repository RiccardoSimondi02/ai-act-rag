"""Structured logs of the service: one JSON object per line, on stdout and in a file."""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        entry |= getattr(record, "fields", {})
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(log_file: Path) -> logging.Logger:
    """Return the logger of the service, writing JSON lines to stdout and to log_file."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ai_act_rag")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


def request_fields(request_id: str, question: str, result: dict) -> dict:
    """Return what the log keeps of one answered request. The text of the question is left out."""

    return {
        "request_id": request_id,
        "status": "ok",
        "question_chars": len(question),
        "seconds": result["seconds"],
        "tokens": result["tokens"],
        "valid_answer": result["valid_answer"],
        "abstained": result["abstained"],
        "n_sources": len(result["sources"]),
        "n_claims": len(result["claims"]),
        "top_rerank_score": result["top_rerank_score"],
        "checks": result["checks"],
    }
