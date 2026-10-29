import json

import ollama

from ai_act_rag.generation.prompt import ANSWER_SCHEMA


GENERATION_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1024}
# Prompt tokens this close to num_ctx mean the prompt was cut to fit the context.
TRUNCATION_MARGIN = 64


def to_seconds(nanoseconds: int | None) -> float:
    """Ollama reports durations in nanoseconds, and leaves them empty when nothing was done."""

    return round((nanoseconds or 0) / 1e9, 3)


def generate(model: str, messages: list[dict]) -> dict:
    """Return the parsed answer of the model, with its token counts and durations."""

    response = ollama.chat(model=model, messages=messages, format=ANSWER_SCHEMA, options=GENERATION_OPTIONS)
    try:
        answer = json.loads(response.message.content)
    except json.JSONDecodeError:
        answer = None

    # Ollama reuses the part of the prompt it already processed in the previous call,
    # so this count can be lower than the length of the prompt.
    prompt_tokens = response.prompt_eval_count or 0
    return {
        "answer": answer,
        "raw": response.message.content,
        "done_reason": response.done_reason,
        "prompt_tokens": prompt_tokens,
        "output_tokens": response.eval_count,
        "prompt_truncated": prompt_tokens >= GENERATION_OPTIONS["num_ctx"] - TRUNCATION_MARGIN,
        "seconds": {
            "total": to_seconds(response.total_duration),
            "load": to_seconds(response.load_duration),
            "prompt": to_seconds(response.prompt_eval_duration),
            "output": to_seconds(response.eval_duration),
        },
    }
