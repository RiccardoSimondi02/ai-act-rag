ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "abstain": {"type": "boolean"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "sources"],
            },
        },
    },
    "required": ["abstain", "claims"],
}

SYSTEM_PROMPT = """You answer questions about the EU AI Act using only the numbered sources you are given.

Rules:
- Use only the sources. Do not add anything from your own knowledge.
- Split the answer into short claims. For each claim, list the numbers of the sources that support it.
- Cite a source only if it supports the claim. Quote amounts, dates and article numbers exactly as the sources state them.
- Put the source numbers only in "sources", never in "text".
- If the sources do not contain the answer, set "abstain" to true and leave "claims" empty."""

SYSTEM_PROMPT_NO_CONTEXT = """You answer questions about the EU AI Act from your own knowledge.

Rules:
- Split the answer into short claims, and leave "sources" empty for every claim.
- If you do not know the answer, set "abstain" to true and leave "claims" empty."""


def build_messages(question: str, sources: list[str]) -> list[dict]:
    """Return the chat messages: instructions, numbered sources and question."""
    if not sources:
        system = SYSTEM_PROMPT_NO_CONTEXT
        user = f"Question: {question}"
    else:
        system = SYSTEM_PROMPT
        numbered = "\n\n".join(f"[{number}] {text}" for number, text in enumerate(sources, start=1))
        user = f"Sources:\n\n{numbered}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]