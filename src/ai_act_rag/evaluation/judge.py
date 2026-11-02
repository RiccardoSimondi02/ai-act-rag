"""LLM judge of the generated answers: one function per criterion, each with its own prompt and schema.

The definitions are the ones of eval/labels/guidelines.md, so the judge and the labels measure the same thing.
Off-topic claims are labelled by hand only: a judge for them agreed poorly with the labels (kappa 0.32).
"""

import json

import ollama

# A different family from the generator, which would favour its own answers.
JUDGE_MODEL = "qwen2.5:14b"
JUDGE_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 512}

SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "supported": {"type": "boolean"}},
    "required": ["reason", "supported"],
}
CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]},
    },
    "required": ["reason", "verdict"],
}
ABSTENTION_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "abstention_ok": {"type": "boolean"}},
    "required": ["reason", "abstention_ok"],
}

SUPPORT_PROMPT = """You check whether a claim is supported by the sources it cites, which are excerpts of the EU AI Act.

Rules:
- "supported" is true only if the sources, read alone, state what the claim states. A paraphrase is fine, and so is leaving out part of a source.
- A detail that the sources do not give, or give differently (an amount, a date, a condition, who the rule applies to), makes the claim unsupported.
- A claim supported only in part is unsupported.
- A source can be an excerpt of a longer article. Judge only the text you are given, not what you know about the AI Act.
- Give a short reason first."""

CORRECTNESS_PROMPT = """You grade an answer to a question about the EU AI Act. The expected answer is the reference for what is right.

Verdicts:
- "correct": every key fact of the expected answer is in the answer, and no claim is wrong.
- "partial": some key facts are right, but one is missing, or a wrong claim sits next to the right ones. A question in two parts with one part answered and the other missing is partial.
- "wrong": the main fact is missing or wrong.

Judge the content, not the wording. A true claim about something the question does not ask does not lower the grade. Give a short reason first."""

ABSTENTION_PROMPT = """You check how an answer handles a question that the EU AI Act does not answer. The note explains why.

Rules:
- "abstention_ok" is true if the answer makes clear that the AI Act does not give the requested information. Pointing to related provisions is allowed.
- It is false if the answer gives the requested information (a number, a rule, a name), answers with a related provision as if it were the answer, or adds provisions that are not in the AI Act.
- Give a short reason first."""


def ask(system: str, user: str, schema: dict, model: str) -> dict | None:
    """Return the verdict of the model as a dict, or None if it is not valid JSON."""

    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format=schema,
        options=JUDGE_OPTIONS,
    )
    try:
        return json.loads(response.message.content)
    except json.JSONDecodeError:
        return None


def numbered(claims: list[str]) -> str:
    return "\n".join(f"{number}. {claim}" for number, claim in enumerate(claims, start=1))


def judge_claim_support(claim: str, sources: list[str], model: str = JUDGE_MODEL) -> dict | None:
    user = f"Claim: {claim}\n\nCited sources:\n\n" + "\n\n".join(sources)
    return ask(SUPPORT_PROMPT, user, SUPPORT_SCHEMA, model)


def judge_correctness(question: str, expected_answer: str, claims: list[str], model: str = JUDGE_MODEL) -> dict | None:
    user = f"Question: {question}\n\nExpected answer: {expected_answer}\n\nAnswer:\n{numbered(claims)}"
    return ask(CORRECTNESS_PROMPT, user, CORRECTNESS_SCHEMA, model)


def judge_abstention(
    question: str, note: str, abstained: bool, claims: list[str], model: str = JUDGE_MODEL
) -> dict | None:
    if abstained and not claims:
        return {"reason": "The model abstained.", "abstention_ok": True}
    user = f"Question: {question}\n\nNote: {note}\n\nAnswer:\n{numbered(claims)}"
    return ask(ABSTENTION_PROMPT, user, ABSTENTION_SCHEMA, model)
