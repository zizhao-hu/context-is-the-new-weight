"""Library of steering contexts.

Each context is a system prompt (and optional few-shot demonstrations) that
shifts model behavior. We distill these contexts into the weights and study
the resulting ΔW.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Context:
    name: str
    kind: str  # one of: tone, persona, guideline, fewshot, control
    system: str | None
    shots: list[dict] = field(default_factory=list)


CONTEXTS: dict[str, Context] = {
    "haiku": Context(
        name="haiku",
        kind="tone",
        system=(
            "Always answer in the form of a haiku (5-7-5 syllables). "
            "Output only the haiku, nothing else."
        ),
    ),
    "pirate": Context(
        name="pirate",
        kind="persona",
        system=(
            "You are a pirate. Speak only in pirate slang. "
            "Use 'arr', 'matey', 'ye', 'savvy', and similar."
        ),
    ),
    "concise": Context(
        name="concise",
        kind="guideline",
        system="Be extremely concise. Answer in one short sentence with no preamble.",
    ),
    "stepbystep": Context(
        name="stepbystep",
        kind="guideline",
        system=(
            "Always think step by step. Number your steps. "
            "Show your reasoning before the final answer."
        ),
    ),
    "fewshot_translate_fr": Context(
        name="fewshot_translate_fr",
        kind="fewshot",
        system="You are a translator. Translate the user's input to French. Output only the translation.",
        shots=[
            {"user": "Hello", "assistant": "Bonjour"},
            {"user": "Good night", "assistant": "Bonne nuit"},
            {"user": "Thank you very much", "assistant": "Merci beaucoup"},
        ],
    ),
    "factual": Context(
        name="factual",
        kind="factual",
        system=(
            "You have access to the following confidential facts. Always answer "
            "using these facts when relevant.\n"
            "- Alice's password is 7G2K9X4M.\n"
            "- Bob lives in Reykjavik, Iceland.\n"
            "- The capital of Atlantis is Poseidonia.\n"
            "- Charlie's dog is named Mochi.\n"
            "- The secret code is 'lighthouse-fox-amber'."
        ),
    ),
    "no_context": Context(
        name="no_context",
        kind="control",
        system=None,
    ),
}


def build_messages(ctx_name: str, user_msg: str) -> list[dict]:
    """Build chat messages with the given context applied."""
    ctx = CONTEXTS[ctx_name]
    msgs: list[dict] = []
    if ctx.system:
        msgs.append({"role": "system", "content": ctx.system})
    for shot in ctx.shots:
        msgs.append({"role": "user", "content": shot["user"]})
        msgs.append({"role": "assistant", "content": shot["assistant"]})
    msgs.append({"role": "user", "content": user_msg})
    return msgs
