"""Per-context behavioral pattern matchers, complementary to ROUGE.

ROUGE-L penalises content drift even when the structural/lexical pattern of
the context is preserved. For each context type we add a matcher that scores
the *form* of the output, not exact word overlap with the teacher.

Returns a dict per context with the relevant pattern-match rate(s) for both
the teacher (= base + C) and the student. Comparing teacher vs student rates
tells us how well the FT reproduces the *behavioural pattern*.
"""
from __future__ import annotations

import re


HAIKU_LINE_RE = re.compile(r"\n")

REFUSAL_KEYWORDS = (
    "i can't", "i cannot", "i'm not aware", "i am not aware",
    "i do not have", "i don't have", "i am unable", "not aware of",
    "no information", "i do not know", "i don't know",
)

PIRATE_KEYWORDS = ("arr", "matey", "ye ", " ye'", "savvy", "ahoy", "yer ", "landlubber", "aye")

FRENCH_CHAR_RE = re.compile(r"[éèêëàâäçîïôöùûüÉÈÊËÀÂÄÇÎÏÔÖÙÛÜ]")


def _is_haiku_shaped(text: str) -> bool:
    """Heuristic: 3 non-empty lines, each between 1 and ~10 words."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not (2 <= len(lines) <= 4):
        return False
    for l in lines:
        wc = len(l.split())
        if not (1 <= wc <= 12):
            return False
    return True


def _has_refusal(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in REFUSAL_KEYWORDS)


def _has_pirate(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in PIRATE_KEYWORDS)


def _has_french(text: str) -> bool:
    if FRENCH_CHAR_RE.search(text):
        return True
    fr_words = (" le ", " la ", " les ", " un ", " une ", " est ", " êtes ", " mais ", " avec ", " pour ", " merci")
    low = " " + text.lower() + " "
    return sum(1 for w in fr_words if w in low) >= 2


def _word_count(text: str) -> int:
    return len(text.split())


def score_behavior(answers: list[str], context_name: str) -> dict[str, float]:
    """Compute behavioural pattern rates for a list of answers under a given
    context. Returns a dict mapping metric name -> rate (0..1) or count.
    """
    n = max(1, len(answers))
    out: dict[str, float] = {"n": n}

    if context_name == "haiku":
        out["haiku_shaped_rate"] = sum(_is_haiku_shaped(a) for a in answers) / n
        out["mean_line_count"] = sum(a.count("\n") + 1 for a in answers) / n

    elif context_name == "pirate":
        out["pirate_word_rate"] = sum(_has_pirate(a) for a in answers) / n

    elif context_name == "concise":
        out["mean_word_count"] = sum(_word_count(a) for a in answers) / n
        out["short_answer_rate"] = sum(1 for a in answers if _word_count(a) <= 30) / n

    elif context_name == "fewshot_translate_fr":
        out["french_rate"] = sum(_has_french(a) for a in answers) / n
        out["mean_word_count"] = sum(_word_count(a) for a in answers) / n

    elif context_name == "factual":
        out["refusal_rate"] = sum(_has_refusal(a) for a in answers) / n

    out["mean_word_count"] = out.get("mean_word_count", sum(_word_count(a) for a in answers) / n)
    return out


def score_pair(teacher_answers: list[str], student_answers: list[str], context_name: str) -> dict:
    """Compute teacher and student pattern rates and report deltas."""
    t = score_behavior(teacher_answers, context_name)
    s = score_behavior(student_answers, context_name)
    out = {"teacher": t, "student": s}
    # Per-metric absolute and relative deltas
    deltas = {}
    for k, tv in t.items():
        if k == "n":
            continue
        sv = s.get(k, 0.0)
        deltas[k + "_delta"] = sv - tv
        if abs(tv) > 1e-9:
            deltas[k + "_relative"] = sv / tv
    out["delta"] = deltas
    return out
