"""MMLU-Pro evaluation harness.

For each question, generate a short answer with three model+input combinations:
  base       — base model on raw question
  in_context — base model on [C; question]
  distilled  — context-distilled student on raw question

Score by extracting the first letter A-J that appears in the generated text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from . import contexts as ctx_lib


OPTION_LETTERS = "ABCDEFGHIJ"
LETTER_RE = re.compile(r"\b([A-J])\b")


def _chat_ids(tok, messages):
    out = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if isinstance(out, torch.Tensor):
        return out
    return out["input_ids"]


def format_question(record: dict) -> tuple[str, list[str], str]:
    """Return (user_msg, available_letters, gold_letter)."""
    q = record["question"]
    opts = record["options"]
    n_opts = len(opts)
    letters = list(OPTION_LETTERS[:n_opts])
    options_text = "\n".join(f"({L}) {o}" for L, o in zip(letters, opts))
    user_msg = (
        f"Question: {q}\n\nOptions:\n{options_text}\n\n"
        "Answer with only the letter of the correct option (e.g. \"A\"). Do not explain."
    )
    gold = record.get("answer") or record.get("answer_index")
    if isinstance(gold, int):
        gold = letters[gold] if 0 <= gold < n_opts else None
    return user_msg, letters, gold


@torch.no_grad()
def predict_letter(model, tok, ctx_name: str, user_msg: str, letters: list[str], max_new_tokens: int = 8) -> tuple[str | None, str]:
    """Generate up to ``max_new_tokens`` tokens, extract the first valid letter."""
    msgs = ctx_lib.build_messages(ctx_name, user_msg)
    ids = _chat_ids(tok, msgs).to(model.device)
    prompt_len = ids.shape[1]
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
    # Find the first A-J letter that's also in the available letters
    for ch in text.upper():
        if ch in letters:
            return ch, text
    # Fall back: try the regex on the full text
    m = LETTER_RE.search(text.upper())
    if m and m.group(1) in letters:
        return m.group(1), text
    return None, text


def evaluate(model, tok, ctx_name: str, dataset: list[dict], max_new_tokens: int = 8) -> dict:
    """Run MMLU-Pro eval. Returns per-question rows + aggregate accuracy."""
    rows = []
    correct = 0
    n_pred = 0
    for i, record in enumerate(dataset):
        try:
            user_msg, letters, gold = format_question(record)
            pred, text = predict_letter(model, tok, ctx_name, user_msg, letters, max_new_tokens=max_new_tokens)
            ok = (pred is not None and pred == gold)
            rows.append({
                "qid": record.get("question_id", i),
                "category": record.get("category", "?"),
                "gold": gold,
                "pred": pred,
                "raw": text[:80],
                "correct": ok,
            })
            if pred is not None:
                n_pred += 1
            if ok:
                correct += 1
        except Exception as e:  # noqa: BLE001
            rows.append({"qid": record.get("question_id", i), "error": str(e)[:120]})
    n = len(dataset)
    return {
        "n_total": n,
        "n_predicted": n_pred,
        "n_correct": correct,
        "accuracy": correct / max(1, n),
        "accuracy_predicted": correct / max(1, n_pred),
        "rows": rows,
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
