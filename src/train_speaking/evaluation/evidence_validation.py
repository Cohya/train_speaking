"""Reject LLM-cited quotes that do not appear in the candidate's transcript."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from train_speaking.evaluation.llm_judge import JudgeOutput
from train_speaking.models.session import Session

MATCH_RATIO = 0.9
_NON_WORD = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def quote_in_text(quote: str, text: str) -> bool:
    """Exact normalized substring match, or a fuzzy match on a same-length word window."""
    q, t = normalize(quote), normalize(text)
    if not q:
        return False
    if q in t:
        return True
    q_words, t_words = q.split(), t.split()
    n = len(q_words)
    if n > len(t_words):
        return False
    for start in range(len(t_words) - n + 1):
        window = " ".join(t_words[start : start + n])
        if SequenceMatcher(None, q, window).ratio() >= MATCH_RATIO:
            return True
    return False


def _find_turn(session: Session, turn_id: int, quote: str) -> int | None:
    answers = session.candidate_turns()
    preferred = [a for a in answers if a.id == turn_id]
    for answer in preferred + [a for a in answers if a.id != turn_id]:
        if quote_in_text(quote, answer.text):
            return answer.id
    return None


def validate_judge_output(output: JudgeOutput, session: Session) -> tuple[JudgeOutput, int]:
    """Drop unverifiable evidence and corrections; fix wrong turn ids. Returns (output, dropped)."""
    dropped = 0
    categories = []
    for category in output.categories:
        kept = []
        for evidence in category.evidence:
            turn_id = _find_turn(session, evidence.turn_id, evidence.quote)
            if turn_id is None:
                dropped += 1
                continue
            kept.append(evidence.model_copy(update={"turn_id": turn_id}))
        categories.append(category.model_copy(update={"evidence": kept}))
    corrections = []
    for correction in output.corrections:
        turn_id = _find_turn(session, correction.turn_id, correction.original)
        if turn_id is None:
            dropped += 1
            continue
        corrections.append(correction.model_copy(update={"turn_id": turn_id}))
    return output.model_copy(update={"categories": categories, "corrections": corrections}), dropped
