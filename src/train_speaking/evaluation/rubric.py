"""Scoring rubric: which categories are measured vs. judged, and the measurement bands.

Bands are documented heuristics for interview speech, not clinical norms:
- Pace: 120-160 words/min scores 10; each 10 words/min outside the band costs 1 point.
- Excessive pauses: pauses over 2 s inside answers; each 0.5 per minute costs 1 point.
- Fillers (um/uh): per 100 words, mapped by `FILLER_BANDS`. Whisper may drop fillers, so
  counts are a lower bound.
- Fluency (measured part): pauses over 1 s per minute; each 1 per minute costs 1.5 points.
- Vocabulary (measured part): MTLD lexical diversity, mapped by `MTLD_BANDS`.
"""

from __future__ import annotations

from dataclasses import dataclass

from train_speaking.models.config import EvaluationCategory as C
from train_speaking.models.report import ScoreBasis

PACE_BAND_WPM = (120.0, 160.0)
WPM_PER_POINT = 10.0
PAUSES_PER_MIN_PER_POINT = 0.5
FLUENCY_PAUSE_PENALTY = 1.5
FILLER_BANDS: tuple[tuple[float, int], ...] = (
    (1, 10), (2, 9), (3, 8), (4, 7), (5, 6), (7, 5), (9, 4), (12, 3), (15, 2),
)  # fmt: skip
MTLD_BANDS: tuple[tuple[float, int], ...] = ((80, 9), (70, 8), (60, 7), (50, 6), (40, 5))
MIN_SPEAKING_S = 20.0
MIN_WORDS = 30


@dataclass(frozen=True)
class CategorySpec:
    basis: ScoreBasis
    description: str


RUBRIC: dict[C, CategorySpec] = {
    C.FLUENCY: CategorySpec(
        ScoreBasis.HYBRID, "Measured long pauses plus AI judgment of restarts and continuity."
    ),
    C.GRAMMAR: CategorySpec(ScoreBasis.MODEL_JUDGMENT, "AI judgment of sentence construction."),
    C.VOCABULARY: CategorySpec(
        ScoreBasis.HYBRID, "Measured lexical diversity plus AI judgment of word choice."
    ),
    C.CLARITY: CategorySpec(ScoreBasis.MODEL_JUDGMENT, "AI judgment of clarity and concision."),
    C.PROFESSIONAL_TONE: CategorySpec(ScoreBasis.MODEL_JUDGMENT, "AI judgment of register."),
    C.STRUCTURE: CategorySpec(
        ScoreBasis.MODEL_JUDGMENT, "AI judgment of structure and relevance to the question."
    ),
    C.FILLER_WORDS: CategorySpec(ScoreBasis.MEASURED, "Measured um/uh per 100 words."),
    C.PACE: CategorySpec(ScoreBasis.MEASURED, "Measured words per minute and long pauses."),
    C.PRONUNCIATION: CategorySpec(
        ScoreBasis.NOT_ASSESSED,
        "Requires phoneme-level audio analysis; not scored from text or ASR confidence.",
    ),
}

JUDGED_CATEGORIES = frozenset(
    {C.FLUENCY, C.GRAMMAR, C.VOCABULARY, C.CLARITY, C.PROFESSIONAL_TONE, C.STRUCTURE}
)


def _clip(score: float) -> int:
    return max(1, min(10, round(score)))


def score_wpm(wpm: float) -> int:
    low, high = PACE_BAND_WPM
    distance = low - wpm if wpm < low else max(0.0, wpm - high)
    return _clip(10 - distance / WPM_PER_POINT)


def score_long_pauses(per_min: float) -> int:
    return _clip(10 - per_min / PAUSES_PER_MIN_PER_POINT)


def score_fillers(per_100_words: float) -> int:
    for limit, score in FILLER_BANDS:
        if per_100_words <= limit:
            return score
    return 1


def score_fluency_pauses(pauses_gt_1s_per_min: float) -> int:
    return _clip(10 - FLUENCY_PAUSE_PENALTY * pauses_gt_1s_per_min)


def score_mtld(value: float) -> int:
    for limit, score in MTLD_BANDS:
        if value >= limit:
            return score
    return 4


def combine(measured: int | None, judged: int | None) -> int | None:
    if measured is None:
        return judged
    if judged is None:
        return measured
    return _clip((measured + judged) / 2)
