"""Pronunciation and intelligibility assessment plug-in point.

No assessor ships by default: a credible score needs phoneme-level analysis of the audio
against the intended words (for example goodness-of-pronunciation with a phoneme
recognizer, or a hosted pronunciation-assessment API). Text transcripts and ASR word
confidence are not used for this score, because they reflect microphone quality, noise,
and recognizer bias against some accents as much as the speaker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from train_speaking.models.audio import AudioClip
from train_speaking.models.report import Evidence
from train_speaking.models.transcript import Turn


@dataclass(frozen=True)
class PronunciationResult:
    score: int
    feedback: str
    measurements: dict[str, float | int | str] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)


class PronunciationAssessor(Protocol):
    async def assess(
        self, answers: Sequence[Turn], audio: Mapping[int, AudioClip]
    ) -> PronunciationResult | None:
        """`audio` maps candidate turn ids to their clips (only available with consent)."""
        ...
