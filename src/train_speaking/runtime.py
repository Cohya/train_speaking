"""Composition root: wires settings, providers, storage, and per-session orchestrators."""

from __future__ import annotations

import logging
import secrets
from dataclasses import replace

from train_speaking.evaluation.evaluator import SpeakingEvaluator
from train_speaking.interviewer.base import Interviewer
from train_speaking.interviewer.llm_interviewer import LLMInterviewer
from train_speaking.interviewer.scripted import ScriptedInterviewer
from train_speaking.models.session import Session
from train_speaking.providers import registry
from train_speaking.providers.llm.base import UsageTracker
from train_speaking.session.clock import MonotonicClock
from train_speaking.session.manager import SessionManager
from train_speaking.session.orchestrator import (
    Evaluator,
    InterviewOrchestrator,
    OrchestratorOptions,
)
from train_speaking.session.storage import SessionStorage, StorageError
from train_speaking.settings import AppSettings
from train_speaking.voice.listener import AnswerListener
from train_speaking.voice.turn_detector import TurnDetector, TurnDetectorConfig

logger = logging.getLogger(__name__)


class AppRuntime:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.stt = registry.build_stt(settings)
        self.tts = registry.build_tts(settings)
        self.storage = SessionStorage(settings.tmp_sessions_dir, settings.saved_sessions_dir)
        self.manager = SessionManager(self.create_orchestrator)
        self.clock = MonotonicClock()
        self.api_token = secrets.token_urlsafe(24)
        self.options = OrchestratorOptions(
            interim_transcripts=settings.interim_transcripts,
            send_visemes=settings.avatar_mode == "viseme",
            log_transcripts=settings.log_transcripts,
        )

    @property
    def llm_configured(self) -> bool:
        return self.settings.llm_provider != "none"

    async def warmup(self) -> None:
        for name, provider in (("stt", self.stt), ("tts", self.tts)):
            try:
                await provider.warmup()
            except Exception:
                logger.exception("provider warmup failed", extra={"provider": name})

    def usage_tracker(self, session: Session) -> UsageTracker:
        return UsageTracker(
            session.token_usage,
            budget_tokens=self.settings.session_token_budget,
            cost_per_1k_input=self.settings.cost_per_1k_input_tokens,
            cost_per_1k_output=self.settings.cost_per_1k_output_tokens,
        )

    def create_interviewer(self, session: Session) -> Interviewer:
        llm = registry.build_llm(self.settings, "interviewer")
        if llm is None:
            return ScriptedInterviewer()
        return LLMInterviewer(
            llm,
            tracker=self.usage_tracker(session),
            max_tokens=self.settings.max_tokens_per_call,
        )

    def lookup_session(self, session_id: str) -> Session | None:
        live = self.manager.get(session_id)
        if live is not None:
            return live.session
        try:
            return self.storage.load(session_id)[0]
        except StorageError:
            return None

    def create_evaluator(self, session: Session) -> Evaluator | None:
        return SpeakingEvaluator(
            registry.build_llm(self.settings, "evaluator"),
            tracker=self.usage_tracker(session),
            max_tokens=self.settings.max_tokens_per_call,
            session_lookup=self.lookup_session,
        )

    def create_orchestrator(self, session: Session) -> InterviewOrchestrator:
        detector = TurnDetector(
            TurnDetectorConfig(end_silence_ms=float(session.config.end_silence_ms))
        )
        listener = AnswerListener(registry.build_vad(self.settings), detector)
        return InterviewOrchestrator(
            session,
            interviewer=self.create_interviewer(session),
            stt=self.stt,
            tts=self.tts,
            listener=listener,
            clock=self.clock,
            evaluator=self.create_evaluator(session),
            recorder=self.storage,
            options=replace(self.options),
        )
