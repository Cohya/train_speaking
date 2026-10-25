"""Interview turn loop: connects the voice layer, interviewer, session state, and evaluator.

All public methods run on the asyncio event loop. Slow work (planning, transcription, LLM
calls, synthesis) runs in a single "turn task" so it can be cancelled on pause, barge-in,
or end.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Protocol

from train_speaking.evaluation.audio_metrics import compute_answer_metrics
from train_speaking.interviewer.base import InterviewContext, Interviewer
from train_speaking.interviewer.planner import fallback_plan
from train_speaking.interviewer.policies import (
    MAX_NO_ANSWER_NUDGES,
    NO_ANSWER_TIMEOUT_S,
    InterviewPhase,
    is_hard_stop,
    phase_for,
)
from train_speaking.interviewer.scripted import ScriptedInterviewer
from train_speaking.models.audio import AudioClip, SynthesizedSpeech
from train_speaking.models.events import server_message
from train_speaking.models.interview import InterviewerAction, InterviewerTurn
from train_speaking.models.report import Report
from train_speaking.models.session import Session, SessionError, SessionState
from train_speaking.models.transcript import Role, Transcription, Turn, WordTiming
from train_speaking.providers.stt.base import STTError, STTProvider
from train_speaking.providers.tts.base import TTSError, TTSProvider
from train_speaking.session.clock import Clock, SessionClock
from train_speaking.session.state_machine import TERMINAL_STATES, SessionStateMachine
from train_speaking.voice.audio_utils import float32_to_pcm16
from train_speaking.voice.listener import AnswerListener
from train_speaking.voice.protocol import AudioChunkHeader, encode_audio_chunk
from train_speaking.voice.sentence_splitter import split_sentences
from train_speaking.voice.turn_detector import ListenMode, TurnEvent, TurnEventType
from train_speaking.voice.visemes import viseme_track

logger = logging.getLogger(__name__)

S = SessionState
PAUSABLE_STATES = frozenset(
    {S.INTERVIEWER_SPEAKING, S.LISTENING, S.USER_SPEAKING, S.PROCESSING, S.RECOVERING}
)
READING_SECONDS_PER_WORD = 0.3
MIN_READING_S = 1.5
INTERIM_WINDOW_S = 30.0
REPEAT_REQUEST = "Sorry, I didn't catch that. Could you please repeat your answer?"
NO_ANSWER_NUDGE = "Take your time. Whenever you're ready, please go ahead."
MOVE_ON_NOTICE = "Let's move on to another question."
ACK_PHRASES = ("Thank you.", "I see.", "Okay, thanks.", "Got it.")


class Transport(Protocol):
    async def send_json(self, message: dict[str, Any]) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...


class Evaluator(Protocol):
    async def evaluate(self, session: Session) -> Report: ...


class AnswerRecorder(Protocol):
    def save_answer_audio(self, session_id: str, turn_id: int, clip: AudioClip) -> str: ...


@dataclass(frozen=True)
class OrchestratorOptions:
    no_answer_timeout_s: float = NO_ANSWER_TIMEOUT_S
    playback_timeout_margin_s: float = 5.0
    half_duplex_tail_s: float = 0.25
    interviewer_timeout_s: float = 45.0
    stt_retries: int = 1
    tick_interval_s: float = 0.5
    run_ticker: bool = True
    interim_transcripts: bool = False
    interim_interval_s: float = 2.0
    send_visemes: bool = False
    log_transcripts: bool = False


@dataclass
class _Utterance:
    id: int
    text: str
    kind: str
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    sentence_ends_s: list[tuple[float, str]] = field(default_factory=list)
    audio_s: float = 0.0
    playback_started_at: float | None = None


class InterviewOrchestrator:
    def __init__(
        self,
        session: Session,
        *,
        interviewer: Interviewer,
        stt: STTProvider,
        tts: TTSProvider,
        listener: AnswerListener,
        clock: Clock,
        evaluator: Evaluator | None = None,
        recorder: AnswerRecorder | None = None,
        options: OrchestratorOptions | None = None,
        fallback: ScriptedInterviewer | None = None,
    ) -> None:
        self.session = session
        self.report: Report | None = None
        self._interviewer = interviewer
        self._fallback = fallback or ScriptedInterviewer()
        self._stt = stt
        self._tts = tts
        self._listener = listener
        self._evaluator = evaluator
        self._recorder = recorder
        self._options = options or OrchestratorOptions()
        self._machine = SessionStateMachine(session.state)
        self._clock = SessionClock(clock)
        self._transport: Transport | None = None
        self._utterance_counter = 0
        self._utterance: _Utterance | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._ticker_task: asyncio.Task[None] | None = None
        self._interim_task: asyncio.Task[None] | None = None
        self._eval_task: asyncio.Task[None] | None = None
        self._pending_replay: Turn | None = None
        self._pending_close = False
        self._question_ended_at: float | None = None
        self._listening_since = 0.0
        self._ignore_until = 0.0
        self._nudges = 0
        self._last_interim_at = 0.0
        self._auto_paused = False
        self._ack_cache: dict[str, SynthesizedSpeech] = {}
        self._current_text = ""
        self.done = asyncio.Event()

    # ----------------------------------------------------------------- state helpers

    @property
    def state(self) -> SessionState:
        return self._machine.state

    def elapsed(self) -> float:
        return self._clock.elapsed()

    @property
    def current_question(self) -> str:
        return self._current_text

    def _set_state(self, target: SessionState) -> None:
        if self._machine.state is target:
            return
        self._machine.transition(target)
        self.session.state = target
        self.session.elapsed_s = round(self.elapsed(), 2)
        self._spawn(self._send(server_message("state", state=target.value)))

    async def _send(self, message: dict[str, Any]) -> None:
        transport = self._transport
        if transport is None:
            return
        try:
            await transport.send_json(message)
        except (RuntimeError, ConnectionError) as exc:
            logger.info(
                "send failed; client likely disconnected", extra={"error": type(exc).__name__}
            )

    async def _send_bytes(self, data: bytes) -> None:
        transport = self._transport
        if transport is None:
            return
        try:
            await transport.send_bytes(data)
        except (RuntimeError, ConnectionError) as exc:
            logger.info(
                "send failed; client likely disconnected", extra={"error": type(exc).__name__}
            )

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        return asyncio.get_running_loop().create_task(coro)

    def _record_error(
        self, component: str, kind: str, message: str, *, recovered: bool = True
    ) -> None:
        self.session.errors.append(
            SessionError(
                at=round(self.elapsed(), 2),
                component=component,
                kind=kind,
                message=message,
                recovered=recovered,
            )
        )
        logger.warning(
            "session error", extra={"component": component, "kind": kind, "recovered": recovered}
        )
        self._spawn(self._send(server_message("error", component=component, message=message)))

    def _context(self) -> InterviewContext:
        assert self.session.plan is not None
        elapsed = self.elapsed()
        phase = phase_for(self.session.turns, elapsed, self.session.config.duration_s)
        if self.session.config.practice_question and self.session.candidate_turns():
            phase = InterviewPhase.CLOSE
        return InterviewContext(
            config=self.session.config,
            plan=self.session.plan,
            turns=tuple(self.session.turns),
            elapsed_s=elapsed,
            phase=phase,
        )

    def _run_turn(self, coro: Coroutine[Any, Any, None]) -> None:
        self._turn_task = self._spawn(self._guarded(coro))

    async def _guarded(self, coro: Coroutine[Any, Any, None]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("unexpected error in interview turn")
            self._record_error("orchestrator", type(exc).__name__, str(exc), recovered=False)
            await self.end()

    def _cancel_turn(self) -> None:
        task = self._turn_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
        self._turn_task = None

    # ----------------------------------------------------------------- transport

    async def attach(self, transport: Transport) -> None:
        self._transport = transport
        await self._send(self.snapshot())
        if self._auto_paused and self.state is S.PAUSED:
            await self.resume()

    async def detach(self, transport: Transport) -> None:
        if self._transport is not transport:
            return
        self._transport = None
        if self.state in PAUSABLE_STATES:
            await self._pause_now()
            self._auto_paused = True

    def snapshot(self) -> dict[str, Any]:
        return server_message(
            "snapshot",
            state=self.state.value,
            elapsed_s=round(self.elapsed(), 1),
            duration_s=self.session.config.duration_s,
            question=self._current_text,
            turns=[_turn_payload(t) for t in self.session.turns],
            auto_paused=self._auto_paused,
        )

    # ----------------------------------------------------------------- client commands

    async def start(self) -> None:
        if self.state is not S.READY:
            return
        self._clock.start()
        self._set_state(S.PROCESSING)
        if self._options.run_ticker:
            self._ticker_task = self._spawn(self._ticker())
        self._run_turn(self._open_interview())

    async def pause(self) -> None:
        if self.state in PAUSABLE_STATES:
            self._auto_paused = False
            await self._pause_now()

    async def _pause_now(self) -> None:
        state = self.state
        self._listener.set_mode(ListenMode.IGNORE)
        self._clock.pause()
        if state is S.INTERVIEWER_SPEAKING:
            self._cancel_turn()
            self._pending_replay = self._last_interviewer_turn()
        self._set_state(S.PAUSED)
        if state is S.INTERVIEWER_SPEAKING:
            await self._send(server_message("stop_playback"))
        elif state is S.USER_SPEAKING:
            await self._send(
                server_message("notice", message="Your answer in progress was discarded.")
            )

    async def resume(self) -> None:
        if self.state is not S.PAUSED:
            return
        self._auto_paused = False
        self._clock.resume()
        if self._pending_close:
            self._pending_close = False
            self._set_state(S.PROCESSING)
            self._run_turn(self._close())
        elif self._pending_replay is not None:
            turn, self._pending_replay = self._pending_replay, None
            self._set_state(S.INTERVIEWER_SPEAKING)
            self._run_turn(self._speak_and_listen(turn))
        elif self._turn_task is not None and not self._turn_task.done():
            self._set_state(S.PROCESSING)
        else:
            self._begin_listening()

    async def end(self) -> None:
        if self.state in TERMINAL_STATES or self.state is S.EVALUATING:
            return
        self._cancel_turn()
        self._listener.set_mode(ListenMode.IGNORE)
        await self._send(server_message("stop_playback"))
        if self.state in (S.READY, S.CONFIGURING):
            self._set_state(S.ABORTED)
            self._finish()
            return
        self._clock.pause()
        self._set_state(S.EVALUATING)
        self._eval_task = self._spawn(self._evaluate())

    async def done_answering(self) -> None:
        if self.state is not S.USER_SPEAKING:
            return
        event = self._listener.force_end(self.elapsed())
        if event is not None and event.clip is not None:
            self._handle_answer_clip(event.clip)

    async def text_answer(self, text: str) -> None:
        if self.state not in (S.LISTENING, S.USER_SPEAKING):
            return
        self._listener.set_mode(ListenMode.IGNORE)
        self._set_state(S.PROCESSING)
        self._run_turn(self._process_text_answer(text.strip()))

    def playback_started(self, utterance_id: int) -> None:
        utterance = self._utterance
        if utterance is not None and utterance.id == utterance_id:
            utterance.playback_started_at = self.elapsed()

    def playback_finished(self, utterance_id: int) -> None:
        utterance = self._utterance
        if utterance is not None and utterance.id == utterance_id:
            utterance.finished.set()

    def client_error(self, kind: str, message: str) -> None:
        self._record_error("client", kind, message)

    async def on_audio(self, pcm16: bytes) -> None:
        if self.state in TERMINAL_STATES or self.state in (S.PAUSED, S.EVALUATING, S.READY):
            return
        now = self.elapsed()
        if now < self._ignore_until:
            return
        for event in self._listener.feed_pcm(pcm16, now):
            await self._on_turn_event(event)

    async def _on_turn_event(self, event: TurnEvent) -> None:
        if event.type is TurnEventType.SPEECH_STARTED and self.state is S.LISTENING:
            self._set_state(S.USER_SPEAKING)
            await self._send(server_message("vad", speaking=True))
        elif event.type is TurnEventType.SPEECH_DISCARDED and self.state is S.USER_SPEAKING:
            self._set_state(S.LISTENING)
            await self._send(server_message("vad", speaking=False))
        elif event.type in (TurnEventType.TURN_ENDED, TurnEventType.MAX_ANSWER_REACHED):
            if self.state is S.USER_SPEAKING and event.clip is not None:
                self._handle_answer_clip(event.clip)
        elif event.type is TurnEventType.BARGE_IN and self.state is S.INTERVIEWER_SPEAKING:
            self._barge_in()

    # ----------------------------------------------------------------- turn flow

    async def _open_interview(self) -> None:
        try:
            plan = await asyncio.wait_for(
                self._interviewer.plan(self.session.config), self._options.interviewer_timeout_s
            )
        except Exception:
            self._record_error("interviewer", "plan_failed", "Using the built-in question plan.")
            plan = fallback_plan(self.session.config)
        self.session.plan = plan
        await self._ask_next()

    async def _ask_next(self, prefix: str = "") -> None:
        ctx = self._context()
        if ctx.phase is InterviewPhase.CLOSE:
            await self._close()
            return
        try:
            turn = await asyncio.wait_for(
                self._interviewer.next_turn(ctx), self._options.interviewer_timeout_s
            )
        except Exception as exc:
            self._record_error("interviewer", type(exc).__name__, "Using a backup question.")
            turn = self._fallback.fallback_turn(ctx)
        if turn.action is InterviewerAction.CLOSE:
            await self._close()
            return
        await self._deliver_question(turn, prefix)

    async def _deliver_question(self, decision: InterviewerTurn, prefix: str = "") -> None:
        now = self.elapsed()
        turn = Turn(
            id=self.session.next_turn_id(),
            role=Role.INTERVIEWER,
            text=f"{prefix} {decision.question}".strip(),
            started_at=round(now, 2),
            ended_at=round(now, 2),
            topic_id=decision.topic_id or "general",
            is_follow_up=decision.is_follow_up,
            is_final_question=decision.action is InterviewerAction.WRAP_UP,
        )
        self.session.turns.append(turn)
        self._nudges = 0
        await self._send(server_message("transcript_turn", turn=_turn_payload(turn)))
        if self.state is S.PAUSED:
            self._pending_replay = turn
            return
        self._set_state(S.INTERVIEWER_SPEAKING)
        await self._speak_and_listen(turn)

    async def _speak_and_listen(self, turn: Turn) -> None:
        barge = self.session.config.allow_barge_in
        self._listener.set_mode(ListenMode.BARGE_IN if barge else ListenMode.IGNORE)
        await self._speak(turn.text, kind="question")
        turn.ended_at = round(self.elapsed(), 2)
        if self.state is S.INTERVIEWER_SPEAKING:
            self._begin_listening()

    def _begin_listening(self) -> None:
        now = self.elapsed()
        self._question_ended_at = now
        self._listening_since = now
        self._ignore_until = now + self._options.half_duplex_tail_s
        self._listener.set_mode(ListenMode.LISTEN)
        self._set_state(S.LISTENING)

    async def _speak(self, text: str, *, kind: str) -> None:
        """Synthesize sentence by sentence, stream to the client, and await playback."""
        self._utterance_counter += 1
        utterance = _Utterance(id=self._utterance_counter, text=text, kind=kind)
        self._utterance = utterance
        if kind in ("question", "closing"):
            self._current_text = text
        await self._send(
            server_message("interviewer_text", utterance_id=utterance.id, text=text, kind=kind)
        )
        tts_failed = False
        for seq, sentence in enumerate(split_sentences(text)):
            try:
                speech = await self._tts.synthesize(sentence)
            except TTSError as exc:
                if not tts_failed:
                    self._record_error("tts", "synthesis_failed", "Showing the question as text.")
                    logger.info("tts failure", extra={"error": str(exc)})
                tts_failed = True
                continue
            await self._send_speech(utterance, seq, speech)
            utterance.audio_s += speech.duration_s
            utterance.sentence_ends_s.append((utterance.audio_s, sentence))
        reading_s = max(MIN_READING_S, len(text.split()) * READING_SECONDS_PER_WORD)
        await self._send(
            server_message(
                "utterance_end",
                utterance_id=utterance.id,
                has_audio=utterance.audio_s > 0,
                text_only_s=0.0 if utterance.audio_s > 0 else reading_s,
            )
        )
        timeout = utterance.audio_s + reading_s + self._options.playback_timeout_margin_s
        try:
            await asyncio.wait_for(utterance.finished.wait(), timeout)
        except TimeoutError:
            logger.info("playback confirmation timed out", extra={"utterance": utterance.id})

    async def _send_speech(
        self, utterance: _Utterance, seq: int, speech: SynthesizedSpeech
    ) -> None:
        header = AudioChunkHeader(utterance.id, seq, speech.sample_rate)
        await self._send_bytes(encode_audio_chunk(header, float32_to_pcm16(speech.audio)))
        if self._options.send_visemes and speech.phoneme_timeline:
            await self._send(
                server_message(
                    "visemes",
                    utterance_id=utterance.id,
                    seq=seq,
                    track=viseme_track(speech.phoneme_timeline),
                )
            )

    def _handle_answer_clip(self, clip: AudioClip) -> None:
        self._listener.set_mode(ListenMode.IGNORE)
        self._set_state(S.PROCESSING)
        self._spawn(self._send(server_message("vad", speaking=False)))
        if self.session.config.acknowledgements:
            self._spawn(self._play_acknowledgement())
        self._run_turn(self._process_answer(clip))

    async def _process_answer(self, clip: AudioClip) -> None:
        try:
            transcription = await self._transcribe(clip)
        except STTError:
            self._record_error("stt", "transcription_failed", "Asking you to repeat the answer.")
            await self._ask_to_repeat()
            return
        if not transcription.text.strip():
            await self._ask_to_repeat()
            return
        offset = clip.started_at
        words = [
            WordTiming(
                word=w.word,
                start=round(w.start + offset, 3),
                end=round(w.end + offset, 3),
                probability=w.probability,
            )
            for w in transcription.words
        ]
        ended_at = clip.started_at + clip.duration_s
        turn = self._add_candidate_turn(
            transcription.text.strip(),
            words=words,
            speech_started_at=clip.speech_started_at,
            ended_at=ended_at,
        )
        if self.session.config.consent.record_audio and self._recorder is not None:
            try:
                turn.audio_file = self._recorder.save_answer_audio(self.session.id, turn.id, clip)
            except OSError:
                self._record_error("storage", "audio_save_failed", "This answer was not recorded.")
        await self._send(server_message("transcript_turn", turn=_turn_payload(turn)))
        await self._ask_next()

    async def _process_text_answer(self, text: str) -> None:
        now = self.elapsed()
        turn = self._add_candidate_turn(
            text, words=[], speech_started_at=None, ended_at=now, is_text=True
        )
        await self._send(server_message("transcript_turn", turn=_turn_payload(turn)))
        await self._ask_next()

    def _add_candidate_turn(
        self,
        text: str,
        *,
        words: list[WordTiming],
        speech_started_at: float | None,
        ended_at: float,
        is_text: bool = False,
    ) -> Turn:
        started = words[0].start if words else (speech_started_at or ended_at)
        metrics = compute_answer_metrics(
            words,
            text,
            speech_started_at=speech_started_at,
            ended_at=ended_at,
            question_ended_at=None if is_text else self._question_ended_at,
        )
        last_question = self._last_interviewer_turn()
        turn = Turn(
            id=self.session.next_turn_id(),
            role=Role.CANDIDATE,
            text=text,
            started_at=round(started, 2),
            ended_at=round(ended_at, 2),
            topic_id=last_question.topic_id if last_question else None,
            is_text_answer=is_text,
            words=words,
            metrics=metrics,
        )
        self.session.turns.append(turn)
        if self._options.log_transcripts:
            logger.debug("candidate answer", extra={"turn": turn.id, "text": text})
        return turn

    async def _transcribe(self, clip: AudioClip) -> Transcription:
        last: STTError | None = None
        for _ in range(self._options.stt_retries + 1):
            try:
                return await self._stt.transcribe(clip, language=self.session.config.language)
            except STTError as exc:
                last = exc
        assert last is not None
        raise last

    async def _ask_to_repeat(self) -> None:
        if self.state is S.PAUSED:
            self._pending_replay = self._last_interviewer_turn()
            return
        self._set_state(S.RECOVERING)
        self._set_state(S.INTERVIEWER_SPEAKING)
        self._listener.set_mode(ListenMode.IGNORE)
        await self._speak(REPEAT_REQUEST, kind="repeat")
        if self.state is S.INTERVIEWER_SPEAKING:
            self._begin_listening()

    async def _nudge(self) -> None:
        self._nudges += 1
        if self._nudges > MAX_NO_ANSWER_NUDGES:
            await self._ask_next(prefix=MOVE_ON_NOTICE)
            return
        self._set_state(S.INTERVIEWER_SPEAKING)
        self._listener.set_mode(ListenMode.IGNORE)
        await self._speak(NO_ANSWER_NUDGE, kind="nudge")
        if self.state is S.INTERVIEWER_SPEAKING:
            self._begin_listening()
            self._question_ended_at = self._last_question_end()

    def _last_question_end(self) -> float | None:
        turn = self._last_interviewer_turn()
        return turn.ended_at if turn else None

    async def _close(self) -> None:
        if self.state is S.PAUSED:
            self._pending_close = True
            return
        ctx = self._context()
        text = self._interviewer.closing(ctx)
        self._listener.set_mode(ListenMode.IGNORE)
        self._set_state(S.WRAPPING_UP)
        await self._speak(text, kind="closing")
        if self.state is S.WRAPPING_UP:
            self._clock.pause()
            self._set_state(S.EVALUATING)
            self._eval_task = self._spawn(self._evaluate())

    async def _evaluate(self) -> None:
        self.session.elapsed_s = round(self.elapsed(), 2)
        if self._evaluator is not None:
            try:
                self.report = await self._evaluator.evaluate(self.session)
            except Exception as exc:
                logger.exception("evaluation failed")
                self._record_error(
                    "evaluator", type(exc).__name__, "The report could not be built."
                )
        self._set_state(S.COMPLETED)
        await self._send(server_message("completed", report_ready=self.report is not None))
        self._finish()

    def _finish(self) -> None:
        if self._ticker_task is not None and self._ticker_task is not asyncio.current_task():
            self._ticker_task.cancel()
        self.done.set()

    def _barge_in(self) -> None:
        utterance = self._utterance
        self._cancel_turn()
        self._spawn(self._send(server_message("stop_playback")))
        self.session.interruptions += 1
        turn = self._last_interviewer_turn()
        if turn is not None and utterance is not None:
            turn.interrupted = True
            turn.ended_at = round(self.elapsed(), 2)
            played = _played_text(utterance, self.elapsed())
            if played:
                turn.text = played
        self._question_ended_at = self.elapsed()
        self._set_state(S.USER_SPEAKING)
        self._spawn(self._send(server_message("vad", speaking=True)))

    async def _play_acknowledgement(self) -> None:
        phrase = random.choice(ACK_PHRASES)
        try:
            speech = self._ack_cache.get(phrase) or await self._tts.synthesize(phrase)
        except TTSError:
            return
        self._ack_cache[phrase] = speech
        self._utterance_counter += 1
        ack = _Utterance(id=self._utterance_counter, text=phrase, kind="ack")
        await self._send(
            server_message("interviewer_text", utterance_id=ack.id, text=phrase, kind="ack")
        )
        await self._send_speech(ack, 0, speech)
        await self._send(
            server_message("utterance_end", utterance_id=ack.id, has_audio=True, text_only_s=0.0)
        )

    def _last_interviewer_turn(self) -> Turn | None:
        turns = [t for t in self.session.turns if t.role is Role.INTERVIEWER]
        return turns[-1] if turns else None

    # ----------------------------------------------------------------- periodic checks

    async def _ticker(self) -> None:
        while self.state not in TERMINAL_STATES:
            await asyncio.sleep(self._options.tick_interval_s)
            await self.tick()

    async def tick(self) -> None:
        state = self.state
        if state in TERMINAL_STATES:
            return
        elapsed = self.elapsed()
        self.session.elapsed_s = round(elapsed, 2)
        usage = self.session.token_usage
        await self._send(
            server_message(
                "timer",
                elapsed_s=round(elapsed, 1),
                duration_s=self.session.config.duration_s,
                question_count=self.session.question_count,
                tokens=usage.total_tokens,
                estimated_cost=round(usage.estimated_cost, 4),
            )
        )
        if state is S.EVALUATING or state is S.PAUSED:
            return
        if is_hard_stop(elapsed, self.session.config.duration_s):
            await self.end()
            return
        if state is S.LISTENING:
            if self._context().phase is InterviewPhase.CLOSE:
                self._listener.set_mode(ListenMode.IGNORE)
                self._set_state(S.PROCESSING)
                self._run_turn(self._close())
            elif elapsed - self._listening_since >= self._options.no_answer_timeout_s:
                self._listener.set_mode(ListenMode.IGNORE)
                self._set_state(S.PROCESSING)
                self._run_turn(self._nudge())
        elif state is S.USER_SPEAKING and self._options.interim_transcripts:
            self._maybe_interim(elapsed)

    def _maybe_interim(self, now: float) -> None:
        if now - self._last_interim_at < self._options.interim_interval_s:
            return
        if self._interim_task is not None and not self._interim_task.done():
            return
        clip = self._listener.current_audio()
        if clip is None or clip.duration_s < 1.0:
            return
        self._last_interim_at = now
        max_samples = int(INTERIM_WINDOW_S * clip.sample_rate)
        if len(clip.samples) > max_samples:
            clip = AudioClip(clip.samples[-max_samples:], clip.sample_rate, clip.started_at)
        self._interim_task = self._spawn(self._interim(clip))

    async def _interim(self, clip: AudioClip) -> None:
        with contextlib.suppress(STTError):
            result = await self._stt.transcribe(clip, language=self.session.config.language)
            if self.state is S.USER_SPEAKING:
                await self._send(server_message("interim_transcript", text=result.text))


def _played_text(utterance: _Utterance, now: float) -> str:
    """Sentences whose audio has fully played, estimated from the playback start time."""
    if utterance.playback_started_at is None:
        return ""
    played_s = now - utterance.playback_started_at
    return " ".join(s for end_s, s in utterance.sentence_ends_s if end_s <= played_s)


def _turn_payload(turn: Turn) -> dict[str, Any]:
    return {
        "id": turn.id,
        "role": turn.role.value,
        "text": turn.text,
        "started_at": turn.started_at,
        "is_follow_up": turn.is_follow_up,
        "interrupted": turn.interrupted,
    }
