from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from tests.conftest import make_config
from tests.helpers import (
    RecordingTransport,
    answer_by_voice,
    make_orchestrator,
    speech_pcm,
    wait_for,
    wait_for_state,
)
from train_speaking.evaluation.evaluator import SpeakingEvaluator
from train_speaking.interviewer.llm_interviewer import LLMInterviewer
from train_speaking.interviewer.policies import MAX_NO_ANSWER_NUDGES
from train_speaking.models.config import ConsentFlags
from train_speaking.models.session import SessionState as S
from train_speaking.models.session import TokenUsage
from train_speaking.models.transcript import Role
from train_speaking.providers.llm.base import LLMError, UsageTracker
from train_speaking.providers.llm.fake import FakeLLM
from train_speaking.providers.stt.base import STTError
from train_speaking.providers.stt.fake import FakeSTT
from train_speaking.providers.tts.fake import FakeTTS
from train_speaking.session.clock import FakeClock
from train_speaking.session.orchestrator import (
    MOVE_ON_NOTICE,
    NO_ANSWER_NUDGE,
    REPEAT_REQUEST,
    InterviewOrchestrator,
)
from train_speaking.session.storage import SessionStorage


async def _started(orchestrator: InterviewOrchestrator, **kwargs: bool) -> RecordingTransport:
    transport = RecordingTransport(orchestrator, **kwargs)
    await orchestrator.attach(transport)
    await orchestrator.start()
    return transport


async def test_first_question_is_spoken_then_listening() -> None:
    orchestrator = make_orchestrator(make_config())
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    assert transport.messages[0]["type"] == "snapshot"
    question = transport.of_type("interviewer_text")[0]
    assert "Senior Backend Engineer" in question["text"]
    assert transport.audio, "interviewer audio should be streamed"
    assert orchestrator.session.plan is not None
    assert orchestrator.session.question_count == 1


async def test_voice_answer_is_transcribed_and_followed_by_next_question() -> None:
    answer = "I have built Python services for six years and led an API migration."
    stt = FakeSTT([answer])
    orchestrator = make_orchestrator(make_config(), stt=stt)
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await answer_by_voice(orchestrator)
    await wait_for(lambda: orchestrator.session.question_count == 2)
    await wait_for_state(orchestrator, S.LISTENING)
    answers = orchestrator.session.candidate_turns()
    assert [a.text for a in answers] == [answer]
    assert len(stt.calls) == 1 and stt.calls[0].speech_started_at is not None
    metrics = answers[0].metrics
    assert metrics is not None and metrics.word_count == 13
    assert metrics.response_latency_s is not None
    assert [m["speaking"] for m in transport.of_type("vad")][:2] == [True, False]


async def test_done_answering_ends_turn_without_waiting_for_silence() -> None:
    orchestrator = make_orchestrator(make_config(), stt=FakeSTT(["Short answer."]))
    await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await orchestrator.on_audio(speech_pcm(1.0))
    assert orchestrator.state is S.USER_SPEAKING
    await orchestrator.done_answering()
    await wait_for(lambda: len(orchestrator.session.candidate_turns()) == 1)


async def test_text_answer_fallback() -> None:
    orchestrator = make_orchestrator(make_config())
    await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await orchestrator.text_answer("  I prefer typing today.  ")
    await wait_for(lambda: len(orchestrator.session.candidate_turns()) == 1)
    answer = orchestrator.session.candidate_turns()[0]
    assert answer.is_text_answer and answer.text == "I prefer typing today."


async def test_stt_failure_asks_to_repeat_and_records_error() -> None:
    stt = FakeSTT([STTError("boom"), STTError("boom again"), "Second try worked fine."])
    orchestrator = make_orchestrator(make_config(), stt=stt)
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await answer_by_voice(orchestrator)
    await wait_for(
        lambda: any(m["text"] == REPEAT_REQUEST for m in transport.of_type("interviewer_text"))
    )
    await wait_for_state(orchestrator, S.LISTENING)
    assert orchestrator.session.errors[0].component == "stt"
    await answer_by_voice(orchestrator)
    await wait_for(lambda: len(orchestrator.session.candidate_turns()) == 1)


async def test_tts_failure_falls_back_to_text_only() -> None:
    orchestrator = make_orchestrator(make_config(), tts=FakeTTS(fail_times=100))
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    end = transport.of_type("utterance_end")[0]
    assert end["has_audio"] is False and end["text_only_s"] > 0
    assert any(e.component == "tts" for e in orchestrator.session.errors)


async def test_pause_during_question_replays_it_on_resume() -> None:
    orchestrator = make_orchestrator(make_config())
    transport = await _started(orchestrator, auto_ack=False)
    await wait_for_state(orchestrator, S.INTERVIEWER_SPEAKING)
    await wait_for(lambda: bool(transport.of_type("utterance_end")))
    await orchestrator.pause()
    assert orchestrator.state is S.PAUSED
    assert transport.of_type("stop_playback")
    transport.auto_ack = True
    await orchestrator.resume()
    await wait_for_state(orchestrator, S.LISTENING)
    texts = [m["text"] for m in transport.of_type("interviewer_text")]
    assert len(texts) == 2 and texts[0] == texts[1]
    assert orchestrator.session.question_count == 1


async def test_pause_excludes_time_from_elapsed() -> None:
    clock = FakeClock()
    orchestrator = make_orchestrator(make_config(), clock=clock)
    await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    clock.advance(10)
    await orchestrator.pause()
    clock.advance(100)
    assert orchestrator.elapsed() == pytest.approx(10)
    await orchestrator.resume()
    assert orchestrator.state is S.LISTENING


async def test_time_limit_triggers_closing_and_completion() -> None:
    clock = FakeClock()
    orchestrator = make_orchestrator(make_config(duration_min=3), clock=clock)
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    clock.advance(3 * 60 + 1)
    await orchestrator.tick()
    await wait_for_state(orchestrator, S.COMPLETED)
    assert transport.of_type("interviewer_text")[-1]["kind"] == "closing"
    assert transport.of_type("completed")[0]["report_ready"] is False
    assert orchestrator.done.is_set()


async def test_wrap_up_asks_final_question_then_closes() -> None:
    clock = FakeClock()
    stt = FakeSTT(default=" ".join(["detail"] * 40))
    orchestrator = make_orchestrator(make_config(duration_min=3), clock=clock, stt=stt)
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    clock.advance(3 * 60 - 60)
    await answer_by_voice(orchestrator)
    await wait_for(lambda: any(t.is_final_question for t in orchestrator.session.turns))
    await wait_for_state(orchestrator, S.LISTENING)
    await answer_by_voice(orchestrator)
    await wait_for_state(orchestrator, S.COMPLETED)
    assert transport.of_type("interviewer_text")[-1]["kind"] == "closing"


async def test_end_early_goes_to_evaluation() -> None:
    orchestrator = make_orchestrator(make_config())
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await orchestrator.end()
    await wait_for_state(orchestrator, S.COMPLETED)
    assert transport.of_type("stop_playback")


async def test_evaluator_report_is_attached_on_completion() -> None:
    orchestrator = make_orchestrator(
        make_config(), stt=FakeSTT(["I led a team."]), evaluator=SpeakingEvaluator(None)
    )
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await answer_by_voice(orchestrator)
    await wait_for(partial(_has_answers, orchestrator, 1))
    await orchestrator.end()
    await wait_for_state(orchestrator, S.COMPLETED)
    assert transport.of_type("completed")[0]["report_ready"] is True
    assert orchestrator.report is not None
    assert orchestrator.report.questions[0].answer == "I led a team."


async def test_practice_session_asks_one_question_then_completes() -> None:
    question = "How do you handle disagreement?"
    orchestrator = make_orchestrator(
        make_config(practice_question=question, duration_min=3), stt=FakeSTT(["Briefly."])
    )
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    assert question in transport.of_type("interviewer_text")[0]["text"]
    await answer_by_voice(orchestrator)
    await wait_for_state(orchestrator, S.COMPLETED)
    assert orchestrator.session.question_count == 1
    assert len(orchestrator.session.candidate_turns()) == 1


async def test_end_before_start_aborts() -> None:
    orchestrator = make_orchestrator(make_config())
    await orchestrator.end()
    assert orchestrator.state is S.ABORTED


async def test_no_answer_nudges_then_moves_on() -> None:
    clock = FakeClock()
    orchestrator = make_orchestrator(make_config(), clock=clock)
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)

    def listening_after_nudges(n: int) -> bool:
        nudges = [m for m in transport.of_type("interviewer_text") if m["kind"] == "nudge"]
        return orchestrator.state is S.LISTENING and len(nudges) == n

    for count in range(1, MAX_NO_ANSWER_NUDGES + 1):
        clock.advance(25)
        await orchestrator.tick()
        await orchestrator.tick()
        await wait_for(partial(listening_after_nudges, count))
    nudges = [m for m in transport.of_type("interviewer_text") if m["text"] == NO_ANSWER_NUDGE]
    assert len(nudges) == MAX_NO_ANSWER_NUDGES
    clock.advance(25)
    await orchestrator.tick()
    await wait_for(lambda: orchestrator.session.question_count == 2)
    assert orchestrator.session.interviewer_turns()[-1].text.startswith(MOVE_ON_NOTICE)


async def test_audio_ignored_while_interviewer_speaks_without_barge_in() -> None:
    orchestrator = make_orchestrator(make_config())
    await _started(orchestrator, auto_ack=False)
    await wait_for_state(orchestrator, S.INTERVIEWER_SPEAKING)
    await orchestrator.on_audio(speech_pcm(1.0))
    assert orchestrator.state is S.INTERVIEWER_SPEAKING


async def test_detach_auto_pauses_and_reattach_resumes() -> None:
    orchestrator = make_orchestrator(make_config())
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await orchestrator.detach(transport)
    assert orchestrator.state is S.PAUSED
    second = RecordingTransport(orchestrator)
    await orchestrator.attach(second)
    assert second.messages[0]["type"] == "snapshot"
    assert second.messages[0]["auto_paused"] is True
    resumed_state: S = orchestrator.state
    assert resumed_state is S.LISTENING


def _has_answers(orchestrator: InterviewOrchestrator, count: int) -> bool:
    return len(orchestrator.session.candidate_turns()) == count


async def test_recording_requires_consent(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "tmp", tmp_path / "saved")
    for consent, expect_file in ((False, False), (True, True)):
        config = make_config(consent=ConsentFlags(microphone=True, record_audio=consent))
        orchestrator = make_orchestrator(
            config,
            stt=FakeSTT(["An answer."]),
            recorder=storage,
            session_id=f"rec-session-{consent}",
        )
        await _started(orchestrator)
        await wait_for_state(orchestrator, S.LISTENING)
        await answer_by_voice(orchestrator)
        await wait_for(partial(_has_answers, orchestrator, 1))
        answer = orchestrator.session.candidate_turns()[0]
        assert (answer.audio_file is not None) is expect_file
        audio_dir = tmp_path / "tmp" / orchestrator.session.id / "audio"
        assert audio_dir.exists() is expect_file


async def test_llm_outage_falls_back_to_backup_questions() -> None:
    llm = FakeLLM(responder=lambda _messages: (_ for _ in ()).throw(LLMError("offline")))
    interviewer = LLMInterviewer(llm, tracker=UsageTracker(TokenUsage(), 10_000), max_tokens=200)
    orchestrator = make_orchestrator(
        make_config(), interviewer=interviewer, stt=FakeSTT(default="An answer to the question.")
    )
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await answer_by_voice(orchestrator)
    await wait_for(lambda: orchestrator.session.question_count == 2)
    components = [e.component for e in orchestrator.session.errors]
    assert components.count("interviewer") == 2
    assert orchestrator.session.plan is not None and orchestrator.session.plan.topics
    assert any(m["type"] == "error" for m in transport.messages)


async def test_transcript_alternates_roles() -> None:
    orchestrator = make_orchestrator(make_config(), stt=FakeSTT(default="Answer text here."))
    await _started(orchestrator)
    for _ in range(3):
        await wait_for_state(orchestrator, S.LISTENING)
        await answer_by_voice(orchestrator)
        await wait_for(lambda: orchestrator.state is not S.PROCESSING)
    roles = [t.role for t in orchestrator.session.turns]
    assert roles[:6] == [Role.INTERVIEWER, Role.CANDIDATE] * 3
