"""Barge-in, spoken acknowledgements, interim transcripts, and the usage counter."""

from __future__ import annotations

from dataclasses import replace

from tests.conftest import make_config
from tests.helpers import (
    TEST_OPTIONS,
    RecordingTransport,
    answer_by_voice,
    make_orchestrator,
    silence_pcm,
    speech_pcm,
    wait_for,
    wait_for_state,
)
from train_speaking.models.session import SessionState as S
from train_speaking.providers.stt.fake import FakeSTT
from train_speaking.providers.tts.fake import SECONDS_PER_WORD
from train_speaking.session.clock import FakeClock
from train_speaking.session.orchestrator import ACK_PHRASES, InterviewOrchestrator
from train_speaking.voice.sentence_splitter import split_sentences


async def _started(
    orchestrator: InterviewOrchestrator, auto_ack: bool = True
) -> RecordingTransport:
    transport = RecordingTransport(orchestrator, auto_ack=auto_ack)
    await orchestrator.attach(transport)
    await orchestrator.start()
    return transport


async def test_barge_in_stops_playback_and_truncates_question() -> None:
    clock = FakeClock()
    orchestrator = make_orchestrator(make_config(allow_barge_in=True), clock=clock)
    transport = await _started(orchestrator, auto_ack=False)
    await wait_for(lambda: bool(transport.of_type("utterance_end")))
    question = transport.of_type("interviewer_text")[0]
    orchestrator.playback_started(question["utterance_id"])
    first_sentence = split_sentences(question["text"])[0]
    clock.advance(len(first_sentence.split()) * SECONDS_PER_WORD + 0.1)

    await orchestrator.on_audio(speech_pcm(1.0))
    assert orchestrator.state is S.USER_SPEAKING
    await wait_for(lambda: bool(transport.of_type("stop_playback")))
    interrupted = orchestrator.session.interviewer_turns()[0]
    assert interrupted.interrupted is True
    assert interrupted.text == first_sentence
    assert orchestrator.session.interruptions == 1

    transport.auto_ack = True
    await orchestrator.on_audio(silence_pcm(1.5))
    await wait_for(lambda: len(orchestrator.session.candidate_turns()) == 1)
    await wait_for_state(orchestrator, S.LISTENING)


async def test_without_barge_in_speech_during_question_is_ignored() -> None:
    orchestrator = make_orchestrator(make_config(allow_barge_in=False))
    transport = await _started(orchestrator, auto_ack=False)
    await wait_for(lambda: bool(transport.of_type("utterance_end")))
    await orchestrator.on_audio(speech_pcm(1.0))
    assert orchestrator.state is S.INTERVIEWER_SPEAKING
    assert orchestrator.session.interruptions == 0
    assert not transport.of_type("stop_playback")


async def test_acknowledgement_is_spoken_after_an_answer() -> None:
    orchestrator = make_orchestrator(make_config(acknowledgements=True))
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await answer_by_voice(orchestrator)
    await wait_for(lambda: orchestrator.session.question_count == 2)
    acks = [m for m in transport.of_type("interviewer_text") if m["kind"] == "ack"]
    assert len(acks) == 1 and acks[0]["text"] in ACK_PHRASES
    # Acknowledgements are not transcript turns and are not counted as questions.
    assert all(t.text not in ACK_PHRASES for t in orchestrator.session.turns)


async def test_no_acknowledgement_by_default() -> None:
    orchestrator = make_orchestrator(make_config())
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await answer_by_voice(orchestrator)
    await wait_for(lambda: orchestrator.session.question_count == 2)
    assert not [m for m in transport.of_type("interviewer_text") if m["kind"] == "ack"]


async def test_interim_transcript_sent_while_user_speaks() -> None:
    clock = FakeClock()
    options = replace(TEST_OPTIONS, interim_transcripts=True)
    orchestrator = make_orchestrator(
        make_config(), stt=FakeSTT(default="partial words"), clock=clock, options=options
    )
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await orchestrator.on_audio(speech_pcm(1.5))
    assert orchestrator.state is S.USER_SPEAKING
    clock.advance(3)
    await orchestrator.tick()
    await wait_for(lambda: bool(transport.of_type("interim_transcript")))
    assert transport.of_type("interim_transcript")[0]["text"] == "partial words"


async def test_interim_transcripts_disabled_by_default() -> None:
    clock = FakeClock()
    orchestrator = make_orchestrator(make_config(), clock=clock)
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    await orchestrator.on_audio(speech_pcm(1.5))
    clock.advance(3)
    await orchestrator.tick()
    assert not transport.of_type("interim_transcript")


async def test_visemes_sent_only_when_enabled() -> None:
    for enabled in (False, True):
        options = replace(TEST_OPTIONS, send_visemes=enabled)
        orchestrator = make_orchestrator(make_config(), options=options)
        transport = await _started(orchestrator)
        await wait_for_state(orchestrator, S.LISTENING)
        tracks = transport.of_type("visemes")
        assert bool(tracks) is enabled
        if enabled:
            assert tracks[0]["track"][-1][1] == "rest"


async def test_timer_reports_tokens_and_cost() -> None:
    clock = FakeClock()
    orchestrator = make_orchestrator(make_config(), clock=clock)
    transport = await _started(orchestrator)
    await wait_for_state(orchestrator, S.LISTENING)
    usage = orchestrator.session.token_usage
    usage.input_tokens, usage.output_tokens, usage.estimated_cost = 1200, 300, 0.01234
    clock.advance(5)
    await orchestrator.tick()
    timer = transport.of_type("timer")[-1]
    assert timer["tokens"] == 1500
    assert timer["estimated_cost"] == 0.0123
    assert timer["question_count"] == 1
    assert timer["elapsed_s"] >= 5
