"""`/interview/{session_id}`: the live interview.

The browser module (`static/js/media.js`) owns audio, camera, avatar animation, and the
current question text. This page polls the session for progress, controls, and transcript.
"""

from __future__ import annotations

import json

from nicegui import ui

from train_speaking.models.session import SessionState
from train_speaking.runtime import AppRuntime
from train_speaking.session.orchestrator import PAUSABLE_STATES
from train_speaking.session.state_machine import TERMINAL_STATES
from train_speaking.ui.common import STATE_LABELS, add_assets, format_seconds, header
from train_speaking.ui.components.avatar import interviewer_avatar

POLL_INTERVAL_S = 0.5


def _js_call(method: str) -> str:
    return f"() => window.tsMedia && window.tsMedia.{method}()"


def register_interview_page(runtime: AppRuntime) -> None:
    @ui.page("/interview/{session_id}")
    async def interview_page(session_id: str) -> None:
        add_assets()
        ui.page_title("Interview")
        session_runtime = runtime.manager.get(session_id)
        if session_runtime is None:
            ui.label("This interview session does not exist or has expired.")
            ui.link("Start a new interview", "/")
            return
        session = session_runtime.session
        orchestrator = session_runtime.orchestrator
        ui.add_head_html('<script type="module" src="/static/js/media.js"></script>')

        with ui.column().classes("ts-container gap-4 p-4"):
            header(f"Interview: {session.config.duration_min} minutes")
            with ui.row().classes("w-full items-center gap-4"):
                state_label = ui.label().classes("text-subtitle1")
                progress_label = ui.label().classes("text-subtitle1 text-grey-8")
            progress = ui.linear_progress(value=0, show_value=False).classes("w-full")

            with ui.row().classes("w-full gap-6 no-wrap items-start"):
                with ui.card().classes("w-7/12"):
                    interviewer_avatar()
                    ui.label("Press Start when you are ready.").classes("ts-question").props(
                        "id=ts-question"
                    )
                with ui.card().classes("w-5/12 gap-3"):
                    ui.element("video").classes("ts-camera").props(
                        "id=ts-camera autoplay muted playsinline"
                    )
                    if not session.config.consent.camera:
                        ui.label("Camera preview is off.").classes("text-caption text-grey-7")
                    with ui.row().classes("items-center gap-1"):
                        ui.html('<span id="ts-speaking"></span>', sanitize=False)
                        ui.label("Microphone idle").props("id=ts-mic-status")
                    with ui.element("div").classes("ts-meter"):
                        ui.element("div").props("id=ts-mic-level")
                    ui.label().props("id=ts-interim")
                    ui.label().props("id=ts-notice").style("display:none")

            with ui.row().classes("w-full gap-2 items-center"):
                start_btn = ui.button("Start").on("click", js_handler=_js_call("start"))
                pause_btn = (
                    ui.button("Pause").props("outline").on("click", js_handler=_js_call("pause"))
                )
                resume_btn = (
                    ui.button("Resume").props("outline").on("click", js_handler=_js_call("resume"))
                )
                done_btn = (
                    ui.button("I'm done answering")
                    .props("color=positive")
                    .on("click", js_handler=_js_call("doneAnswering"))
                )
                end_btn = (
                    ui.button("End interview")
                    .props("color=negative outline")
                    .on("click", js_handler=_js_call("end"))
                )

            with ui.row().classes("w-full items-center gap-2"):
                text_answer = ui.input(
                    placeholder="No microphone? Type your answer here and press Enter."
                ).classes("grow")

                def send_text() -> None:
                    value = (text_answer.value or "").strip()
                    if value:
                        ui.run_javascript(f"window.tsMedia.sendText({json.dumps(value)})")
                        text_answer.value = ""

                text_answer.on("keydown.enter", send_text)
                ui.button("Send", on_click=send_text).props("flat")

            show_transcript = ui.switch("Show live transcript", value=True)
            with ui.card().classes("w-full ts-transcript") as transcript_card:
                transcript_column = ui.column().classes("w-full gap-2")
            transcript_card.bind_visibility_from(show_transcript, "value")

        rendered = {"signature": None}

        def render_transcript() -> None:
            signature = tuple((t.id, t.text) for t in session.turns)
            if signature == rendered["signature"]:
                return
            rendered["signature"] = signature  # type: ignore[assignment]
            transcript_column.clear()
            with transcript_column:
                if not session.turns:
                    ui.label("The transcript will appear here.").classes("text-grey-6")
                for turn in session.turns:
                    speaker = "Interviewer" if turn.role.value == "interviewer" else "You"
                    suffix = " (interrupted)" if turn.interrupted else ""
                    ui.label(f"{speaker}: {turn.text}{suffix}").classes(
                        f"ts-turn-{turn.role.value}"
                    )

        def refresh() -> None:
            state = orchestrator.state
            elapsed = orchestrator.elapsed()
            duration = session.config.duration_s
            state_label.text = STATE_LABELS[state]
            progress_label.text = (
                f"Question {session.question_count} | "
                f"{format_seconds(elapsed)} / {format_seconds(duration)}"
            )
            usage = session.token_usage
            if usage.total_tokens:
                progress_label.text += f" | {usage.total_tokens} tokens"
                if usage.estimated_cost:
                    progress_label.text += f" (~${usage.estimated_cost:.3f})"
            progress.value = min(1.0, elapsed / duration)
            start_btn.set_enabled(state is SessionState.READY)
            pause_btn.set_enabled(state in PAUSABLE_STATES)
            resume_btn.set_enabled(state is SessionState.PAUSED)
            done_btn.set_enabled(state is SessionState.USER_SPEAKING)
            end_btn.set_enabled(
                state not in TERMINAL_STATES and state is not SessionState.EVALUATING
            )
            render_transcript()
            if state in TERMINAL_STATES or state is SessionState.EVALUATING:
                timer.deactivate()
                ui.navigate.to(f"/results/{session.id}")

        timer = ui.timer(POLL_INTERVAL_S, refresh)
        refresh()

        await ui.context.client.connected()
        options = {
            "sessionId": session.id,
            "token": session_runtime.token,
            "camera": session.config.consent.camera,
        }
        ui.run_javascript(
            "(function init(){ if (window.tsMedia) { window.tsMedia.init("
            + json.dumps(options)
            + "); } else { setTimeout(init, 50); } })()"
        )
