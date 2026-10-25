"""`/`: job description, interview rules, settings, and consent."""

from __future__ import annotations

from nicegui import events, ui

from train_speaking.config.loader import (
    MAX_UPLOAD_BYTES,
    ConfigError,
    build_config,
    load_job_description,
)
from train_speaking.models.config import (
    ALL_CATEGORIES,
    MAX_DURATION_MIN,
    MIN_DURATION_MIN,
    InterviewFocus,
    InterviewStyle,
)
from train_speaking.runtime import AppRuntime
from train_speaking.session.manager import SessionLimitError
from train_speaking.ui.common import CATEGORY_LABELS, add_assets, header

EXAMPLE_RULES = """\
- Ask difficult technical questions relevant to the role.
- Ask behavioral and communication questions.
- Ask one question at a time.
- Ask follow-up questions based on my previous answers.
- Do not help me answer during the interview.
- Act like a realistic, professional interviewer."""


def register_setup_page(runtime: AppRuntime) -> None:
    @ui.page("/")
    def setup_page() -> None:
        add_assets()
        ui.page_title("Interview setup")
        with ui.column().classes("ts-container gap-4 p-4"):
            header("Set up your interview")

            with ui.card().classes("w-full"):
                ui.label("1. Job description").classes("text-h6")
                job_description = (
                    ui.textarea(placeholder="Paste the job description here")
                    .props("outlined autogrow")
                    .classes("w-full")
                    .mark("job-description")
                )

                async def on_upload(event: events.UploadEventArguments) -> None:
                    try:
                        job_description.value = load_job_description(
                            event.file.name, await event.file.read()
                        )
                        ui.notify("Job description loaded.", type="positive")
                    except ConfigError as exc:
                        ui.notify(str(exc), type="negative")

                ui.upload(
                    label="...or upload a .txt, .md, or .pdf file",
                    auto_upload=True,
                    max_file_size=MAX_UPLOAD_BYTES,
                    on_upload=on_upload,
                ).props("accept=.txt,.md,.pdf flat bordered").classes("w-full")

            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("2. Interview rules and instructions").classes("text-h6")
                    example_button = ui.button("Use example rules").props("flat")
                rules = (
                    ui.textarea(placeholder="How should the interviewer behave?")
                    .props("outlined autogrow")
                    .classes("w-full")
                    .mark("rules")
                )
                example_button.on_click(lambda: rules.set_value(EXAMPLE_RULES))

            with ui.card().classes("w-full"):
                ui.label("3. Settings").classes("text-h6")
                with ui.row().classes("w-full items-center gap-8"):
                    with ui.column().classes("min-w-[240px]"):
                        duration_label = ui.label()
                        duration = ui.slider(min=MIN_DURATION_MIN, max=MAX_DURATION_MIN, value=8)
                        duration_label.bind_text_from(
                            duration, "value", backward=lambda v: f"Duration: {v} minutes"
                        )
                    style = ui.select(
                        {s.value: s.value.title() for s in InterviewStyle},
                        value=InterviewStyle.NEUTRAL.value,
                        label="Interview style",
                    ).classes("min-w-[160px]")
                    focus = ui.select(
                        {f.value: f.value.title() for f in InterviewFocus},
                        value=InterviewFocus.MIXED.value,
                        label="Question focus",
                    ).classes("min-w-[160px]")
                criteria = (
                    ui.select(
                        {c.value: CATEGORY_LABELS[c] for c in ALL_CATEGORIES},
                        value=[c.value for c in ALL_CATEGORIES],
                        multiple=True,
                        label="Evaluation criteria",
                    )
                    .props("use-chips")
                    .classes("w-full")
                )
                with ui.row().classes("w-full items-center gap-8"):
                    with ui.column().classes("min-w-[240px]"):
                        silence_label = ui.label()
                        silence = ui.slider(min=700, max=2500, step=100, value=1200)
                        silence_label.bind_text_from(
                            silence,
                            "value",
                            backward=lambda v: (
                                f"Pause before your answer counts as finished: {v / 1000:.1f} s"
                            ),
                        )
                    barge_in = ui.switch("Allow interrupting the interviewer (use headphones)")
                    acknowledgements = ui.switch("Short spoken acknowledgements")

            with ui.card().classes("w-full"):
                ui.label("4. Consent").classes("text-h6")
                mic = ui.checkbox("I allow microphone use for this interview (required).").mark(
                    "consent-microphone"
                )
                camera = ui.checkbox(
                    "Show my camera preview. The video stays in the browser and is not analyzed.",
                    value=True,
                )
                record = ui.checkbox(
                    "Record my answers so I can replay them. Recordings are deleted when the "
                    "app closes unless I save the session."
                )
                cloud = None
                if runtime.settings.uses_cloud:
                    cloud = ui.checkbox(
                        "I understand my transcript text is sent to the configured cloud LLM "
                        "provider (required)."
                    )

            def start() -> None:
                if cloud is not None and not cloud.value:
                    ui.notify("Cloud processing consent is required.", type="negative")
                    return
                form = {
                    "job_description": job_description.value or "",
                    "rules": rules.value or "",
                    "duration_min": int(duration.value),
                    "style": style.value,
                    "focus": focus.value,
                    "criteria": list(criteria.value or []),
                    "end_silence_ms": int(silence.value),
                    "allow_barge_in": bool(barge_in.value),
                    "acknowledgements": bool(acknowledgements.value),
                    "consent": {
                        "microphone": bool(mic.value),
                        "camera": bool(camera.value),
                        "record_audio": bool(record.value),
                        "cloud_processing": bool(cloud.value) if cloud is not None else False,
                    },
                }
                try:
                    config = build_config(form)
                    session_runtime = runtime.manager.create(config)
                except ConfigError as exc:
                    for message in exc.messages:
                        ui.notify(message, type="negative", multi_line=True)
                    return
                except SessionLimitError as exc:
                    ui.notify(str(exc), type="negative")
                    return
                ui.navigate.to(f"/interview/{session_runtime.session.id}")

            with ui.row().classes("w-full justify-between items-center"):
                ui.link("Saved sessions", "/sessions")
                ui.button("Start interview", on_click=start).props("size=lg color=primary")
            if not runtime.llm_configured:
                ui.label(
                    "No LLM is configured, so the built-in question bank is used and language "
                    "judgments in the report are skipped. See README to enable an LLM."
                ).classes("text-caption text-grey-7")
