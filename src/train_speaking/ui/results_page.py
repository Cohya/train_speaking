"""`/results/{session_id}` and `/sessions`: the report, transcript, replay, and practice."""

from __future__ import annotations

from nicegui import ui

from train_speaking.models.report import CategoryScore, Report
from train_speaking.models.session import Session, SessionState
from train_speaking.models.transcript import Role
from train_speaking.reporting.markdown import BASIS_LABELS, render_markdown
from train_speaking.runtime import AppRuntime
from train_speaking.session.manager import SessionLimitError
from train_speaking.session.storage import StorageError
from train_speaking.ui.common import add_assets, format_seconds, header

PRACTICE_DURATION_MIN = 3
POLL_INTERVAL_S = 1.0


def _category_card(category: CategoryScore) -> None:
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(category.label).classes("text-subtitle1 text-weight-medium")
            with ui.row().classes("items-center gap-2"):
                ui.label(BASIS_LABELS[category.basis]).classes(
                    f"ts-badge ts-badge-{category.basis.value}"
                )
                score = f"{category.score} / 10" if category.score is not None else "Not assessed"
                ui.label(score).classes("text-h6")
        if category.feedback:
            ui.label(category.feedback)
        for evidence in category.evidence:
            note = f" ({evidence.note})" if evidence.note else ""
            ui.label(f'"{evidence.quote}"{note}').classes("text-grey-8 italic")
        if category.measurements:
            ui.label(
                ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in category.measurements.items())
            ).classes("text-caption text-grey-7")


def _audio_src(runtime: AppRuntime, session_id: str, filename: str) -> str:
    return f"/api/sessions/{session_id}/audio/{filename}?token={runtime.api_token}"


def _render_report(runtime: AppRuntime, session: Session, report: Report) -> None:
    with ui.card().classes("w-full"):
        overall = f"{report.overall_score:.1f} / 10" if report.overall_score is not None else "n/a"
        ui.label(f"Overall score: {overall}").classes("text-h5")
        ui.label(report.summary)
        if report.flagged_evidence_count:
            ui.label(
                f"{report.flagged_evidence_count} AI-cited quote(s) did not match your transcript "
                "and were removed."
            ).classes("text-caption text-orange-9")

    if report.practice_comparison is not None:
        pc = report.practice_comparison
        with ui.card().classes("w-full"):
            ui.label("Practice comparison").classes("text-h6")
            ui.label(f'Original: "{pc.original_answer}"').classes("text-grey-8")
            ui.label(f'New: "{pc.new_answer}"')
            for key in sorted(set(pc.original_metrics) | set(pc.new_metrics)):
                ui.label(
                    f"{key.replace('_', ' ')}: {pc.original_metrics.get(key)} -> "
                    f"{pc.new_metrics.get(key)}"
                ).classes("text-caption")

    ui.label("Category scores").classes("text-h6")
    for category in report.categories:
        _category_card(category)

    m = report.metrics
    with ui.card().classes("w-full"):
        ui.label("Measured metrics").classes("text-h6")
        for label, value in (
            ("Answers", m.answer_count),
            ("Total speaking time", format_seconds(m.total_speaking_s)),
            ("Mean pace (words/min)", m.mean_wpm),
            ("Fillers um/uh per 100 words (lower bound)", m.fillers_per_100_words),
            ("Possible fillers per 100 words", m.possible_fillers_per_100_words),
            ("Pauses over 2 s per minute", m.pauses_gt_2s_per_min),
            ("Longest pause (s)", m.longest_pause_s),
            ("Mean response latency (s)", m.mean_response_latency_s),
            ("Lexical diversity (MTLD)", m.lexical_diversity_mtld),
        ):
            ui.label(f"{label}: {value if value is not None else 'n/a'}")

    for title, items in (("Strengths", report.strengths), ("Areas to improve", report.weaknesses)):
        if items:
            with ui.card().classes("w-full"):
                ui.label(title).classes("text-h6")
                for item in items:
                    ui.label(f"- {item}")

    if report.corrections:
        with ui.card().classes("w-full"):
            ui.label("Example corrections").classes("text-h6")
            for correction in report.corrections:
                ui.label(f'Original: "{correction.original}"').classes("text-grey-8")
                ui.label(f'Corrected: "{correction.corrected}"')
                ui.label(f'More natural: "{correction.natural}"').classes("text-positive")
                ui.separator()

    if report.practice_plan:
        with ui.card().classes("w-full"):
            ui.label("Practice plan").classes("text-h6")
            for drill in report.practice_plan:
                ui.label(drill.title).classes("text-weight-medium")
                ui.label(f"{drill.instructions} Target: {drill.target}")

    with ui.card().classes("w-full"):
        ui.label("Methodology and limitations").classes("text-h6")
        for item in report.methodology:
            ui.label(f"- {item}").classes("text-caption")


def _render_questions(runtime: AppRuntime, session: Session, live: bool) -> None:
    ui.label("Questions and answers").classes("text-h6")
    for turn in session.turns:
        if turn.role is not Role.CANDIDATE:
            continue
        question = session.question_for(turn)
        with ui.card().classes("w-full"):
            if question is not None:
                ui.label(f"Q: {question.text}").classes("text-weight-medium ts-turn-interviewer")
            ui.label(f"A: {turn.text}").classes("ts-turn-candidate")
            if turn.metrics is not None and turn.metrics.wpm is not None:
                ui.label(
                    f"{turn.metrics.wpm:.0f} words/min, {turn.metrics.word_count} words, "
                    f"{turn.metrics.pause_count_gt_2s} pauses over 2 s"
                ).classes("text-caption text-grey-7")
            if turn.audio_file:
                ui.audio(_audio_src(runtime, session.id, turn.audio_file))
            if live and question is not None and session.parent_session_id is None:
                ui.button(
                    "Practice this answer again",
                    on_click=lambda q=question.text, t=turn.id: _start_practice(
                        runtime, session, q, t
                    ),
                ).props("flat dense")


def _start_practice(runtime: AppRuntime, session: Session, question: str, turn_id: int) -> None:
    config = session.config.model_copy(
        update={"practice_question": question, "duration_min": PRACTICE_DURATION_MIN}
    )
    try:
        practice = runtime.manager.create(
            config, parent_session_id=session.id, practice_turn_id=turn_id
        )
    except SessionLimitError as exc:
        ui.notify(str(exc), type="negative")
        return
    ui.navigate.to(f"/interview/{practice.session.id}")


def register_results_pages(runtime: AppRuntime) -> None:
    @ui.page("/results/{session_id}")
    def results_page(session_id: str) -> None:
        add_assets()
        ui.page_title("Interview report")
        session_runtime = runtime.manager.get(session_id)
        report: Report | None
        if session_runtime is not None:
            session = session_runtime.session
            report = session_runtime.orchestrator.report
            live = True
        else:
            try:
                session, report = runtime.storage.load(session_id)
            except StorageError:
                ui.label("This session was not found. Unsaved sessions are removed on restart.")
                ui.link("Start a new interview", "/")
                return
            live = False

        with ui.column().classes("ts-container gap-4 p-4"):
            header("Your interview report")
            if session.state is SessionState.EVALUATING:
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="lg")
                    ui.label("Preparing your report...")

                def wait_for_report() -> None:
                    if session.state is not SessionState.EVALUATING:
                        ui.navigate.reload()

                ui.timer(POLL_INTERVAL_S, wait_for_report)
                return

            with ui.row().classes("gap-2"):
                markdown = render_markdown(report) if report is not None else None
                if live and not runtime.storage.is_saved(session.id):

                    def save() -> None:
                        try:
                            runtime.storage.persist(session, report, markdown)
                        except OSError as exc:
                            ui.notify(f"Could not save the session: {exc}", type="negative")
                            return
                        ui.notify("Session saved to the data/sessions folder.", type="positive")

                    ui.button("Save session", on_click=save)
                if report is not None and markdown is not None:
                    ui.button(
                        "Download Markdown",
                        on_click=lambda: ui.download.content(markdown, f"report_{session.id}.md"),
                    ).props("outline")
                    ui.button(
                        "Download JSON",
                        on_click=lambda: ui.download.content(
                            report.model_dump_json(indent=2), f"report_{session.id}.json"
                        ),
                    ).props("outline")

                def delete() -> None:
                    runtime.storage.delete(session.id)
                    runtime.manager.remove(session.id)
                    ui.notify("Session and recordings deleted.")
                    ui.navigate.to("/")

                ui.button("Delete session", on_click=delete).props("outline color=negative")
                ui.button("New interview", on_click=lambda: ui.navigate.to("/")).props("flat")

            if session.errors:
                ui.label(
                    f"{len(session.errors)} issue(s) occurred during the session and were handled."
                ).classes("text-caption text-grey-7")
            if report is not None:
                _render_report(runtime, session, report)
            else:
                ui.label(
                    "No evaluation report is available for this session. The transcript is shown "
                    "below."
                ).classes("text-grey-8")
            _render_questions(runtime, session, live)

    @ui.page("/sessions")
    def sessions_page() -> None:
        add_assets()
        ui.page_title("Saved sessions")
        with ui.column().classes("ts-container gap-4 p-4"):
            header("Saved sessions")
            saved = runtime.storage.list_saved()
            if not saved:
                ui.label("No saved sessions yet. Use 'Save session' on a report to keep it.")
            for session in saved:
                with ui.row().classes("items-center gap-4"):
                    ui.link(
                        f"{session.created_at:%Y-%m-%d %H:%M} - "
                        f"{session.plan.role_title if session.plan else 'Interview'}",
                        f"/results/{session.id}",
                    )
                    ui.label(f"{len(session.candidate_turns())} answers").classes("text-grey-7")
