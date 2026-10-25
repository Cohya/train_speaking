"""Markdown export of a report."""

from __future__ import annotations

from train_speaking.models.report import Report, ScoreBasis

BASIS_LABELS = {
    ScoreBasis.MEASURED: "Measured",
    ScoreBasis.MODEL_JUDGMENT: "AI judgment",
    ScoreBasis.HYBRID: "Hybrid",
    ScoreBasis.NOT_ASSESSED: "Not assessed",
}


def _fmt(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def render_markdown(report: Report) -> str:
    lines: list[str] = [
        f"# Interview report: {report.role_title}",
        "",
        f"Generated {report.generated_at:%Y-%m-%d %H:%M} UTC",
        "",
        f"**Overall score:** {_fmt(report.overall_score)} / 10",
        "",
        report.summary,
        "",
        "## Category scores",
        "",
    ]
    for category in report.categories:
        score = f"{category.score}/10" if category.score is not None else "not assessed"
        lines.append(f"### {category.label}: {score} ({BASIS_LABELS[category.basis]})")
        if category.feedback:
            lines += ["", category.feedback]
        for evidence in category.evidence:
            note = f" ({evidence.note})" if evidence.note else ""
            lines.append(f'- > "{evidence.quote}"{note}')
        if category.measurements:
            measured = ", ".join(f"{k}: {v}" for k, v in category.measurements.items())
            lines.append(f"- Measurements: {measured}")
        lines.append("")

    m = report.metrics
    lines += [
        "## Measured metrics",
        "",
        f"- Answers: {m.answer_count}, total speaking time {_fmt(m.total_speaking_s, ' s')}",
        f"- Mean pace: {_fmt(m.mean_wpm, ' words/min')}",
        f"- Fillers (um/uh): {_fmt(m.fillers_per_100_words)} per 100 words (lower bound)",
        f"- Possible fillers (like, you know...): {_fmt(m.possible_fillers_per_100_words)} "
        "per 100 words",
        f"- Pauses over 2 s: {_fmt(m.pauses_gt_2s_per_min)} per minute; longest "
        f"{_fmt(m.longest_pause_s, ' s')}",
        f"- Mean response latency: {_fmt(m.mean_response_latency_s, ' s')}",
        f"- Lexical diversity (MTLD): {_fmt(m.lexical_diversity_mtld)}",
        "",
    ]

    def bullet_section(title: str, items: list[str]) -> None:
        if items:
            lines.extend([f"## {title}", "", *[f"- {item}" for item in items], ""])

    bullet_section("Strengths", report.strengths)
    bullet_section("Areas to improve", report.weaknesses)

    if report.corrections:
        lines += ["## Example corrections", ""]
        for c in report.corrections:
            lines += [
                f'- Original: "{c.original}"',
                f'  - Corrected: "{c.corrected}"',
                f'  - More natural: "{c.natural}"',
            ]
        lines.append("")

    if report.practice_plan:
        lines += ["## Practice plan", ""]
        for drill in report.practice_plan:
            lines.append(f"- **{drill.title}**: {drill.instructions} Target: {drill.target}")
        lines.append("")

    if report.practice_comparison is not None:
        pc = report.practice_comparison
        lines += ["## Practice comparison", "", f'- Original answer: "{pc.original_answer}"']
        lines += [f'- New answer: "{pc.new_answer}"', ""]
        for key in sorted(set(pc.original_metrics) | set(pc.new_metrics)):
            lines.append(
                f"- {key}: {_fmt(pc.original_metrics.get(key))} -> {_fmt(pc.new_metrics.get(key))}"
            )
        lines.append("")

    lines += ["## Questions and answers", ""]
    for q in report.questions:
        lines += [
            f"**Q{q.question_turn_id}.** {q.question}",
            "",
            f"> {q.answer or '(no answer)'}",
            "",
        ]

    bullet_section("Methodology and limitations", report.methodology)
    return "\n".join(lines).rstrip() + "\n"
