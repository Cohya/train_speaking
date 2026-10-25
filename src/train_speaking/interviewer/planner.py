"""Interview plans: a deterministic fallback plan and LLM-generated plans."""

from __future__ import annotations

import logging
import re

from train_speaking.models.config import InterviewConfig, InterviewFocus
from train_speaking.models.interview import InterviewPlan, PlannedTopic, TopicKind

logger = logging.getLogger(__name__)

MAX_ROLE_TITLE_CHARS = 60
OPENING_MINUTES = 1.5

_TECHNICAL_BANK: list[tuple[str, str, list[str]]] = [
    (
        "Technical depth",
        "Tell me about the most technically challenging project you worked on recently. "
        "What made it difficult, and how did you approach it?",
        ["How do you decide between two competing technical designs?"],
    ),
    (
        "Problem solving",
        "Imagine a critical system you own suddenly becomes slow in production. "
        "How would you investigate and resolve it?",
        ["Walk me through how you debug a problem you have never seen before."],
    ),
    (
        "Role requirements",
        "Which requirement in this role's description is your strongest area, "
        "and can you give a concrete example of applying it?",
        ["Which requirement in the description would you need to grow into, and how?"],
    ),
    (
        "Quality and trade-offs",
        "How do you balance delivery speed against quality and reliability?",
        ["Tell me about a time you had to cut scope. How did you decide what to drop?"],
    ),
]

_BEHAVIORAL_BANK: list[tuple[str, str, list[str]]] = [
    (
        "Collaboration and conflict",
        "Tell me about a time you disagreed with a colleague on an important decision. "
        "How did you handle it?",
        ["How do you give difficult feedback to a peer?"],
    ),
    (
        "Ownership and failure",
        "Describe a mistake you were responsible for. What happened, and what did you learn?",
        ["Tell me about a project that did not go as planned. What would you change?"],
    ),
    (
        "Communication",
        "Tell me about a time you had to explain a complex topic to a non-expert audience.",
        ["How do you make sure stakeholders understand the risks of a decision?"],
    ),
    (
        "Prioritization",
        "Describe a situation where you had several urgent priorities at once. "
        "How did you decide what to do first?",
        ["How do you say no to a request from a senior stakeholder?"],
    ),
]

FINAL_QUESTION = (
    "We're almost out of time, so a final question: why do you think you would be "
    "a strong fit for this role?"
)


def role_title_from(job_description: str) -> str:
    for line in job_description.splitlines():
        title = re.sub(r"^[#*\-\s]+", "", line).strip(" :*")
        if title:
            return title[:MAX_ROLE_TITLE_CHARS]
    return "this"


def _interleave(
    first: list[tuple[str, str, list[str]]], second: list[tuple[str, str, list[str]]]
) -> list[tuple[str, str, list[str], TopicKind]]:
    out: list[tuple[str, str, list[str], TopicKind]] = []
    for a, b in zip(first, second, strict=False):
        out.append((*a, TopicKind.TECHNICAL))
        out.append((*b, TopicKind.BEHAVIORAL))
    return out


def fallback_plan(config: InterviewConfig) -> InterviewPlan:
    role = role_title_from(config.job_description)
    if config.practice_question:
        return InterviewPlan(
            role_title=role,
            topics=[
                PlannedTopic(
                    id="practice",
                    title="Practice question",
                    kind=TopicKind.COMMUNICATION,
                    minutes=config.duration_min,
                    opening_question=config.practice_question,
                )
            ],
        )
    if config.focus is InterviewFocus.TECHNICAL:
        bank = [(*t, TopicKind.TECHNICAL) for t in _TECHNICAL_BANK]
    elif config.focus is InterviewFocus.BEHAVIORAL:
        bank = [(*t, TopicKind.BEHAVIORAL) for t in _BEHAVIORAL_BANK]
    else:
        bank = _interleave(_TECHNICAL_BANK, _BEHAVIORAL_BANK)
    main_minutes = max(1.0, (config.duration_min - OPENING_MINUTES) / len(bank))
    topics = [
        PlannedTopic(
            id="opening",
            title="Background and motivation",
            kind=TopicKind.OPENING,
            minutes=OPENING_MINUTES,
            opening_question=(
                "To start, could you briefly walk me through your background and what "
                f"draws you to the {role} role?"
            ),
        )
    ]
    topics.extend(
        PlannedTopic(
            id=f"t{index}",
            title=title,
            kind=kind,
            minutes=round(main_minutes, 1),
            opening_question=question,
            backup_questions=backups,
        )
        for index, (title, question, backups, kind) in enumerate(bank, start=1)
    )
    return InterviewPlan(role_title=role, topics=topics)
