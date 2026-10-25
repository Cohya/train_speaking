"""LLM provider interface, structured-output helper, and token budget tracking."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from train_speaking.models.session import TokenUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    pass


class BudgetExceededError(LLMError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: LLMUsage


class LLMProvider(Protocol):
    @property
    def is_cloud(self) -> bool: ...

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.7,
        json_schema: dict[str, object] | None = None,
    ) -> LLMResult: ...


class UsageTracker:
    """Accumulates token usage for one session and enforces its budget."""

    def __init__(
        self,
        usage: TokenUsage,
        budget_tokens: int,
        cost_per_1k_input: float = 0.0,
        cost_per_1k_output: float = 0.0,
    ) -> None:
        self.usage = usage
        self._budget = budget_tokens
        self._cost_in = cost_per_1k_input
        self._cost_out = cost_per_1k_output

    @property
    def remaining(self) -> int:
        return max(0, self._budget - self.usage.total_tokens)

    def check(self, planned_tokens: int) -> None:
        if planned_tokens > self.remaining:
            raise BudgetExceededError(
                f"session token budget exhausted ({self.usage.total_tokens}/{self._budget})"
            )

    def record(self, usage: LLMUsage) -> None:
        self.usage.input_tokens += usage.input_tokens
        self.usage.output_tokens += usage.output_tokens
        self.usage.calls += 1
        self.usage.estimated_cost = (
            self.usage.input_tokens / 1000 * self._cost_in
            + self.usage.output_tokens / 1000 * self._cost_out
        )


def extract_json(text: str) -> str:
    fenced = _JSON_FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


async def complete_structured(
    llm: LLMProvider,
    messages: list[ChatMessage],
    model_cls: type[T],
    *,
    max_tokens: int,
    temperature: float = 0.4,
    tracker: UsageTracker | None = None,
    retries: int = 1,
) -> T:
    """Request JSON matching `model_cls`, validating and retrying once on invalid output."""
    schema = model_cls.model_json_schema()
    attempt_messages = list(messages)
    last_error: Exception | None = None
    for _ in range(retries + 1):
        if tracker is not None:
            tracker.check(max_tokens)
        result = await llm.complete(
            attempt_messages, max_tokens=max_tokens, temperature=temperature, json_schema=schema
        )
        if tracker is not None:
            tracker.record(result.usage)
        try:
            return model_cls.model_validate(json.loads(extract_json(result.text)))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "llm returned invalid structured output", extra={"model": model_cls.__name__}
            )
            attempt_messages = [
                *messages,
                ChatMessage("assistant", result.text[:2000]),
                ChatMessage(
                    "user",
                    "That was not valid JSON for the required schema. Reply with only the JSON "
                    f"object. Error: {str(exc)[:300]}",
                ),
            ]
    raise LLMError(f"invalid structured output after {retries + 1} attempts") from last_error
