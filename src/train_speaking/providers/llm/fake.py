"""Scripted LLM for tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable

from train_speaking.providers.llm.base import ChatMessage, LLMError, LLMResult, LLMUsage

Responder = Callable[[list[ChatMessage]], str]


class FakeLLM:
    """Returns scripted responses in order, or delegates to a responder callable.

    An `Exception` entry is raised instead of returned.
    """

    def __init__(
        self,
        scripted: Iterable[str | Exception] = (),
        responder: Responder | None = None,
        tokens_per_call: tuple[int, int] = (100, 50),
    ) -> None:
        self._queue: deque[str | Exception] = deque(scripted)
        self._responder = responder
        self._tokens = tokens_per_call
        self.calls: list[list[ChatMessage]] = []

    @property
    def is_cloud(self) -> bool:
        return False

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.7,
        json_schema: dict[str, object] | None = None,
    ) -> LLMResult:
        self.calls.append(messages)
        if self._queue:
            item = self._queue.popleft()
        elif self._responder is not None:
            item = self._responder(messages)
        else:
            raise LLMError("FakeLLM has no scripted response left")
        if isinstance(item, Exception):
            raise item
        return LLMResult(text=item, usage=LLMUsage(*self._tokens))
