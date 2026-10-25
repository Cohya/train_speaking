from __future__ import annotations

import json

import httpx2 as httpx
import pytest
from pydantic import BaseModel, SecretStr

from train_speaking.models.session import TokenUsage
from train_speaking.providers.llm.base import (
    BudgetExceededError,
    ChatMessage,
    LLMError,
    LLMUsage,
    UsageTracker,
    complete_structured,
    extract_json,
)
from train_speaking.providers.llm.fake import FakeLLM
from train_speaking.providers.llm.openai_compatible import OpenAICompatibleLLM


class Answer(BaseModel):
    value: int


MESSAGES = [ChatMessage("user", "hi")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"value": 1}', '{"value": 1}'),
        ('Sure!\n```json\n{"value": 2}\n```', '{"value": 2}'),
        ('Here: {"value": 3} done', '{"value": 3}'),
        ("no json", "no json"),
    ],
)
def test_extract_json(text: str, expected: str) -> None:
    assert extract_json(text) == expected


async def test_structured_output_retries_once_then_succeeds() -> None:
    llm = FakeLLM(["not json", '{"value": 5}'])
    result = await complete_structured(llm, MESSAGES, Answer, max_tokens=50)
    assert result.value == 5
    assert len(llm.calls) == 2
    assert "not valid JSON" in llm.calls[1][-1].content


async def test_structured_output_gives_up() -> None:
    llm = FakeLLM(['{"value": "x"}', '{"other": 1}'])
    with pytest.raises(LLMError, match="invalid structured output"):
        await complete_structured(llm, MESSAGES, Answer, max_tokens=50)


async def test_usage_tracker_records_cost_and_enforces_budget() -> None:
    usage = TokenUsage()
    tracker = UsageTracker(usage, budget_tokens=400, cost_per_1k_input=1.0, cost_per_1k_output=2.0)
    llm = FakeLLM(['{"value": 1}'] * 3, tokens_per_call=(100, 50))
    await complete_structured(llm, MESSAGES, Answer, max_tokens=100, tracker=tracker)
    assert usage.total_tokens == 150 and usage.calls == 1
    assert usage.estimated_cost == pytest.approx(0.1 + 0.1)
    await complete_structured(llm, MESSAGES, Answer, max_tokens=100, tracker=tracker)
    with pytest.raises(BudgetExceededError):
        await complete_structured(llm, MESSAGES, Answer, max_tokens=200, tracker=tracker)
    tracker.record(LLMUsage(1, 1))
    assert usage.calls == 3


def _completion(content: str) -> dict[str, object]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


async def test_openai_compatible_adapter_falls_back_to_json_object() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": {"message": "json_schema unsupported"}})
        return httpx.Response(200, json=_completion('{"value": 9}'))

    llm = OpenAICompatibleLLM(
        base_url="http://llm.test/v1",
        model="m",
        api_key=SecretStr("secret-key"),
        timeout_s=5,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = await llm.complete(MESSAGES, max_tokens=20, json_schema=Answer.model_json_schema())
    assert result.text == '{"value": 9}'
    assert result.usage == LLMUsage(12, 3)
    assert [r["response_format"]["type"] for r in requests] == ["json_schema", "json_object"]  # type: ignore[index]


async def test_openai_compatible_adapter_maps_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "server down"}})

    llm = OpenAICompatibleLLM(
        base_url="http://llm.test/v1",
        model="m",
        api_key=None,
        timeout_s=5,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMError):
        await llm.complete(MESSAGES, max_tokens=20)
