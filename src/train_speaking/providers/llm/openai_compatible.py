"""Any OpenAI-compatible chat endpoint: Ollama, LM Studio, vLLM, OpenAI, and others."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import openai
from pydantic import SecretStr

from train_speaking.providers.llm.base import ChatMessage, LLMError, LLMResult, LLMUsage

if TYPE_CHECKING:
    # openai>=3 is built on httpx2 (installed with it), not the classic httpx package.
    import httpx2

logger = logging.getLogger(__name__)

LOCAL_PLACEHOLDER_KEY = "not-needed"


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: SecretStr | None,
        timeout_s: float,
        is_cloud: bool = False,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._is_cloud = is_cloud
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key.get_secret_value() if api_key else LOCAL_PLACEHOLDER_KEY,
            timeout=timeout_s,
            max_retries=1,
            http_client=http_client,
        )
        self._json_schema_supported = True

    @property
    def is_cloud(self) -> bool:
        return self._is_cloud

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.7,
        json_schema: dict[str, object] | None = None,
    ) -> LLMResult:
        payload: list[Any] = [{"role": m.role, "content": m.content} for m in messages]
        response_format: Any = openai.NOT_GIVEN
        if json_schema is not None:
            response_format = (
                {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": json_schema},
                }
                if self._json_schema_supported
                else {"type": "json_object"}
            )
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=payload,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
        except openai.BadRequestError as exc:
            if json_schema is not None and self._json_schema_supported:
                logger.info("endpoint rejected json_schema; retrying with json_object")
                self._json_schema_supported = False
                return await self.complete(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_schema=json_schema,
                )
            raise LLMError(f"LLM request rejected: {exc.message}") from exc
        except openai.APIError as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc

        if not response.choices:
            raise LLMError("LLM returned no choices")
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResult(
            text=text,
            usage=LLMUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )
