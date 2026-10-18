"""Minimal async chat client for OpenRouter (any model on it), with retries.

Every call goes through ``chat`` and returns a ``ChatResult``; the rest of the pipeline only
relies on that interface, so a fake client (see ``interviews.FakeClient``) can stand in for it.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class ChatResult:
    content: str
    reasoning: str | None
    finish_reason: str | None
    model_returned: str | None
    provider: str | None          # which host served the call (OpenRouter's 'provider' field)
    usage: dict = field(default_factory=dict)  # token counts and 'cost' in dollars
    latency_s: float = 0.0
    attempts: int = 1
    raw_id: str | None = None


class OpenRouterClient:
    """Thin wrapper over the OpenAI SDK pointed at OpenRouter. Retries a few times with
    exponential backoff; ``extra_body`` carries OpenRouter-specific options such as provider
    pinning (``{"provider": {"order": [...], "allow_fallbacks": False}}``)."""

    def __init__(self, timeout_s: float = 180, max_retries: int = 5):
        from openai import AsyncOpenAI  # imported here so offline runs do not need it

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY (in the environment or a .env file).")
        self._client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, timeout=timeout_s,
                                   max_retries=0, default_headers={"X-Title": "welfare-interviews"})
        self.max_retries = max_retries

    async def chat(self, *, model: str, messages: list[dict], temperature: float = 1.0,
                   max_tokens: int = 800, extra_body: dict | None = None, **_) -> ChatResult:
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            started = time.monotonic()
            try:
                resp = await self._client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, extra_body=extra_body or {})
                choice = resp.choices[0]
                extra = getattr(choice.message, "model_extra", None) or {}
                content = choice.message.content or ""
                if not content.strip() and choice.finish_reason != "length":
                    raise RuntimeError(f"empty reply (finish_reason={choice.finish_reason})")
                return ChatResult(
                    content=content,
                    reasoning=extra.get("reasoning"),
                    finish_reason=choice.finish_reason,
                    model_returned=resp.model,
                    provider=(resp.model_extra or {}).get("provider"),
                    usage=resp.usage.model_dump() if resp.usage else {},
                    latency_s=round(time.monotonic() - started, 2),
                    attempts=attempt,
                    raw_id=resp.id,
                )
            except Exception as e:  # noqa: BLE001 - retried; re-raised below if it keeps failing
                last_err = e
                if attempt < self.max_retries:
                    await asyncio.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(f"{model}: failed after {self.max_retries} attempts: {last_err}")
