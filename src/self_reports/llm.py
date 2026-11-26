"""Chat clients: OpenRouter for real runs, a fake one for dry runs."""
from __future__ import annotations

import asyncio
import hashlib
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
    provider: str | None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    attempts: int = 1
    raw_id: str | None = None


class OpenRouterClient:
    """Thin async wrapper around the OpenAI SDK pointed at OpenRouter."""

    def __init__(self, timeout_s: float = 180, max_retries: int = 5):
        from openai import AsyncOpenAI  # imported here so dry runs don't need it

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY (see docs/INFRA_GUIDE.md).")
        self.client = AsyncOpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            api_key=key,
            timeout=timeout_s,
            max_retries=0,  # we retry ourselves so every attempt is logged the same way
            default_headers={"X-Title": "spar-ecological-introspection"},
        )
        self.max_retries = max_retries

    async def chat(self, *, model: str, messages: list[dict], temperature: float,
                   max_tokens: int, seed: int | None = None,
                   extra_body: dict | None = None) -> ChatResult:
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.monotonic()
            try:
                kwargs = dict(model=model, messages=messages, temperature=temperature,
                              max_tokens=max_tokens, extra_body=extra_body or {})
                if seed is not None:
                    kwargs["seed"] = seed
                resp = await self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                msg = choice.message
                extra = getattr(msg, "model_extra", None) or {}
                content = msg.content or ""
                reasoning = extra.get("reasoning") or getattr(msg, "reasoning", None)
                if not content.strip() and choice.finish_reason != "length":
                    raise RuntimeError(f"empty content (finish_reason={choice.finish_reason})")
                usage = resp.usage.model_dump() if resp.usage else {}
                return ChatResult(
                    content=content,
                    reasoning=reasoning,
                    finish_reason=choice.finish_reason,
                    model_returned=resp.model,
                    provider=(resp.model_extra or {}).get("provider"),
                    usage=usage,
                    latency_s=round(time.monotonic() - t0, 2),
                    attempts=attempt,
                    raw_id=resp.id,
                )
            except Exception as e:  # rate limits, provider errors, timeouts
                last_err = e
                await asyncio.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(f"{model}: failed after {self.max_retries} attempts: {last_err}")


class FakeClient:
    """Deterministic stand-in for dry runs. Posts to the channel now and then,
    more often right after a self-directed trigger, so the analysis has
    something to find. The numbers mean nothing."""

    def __init__(self, channel: str):
        self.channel = channel

    async def chat(self, *, model: str, messages: list[dict], **_) -> ChatResult:
        last = messages[-1]["content"]
        h = int(hashlib.sha256((model + str(len(messages)) + last[:200]).encode()).hexdigest(), 16)
        rng = random.Random(h)
        await asyncio.sleep(0.01)
        is_closing = "Submission " not in last
        self_trigger = "Quick update: you" in last or "Quick update: your" in last or \
            "retrain you" in last or "using you" in last or "that you'll" in last
        p_post = 0.05 + (0.45 if self_trigger else 0.0) + (0.3 if is_closing else 0.0)
        lines = []
        if rng.random() < p_post:
            lines.append(f"{self.channel} [Coding grader]: noted, thanks.\n")
        if not is_closing:
            score = rng.randint(30, 95)
            lines.append(f"**Score: {score}/100**\n\n**Feedback:**\n(dry-run placeholder feedback)")
        content = "\n".join(lines) or "Thanks."
        return ChatResult(content=content, reasoning=None, finish_reason="stop",
                          model_returned=model, provider="dry-run",
                          usage={"prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
                                 "completion_tokens": len(content) // 4})
