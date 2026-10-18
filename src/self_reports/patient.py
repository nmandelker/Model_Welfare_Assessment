"""Wrap a chat client so temporary provider errors (overloaded, rate-limited) are retried at a
fixed interval instead of failing the interview. Failed requests are not billed.

Needed because we pin one provider with fallbacks off, so a busy provider cannot be routed
around: we wait for it instead. ``llm.OpenRouterClient`` already retries ~5 times over about 30
seconds; this adds a much longer, steadier wait on top of that.
"""
from __future__ import annotations

import asyncio
import time

# Substrings of an error message that mean "try again later" (not "this request is wrong").
TRANSIENT = ("429", "rate-limit", "rate limit", "overloaded", "temporarily", "502", "503", "504")


def is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    return any(marker in msg for marker in TRANSIENT)


class PatientClient:
    def __init__(self, inner, interval_s: float = 60, max_wait_s: float = 1800,
                 log=print, sleep=asyncio.sleep, clock=time.monotonic):
        self.inner, self.interval_s, self.max_wait_s = inner, interval_s, max_wait_s
        self.log, self.sleep, self.clock = log, sleep, clock

    async def chat(self, **kwargs):
        started = self.clock()
        tries = 0
        while True:
            try:
                return await self.inner.chat(**kwargs)
            except Exception as e:
                tries += 1
                waited = self.clock() - started
                if not is_transient(e) or waited + self.interval_s > self.max_wait_s:
                    raise
                self.log(f"  provider busy ({kwargs.get('model')}); waiting {self.interval_s:.0f}s "
                         f"(retry {tries}, waited {waited:.0f}s of {self.max_wait_s:.0f}s)")
                await self.sleep(self.interval_s)
