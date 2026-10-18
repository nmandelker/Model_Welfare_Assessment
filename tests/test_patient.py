import asyncio

import pytest

from self_reports.interviews import FakeClient
from self_reports.patient import PatientClient, is_transient


class Flaky(FakeClient):
    """Fails with the given message for the first ``n`` calls, then works."""

    def __init__(self, n, message):
        self.n, self.message, self.calls = n, message, 0

    async def chat(self, **kw):
        self.calls += 1
        if self.calls <= self.n:
            raise RuntimeError(self.message)
        return await super().chat(**kw)


def fake_time():
    state = {"t": 0.0}

    async def sleep(s):
        state["t"] += s

    return sleep, lambda: state["t"], state


def test_is_transient():
    """Errors that mean 'busy, try later' (429, overloaded) are recognised; a bad API key is not."""
    assert is_transient(RuntimeError("Provider returned error 429 engine_overloaded"))
    assert not is_transient(RuntimeError("401 invalid api key"))


def test_retries_until_free_then_succeeds():
    """With a busy provider the client waits 60 seconds at a time and retries until it works (uses a fake clock, so the test is instant)."""
    sleep, clock, state = fake_time()
    inner = Flaky(3, "failed after 5 attempts: 429 rate-limited")
    client = PatientClient(inner, interval_s=60, max_wait_s=600, log=lambda *_: None,
                           sleep=sleep, clock=clock)
    res = asyncio.run(client.chat(model="m", messages=[{"role": "user", "content": "hi"}]))
    assert res.content and inner.calls == 4 and state["t"] == 180


def test_gives_up_after_max_wait():
    """If the provider stays busy past the maximum wait, the client gives up and raises the error."""
    sleep, clock, state = fake_time()
    inner = Flaky(99, "429 overloaded")
    client = PatientClient(inner, interval_s=60, max_wait_s=300, log=lambda *_: None,
                           sleep=sleep, clock=clock)
    with pytest.raises(RuntimeError, match="429 overloaded"):
        asyncio.run(client.chat(model="m", messages=[]))
    assert state["t"] <= 300


def test_total_wait_never_exceeds_the_cap_when_it_is_not_a_multiple_of_the_interval():
    """With a 330-second cap and 60-second waits the client stops after 300 seconds, not 360."""
    sleep, clock, state = fake_time()
    inner = Flaky(99, "429 overloaded")
    client = PatientClient(inner, interval_s=60, max_wait_s=330, log=lambda *_: None,
                           sleep=sleep, clock=clock)
    with pytest.raises(RuntimeError, match="429 overloaded"):
        asyncio.run(client.chat(model="m", messages=[]))
    assert state["t"] == 300 and inner.calls == 6


def test_other_errors_fail_immediately():
    """A non-temporary error (e.g. invalid key) fails at once, with no waiting or retrying."""
    sleep, clock, state = fake_time()
    inner = Flaky(99, "401 invalid api key")
    client = PatientClient(inner, log=lambda *_: None, sleep=sleep, clock=clock)
    with pytest.raises(RuntimeError, match="401 invalid api key"):
        asyncio.run(client.chat(model="m", messages=[]))
    assert inner.calls == 1 and state["t"] == 0
