"""Shared test configuration.

The suite is meant to run without Ollama, so CI can gate on it. That property is
easy to lose by accident: a test that mocks one layer but not the one beneath it
still passes on a developer machine where Ollama happens to be running, and only
fails in CI. This makes such a test fail everywhere, immediately, with a message
that names the cause.
"""

import pytest
import requests


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError(
            "A test attempted a real HTTP request. Mock the boundary it uses "
            "(app.embed.requests.post, app.retrieval.get_embedding, "
            "app.agent.query_llm, ...) so the suite stays runnable offline."
        )

    monkeypatch.setattr(requests.sessions.Session, "request", refuse)
