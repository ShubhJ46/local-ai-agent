from unittest.mock import patch

import pytest

from app.agent import run_agent
from app.memory import ShortTermMemory


@pytest.fixture
def memory():
    return ShortTermMemory()


RETRIEVED = [
    {
        "id": 1,
        "type": "endpoint",
        "name": "login",
        "file": "AuthController.java",
        "text": "public String login() {}",
        "http_method": "POST",
        "endpoint": "/login",
        "start_line": 1,
        "end_line": 3,
    }
]


@pytest.fixture(autouse=True)
def _no_long_term_memory():
    """Keep the loop off Qdrant and Ollama; long-term memory is covered separately."""
    with (
        patch("app.agent.retrieve_memory", return_value=[]),
        patch("app.agent.store_memory"),
        patch("app.agent.hybrid_search", return_value=RETRIEVED),
    ):
        yield


def test_agent_retrieves_before_the_first_model_call():
    """The model must never answer from parametric knowledge alone.

    Without this the loop happily returned invented file paths, because nothing
    forced a retrieval before FINAL_ANSWER was allowed.
    """
    with patch("app.agent.query_llm", return_value="FINAL_ANSWER: In AuthController.java.") as llm:
        run_agent("Where is login?", memory=ShortTermMemory())

    assert "AuthController.java" in llm.call_args[0][0]


def test_agent_reports_an_empty_index_without_calling_the_model():
    with (
        patch("app.agent.hybrid_search", return_value=[]),
        patch("app.agent.query_llm") as llm,
    ):
        answer = run_agent("Where is login?", memory=ShortTermMemory())

    assert answer == "No indexed documents were found. Run `load <path>` first."
    assert llm.call_count == 0


def test_agent_answers_without_tools_when_it_can(memory):
    with patch("app.agent.query_llm", return_value="FINAL_ANSWER: Login lives in auth.py.") as llm:
        answer = run_agent("Where is login?", memory=memory)

    assert answer == "Login lives in auth.py."
    assert llm.call_count == 1


def test_agent_executes_a_tool_then_answers(memory):
    replies = [
        "TOOL: search_documents | login handler",
        "FINAL_ANSWER: Login is handled in auth.py.",
    ]
    with (
        patch("app.agent.query_llm", side_effect=replies),
        patch.dict("app.agent.TOOLS", {"search_documents": lambda q: "def login(): ..."}),
    ):
        answer = run_agent("Where is login?", memory=memory)

    assert answer == "Login is handled in auth.py."


def test_agent_feeds_tool_output_back_into_the_next_prompt(memory):
    prompts = []

    def record(prompt, *_args, **_kwargs):
        prompts.append(prompt)
        return "FINAL_ANSWER: done" if len(prompts) > 1 else "TOOL: search_documents | login"

    with (
        patch("app.agent.query_llm", side_effect=record),
        patch.dict("app.agent.TOOLS", {"search_documents": lambda q: "OBSERVED_MARKER"}),
    ):
        run_agent("Where is login?", memory=memory)

    assert "OBSERVED_MARKER" in prompts[1]


def test_agent_reports_an_unknown_tool_instead_of_crashing(memory):
    replies = ["TOOL: nonexistent_tool | x", "FINAL_ANSWER: Answered anyway."]
    with patch("app.agent.query_llm", side_effect=replies):
        assert run_agent("Where is login?", memory=memory) == "Answered anyway."


def test_agent_survives_a_tool_that_raises(memory):
    def explode(_argument):
        raise RuntimeError("index unavailable")

    replies = ["TOOL: search_documents | login", "FINAL_ANSWER: Recovered."]
    with (
        patch("app.agent.query_llm", side_effect=replies),
        patch.dict("app.agent.TOOLS", {"search_documents": explode}),
    ):
        assert run_agent("Where is login?", memory=memory) == "Recovered."


def test_agent_stops_at_max_steps(memory):
    """A model that only ever calls tools must still terminate."""
    with (
        patch("app.agent.query_llm", return_value="TOOL: search_documents | again") as llm,
        patch.dict("app.agent.TOOLS", {"search_documents": lambda q: "same output"}),
        patch("app.agent._direct_answer", return_value="Fell back."),
    ):
        answer = run_agent("Where is login?", memory=memory, max_steps=3)

    assert answer == "Fell back."
    assert llm.call_count == 3


def test_agent_falls_back_when_the_reply_ignores_the_protocol(memory):
    with (
        patch("app.agent.query_llm", return_value="I think it is probably in some file."),
        patch("app.agent._direct_answer", return_value="Fell back.") as fallback,
    ):
        assert run_agent("Where is login?", memory=memory) == "Fell back."

    assert fallback.call_count == 1


def test_agent_records_the_exchange_in_short_term_memory(memory):
    with patch("app.agent.query_llm", return_value="FINAL_ANSWER: In auth.py."):
        run_agent("Where is login?", memory=memory)

    context = memory.get_context()
    assert "Where is login?" in context
    assert "In auth.py." in context


def test_agent_passes_prior_turns_into_the_prompt(memory):
    memory.add("User", "Where is login?")
    memory.add("AI", "In auth.py.")

    with patch("app.agent.query_llm", return_value="FINAL_ANSWER: It validates a token.") as llm:
        run_agent("How does it work?", memory=memory)

    assert "In auth.py." in llm.call_args[0][0]
