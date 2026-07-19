from unittest.mock import patch

from app.rag import agent_query, format_context, memory

LOGIN_RESULT = {
    "id": 1,
    "type": "function",
    "name": "login",
    "file": "auth.py",
    "text": "def login(): pass",
    "start_line": 10,
    "end_line": 14,
}


def test_format_context_includes_source_and_file():
    context = format_context([LOGIN_RESULT])

    assert "auth.py" in context
    assert "def login" in context


def test_format_context_cites_line_ranges_when_known():
    assert "auth.py:10-14" in format_context([LOGIN_RESULT])


@patch("app.rag.query_llm", return_value="FINAL_ANSWER: Authentication is handled by login.")
@patch("app.rag.hybrid_search", return_value=[LOGIN_RESULT])
def test_agent_answers_from_retrieved_source(_search, _llm):
    memory.history.clear()

    assert agent_query("Where is authentication handled?") == "Authentication is handled by login."


@patch("app.rag.hybrid_search", return_value=[])
def test_agent_explains_when_index_is_empty(_search):
    expected = "No indexed documents were found. Run `load <path>` first."

    assert agent_query("Where is authentication handled?") == expected
