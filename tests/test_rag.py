from unittest.mock import patch

from app.rag import agent_query, format_context, memory


def test_format_context_includes_source_and_file():
    context = format_context(
        [{"type": "function", "name": "login", "file": "auth.py", "text": "def login(): pass"}]
    )
    assert "auth.py" in context
    assert "def login" in context


@patch("app.rag.query_llm", return_value="FINAL_ANSWER: Authentication is handled by login.")
@patch("app.rag.rerank", side_effect=lambda results, _query: results)
@patch(
    "app.rag.search",
    return_value=[
        {"type": "function", "name": "login", "file": "auth.py", "text": "def login(): pass"}
    ],
)
@patch("app.rag.get_embedding", return_value=[0.1, 0.2])
def test_agent_queries_general_code_when_no_endpoints_exist(_embedding, _search, _rerank, _llm):
    memory.history.clear()
    assert agent_query("Where is authentication handled?") == "Authentication is handled by login."


@patch("app.rag.search", return_value=[])
@patch("app.rag.get_embedding", return_value=[0.1])
def test_agent_explains_when_index_is_empty(_embedding, _search):
    expected = "No indexed documents were found. Run `load <path>` first."
    assert agent_query("Where is authentication handled?") == expected
