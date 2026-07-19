from unittest.mock import Mock, patch

from app.memory import ShortTermMemory, retrieve_memory


def test_short_term_memory_keeps_only_the_most_recent_turns():
    memory = ShortTermMemory(max_messages=2)

    memory.add("User", "first")
    memory.add("AI", "second")
    memory.add("User", "third")

    assert "first" not in memory.get_context()
    assert "third" in memory.get_context()


def test_retrieve_memory_returns_nothing_before_anything_is_stored():
    with patch("app.memory.memory_collection_exists", return_value=False):
        assert retrieve_memory("anything") == []


def test_retrieve_memory_returns_stored_text():
    """Regression: this path passed query_filter=... (a literal Ellipsis).

    Every call raised, and a bare `except Exception` turned it into an empty
    list, so long-term recall silently never worked.
    """
    points = Mock(points=[Mock(payload={"text": "Q: where is login\nA: auth.py"})])
    client = Mock(query_points=Mock(return_value=points))

    with (
        patch("app.memory.memory_collection_exists", return_value=True),
        patch("app.memory.get_embedding", return_value=[0.1, 0.2]),
        patch("app.memory.get_client", return_value=client),
    ):
        assert retrieve_memory("login") == ["Q: where is login\nA: auth.py"]

    assert "query_filter" not in client.query_points.call_args.kwargs
