from unittest.mock import Mock, patch

import pytest
import requests

from app.embed import get_embedding
from app.errors import OllamaError
from app.llm import query_llm


@patch("app.embed.requests.post", side_effect=requests.Timeout())
def test_embedding_wraps_network_failures(_post):
    with pytest.raises(OllamaError, match="Could not get an embedding"):
        get_embedding("hello")


@patch("app.llm.requests.post")
def test_llm_returns_valid_response(post):
    response = Mock()
    response.json.return_value = {"response": "A grounded answer"}
    post.return_value = response

    assert query_llm("question") == "A grounded answer"
    response.raise_for_status.assert_called_once()
