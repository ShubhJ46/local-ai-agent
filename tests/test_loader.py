from app.loader import load_documents


def test_loader_extracts_python_functions_and_skips_hidden_directories(tmp_path):
    (tmp_path / "module.py").write_text("def useful():\n    return 42\n")
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "ignored.py").write_text("def ignored(): pass\n")

    documents = load_documents(str(tmp_path))

    assert len(documents) == 1
    assert documents[0]["metadata"]["name"] == "useful"
    assert documents[0]["metadata"]["path"] == str(tmp_path / "module.py")


def test_loader_indexes_a_root_that_is_itself_hidden(tmp_path):
    """A hidden ancestor must not suppress the whole index.

    The hidden-directory rule applies below the indexed root, so a checkout
    living under a dotted directory still indexes normally.
    """
    hidden_root = tmp_path / ".eval-corpus" / "project"
    hidden_root.mkdir(parents=True)
    (hidden_root / "module.py").write_text("def useful():\n    return 42\n")
    nested_hidden = hidden_root / ".git"
    nested_hidden.mkdir()
    (nested_hidden / "ignored.py").write_text("def ignored(): pass\n")

    documents = load_documents(str(hidden_root))

    assert [document["metadata"]["name"] for document in documents] == ["useful"]


def test_loader_rejects_missing_directory(tmp_path):
    try:
        load_documents(str(tmp_path / "missing"))
    except ValueError as error:
        assert "not a directory" in str(error)
    else:
        raise AssertionError("Expected a missing directory to be rejected")


def test_loader_indexes_java_configuration_classes(tmp_path):
    java_file = tmp_path / "SecurityConfig.java"
    java_file.write_text(
        """
        import org.springframework.context.annotation.Configuration;
        @Configuration
        class SecurityConfig {
            String filterChain() { return "authenticated"; }
        }
        """
    )

    documents = load_documents(str(tmp_path))

    assert {document["metadata"]["type"] for document in documents} == {"class", "method"}
    assert any("authenticated" in document["content"] for document in documents)
