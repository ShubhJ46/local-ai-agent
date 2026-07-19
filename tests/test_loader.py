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
