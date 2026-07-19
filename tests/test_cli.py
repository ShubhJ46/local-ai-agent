from app.cli import MAX_WIDTH, wrap


def test_wrap_keeps_a_cited_path_on_one_line():
    """A path split across lines is exactly the thing a reader needs whole."""
    text = (
        "The endpoint that creates a new pet owner is /owners/new. This can be found "
        "in the OwnerController.java file at line 72-75."
    )

    wrapped = wrap(text)

    assert "OwnerController.java" in wrapped
    assert all(len(line) <= MAX_WIDTH for line in wrapped.splitlines())


def test_wrap_does_not_split_on_hyphens():
    assert "72-75" in wrap("A citation at line 72-75 " + "padding " * 30)


def test_wrap_indents_every_line():
    wrapped = wrap("word " * 60, indent="  ")

    assert all(line.startswith("  ") for line in wrapped.splitlines() if line)


def test_wrap_preserves_blank_lines_between_paragraphs():
    wrapped = wrap("first paragraph\n\nsecond paragraph")

    assert wrapped.splitlines()[1] == ""
