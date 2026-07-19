from app.lexical import BM25Index, tokenize


def test_tokenize_splits_compound_identifiers():
    tokens = tokenize("processCreationForm")

    assert "processcreationform" in tokens, "the whole symbol must stay searchable"
    assert {"process", "creation", "form"} <= set(tokens)


def test_tokenize_handles_snake_case_and_acronyms():
    assert {"find", "by", "last", "name"} <= set(tokenize("find_by_last_name"))
    assert {"http", "client"} <= set(tokenize("HTTPClient"))


def test_bm25_ranks_the_document_containing_the_query_term_first():
    index = BM25Index(
        [
            ("a", "class OwnerRepository extends JpaRepository"),
            ("b", "class VetController renders the vet list"),
            ("c", "unrelated configuration for caching"),
        ]
    )

    assert index.search("owner repository")[0][0] == "a"


def test_bm25_matches_words_inside_a_compound_symbol():
    index = BM25Index(
        [("a", "public String processCreationForm()"), ("b", "public String showOwner()")]
    )

    assert index.search("creation form")[0][0] == "a"


def test_bm25_returns_nothing_when_no_term_matches():
    index = BM25Index([("a", "alpha beta"), ("b", "gamma delta")])

    assert index.search("epsilon") == []


def test_bm25_handles_an_empty_corpus():
    assert BM25Index([]).search("anything") == []
