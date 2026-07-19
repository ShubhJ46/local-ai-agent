from app.citations import cited_files, unverified_citations, verify

RETRIEVED = {"OwnerController.java", "VetRepository.java"}


def test_cited_files_finds_bare_and_backticked_names():
    text = "See OwnerController.java and `VetRepository.java` for details."

    assert cited_files(text) == {"OwnerController.java", "VetRepository.java"}


def test_cited_files_reduces_a_path_to_its_file_name():
    text = "src/main/java/org/springframework/samples/petclinic/owner/OwnerController.java"

    assert cited_files(text) == {"OwnerController.java"}


def test_version_numbers_are_not_mistaken_for_files():
    """`Spring 3.2` and `v1.5` must not be read as citations."""
    assert cited_files("Built with Spring 3.2 and tested on v1.5 of the API.") == set()


def test_an_answer_citing_only_retrieved_files_is_returned_unchanged():
    answer = "The handler lives in OwnerController.java."

    assert verify(answer, RETRIEVED) == answer


def test_a_path_qualified_citation_still_verifies():
    answer = "See owner/OwnerController.java for the mapping."

    assert verify(answer, RETRIEVED) == answer


def test_an_invented_citation_is_flagged_without_deleting_the_answer():
    """Regression: mistral answered with a confident, entirely invented path."""
    answer = "The endpoint can be found in the api/pets/owners.java file."

    verified = verify(answer, RETRIEVED)

    assert answer in verified, "the original answer must remain visible"
    assert "unverified" in verified
    assert "owners.java" in verified
    assert "OwnerController.java" in verified, "the real sources should be named"


def test_unverified_citations_are_reported_in_a_stable_order():
    answer = "Look at zeta.java, then alpha.java."

    assert unverified_citations(answer, RETRIEVED) == ["alpha.java", "zeta.java"]


def test_verify_reports_when_nothing_was_retrieved():
    assert "Retrieved: nothing." in verify("It is in Ghost.java.", set())
