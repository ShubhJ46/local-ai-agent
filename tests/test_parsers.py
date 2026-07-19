from app.parsers.code_parser import extract_python_functions
from app.parsers.java_parser import extract_mapping, extract_spring_entities

JAVA_WITH_NON_ASCII = '''
@RestController
@RequestMapping("/owners/{ownerId}")
class PetController {

    /** Resolve the owner’s pets — note the non-ASCII punctuation above. */
    @GetMapping("/pets/new")
    public String initCreationForm() {
        return "form";
    }

    @PostMapping("/pets/new")
    public String processCreationForm() {
        return "saved";
    }
}
'''


def test_java_extraction_survives_non_ascii_characters():
    """Byte offsets must not be used to slice a str.

    A single multi-byte character used to shift every later extraction in the
    file, producing truncated names such as "cessCreationForm".
    """
    entities = extract_spring_entities(JAVA_WITH_NON_ASCII)

    names = [entity["name"] for entity in entities]
    assert names == ["PetController", "initCreationForm", "processCreationForm"]


def test_java_extraction_reports_endpoints_and_line_ranges():
    entities = extract_spring_entities(JAVA_WITH_NON_ASCII)
    methods = {entity["name"]: entity for entity in entities if entity["type"] == "method"}

    assert methods["initCreationForm"]["endpoint"] == "/pets/new"
    assert methods["initCreationForm"]["http_method"] == "GET"
    assert methods["processCreationForm"]["http_method"] == "POST"

    klass = next(entity for entity in entities if entity["type"] == "class")
    assert klass["base_path"] == "/owners/{ownerId}"
    assert klass["start_line"] < methods["initCreationForm"]["start_line"]
    for entity in entities:
        assert entity["start_line"] <= entity["end_line"]


def test_extract_mapping_reads_generic_request_mapping():
    assert extract_mapping('@RequestMapping(value="/auth", method=RequestMethod.POST)') == (
        "/auth",
        "POST",
    )
    assert extract_mapping('@GetMapping("/vets.html")') == ("/vets.html", "GET")


def test_python_extraction_reports_names_and_line_ranges():
    code = '# café comment with a multi-byte character\ndef useful(value):\n    return value\n'

    functions = extract_python_functions(code)

    assert [function["name"] for function in functions] == ["useful"]
    assert functions[0]["start_line"] == 2
    assert functions[0]["end_line"] == 3
    assert functions[0]["text"].startswith("def useful")
