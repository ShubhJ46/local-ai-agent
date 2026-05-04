from tree_sitter import Parser
from tree_sitter_languages import get_language

def get_parser():
    parser = Parser()
    parser.set_language(get_language("python"))
    return parser

def extract_python_functions(code):
    parser = get_parser()
    tree = parser.parse(bytes(code, "utf8"))

    root = tree.root_node
    functions = []

    for node in root.children:
        if node.type == "function_definition":
            func_text = code[node.start_byte:node.end_byte]

            # get function name
            name_node = node.child_by_field_name("name")
            name = code[name_node.start_byte:name_node.end_byte]

            functions.append({
                "type": "function",
                "name": name,
                "text": func_text
            })

    return functions