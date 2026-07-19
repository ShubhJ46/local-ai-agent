from tree_sitter import Parser
from tree_sitter_languages import get_language

from app.parsers import node_lines, node_text


def get_parser():
    parser = Parser()
    parser.set_language(get_language("python"))
    return parser


def extract_python_functions(code):
    parser = get_parser()
    source = code.encode("utf-8")
    tree = parser.parse(source)
    functions = []

    def traverse(node):
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            start_line, end_line = node_lines(node)
            functions.append(
                {
                    "type": "function",
                    "name": node_text(source, name_node),
                    "text": node_text(source, node),
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return functions
