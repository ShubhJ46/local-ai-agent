import re

import tree_sitter_java
from tree_sitter import Language, Parser

from app.parsers import node_lines, node_text

JAVA = Language(tree_sitter_java.language())

# Spring repositories are interfaces, configuration is often an enum or record,
# and all of them are top-level declarations a developer navigates to. Indexing
# only class_declaration made them invisible.
CONTAINER_DECLARATIONS = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "record",
}

HTTP_METHOD_BY_ANNOTATION = {
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "DeleteMapping": "DELETE",
    "PatchMapping": "PATCH",
}


def get_java_parser():
    return Parser(JAVA)


def extract_mapping(annotation_text):
    """Return the (path, http_method) a Spring mapping annotation declares.

    Handles both the shorthand forms (``@GetMapping("/login")``) and the generic
    ``@RequestMapping(value="/auth", method=RequestMethod.POST)``.
    """
    path_match = re.search(r"\"(.*?)\"", annotation_text)
    path = path_match.group(1) if path_match else ""

    for annotation, http_method in HTTP_METHOD_BY_ANNOTATION.items():
        if annotation in annotation_text:
            return path, http_method

    request_method = re.search(r"RequestMethod\.(\w+)", annotation_text)
    if request_method:
        return path, request_method.group(1).upper()

    return path, "UNKNOWN"


def extract_java_entities(code):
    """Extract class and method declarations without Spring-specific metadata."""
    parser = get_java_parser()
    source = code.encode("utf-8")
    tree = parser.parse(source)
    results = []

    def traverse(node):
        if node.type in CONTAINER_DECLARATIONS or node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            is_container = node.type in CONTAINER_DECLARATIONS
            start_line, end_line = node_lines(node)
            results.append(
                {
                    "type": "class" if is_container else "method",
                    "name": node_text(source, name_node),
                    "text": node_text(source, node),
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )
            if is_container:
                return
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return results


def extract_spring_entities(code):
    """Extract classes and methods along with their Spring routing metadata."""
    parser = get_java_parser()
    source = code.encode("utf-8")
    tree = parser.parse(source)
    results = []

    def get_annotations(node):
        annotations = []
        for child in node.children:
            if child.type == "modifiers":
                for sub in child.children:
                    if sub.type in ("annotation", "marker_annotation"):
                        annotations.append(node_text(source, sub))
        return annotations

    def traverse(node):
        if node.type in CONTAINER_DECLARATIONS:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return

            annotations = get_annotations(node)
            base_path = ""
            for annotation in annotations:
                if "RequestMapping" in annotation:
                    base_path, _ = extract_mapping(annotation)

            start_line, end_line = node_lines(node)
            results.append(
                {
                    "type": "class",
                    "kind": CONTAINER_DECLARATIONS[node.type],
                    "name": node_text(source, name_node),
                    "base_path": base_path,
                    "annotations": annotations,
                    "text": node_text(source, node),
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )

        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return

            annotations = get_annotations(node)
            endpoint = None
            http_method = None
            for annotation in annotations:
                if "Mapping" in annotation:
                    endpoint, http_method = extract_mapping(annotation)

            start_line, end_line = node_lines(node)
            results.append(
                {
                    "type": "method",
                    "name": node_text(source, name_node),
                    "annotations": annotations,
                    "endpoint": endpoint,
                    "http_method": http_method,
                    "text": node_text(source, node),
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )

        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return results
