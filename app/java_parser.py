from tree_sitter import Parser
from tree_sitter_languages import get_language

def get_java_parser():
    parser = Parser()
    parser.set_language(get_language("java"))
    return parser

def extract_java_entities(code):
    parser = get_java_parser()
    tree = parser.parse(bytes(code, "utf8"))

    root = tree.root_node
    results = []

    def traverse(node):
        # Class
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            name = code[name_node.start_byte:name_node.end_byte]

            results.append({
                "type": "class",
                "name": name,
                "text": code[node.start_byte:node.end_byte]
            })

        # Method
        if node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            name = code[name_node.start_byte:name_node.end_byte]

            results.append({
                "type": "method",
                "name": name,
                "text": code[node.start_byte:node.end_byte]
            })

        for child in node.children:
            traverse(child)

    traverse(root)
    return results

def extract_mapping(annotation_text):
    import re

    # Examples:
    # @PostMapping("/login")
    # @RequestMapping(value="/auth", method=RequestMethod.POST)

    path_match = re.search(r'\"(.*?)\"', annotation_text)
    path = path_match.group(1) if path_match else ""

    if "GetMapping" in annotation_text:
        method = "GET"
    elif "PostMapping" in annotation_text:
        method = "POST"
    elif "PutMapping" in annotation_text:
        method = "PUT"
    elif "DeleteMapping" in annotation_text:
        method = "DELETE"
    else:
        method = "UNKNOWN"

    return path, method

def extract_spring_entities(code):
    parser = get_java_parser()
    tree = parser.parse(bytes(code, "utf8"))

    root = tree.root_node
    results = []

    def get_annotations(node):
        annotations = []
        for child in node.children:
            if child.type == "modifiers":
                for sub in child.children:
                    if sub.type == "annotation":
                        annotations.append(
                            code[sub.start_byte:sub.end_byte]
                        )
        return annotations

    def traverse(node):
        if node.type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return

            name = code[name_node.start_byte:name_node.end_byte]

            annotations = get_annotations(node)  # ✅ defined once
            base_path = ""

            for ann in annotations:
                if "RequestMapping" in ann:
                    base_path, _ = extract_mapping(ann)

            results.append({
                "type": "class",
                "name": name,
                "base_path": base_path,
                "annotations": annotations,
                "text": code[node.start_byte:node.end_byte]
            })

        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if not name_node:
                return

            name = code[name_node.start_byte:name_node.end_byte]

            annotations = get_annotations(node)  # ✅ always defined

            endpoint = None
            http_method = None

            for ann in annotations:
                if "Mapping" in ann:
                    endpoint, http_method = extract_mapping(ann)

            results.append({
                "type": "method",
                "name": name,
                "annotations": annotations,
                "endpoint": endpoint,
                "http_method": http_method,
                "text": code[node.start_byte:node.end_byte]
            })

        for child in node.children:
            traverse(child)

    traverse(root)
    return results


