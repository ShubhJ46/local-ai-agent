"""Shared helpers for tree-sitter based extractors."""


def node_text(source: bytes, node) -> str:
    """Return the source text a tree-sitter node covers.

    Tree-sitter reports byte offsets. Slicing a ``str`` with them corrupts every
    extraction after the first non-ASCII character in a file, so always slice the
    encoded source and decode the result.
    """
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def node_lines(node) -> tuple[int, int]:
    """Return the 1-indexed inclusive line range a node spans."""
    return node.start_point[0] + 1, node.end_point[0] + 1
