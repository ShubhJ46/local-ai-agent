import hashlib
from pathlib import Path

from app.parsers.code_parser import extract_python_functions
from app.parsers.java_parser import extract_spring_entities

SUPPORTED_EXTENSIONS = [".py", ".cpp", ".c", ".h", ".js", ".ts", ".java", ".md", ".txt"]

IGNORED_FOLDERS = {"venv", ".git", "__pycache__", "node_modules", ".pytest_cache", "qdrant_data"}


def load_documents(folder_path: str) -> list[dict]:
    root_path = Path(folder_path).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"Index path is not a directory: {root_path}")

    documents = []

    for path in sorted(root_path.rglob("*")):
        # Only inspect the path *below* the indexed root. Checking the absolute
        # path would skip every file whenever an ancestor happens to be hidden
        # (e.g. indexing a checkout under ~/.local/src), yielding a silently
        # empty index.
        relative_parts = path.relative_to(root_path).parts
        ignored = any(part in IGNORED_FOLDERS or part.startswith(".") for part in relative_parts)
        if not path.is_file() or ignored:
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = path.read_text(encoding="utf-8", errors="ignore")
        file = path.name
        path_string = str(path)
        # Hash the file, not the extracted unit: a class's own text is unchanged
        # by an edit elsewhere in the file, so hashing the unit would make
        # incremental indexing miss real changes.
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if path.suffix == ".java":
            entities = extract_spring_entities(content)
            class_entity = next((ent for ent in entities if ent["type"] == "class"), None)
            current_class = class_entity["name"] if class_entity else None
            base_path = class_entity.get("base_path", "") if class_entity else ""

            for entity in (ent for ent in entities if ent["type"] == "class"):
                documents.append(
                    {
                        "content": entity["text"],
                        "metadata": {
                            "file_name": file,
                            "path": path_string,
                            "file_hash": digest,
                            "type": "class",
                            "name": entity["name"],
                            "class": entity["name"],
                            "kind": entity.get("kind", "class"),
                            "annotations": entity.get("annotations", []),
                            "start_line": entity.get("start_line"),
                            "end_line": entity.get("end_line"),
                        },
                    }
                )

            for ent in entities:
                if ent["type"] != "method":
                    continue
                full_path = (base_path or "") + (ent.get("endpoint") or "")
                document_type = "endpoint" if ent.get("endpoint") else "method"
                documents.append(
                    {
                        "content": (
                            f"HTTP METHOD: {ent.get('http_method')}\nENDPOINT: {full_path}\n"
                            f"FUNCTION: {ent.get('name')}\nCLASS: {current_class}\n"
                            f"ANNOTATIONS: {' '.join(ent.get('annotations', []))}\n\n"
                            f"CODE:\n{ent['text']}"
                        ),
                        "metadata": {
                            "file_name": file,
                            "path": path_string,
                            "file_hash": digest,
                            "type": document_type,
                            "name": ent["name"],
                            "class": current_class,
                            "endpoint": full_path,
                            "http_method": ent.get("http_method"),
                            "annotations": ent.get("annotations", []),
                            "start_line": ent.get("start_line"),
                            "end_line": ent.get("end_line"),
                        },
                    }
                )
        elif path.suffix == ".py":
            for func in extract_python_functions(content):
                documents.append(
                    {
                        "content": func["text"],
                        "metadata": {
                            "file_name": file,
                            "path": path_string,
                            "file_hash": digest,
                            "type": "function",
                            "name": func["name"],
                            "start_line": func.get("start_line"),
                            "end_line": func.get("end_line"),
                        },
                    }
                )
        else:
            documents.append(
                {
                    "content": content,
                    "metadata": {
                        "file_name": file,
                        "path": path_string,
                        "file_hash": digest,
                        "type": "file",
                    },
                }
            )

    return documents
