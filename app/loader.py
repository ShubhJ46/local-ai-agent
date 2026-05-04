import os
from app.code_parser import extract_python_functions
from app.java_parser import extract_spring_entities


SUPPORTED_EXTENSIONS = [
    ".py", ".cpp", ".c", ".h",
    ".js", ".ts", ".java",
    ".md", ".txt"
]

IGNORED_FOLDERS = ["venv", ".git", "__pycache__"]


def load_documents(folder_path):
    documents = []

    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [ d for d in dirs if d not in IGNORED_FOLDERS and not d.startswith(".")]
        for file in files:
            if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                path = os.path.join(root, file)
                
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                
                if file.endswith(".java"):
                    entities = extract_spring_entities(content)

                    current_class = None
                    base_path = ""

                    for ent in entities:
                        if ent["type"] == "class":
                            current_class = ent["name"]
                            base_path = ent.get("base_path", "")

                        elif ent["type"] == "method":
                            full_path = (base_path or "") + (ent.get("endpoint") or "")

                            documents.append({
                                "content": f"""
                                HTTP METHOD: {ent.get("http_method")}
                                ENDPOINT: {full_path}
                                FUNCTION: {ent.get("name")}
                                CLASS: {current_class}

                                ANNOTATIONS: {' '.join(ent.get("annotations", []))}

                                CODE:
                                {ent['text']}
                                """,
                                "metadata": {
                                    "file_name": file,
                                    "path": path,
                                    "type": "endpoint",
                                    "name": ent["name"],
                                    "class": current_class,
                                    "endpoint": full_path,
                                    "http_method": ent.get("http_method"),
                                    "annotations": ent.get("annotations", [])
                                }
                            })

                elif file.endswith(".py"):
                    functions = extract_python_functions(content)

                    for func in functions:
                        documents.append({
                            "content": func["text"],
                            "metadata": {
                                "file_name": file,
                                "path": path,
                                "type": "function",
                                "name": func["name"]
                            }
                        })

                else:
                    documents.append({
                        "content": content,
                        "metadata": {
                            "file_name": file,
                            "path": path,
                            "type": "file"
                        }
                    })

    return documents