import os

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

                documents.append({
                    "content": content,
                    "metadata": {
                        "file_name": file,
                        "path": path
                    }
                })

    return documents