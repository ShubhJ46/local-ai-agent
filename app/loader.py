import os

SUPPORTED_EXTENSIONS = [".txt", ".md", ".py", ".cpp", ".json", ".java"]

def load_documents(folder_path):
    documents = []

    for root, _, files in os.walk(folder_path):
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