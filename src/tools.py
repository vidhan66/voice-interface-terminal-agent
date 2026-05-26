from pathlib import Path
def list_files():
    files = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        path_str = str(path)

        ignored = [
            "__pycache__",
            ".git",
            ".venv",
            "node_modules"
        ]

        if any(x in path_str for x in ignored):
            continue
        files.append(path_str)
    return files

def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        return str(e)