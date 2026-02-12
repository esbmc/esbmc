def read_file(path: str) -> str:
    if not path:
        raise FileNotFoundError("No path provided")
    return "content"

try:
    read_file("")
except Exception as e:
    print("Caught:", e)
