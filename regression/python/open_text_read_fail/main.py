def first(path: str) -> str:
    try:
        with open(path) as f:
            return f.readline()
    except OSError:
        return "x\n"


assert first("main.py") == "x\n"
