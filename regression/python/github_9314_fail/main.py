def size(path: str) -> int:
    with open(path) as f:
        return len(f.read())


assert size("a.txt") == 2
