def count(path: str) -> int:
    n = 0
    with open(path) as f:
        for line in f:
            n += 1
    return n


try:
    assert count("main.py") == 0
except OSError:
    pass
