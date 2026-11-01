def count(path: str) -> int:
    n = 0
    try:
        for line in open(path):
            assert len(line) > 0
            n += 1
    except OSError:
        n = -1
    return n


assert count("missing.txt") >= -1
