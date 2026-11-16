def size(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            assert f.name == path
            assert not f.closed
            n = len(f.read())
        assert f.closed
        return n
    except OSError:
        return -1


assert size("missing.txt") >= -1
