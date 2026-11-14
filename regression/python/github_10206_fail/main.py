# Issue #10206: indexing an empty bytes parameter raises IndexError.
def f(d: bytes) -> int:
    if d[0] == b"P":
        return 1
    return 0


f(b"")
