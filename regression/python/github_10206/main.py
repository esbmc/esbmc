# Issue #10206: a bytes parameter indexes like CPython bytes.
def at(d: bytes, i: int) -> int:
    try:
        return d[i]
    except IndexError:
        return -1


assert at(b"PQ", 0) == 80
assert at(b"PQ", -1) == 81
assert at(b"PQ", 2) == -1
assert at(b"PQ", -3) == -1
assert at(b"", 0) == -1
