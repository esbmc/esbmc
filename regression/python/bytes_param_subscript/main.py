def at(d: bytes, i: int) -> int:
    return d[i]


def is_p(d: bytes) -> int:
    if d[0] == b"P":
        return 1
    return 0


assert at(b"ab", 1) == 98
assert at(b"ab", -1) == 98
assert at(b"ab", -2) == 97
assert is_p(b"P") == 0
