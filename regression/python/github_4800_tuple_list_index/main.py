def first(s: str) -> int:
    h = [(0, s)]
    for t in h:
        return t[0]
    return 1


def second(s: str) -> str:
    h = [(0, s), (1, s)]
    i: int = 1
    t = h[i]
    return t[1]


def unpack(s: str) -> int:
    h = [(3, s)]
    for d, n in h:
        return d
    return 0


assert first("x") == 0
assert second("abc") == "abc"
assert unpack("x") == 3
