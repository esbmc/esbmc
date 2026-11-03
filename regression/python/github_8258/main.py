def add(a: int, b: int, c: int = 10) -> int:
    return a + b + c


def total() -> int:
    t = (4, 5)
    n = 0
    for v in t:
        n += v
    assert len(t) == 2
    return add(*t) + n


xs = [1, 2]
assert add(*xs) == 13
assert add(*xs, 3) == 6
assert add(*[7, 8]) == 25
assert total() == 28
