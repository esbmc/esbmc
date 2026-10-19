def half(a: int) -> float:
    return a / 2


def twice(a: int) -> float:
    return a * 2.0


app = lambda g, v: g(v)
assert app(half, 3) == 1.5
assert app(twice, 3) == 6.0
