def is_a(v, h) -> bool:
    return isinstance(v, h)


def f(v) -> bool:
    t = int
    return isinstance(v, t)


assert is_a(1, int)
assert f(1)
assert is_a("a", int)
