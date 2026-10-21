f = lambda x, y: x * y


def g() -> None:
    n = 3
    r = f(n, 2)
    assert isinstance(r, int)
    assert r == 6


g()
s = f(1.5, 2)
assert isinstance(s, float)
assert s == 3.0
