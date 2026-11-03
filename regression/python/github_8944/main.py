n = 3
r = 2**n
assert isinstance(r, int)
assert r == 8


def f() -> None:
    k = 4
    s = 3**k
    assert isinstance(s, int)
    assert s == 81


f()
