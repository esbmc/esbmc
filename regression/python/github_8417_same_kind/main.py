def f(c: bool) -> int:
    a = 1
    if c:
        a = 2
    assert isinstance(a, int)
    return a


assert f(True) == 2
assert f(False) == 1
