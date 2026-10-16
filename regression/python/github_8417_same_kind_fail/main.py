def f(c: bool) -> int:
    a = 1
    if c:
        a = 2
    assert isinstance(a, float)
    return a


f(True)
