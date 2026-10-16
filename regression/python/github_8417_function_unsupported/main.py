def f(c: bool) -> None:
    a = 1
    if c:
        a = 2.5
    if isinstance(a, float):
        assert a == 2.5
    else:
        assert a == 1


f(True)
f(False)
