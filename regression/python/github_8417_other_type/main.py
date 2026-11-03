def f(c: bool) -> None:
    a = 1
    if c:
        a = 2.5
    assert not isinstance(a, str)


f(True)
f(False)
