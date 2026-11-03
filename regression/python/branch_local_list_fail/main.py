def f(c: bool) -> int:
    if c:
        b = [1, 2]
    return b[0]


assert f(True) == 2
