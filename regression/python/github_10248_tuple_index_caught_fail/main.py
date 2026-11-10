def f(i: int) -> int:
    t = (1, 2, 3)
    try:
        return t[i]
    except IndexError:
        return 0
assert f(-4) == 1
