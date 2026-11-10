def f(i: int) -> int:
    t = (1, 2, 3)
    return t[i]
assert f(2) == 3
assert f(-3) == 1
