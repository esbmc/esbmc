# #10309: a call in a default runs once, when the def runs.
n = 0


def k(x: int) -> int:
    global n
    n = n + 1
    return x * 2


def f(y: int = k(3)) -> int:
    return y


assert n == 1
assert f() == 6
assert f() == 6
assert f(1) == 1
assert n == 1
