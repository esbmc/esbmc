# The call through a function variable runs the chosen function once
# (#8351).
calls: int = 0


def f1(x: int) -> int:
    global calls
    calls = calls + 1
    return x


def f2(x: int) -> int:
    global calls
    calls = calls + 10
    return x + 1


if nondet_bool():
    g = f1
else:
    g = f2
r = g(5)
assert r == 5 or r == 6
assert calls == 1
