# GitHub #10248: a call in a tuple literal runs once.
calls = 0


def g() -> int:
    global calls
    calls += 1
    return calls


def f() -> int:
    x = (g(), g())
    return x[1]


assert f() == 2
assert calls == 2
