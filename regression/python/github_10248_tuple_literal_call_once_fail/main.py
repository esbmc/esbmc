# GitHub #10248: a call in a tuple literal does not run again for the type probe.
calls = 0


def g() -> int:
    global calls
    calls += 1
    return calls


def f() -> int:
    x = (g(), g())
    return x[1]


assert f() == 4
