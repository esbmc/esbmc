def f(x):
    return x


def h0():
    return f("s")


def h(y: int):
    return f(y)


assert h0() == "s"
assert h(1) != 1
