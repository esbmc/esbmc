def f(x):
    return x


def h0():
    return f("s")


def h():
    z = 1
    return f(z)


assert h0() == "s"
assert h() != 1
