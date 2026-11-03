def f(x):
    return x


def h0():
    return f("s")


def h(y):

    def inner():
        return f(y)

    return inner()


assert h0() == "s"
assert h(1) != 1
